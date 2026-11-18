# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Armor keypoint dataset: 4 corners per object; mosaic / affine / perspective transform the points exactly.

Label file per image (normalized to the original image, points TL->BL->BR->TR):
    cls color x1 y1 x2 y2 x3 y3 x4 y4 [flag] flag 1 = vertex order not verified (optional column)
Collated targets (pixels of the 640 network input): [img_idx, cls, color, flag, x1, y1, ..., x4, y4].

Validation/test images use exactly 27's yolov5.cpp preprocessing: scale = min(640 / h, 640 / w), resize to
(int(w * scale), int(h * scale)) with INTER_LINEAR, paste top-left on black. Training mosaics use black fill too.
No flips: mirrored armor digits are not valid armors.
"""

import math
import os
import random
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..augmentations import augment_hsv
from ..dataloaders import PIN_MEMORY, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, NUM_THREADS

IMG_SIZE = 640


def load_label(path):
    """Label file -> float32 (n, 11) [cls, color, flag, x1, y1, ..., x4, y4] (normalized)."""
    rows = [x.split() for x in open(path)] if os.path.exists(path) else []
    out = [[float(a[0]), float(a[1]), float(a[10]) if len(a) > 10 else 0.0, *map(float, a[2:10])] for a in rows if len(a) >= 10]
    return np.array(out, np.float32).reshape(-1, 11)


def letterbox27(im0):
    """27 yolov5.cpp preprocessing (top-left paste, black padding) -> (640x640 BGR, scale)."""
    h0, w0 = im0.shape[:2]
    scale = min(IMG_SIZE / h0, IMG_SIZE / w0)
    w, h = int(w0 * scale), int(h0 * scale)
    out = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    out[:h, :w] = cv2.resize(im0, (w, h))
    return out, scale


def visible_fraction(labels, w, h):
    """Fraction of each object's point bounding box inside [0, w] x [0, h]."""
    xs, ys = labels[:, 3:11:2], labels[:, 4:11:2]
    x0, x1, y0, y1 = xs.min(1), xs.max(1), ys.min(1), ys.max(1)
    area = np.maximum(x1 - x0, 1e-3) * np.maximum(y1 - y0, 1e-3)
    cw = np.clip(np.minimum(x1, w) - np.maximum(x0, 0), 0, None)
    ch = np.clip(np.minimum(y1, h) - np.maximum(y0, 0), 0, None)
    return cw * ch / area


def random_perspective_kpt(im, labels, degrees=0, translate=0.1, scale=0.5, shear=0, perspective=0.0, border=(0, 0)):
    """utils.augmentations.random_perspective for 4-point labels (pixels), black border.

    Points are transformed exactly (not clipped). Objects keep only if >= 60% of their box stays in view and the
    box is > 2 px on both sides; partially visible points may lie slightly outside the image.
    """
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=random.uniform(-degrees, degrees), center=(0, 0), scale=random.uniform(1 - scale, 1 + scale))
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(0, 0, 0))
        else:
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

    if n := len(labels):
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, 3:11].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
        labels = labels.copy()
        labels[:, 3:11] = xy.reshape(n, 8)
        xs, ys = labels[:, 3:11:2], labels[:, 4:11:2]
        keep = (visible_fraction(labels, width, height) >= 0.6) & (np.ptp(xs, 1) > 2) & (np.ptp(ys, 1) > 2)
        labels = labels[keep]
    return im, labels


def motion_blur(im, p):
    """Linear motion blur (kernel 3-9 px, random direction) with probability p, in place on BGR uint8."""
    if random.random() < p:
        k = random.choice((3, 5, 7, 9))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1.0
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), random.uniform(0, 180), 1.0), (k, k))
        im[:] = cv2.filter2D(im, -1, kernel / max(kernel.sum(), 1e-6))


class LoadArmorKpts(LoadImagesAndLabels):
    """LoadImagesAndLabels for 4-point armor labels; reuses its load_image (RAM cache), mosaic layout, collate_fn."""

    def __init__(self, list_file, label_dir, augment=False, hyp=None, cache=False, prefix=""):
        """Read an image list (one path per line) and labels from label_dir/<stem>.txt; no YOLO-format cache."""
        self.img_size = IMG_SIZE
        self.augment = augment
        self.hyp = hyp
        self.rect = False
        self.mosaic = augment and hyp["mosaic"] > 0
        self.mosaic_border = [-IMG_SIZE // 2, -IMG_SIZE // 2]
        self.im_files = [x.strip() for x in open(list_file) if x.strip()]
        self.labels = [load_label(os.path.join(label_dir, Path(f).stem + ".txt")) for f in self.im_files]
        self.n = len(self.im_files)
        self.indices = np.arange(self.n)
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.ims, self.im_hw0, self.im_hw = [None] * self.n, [None] * self.n, [None] * self.n
        self.eval_ims = [None] * self.n
        nl = sum(len(x) for x in self.labels)
        LOGGER.info(f"{prefix}{self.n} images, {nl} objects, {sum(int(x[:, 2].sum()) for x in self.labels)} order-uncertain")
        if cache:
            fcn = self.load_image if augment else self.load_eval
            with ThreadPool(NUM_THREADS) as pool:
                for i, x in enumerate(pool.imap(fcn, range(self.n))):
                    if augment:
                        self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    else:
                        self.eval_ims[i] = x
            gb = sum(x.nbytes for x in (self.ims if augment else (e[0] for e in self.eval_ims))) / 1e9
            LOGGER.info(f"{prefix}cached {self.n} images in RAM ({gb:.1f} GB)")

    def load_eval(self, i):
        """27-style letterboxed image and its labels in 640 pixels."""
        if self.eval_ims[i] is not None:
            return self.eval_ims[i]
        im0 = cv2.imread(self.im_files[i])
        assert im0 is not None, f"Image Not Found {self.im_files[i]}"
        img, scale = letterbox27(im0)
        labels = self.labels[i].copy()
        labels[:, 3:11:2] *= im0.shape[1] * scale
        labels[:, 4:11:2] *= im0.shape[0] * scale
        return img, labels

    def load_mosaic(self, index):
        """LoadImagesAndLabels.load_mosaic with 4-point labels and black fill."""
        labels4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index, *random.choices(self.indices, k=3)]
        random.shuffle(indices)
        img4 = np.zeros((s * 2, s * 2, 3), np.uint8)
        for i, mi in enumerate(indices):
            img, _, (h, w) = self.load_image(mi)
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            labels = self.labels[mi].copy()
            labels[:, 3:11:2] = labels[:, 3:11:2] * w + (x1a - x1b)
            labels[:, 4:11:2] = labels[:, 4:11:2] * h + (y1a - y1b)
            labels4.append(labels)
        labels4 = np.concatenate(labels4, 0)
        labels4 = labels4[visible_fraction(labels4, 2 * s, 2 * s) >= 0.6]  # drop objects cut by the canvas edge
        hyp = self.hyp
        return random_perspective_kpt(
            img4, labels4, hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["perspective"], self.mosaic_border
        )

    def __getitem__(self, index):
        """Returns (CHW RGB uint8 tensor, targets (n, 12) [0, cls, color, flag, 8 pts px], path, None)."""
        index = self.indices[index]
        hyp = self.hyp
        if self.mosaic and random.random() < hyp["mosaic"]:
            img, labels = self.load_mosaic(index)
        elif self.augment:  # single image, 27-style top-left placement, then affine/perspective
            im, _, (h, w) = self.load_image(index)
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            img[:h, :w] = im
            labels = self.labels[index].copy()
            labels[:, 3:11:2] *= w
            labels[:, 4:11:2] *= h
            img, labels = random_perspective_kpt(
                img, labels, hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["perspective"]
            )
        else:
            img, labels = self.load_eval(index)
            img = img.copy()
        if self.augment:
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
            motion_blur(img, hyp["blur"])
        labels_out = torch.zeros((len(labels), 12))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])  # HWC to CHW, BGR to RGB
        return torch.from_numpy(img), labels_out, self.im_files[index], None


def create_dataloader(list_file, label_dir, batch_size, hyp=None, augment=False, cache=False, workers=8, shuffle=False, seed=0, prefix=""):
    """Plain DataLoader (workers re-forked each epoch, so toggling dataset.mosaic for close-mosaic takes effect)."""
    dataset = LoadArmorKpts(list_file, label_dir, augment=augment, hyp=hyp, cache=cache, prefix=prefix)
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, workers)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=PIN_MEMORY,
        collate_fn=LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, dataset
