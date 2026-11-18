#!/usr/bin/env python3
"""Train YOLOv5 + DetectKpt (8 keypoints + obj + 4 colors + NC classes) for RoboMaster armors.

Pipeline: utils/kpt (mosaic/affine/perspective/HSV/motion-blur with exact keypoint transforms, keypoint-aware
multi-positive assignment, quality-aware obj + point/OKS/box/class/color/geometry losses, 27-style validation) with
YOLOv5's optimizer groups, nbs-64 accumulation, warmup, cosine LR, EMA, AMP, gradient clipping, early stopping.

Usage:
    python train_kpt.py --model yolov5s --weights yolov5s.pt --epochs 150 --batch 8 --name kpt_s
    python train_kpt.py --resume runs/kpt_s/last.pt
Outputs runs/<name>/: best.pt (EMA, best val fitness), last.pt (resumable), metrics.csv, train.log, opt.yaml,
pred/{val,test}/ (green GT, red predictions >= 0.7) and final_metrics.json.
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

import val_kpt
from models.yolo import Model
from utils.general import LOGGER, init_seeds, intersect_dicts, one_cycle
from utils.kpt.dataloaders import create_dataloader
from utils.kpt.loss import ComputeKptLoss
from utils.torch_utils import EarlyStopping, ModelEMA, smart_optimizer

DATA = "datasets/data.yaml"


def parse_opt():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov5s", help="yolov5s | yolov5n -> models/<model>-kpt22.yaml, or a yaml path")
    ap.add_argument("--weights", default="yolov5s.pt", help="official YOLOv5 .pt (backbone/neck) | our ckpt | none")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--hyp", default="data/hyps/hyp.kpt22.yaml")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nbs", type=int, default=64, help="nominal batch size (gradient accumulation)")
    ap.add_argument("--close-mosaic", type=int, default=15, help="disable mosaic for the last N epochs")
    ap.add_argument("--patience", type=int, default=40, help="early-stop epochs without val fitness improvement")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", action="store_true", help="cache resized images in RAM (~5 GB for large_split)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="kpt_s")
    ap.add_argument("--resume", default="", help="runs/<name>/last.pt")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: limit batches per epoch")
    return ap.parse_args()


def load_pretrained(model, weights):
    """Official YOLOv5 checkpoint (pickled model) or our ckpt (state dict); head kept only if shapes match."""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = sd.float().state_dict() if hasattr(sd, "state_dict") else sd
    own = model.state_dict()
    exclude = [] if "yaml" in ck and ck["yaml"].get("kpt_decode") else ["model.24."]  # new head on transfer
    csd = intersect_dicts(sd, own, exclude=exclude)
    model.load_state_dict(csd, strict=False)
    LOGGER.info(f"weights: transferred {len(csd)}/{len(own)} tensors from {weights}")


def save_ckpt(path, model, ema, extra):
    """Write <path>.tmp then rename: stopping the process mid-save never corrupts the previous checkpoint."""
    tmp = Path(f"{path}.tmp")
    torch.save({"model": deepcopy(ema.ema).float().state_dict(), "yaml": model.yaml, **extra}, tmp)
    os.replace(tmp, path)


def main(opt):
    if opt.resume:
        ck = torch.load(opt.resume, map_location="cpu", weights_only=False)
        resume_path, opt = opt.resume, argparse.Namespace(**ck["opt"])
        opt.resume = resume_path
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(save_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOGGER.addHandler(fh)
    LOGGER.setLevel(logging.INFO)
    hyp = yaml.safe_load(open(opt.hyp))
    data = yaml.safe_load(open(opt.data))
    root, nc, names = Path(data["path"]), int(data["nc"]), data["names"]
    assert len(names) == nc, "data.yaml names / nc mismatch"
    (save_dir / "opt.yaml").write_text(yaml.safe_dump({**vars(opt), "hyp": hyp}, sort_keys=False))
    LOGGER.info(f"opt: {vars(opt)}\nhyp: {hyp}")
    init_seeds(opt.seed)
    device = torch.device(opt.device)

    cfg = opt.model if opt.model.endswith(".yaml") else f"models/{opt.model}-kpt22.yaml"
    model = Model(cfg, ch=3, nc=nc)
    if not opt.resume and opt.weights.lower() != "none":
        load_pretrained(model, opt.weights)
    model = model.to(device)
    model.hyp, model.nc, model.names = hyp, nc, names

    train_loader, train_ds = create_dataloader(
        root / data["train"], root / "labels", opt.batch, hyp, augment=True, cache=opt.cache, workers=opt.workers,
        shuffle=True, seed=opt.seed, prefix="train: "
    )
    val_loader, _ = create_dataloader(root / data["val"], root / "labels", opt.batch * 2, cache=opt.cache, workers=opt.workers, prefix="val: ")
    nb = len(train_loader) if not opt.max_batches else min(opt.max_batches, len(train_loader))

    accumulate = max(round(opt.nbs / opt.batch), 1)
    hyp["weight_decay"] *= opt.batch * accumulate / opt.nbs
    optimizer = smart_optimizer(model, "SGD", hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    lf = one_cycle(1, hyp["lrf"], opt.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper, best_fitness, start_epoch = EarlyStopping(patience=opt.patience), -1.0, 0
    if opt.resume:
        model.load_state_dict(ck["train_model"])
        ema.ema.load_state_dict(ck["model"])
        ema.updates = ck["ema_updates"]
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_fitness = ck["epoch"] + 1, ck["best_fitness"]
        stopper.best_fitness, stopper.best_epoch = best_fitness, ck.get("best_epoch", ck["epoch"])
        LOGGER.info(f"resumed from {opt.resume}: epoch {start_epoch}, best fitness {best_fitness:.4f}")
        del ck
    scheduler.last_epoch = start_epoch - 1
    compute_loss = ComputeKptLoss(model)
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    last_opt_step = -1
    csv_path = save_dir / "metrics.csv"
    LOGGER.info(f"train {len(train_ds)} images, {nb} batches/epoch, accumulate {accumulate}, epochs {opt.epochs}")

    for epoch in range(start_epoch, opt.epochs):
        t0 = time.time()
        model.train()
        train_ds.mosaic = epoch < opt.epochs - opt.close_mosaic and hyp["mosaic"] > 0
        mloss = torch.zeros(len(compute_loss.names), device=device)
        optimizer.zero_grad()
        for i, (imgs, targets, _, _) in enumerate(train_loader):
            if i >= nb:
                break
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255
            if ni <= nw:  # warmup: bias lr 0.1 -> lr0 * lf, other groups 0 -> lr0 * lf, momentum 0.8 -> 0.937
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, opt.nbs / opt.batch]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(imgs)
            loss, items = compute_loss(pred, targets.to(device))
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                last_opt_step = ni
            mloss = (mloss * i + items) / (i + 1)
            if i % 100 == 0:
                LOGGER.info(
                    f"epoch {epoch + 1}/{opt.epochs} batch {i}/{nb} loss {loss.item() / opt.batch:.3f} "
                    + " ".join(f"{n} {v:.4f}" for n, v in zip(compute_loss.names, mloss.tolist()))
                    + f" mem {torch.cuda.memory_reserved() / 1e9 if device.type == 'cuda' else 0:.1f}G"
                )
        lr = [x["lr"] for x in optimizer.param_groups]
        scheduler.step()

        m = val_kpt.run(ema.ema, val_loader, nc)
        fit = m["fitness"]
        improved = fit > best_fitness
        best_fitness = max(best_fitness, fit)
        row = {"epoch": epoch + 1, **dict(zip(compute_loss.names, mloss.tolist())), "lr0": lr[0], "lr2": lr[2]}
        row.update({k: m[k] for k in m if not k.startswith(("TP@", "FP@", "FN@", "n_loc@")) and "_AP50_c" not in k})
        row["time_s"] = time.time() - t0
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
        LOGGER.info(
            f"epoch {epoch + 1} done {row['time_s']:.0f}s | val box mAP50 {m['box_mAP50']:.3f} mAP50-95 "
            f"{m['box_mAP50_95']:.3f} pose mAP50-95 {m['pose_mAP50_95']:.3f} | @0.7 P {m['P@0.7']:.3f} R {m['R@0.7']:.3f} "
            f"F1 {m['F1@0.7']:.3f} cls {m['cls_acc@0.7']:.3f} color {m['color_acc@0.7']:.3f} px med "
            f"{m['px_median@0.7']:.2f} | fitness {fit:.4f}{' *' if improved else ''}"
        )
        extra = {"epoch": epoch, "best_fitness": best_fitness, "val": m, "names": names, "opt": vars(opt), "hyp": hyp}
        if improved:
            save_ckpt(save_dir / "best.pt", model, ema, extra)
        resume = {"train_model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
        save_ckpt(save_dir / "last.pt", model, ema, {**extra, **resume, "ema_updates": ema.updates, "best_epoch": stopper.best_epoch})
        if stopper(epoch=epoch, fitness=fit):
            break

    # final: best.pt on val (model selection set) and test (clean200 40, reported only), with prediction images
    best, _ = val_kpt.load_model(save_dir / "best.pt", device)
    final = {}
    for split in ("val", "test"):
        loader, _ = create_dataloader(root / data[split], root / "labels", opt.batch * 2, workers=opt.workers, prefix=f"{split}: ")
        final[split] = val_kpt.run(best, loader, nc, names, save_dir=save_dir / "pred" / split)
        LOGGER.info(val_kpt.summary(final[split], f"best.pt on {split} ({len(loader.dataset)} images)"))
    (save_dir / "final_metrics.json").write_text(json.dumps(final, indent=1))
    LOGGER.info(f"done: {save_dir}")


if __name__ == "__main__":
    main(parse_opt())
