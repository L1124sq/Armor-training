#!/usr/bin/env python3
"""Validate a DetectKpt checkpoint with 27's preprocessing / post-processing (see utils/kpt/metrics.py for metrics).

Usage:
    python val_kpt.py --weights runs/kpt_s/best.pt --split val   # large_split val (579, model selection)
    python val_kpt.py --weights runs/kpt_s/best.pt --split test         # clean200 40 (final report only)
    python val_kpt.py --weights runs/kpt_large3/best.pt --split test    # legacy baseline through the same metrics
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from models.yolo import Model
from utils.kpt.dataloaders import create_dataloader
from utils.kpt.metrics import KptMetrics, legacy_index, postprocess

DATA = "datasets/data.yaml"
COLORS = ("blue", "red", "dim", "purple")
DEPLOY_CONF = 0.7  # 27 yolov5.hpp score_threshold_


def load_model(weights, device):
    """train_kpt.py or legacy (runs/kpt_large*) checkpoint -> fp32 eval model built from its embedded yaml."""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    model = Model(ck["yaml"], ch=3, nc=ck["yaml"]["nc"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


class OVModel:
    """OpenVINO IR wrapped like 27 yolov5.cpp (u8 NHWC BGR in, PrePostProcessor -> f32 RGB / 255), torch-like call."""

    def __init__(self, xml, torch_model, precision=""):
        import openvino as ov
        from openvino.preprocess import ColorFormat, PrePostProcessor

        core = ov.Core()
        ppp = PrePostProcessor(core.read_model(str(xml)))
        tensor = ppp.input().tensor().set_element_type(ov.Type.u8).set_shape([1, 640, 640, 3])
        tensor.set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.BGR)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).scale(255.0)
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if precision:
            cfg["INFERENCE_PRECISION_HINT"] = precision
        self.net = core.compile_model(ppp.build(), "CPU", cfg)
        self.precision = self.net.get_property("INFERENCE_PRECISION_HINT")
        self.model, self.parameters = torch_model.model, torch_model.parameters  # head metadata for run()

    def __call__(self, x):
        bgr = (x * 255).round().byte().permute(0, 2, 3, 1).flip(-1).cpu().numpy()
        out = [self.net({0: np.ascontiguousarray(im[None])})[self.net.output(0)][0] for im in bgr]
        return torch.from_numpy(np.stack(out))


def draw(img, det, gt, names, conf=DEPLOY_CONF):
    """Green = GT (dot on TL), red = predictions with score >= conf, labelled 'class color score'."""
    im = np.ascontiguousarray(img)
    for r in gt:
        q = r[3:11].reshape(4, 2).round().astype(np.int32)
        cv2.polylines(im, [q], True, (0, 255, 0), 1)
        cv2.circle(im, tuple(int(v) for v in q[0]), 3, (0, 255, 0), -1)
    for d in det[det[:, 8] >= conf]:
        q = d[:8].reshape(4, 2).round().astype(np.int32)
        cv2.polylines(im, [q], True, (0, 0, 255), 1)
        cv2.circle(im, tuple(int(v) for v in q[0]), 2, (0, 0, 255), -1)
        txt = f"{names[int(d[9])]} {COLORS[int(d[10])]} {d[8]:.2f}"
        cv2.putText(im, txt, (int(q[:, 0].min()), max(int(q[:, 1].min()) - 3, 8)), 0, 0.4, (0, 0, 255), 1)
    return im


@torch.no_grad()
def run(model, loader, nc, names=None, conf_thres=0.001, nms_iou=0.3, save_dir=None, max_save=40, n_fail=10):
    """Evaluate on a loader from utils.kpt.dataloaders.create_dataloader -> metrics dict."""
    device = next(model.parameters()).device
    head = model.model[-1]
    anchors_px = (head.anchors * head.stride.view(-1, 1, 1)).tolist()
    metrics, keep = KptMetrics(nc), {}
    for imgs, targets, paths, _ in loader:
        out = model(imgs.to(device, non_blocking=True).float() / 255)
        for si, det in enumerate(postprocess(out, nc, conf_thres, nms_iou)):
            gt = targets[targets[:, 0] == si, 1:]
            rows = None
            if len(gt):
                rows = out[si, legacy_index(gt[:, 3:11].view(-1, 4, 2), anchors_px, head.stride.tolist(), head.na)]
            metrics.update(det, gt, paths[si], rows)
            if save_dir:
                keep[paths[si]] = (det.cpu().numpy(), gt.numpy())
    if save_dir:
        names = names or [str(i) for i in range(nc)]
        ds = loader.dataset
        worst = [p for e, p in sorted(metrics.per_image, key=lambda x: -x[0])[:n_fail] if e > 0]
        for sub, lst in (("", list(keep)[:max_save]), ("failures", worst)):
            d = Path(save_dir) / sub
            d.mkdir(parents=True, exist_ok=True)
            for p in lst:
                img, _ = ds.load_eval(ds.im_files.index(p))
                cv2.imwrite(str(d / Path(p).name), draw(img.copy(), *keep[p], names))
    return metrics.compute()


def summary(m, title=""):
    """Readable multi-line summary of the main metrics."""
    s = [f"== {title}"] if title else []
    s.append(
        f"box  mAP50 {m['box_mAP50']:.3f}  mAP50-95 {m['box_mAP50_95']:.3f}  (class-agnostic AP50 {m['armor_AP50_agnostic']:.3f})"
    )
    s.append("box AP50 per class: " + "  ".join(f"{k[10:]} {v:.3f}" for k, v in m.items() if k.startswith("box_AP50_c")))
    s.append(f"pose mAP50 {m['pose_mAP50']:.3f}  mAP50-95 {m['pose_mAP50_95']:.3f}  P {m['pose_P']:.3f}  R {m['pose_R']:.3f}")
    for c in (0.5, 0.6, 0.7, 0.8):
        s.append(
            f"conf {c}: P {m[f'P@{c}']:.3f} R {m[f'R@{c}']:.3f} F1 {m[f'F1@{c}']:.3f} "
            f"(TP {m[f'TP@{c}']:.0f} FP {m[f'FP@{c}']:.0f} FN {m[f'FN@{c}']:.0f})  cls {m[f'cls_acc@{c}']:.3f} "
            f"color {m[f'color_acc@{c}']:.3f}  px median {m[f'px_median@{c}']:.2f} mean {m[f'px_mean@{c}']:.2f} "
            f"p90 {m[f'px_p90@{c}']:.2f} max {m[f'px_max@{c}']:.2f}"
        )
    if "legacy_n" in m:
        s.append(
            f"legacy GT-matched (center-anchor row): px median {m['legacy_px_median']:.2f} mean {m['legacy_px_mean']:.2f} "
            f"max {m['legacy_px_max']:.2f}  cls {m['legacy_cls_acc']:.3f}  color {m['legacy_color_acc']:.3f}  n {m['legacy_n']:.0f}"
        )
    s.append(f"fitness {m['fitness']:.4f}")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", default="val", help="val | test | train (key in data.yaml)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001, help="pre-NMS score threshold for mAP")
    ap.add_argument("--nms-iou", type=float, default=0.3, help="27 yolov5.hpp nms_threshold_")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="", help="prediction images (default <weights dir>/pred/<split>)")
    ap.add_argument("--max-save", type=int, default=40)
    ap.add_argument("--ir", default="", help="evaluate this OpenVINO IR (.xml) as 27 runs it, on CPU")
    ap.add_argument("--ir-precision", default="", help="f32 | bf16 | empty = CPU default (what 27 gets)")
    args = ap.parse_args()

    data = yaml.safe_load(open(args.data))
    root = Path(data["path"])
    model, ck = load_model(args.weights, "cpu" if args.ir else args.device)
    if args.ir:
        model = OVModel(args.ir, model, args.ir_precision)
        print(f"OpenVINO IR {args.ir}, CPU inference precision {model.precision}")
    nc = model.model[-1].nc
    loader, _ = create_dataloader(root / data[args.split], root / "labels", args.batch, workers=8, prefix=f"{args.split}: ")
    save_dir = args.save_dir or Path(args.weights).parent / "pred" / args.split
    m = run(model, loader, nc, data.get("names"), args.conf, args.nms_iou, save_dir, args.max_save)
    print(summary(m, f"{args.weights} on {args.split} ({len(loader.dataset)} images)"))
    out = Path(save_dir) / "metrics.json"
    out.write_text(json.dumps(m, indent=1))
    print("saved:", save_dir, out)


if __name__ == "__main__":
    main()
