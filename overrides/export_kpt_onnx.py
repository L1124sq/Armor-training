#!/usr/bin/env python3
"""Export a DetectKpt checkpoint to ONNX: images [1, 3, 640, 640] f32 RGB/255 -> [1, 25200, 13 + NC] f32.

Usage:
    python export_kpt_onnx.py --weights runs/kpt_s/best.pt --out runs/kpt_s/kpt_s.onnx
"""

import argparse
from pathlib import Path

import onnx
import torch

from val_kpt import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="", help="default: <weights>.onnx")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.out or Path(args.weights).with_suffix(".onnx"))
    model, _ = load_model(args.weights, "cpu")
    nc = model.model[-1].nc
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 640, 640),
        str(out),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["output"],
        do_constant_folding=True,
        dynamo=False,
    )
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    dims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    assert dims == [1, 25200, 13 + nc], f"output dims {dims}, want [1, 25200, {13 + nc}]"
    print(f"saved {out}  output {dims}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("Next: convert to OpenVINO IR (2024.6) with:")
    print(f"  source /opt/intel/openvino_2024.6.0/setupvars.sh")
    print(f"  python3 -c \"import openvino as ov; ov.save_model(ov.convert_model('{out}'), '{out.with_suffix('.xml')}', compress_to_fp16=False)\"")


if __name__ == "__main__":
    main()
