#!/usr/bin/env python
"""
INT8 YOLOP for the CPU-only study: static post-training quantization with
onnxruntime, calibrated on real Bench2Drive front images (the agent's dumps).

    python tools/b2d/quantize_yolop.py --frames results/b2d/dump_smoke/dump \\
        --out models/yolop/yolop-640-640-int8.onnx

Then compare FP32 vs INT8 with tools/b2d/bench_stack.py (speed) and on the
road with a CPU config pointing 'yolop' at the INT8 file (accuracy: lane
error in tools/b2d/check_perception.py).  QDQ format, per-channel weights;
runs on onnxruntime's CPU provider.
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--model', default=os.path.join(ROOT, 'models/yolop/yolop-640-640.onnx'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models/yolop/yolop-640-640-int8.onnx'))
    ap.add_argument('--n-calib', type=int, default=64)
    args = ap.parse_args()

    import cv2
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from perception.lanes import MEAN, STD, letterbox

    pngs = sorted(glob.glob(os.path.join(args.frames, '*.png')))
    if not pngs:
        sys.exit(f'no frames in {args.frames}')
    pick = [pngs[i] for i in np.linspace(0, len(pngs) - 1, min(args.n_calib, len(pngs))).astype(int)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(pick)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            canvas = letterbox(rgb)[0]
            x = ((canvas / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
            return {'images': x}

    pre = args.out.replace('.onnx', '-pre.onnx')
    quant_pre_process(args.model, pre)
    quantize_static(pre, args.out, Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8)
    os.remove(pre)
    print(f'INT8 YOLOP ({len(pick)} calibration frames) -> {args.out}  '
          f'({os.path.getsize(args.model) / 2**20:.1f} MB -> {os.path.getsize(args.out) / 2**20:.1f} MB)')


if __name__ == '__main__':
    main()
