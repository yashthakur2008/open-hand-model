from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv

from .model import HandPoseModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect 21 hand landmarks in an image")
    parser.add_argument("input", type=Path, help="input image path")
    parser.add_argument("--output", type=Path, help="optional annotated output path")
    parser.add_argument("--model-dir", type=Path, default=None, help="directory containing the two ONNX assets")
    args = parser.parse_args()

    image = cv.imread(str(args.input))
    if image is None:
        parser.error(f"could not read image: {args.input}")
    model = HandPoseModel(model_dir=args.model_dir) if args.model_dir else HandPoseModel()
    detections = model.detect(image)
    print(json.dumps([d.as_dict() for d in detections]))

    if args.output:
        annotated = image.copy()
        for detection in detections:
            points = detection.landmarks[:, :2].round().astype(int)
            for x, y in points:
                cv.circle(annotated, (int(x), int(y)), 2, (0, 0, 255), -1)
            x1, y1, x2, y2 = detection.bbox.round().astype(int)
            cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if not cv.imwrite(str(args.output), annotated):
            parser.error(f"could not write output image: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
