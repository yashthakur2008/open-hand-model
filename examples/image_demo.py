from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv

from open_hand_model import HandPoseModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Open Hand Model on one image")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result.jpg"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    image = cv.imread(str(args.input))
    if image is None:
        parser.error(f"could not read image: {args.input}")
    detections = HandPoseModel(model_dir=args.model_dir).detect(image)
    print(json.dumps([d.as_dict() for d in detections]))

    for detection in detections:
        for x, y in detection.landmarks[:, :2].round().astype(int):
            cv.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
        x1, y1, x2, y2 = detection.bbox.round().astype(int)
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if not cv.imwrite(str(args.output), image):
        parser.error(f"could not write output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
