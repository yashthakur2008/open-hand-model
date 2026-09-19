# Open Hand Model

A standalone, open-source hand-pose model that detects palms and estimates 21
hand landmarks with OpenCV DNN. It is designed for reproducible CPU inference,
small integrations, and testable computer-vision experiments.

## Features

- 21 image-space landmarks plus 21 world-space landmarks.
- Bounding box, confidence, and handedness for each detection.
- CPU-first OpenCV DNN execution with optional backend/target selection.
- Headless image example that writes annotated output.
- Deterministic contract tests plus opt-in real-model smoke tests.
- Apache-2.0 licensed source and model assets.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

## Run an image

```bash
python examples/image_demo.py tests/fixtures/hand169.png --output result.jpg
# or, after installation:
open-hand-model tests/fixtures/hand169.png --output result.jpg --model-dir models
```

The input image must be a BGR `uint8` image when using the Python API. The
returned `HandDetection` objects expose:

```python
from open_hand_model import HandPoseModel
import cv2

image = cv2.imread("hand.jpg")
hands = HandPoseModel(model_dir="models").detect(image)
for hand in hands:
    print(hand.handedness, hand.confidence, hand.landmarks.shape)
```

## Deploy as a Vercel API

The repository includes a small Flask API entrypoint in [`app.py`](app.py).
Vercel detects it through the explicit `tool.vercel.entrypoint` setting in
`pyproject.toml`.

```bash
vercel deploy
curl https://YOUR-DEPLOYMENT.vercel.app/health
curl https://YOUR-DEPLOYMENT.vercel.app/ready
curl https://YOUR-DEPLOYMENT.vercel.app/metadata
curl -X POST -F image=@tests/fixtures/hand169.png \
  https://YOUR-DEPLOYMENT.vercel.app/predict
```

`/ready` confirms that both bundled ONNX assets are available.

`/predict` accepts an image upload in the `image` form field or a JSON body with
an `image` base64 string. Requests are limited to 8 MiB. The model is loaded
lazily and cached per serverless instance.

## Tests

The default suite is deterministic and does not require a camera, GUI, CUDA, or
network access:

```bash
pytest -q
```

The real ONNX smoke test is opt-in because it loads model assets and can be slow:

```bash
RUN_MODEL_SMOKE=1 pytest -q tests/test_model_smoke.py
```

## Model card

See [`MODEL_CARD.md`](MODEL_CARD.md) for provenance, intended use, limitations,
known failure modes, and evaluation notes.

## License and attribution

This repository is released under Apache License 2.0. The palm and hand-pose
implementation and model assets are derived from OpenCV Zoo. See [`NOTICE.md`](NOTICE.md)
and [`LICENSE`](LICENSE).
