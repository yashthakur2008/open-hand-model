# Model Card: Open Hand Model

## Model details

This repository combines the OpenCV Zoo MediaPipe palm detector with the
MediaPipe hand-pose ONNX model converted for OpenCV DNN. The hand stage returns
21 screen landmarks and 21 relative world landmarks.

## Intended use

- Offline image experiments.
- Prototyping gesture and hand-pose interfaces.
- Reproducible CPU-based research and demos.

## Out of scope

- Identity recognition.
- Medical diagnosis.
- Safety-critical control.
- Demographic or biometric inference.

## Limitations

- Detection quality depends on lighting, occlusion, framing, and image quality.
- A hand-shaped face or other structure can produce false positives.
- CPU inference can be slow on high-resolution camera streams.
- The model is not guaranteed to generalize across all skin tones, poses, or
  camera hardware. Validate it on the deployment data before relying on it.
- Handedness follows the model's camera-coordinate convention and may require
  a mirror correction for selfie previews.

## Data and provenance

The model assets and original inference implementation are derived from
OpenCV Zoo's MediaPipe hand-pose example and are redistributed under Apache 2.0.
No training data is bundled here. See `NOTICE.md` and the upstream references
in the repository history.

## Evaluation

The test suite checks input contracts, malformed data handling, model asset
integrity, coordinate shapes, confidence bounds, and an opt-in smoke path using
the bundled fixture. These tests are not a benchmark or a claim of universal
accuracy.
