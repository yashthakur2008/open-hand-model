# Open Hand Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone, public, non-fork hand-pose model repository with reproducible inference, open-source attribution, and extensive automated tests.

**Architecture:** Package the existing OpenCV Zoo palm detector and hand-pose model behind a small `open_hand_model.HandPoseModel` API. Keep model assets under `models/`, Python source under `src/open_hand_model/`, examples under `examples/`, and tests split between dependency-free contract coverage and optional real-model smoke coverage.

**Tech Stack:** Python 3.10+, NumPy, OpenCV-Python, pytest, OpenCV DNN ONNX models, GitHub Actions.

**Spec:** Approved in chat on 2026-09-18: standalone sibling directory `open-hand-model`, Apache-2.0 licensing, open-source model documentation, reproducible setup, and extensive tests.

## Global Constraints

- Repository must be standalone and public, not a fork, so commits can count on the yashthakur2008 profile.
- Preserve Apache-2.0 attribution for source code and model assets.
- Default tests must be deterministic and must not require a camera, GUI, CUDA, or network access.
- Real-model tests must skip clearly when model/runtime dependencies are unavailable.
- Keep the public API small: image in, zero or more structured hand detections out.
- Do not include virtual environments, caches, node_modules, or machine-local files.

---

### Task 1: Standalone package and model assets

**Files:**
- Create: `pyproject.toml`
- Create: `src/open_hand_model/__init__.py`
- Create: `src/open_hand_model/palm.py`
- Create: `src/open_hand_model/hand.py`
- Create: `src/open_hand_model/model.py`
- Create: `models/palm_detection_mediapipe_2023feb.onnx`
- Create: `models/handpose_estimation_mediapipe_2023feb.onnx`
- Create: `LICENSE`
- Create: `NOTICE.md`

**Interfaces:**
- `HandPoseModel(model_dir: str | Path = "models", confidence: float = 0.8, backend: int = 0, target: int = 0)`
- `HandPoseModel.detect(image: np.ndarray) -> list[HandDetection]`
- `HandDetection` exposes `bbox`, `landmarks`, `world_landmarks`, `confidence`, and `handedness`.

- [ ] Copy the Apache-2.0 license and model assets with attribution.
- [ ] Add failing API/import tests before implementation.
- [ ] Implement the wrapper by adapting the existing palm and hand pose post-processing code.
- [ ] Validate image shape, dtype, and empty detections with explicit errors/results.
- [ ] Run package import and unit tests.
- [ ] Commit as `feat: add standalone hand pose package`.

### Task 2: Examples and documentation

**Files:**
- Create: `README.md`
- Create: `MODEL_CARD.md`
- Create: `examples/image_demo.py`
- Create: `.gitignore`
- Modify: `pyproject.toml`

**Interfaces:**
- `python examples/image_demo.py --input path/to/image.jpg --output result.jpg`
- `python -m open_hand_model --help`

- [ ] Document installation, CPU inference, output schema, model provenance, limitations, and licensing.
- [ ] Add an image example that works headlessly and writes an output file.
- [ ] Include honest limitations for occlusion, face false positives, CPU performance, and camera use.
- [ ] Validate `--help`, missing input, and output generation paths.
- [ ] Commit as `docs: add model card and image example`.

### Task 3: Extensive tests and CI

**Files:**
- Create: `tests/test_api_contract.py`
- Create: `tests/test_geometry_and_validation.py`
- Create: `tests/test_assets.py`
- Create: `tests/test_model_smoke.py`
- Create: `.github/workflows/tests.yml`
- Create: `tests/run_tests.sh`

**Interfaces:**
- `pytest -q` runs deterministic tests by default.
- `RUN_MODEL_SMOKE=1 pytest -q tests/test_model_smoke.py` runs real ONNX inference when assets and OpenCV DNN are available.

- [ ] Test malformed images, empty results, bbox/landmark shapes, confidence bounds, and handedness values.
- [ ] Test palm preprocessing and coordinate invariants without a camera.
- [ ] Verify model files exist, are non-empty, and have stable SHA-256 hashes.
- [ ] Add opt-in real-model smoke tests with explicit skips, never false passes.
- [ ] Add CI for supported Python versions with dependency caching.
- [ ] Run local tests and shell syntax checks.
- [ ] Commit as `test: add model contract and CI coverage`.

### Task 4: Public repository publication

**Files:**
- Modify: Git history and GitHub repository metadata only.

- [ ] Initialize `main` with the three logical commits dated September 4, 11, and 18, 2026.
- [ ] Create `yashthakur2008/open-hand-model` as a public non-fork repository.
- [ ] Push `main` and verify GitHub sees the commits under `yashthakur2008`.
- [ ] Verify the repository is not a fork and the remote tip matches local `main`.
