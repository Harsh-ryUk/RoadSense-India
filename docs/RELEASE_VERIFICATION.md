# Portfolio milestone verification

This update publishes completed drivable-area fine-tuning and prior video profiling
evidence, plus the preparation tools. It does not start new training, invent labels
or claim a completed independent dashcam test.

## Pre-publication checks

| Check | Result | Scope |
|---|---|---|
| Complete offline test suite | **206 passed; no skips**, 10.68 seconds | CPU regression and preserved-evidence checks; not new model inference |
| Ruff fatal-error rules | Passed | `E9,F63,F7,F82` on source, scripts and tests |
| Annotation JavaScript | Syntax check passed | No claim of a full browser/accessibility audit |
| Original official-validation report | Identity/arithmetic audit passed | All 204 records; 61 drives; exact pixel accounting |
| Published artifact/source hashes | Regression checks passed | Original reports preserved; exact measured Python snapshot verified |
| Training completion | 50 epochs; epoch 36 selected | History, updates and checkpoint identity agree |
| Documentation links / notebook syntax | 86 file targets and 11 code cells passed | No GPU notebook execution or remote-link guarantee |

The test environment was a separate macOS ARM64 Python 3.12 environment with
Torch 2.14.1, Transformers 4.57.6, Ultralytics 8.4.173, NumPy 2.5.3,
OpenCV 5.0.0.93, Pytest 9.1.1 and PyCOCOTools 2.0.11. It is not the historical
T4 training or video environment. Those original versions remain in their reports.
Hosted CI is separate; inspect the workflow badge/run before calling it successful.

## Recheck the release

```bash
python -m pip install -r requirements-dev.txt -r requirements-eval.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
ruff check --select E9,F63,F7,F82 src scripts tests
python scripts/check_project_docs.py
node --check tools/road_annotation/app.js
```

`check_project_docs.py` checks local file-link targets and clean notebook Python
syntax; it does not run Colab cells or verify external websites. Chart generation
and visual inspection were performed separately on the preserved measurement data.

## Publication boundary

Dataset images, annotation masks, videos, trained weights, account credentials,
virtual environments and private run directories stay outside the Git commit.
The published training manifest contains paths and hashes, not dataset pixels.
The source archives contain reproducible project code, not model binaries.

The evidence path is saved artifacts → identity/count audit → source/hash tests →
charts/documentation → offline regression → scoped Git publication. A new numerical
score always requires a new declared model-inference experiment.
