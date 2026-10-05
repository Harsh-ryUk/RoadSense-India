# IDD Lite drivable-area baseline

Measured on 2026-10-01 using every official validation pair: **79.87% global
drivable-area IoU**, **94.93% pixel precision**, **83.43% pixel recall**.
This is a SegFormer component/postprocessor evaluation on an Apple M1 CPU,
not lane-marking accuracy, pothole AP, full-pipeline accuracy or driving safety.
No IDD fine-tuning or configuration tuning occurred in this experiment.

## Results and protocol

| Item | Measured value / fixed choice |
|---|---|
| Dataset | Official IDD Lite, validation split; 204/204 pairs across 61 drive folders |
| Image and label size | 320 × 227 |
| Target | level1Id 0: road, parking and drivable fallback |
| Other valid labels / ignore | 1–6 / 255 |
| Aggregation | Global TP/(TP+FP+FN); not multiclass mIoU or mean per-image IoU |
| Valid / ignored pixels | 14,815,509 / 3,051 |
| TP / FP / FN / TN | 3,929,523 / 210,052 / 780,420 / 9,895,514 |
| Observed / unknown predictions | 200 / 4; unknown predictions remain in the score |
| Model | ADE20K-pretrained `nvidia/segformer-b0-finetuned-ade-512-512`; output class 6 |
| Model revision | `489d5cd81a0b59fab9b7ea758d3548ebe99677da` |
| Input / crop | 512 × 512 network input; entire source image; no upper-image cut |
| Postprocessing | Current interpolation, morphological close, largest lower-image component and 2% minimum coverage |
| Preprocessing condition | `Normal`; no pipeline scene-condition/CLAHE routing |
| Runtime | CPU, 2 Torch threads, 1 OpenCV thread; Python 3.9.6 |
| Package versions | Torch 2.8.0, Transformers 4.57.6, Ultralytics 8.4.165, NumPy 1.26.4 |
| Source revision | `1ccec21bfa2c2515b16ecc3ae27ff84761e91fa1` |

The official label grouping is documented in
[AutoNUE's label definitions](https://github.com/AutoNUE/public-code/blob/master/helpers/anue_labels.py).
The target is broader than ADE20K's road label. It must be called drivable-area
IoU rather than pure road-only IoU. These low-resolution IDD Lite results are not
IDD 20k leaderboard results. Different profiles, checkpoints or resolutions need
their own reports; the T4 video performance run uses a different crop/workload.

The four unknown predictions have zero drivable-area IoU. The worst nonempty
prediction, `516/0001923_label`, has approximately 5.14% per-image IoU. A good
global score does not remove these failures. The arithmetic audit cannot check
annotation quality or substitute for independently re-running inference.

## Reproduce the measurement

Obtain IDD Lite through the [official portal](https://idd.insaan.iiit.ac.in/) and
accept its licence yourself. Images and annotation masks are not bundled here.
The measured archive SHA-256 is
`064f92c46a1141f290dd2a52f4c84174381445baf9f6d6bf471be0aa006601a1`.

From the repository root, use a compatible Python environment with the runtime
requirements. Extract the licensed archive so it produces
`runs/idd_reproduction/idd20k_lite/`, then run:

```bash
cp benchmarks/idd_lite_cpu_20261001/sample_index.json runs/idd_reproduction/manifest.json
python scripts/evaluate_accuracy.py --manifest runs/idd_reproduction/manifest.json --config benchmarks/idd_lite_cpu_20261001/config.yaml --device cpu --threads 2 --output runs/idd_reproduction/result.json
python scripts/audit_accuracy_report.py --report runs/idd_reproduction/result.json --manifest runs/idd_reproduction/manifest.json
```

Image/mask SHA-256 checks happen before inference. The portable index retains
every ID, input hash, selection rule and label schema from the original manifest;
only local absolute paths were replaced with dataset-relative paths. Consequently
its manifest-file hash differs from the original run's private-path manifest.
[Provenance](provenance.json) records both identities and the transformation.
This does not change model inputs or scores.

For exact historical source, extract [measured_source.tar.gz](measured_source.tar.gz)
into a separate directory and use its evaluator/config instead of later source.
The snapshot contains every Python file hashed by the raw report plus the
preparation/benchmark helper and dependency/config files; no weights or data.
Ensure the downloaded SegFormer checkpoint matches the recorded revision.
The published measurement used cached weights with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; online first-time setup downloads weights separately.
CPU/GPU numerical differences may require a new report rather than assuming an
identical score. No GPU throughput is measured by this accuracy run.

## Evidence, finding and next validation

| Evidence | Finding | Next validation |
|---|---|---|
| [Original result](result.json), with 204 per-image records and frozen source/model hashes | 79.8684% global drivable IoU for this exact component/profile | Keep this baseline immutable; evaluate new models under separately named protocols |
| [Input hash index](sample_index.json), [audit](audit.json) and regression tests | Every pair is included; identities, sums, ignore pixels and metric arithmetic agree | Re-run inference to independently confirm model outputs; review failures privately |
| Four unknown masks; recall 83.4304% | 16.57% of labelled drivable pixels were missed globally | Review conditions and false negatives before selecting a new training experiment |

The evaluation path is: frozen official validation pairs → complete-image
SegFormer inference → current mask postprocessor → ignore-aware pixel counts →
global ratios → separate identity/arithmetic audit. No prediction-based sample
selection or exclusion is allowed at any step.

The [validation roadmap](../../docs/VALIDATION_ROADMAP.md) separates road regions,
painted lanes and road damage. No detection AP is available from these semantic
labels. No lane/pothole score follows from this result. A small benchmark does not
establish night/rain coverage, calibrated confidence or vehicle-level safety.

All `artifact_sha256` entries in [provenance.json](provenance.json) refer to
unchanged machine-generated artifacts. Verify locally without dataset access:

```bash
python scripts/audit_accuracy_report.py --report benchmarks/idd_lite_cpu_20261001/result.json --manifest benchmarks/idd_lite_cpu_20261001/sample_index.json
python -m pytest -q tests/test_idd_accuracy_evidence.py
```
