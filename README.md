# RoadSense India

### Road-scene perception, Indian-road fine-tuning and measured pipeline optimization

A personal computer-vision engineering project combining **YOLOv8 road-user detection,
SegFormer drivable-area segmentation, multi-object tracking and simulation rules**.
The focus is not just running models: it is separating data splits, measuring complete
workloads, finding bottlenecks and preserving evidence that can be checked.

[![Tests](https://github.com/Harsh-ryUk/Hybrid-Transformer-ADAS-India-Perspective/actions/workflows/ci.yml/badge.svg)](https://github.com/Harsh-ryUk/Hybrid-Transformer-ADAS-India-Perspective/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harsh-ryUk/Hybrid-Transformer-ADAS-India-Perspective/blob/main/notebooks/colab_drivable_training.ipynb)

[Accuracy evidence](benchmarks/idd_lite_finetuned_20261004/README.md) ·
[T4 video benchmark](benchmarks/colab_t4_video_20261001/README.md) ·
[Project summary](docs/PROJECT_SUMMARY.md) · [Model card](docs/MODEL_CARD.md)

> Research and simulation prototype—not a vehicle-control product. Accuracy below is
> binary drivable-area segmentation. The video timing experiment uses an earlier
> checkpoint; it is not the fine-tuned model's measured end-to-end performance.

## Results at a glance

| Experiment | Measured result | Evaluation scope |
|---|---:|---|
| IDD Lite fine-tuned SegFormer-B0 | **92.26% drivable IoU** | All 204 official validation images, 61 drive folders |
| Same checkpoint and validation run | **97.47% pixel precision / 94.52% recall** | Global binary pixel counts, ignore-aware scoring |
| Improvement over the preserved ADE20K reference | **+12.39 percentage points IoU** | Same validation image/mask identities and label policy; CPU/GPU runs differ |
| T4 moving-video optimization | **6.95 → 11.05 FPS (+59.1%)** | 1,782 measured frames per revision; decode and rendering included |
| Same paired video workload | **207.14 → 132.35 ms p95 latency (−36.1%)** | Three replays, no frame skipping or network-resolution reduction |
| Fine-tuning completion | **50 epochs; selected epoch 36** | Selection on 190 internal development images, not official validation |

### Drivable-area accuracy

![Official-validation comparison and separate development training curve](benchmarks/idd_lite_finetuned_20261004/comparison.png)

| Global pixel metric | ADE20K reference | IDD Lite fine-tuned |
|---|---:|---:|
| Drivable-area IoU | 79.87% | **92.26%** |
| Precision | 94.93% | **97.47%** |
| Recall | 83.43% | **94.52%** |

The fixed assignment uses **1,213 fit images / 190 internal development images /
204 official validation images**, with drive-folder separation and exact encoded/
decoded-image duplicate checks. Fine-tuning uses a binary head, 512×512 RGB input,
ignore-aware cross entropy plus Dice, AdamW and CUDA FP16 forward passes with
FP32 interpolation/loss. Unsafe AMP updates are skipped without advancing the scheduler.

The target is IDD Lite's **road, parking and drivable fallback** grouping, not
painted lanes. Scores use deployed mask postprocessing at native label resolution;
the training curve uses raw development argmax masks and is a different measurement.
The previously inspected official validation split is not a never-seen final test.
The 95% IoU / 98% precision / 95% recall research targets were **not all reached**.

[Protocol, counts and reproduction](benchmarks/idd_lite_finetuned_20261004/README.md) ·
[Original validation report](benchmarks/idd_lite_finetuned_20261004/official_val_result.json) ·
[50-epoch history](benchmarks/idd_lite_finetuned_20261004/history.json) ·
[Identity/arithmetic audit](benchmarks/idd_lite_finetuned_20261004/audit.json)

### Whole-pipeline profiling

![Paired T4 video throughput, tail latency and processing-stage costs](benchmarks/colab_t4_video_20261001/comparison.png)

The bottleneck was CPU surface filtering: a 61×61 black-hat operation over the full
720×1280 frame. Processing the supported-road bounding region with a 60-pixel halo
reduced its mean cost from **84.75 to 32.19 ms (−62.0%)**. The halo preserves the
morphological operation's input dependencies; image-edge behavior is retained.

All **1,782 paired trace rows** agree on recorded detection/track/anomaly counts,
road coverage/status, traffic signal and rule action. Pixel-equivalence tests use
generated fixtures; traces do not establish full prediction-array equivalence.
OWL was disabled in both video revisions. **11.05 FPS is offline throughput, not
real-time processing of the 59.94 FPS source or a safety qualification.**

[Paired reports, traces and exact source](benchmarks/colab_t4_video_20261001/README.md)

## Architecture

~~~mermaid
flowchart LR
    Frame["Decoded RGB/BGR frame"] --> Detect["YOLOv8 road users"]
    Frame --> Road["SegFormer drivable region"]
    Frame --> Signal["Signal-colour heuristic"]
    Frame --> OWL["Optional OWLv2 prompts"]
    Detect --> Track["Kalman + IoU association"]
    Track --> Surface["Motion / surface candidates"]
    Road --> Surface
    Track --> Rules["Priority simulation rules"]
    Road --> Rules
    Signal --> Rules
    Surface --> Rules
    Rules --> Output["Annotated preview + simulation output"]
    OWL -. "supplemental display only" .-> Output
    Frame --> Timing["Per-frame and per-stage profiler"]
    Timing --> Evidence["CSV traces / JSON reports / hashes"]
~~~

The tracker is SORT-style Kalman/IoU/Hungarian association, without a learned
appearance encoder. OWLv2 is optional supplemental display, not fused into decisions.
Road-mask boundaries are not treated as lane-marking steering targets. An unknown
road mask suppresses default cruise in the simulation rules.

## Run a video

Python 3.10–3.11 is the CI target. The saved GPU experiments used Python 3.13.15;
actual package versions are preserved in each report. Weights download on first use.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt

# Compact CPU demonstration with detection, segmentation, tracking and rules
python -m src.adas_pipeline_l4 --config configs/cpu.yaml --source video.mp4 --device cpu --headless --output result.mp4
~~~

For a GPU/video-quality run, use `configs/dashcam_quality.yaml` and `--device cuda:0`.
Its ROI was configured for the portrait demo; adjust it for a different camera.
These default profiles use pretrained weights. To use the privately retained
fine-tuned checkpoint, configure `segmentation.model_name` to its `best` directory
and `segmentation.road_class_ids: [1]`. Do not assume the recorded video FPS carries over.

## Reproduce an experiment

| Task | Entry point | What it records |
|---|---|---|
| Fine-tune the binary drivable model | [Training guide](docs/DRIVABLE_TRAINING.md), [Colab notebook](notebooks/colab_drivable_training.ipynb) | Frozen splits, epochs, loss, development scores and selected checkpoint hashes |
| Evaluate labelled frames | [Accuracy guide](docs/ACCURACY_EVALUATION.md), `scripts/evaluate_accuracy.py` | All image/mask identities, global confusion counts, model/configuration hashes |
| Benchmark a moving video | `scripts/benchmark_pipeline.py` | Warmup, FPS, p50/p95/p99, stage timings, memory and per-frame traces |
| Audit a saved score | `scripts/audit_accuracy_report.py` | Input identity, pixel accounting, status totals and metric arithmetic; no model inference |
| Prepare independently reviewed road labels | [Annotation studio](docs/ROAD_ANNOTATION_STUDIO.md) | Native polygon masks, review declarations and frozen-input gates |

~~~bash
python -m pip install -r requirements-dev.txt
python scripts/audit_accuracy_report.py --report benchmarks/idd_lite_finetuned_20261004/official_val_result.json --manifest benchmarks/idd_lite_finetuned_20261004/sample_index.json
python scripts/plot_drivable_results.py
python -m pytest -q
ruff check --select E9,F63,F7,F82 src scripts tests
~~~

Training tests need `torch>=2.8` for the public `torch.amp` API. COCO box-metric
tests additionally need `requirements-eval.txt`. Tests use generated fixtures or
preserved records; they do not download models or substitute for new accuracy runs.
CI runs offline on Python 3.10 and 3.11. Published source snapshots identify the
historical measured code even when later evaluation tooling changes.

## Scope and current stopping point

| Capability | Current status |
|---|---|
| Indian drivable-area segmentation | Completed IDD Lite fine-tuning and official-validation evaluation |
| Runtime engineering | Measured M1/T4 sample profiles and paired T4 moving-video optimization |
| Independent dashcam generalization | 40 frames from two new sessions prepared; human masks/review and scores pending |
| Painted lane markings | No integrated, independently validated dedicated model |
| Pothole / road-damage detection | Surface-contrast candidates only; no measured pothole AP or precision/recall |
| Road-user tracking | Implemented association; no measured MOTA/IDF1 on annotated tracks |
| Vehicle actuation / closed-loop safety | Not demonstrated; rule outputs are for research/simulation |

This milestone is a portfolio project with measured segmentation and optimization
results—not an industry-certified ADAS stack. No further training or new-road
accuracy is claimed in this release. Legacy `L4` names are retained for compatibility,
not as evidence of Level 4 autonomy.

## Explore the evidence

[Completed fine-tuning](benchmarks/idd_lite_finetuned_20261004/README.md) ·
[Preserved CPU accuracy reference](benchmarks/idd_lite_cpu_20261001/README.md) ·
[Paired T4 video](benchmarks/colab_t4_video_20261001/README.md) ·
[Historical T4 sample](benchmarks/colab_t4_sample_20260930T202529Z/README.md) ·
[M1 CPU sample](benchmarks/mac_m1_cpu/README.md)

[Architecture details](ARCHITECTURE.md) · [Engineering notes](docs/ENGINEERING.md) ·
[Model card](docs/MODEL_CARD.md) · [Road-quality behavior](docs/ROAD_QUALITY.md) ·
[Future validation](docs/VALIDATION_ROADMAP.md) · [Results protocol](docs/RESULTS.md)

## Attribution

Pretrained models: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
[NVIDIA SegFormer-B0](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512),
and [Google OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble).
Indian-road labels: [IDD / AutoNUE](https://idd.insaan.iiit.ac.in/).
Dataset images, masks and trained weights are not redistributed in this update.
Historical sample previews use Ultralytics' bundled `bus.jpg`, not Indian footage.
Model, dependency and dataset licences remain their own.
