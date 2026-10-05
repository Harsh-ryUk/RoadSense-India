# RoadSense India: project summary and interview notes

## Short explanation

RoadSense India takes a road video, detects road users, estimates the visible
drivable region, tracks objects across frames and produces annotated output with
simulation-only scene rules. I fine-tuned its segmentation component on IDD Lite
and measured a separate pipeline optimization on a Colab T4 video workload.

## CV-ready project entry

**RoadSense India — Indian-road perception and performance engineering**

Python · PyTorch · YOLOv8 · SegFormer · OpenCV · CUDA · GitHub Actions

- Built a modular road-scene perception pipeline integrating YOLOv8 detection,
  SegFormer segmentation, Kalman/IoU tracking and simulation-oriented scene rules.
- Fine-tuned SegFormer-B0 on 1,213 IDD Lite training images with a drive-disjoint
  development split; achieved **92.26% drivable-area IoU, 97.47% pixel precision and
  94.52% recall** on all 204 official validation images—**+12.39 percentage points
  IoU** over the preserved pretrained reference.
- Optimized ROI-bounded surface filtering on a paired Tesla T4 video workload,
  increasing end-to-end throughput **59.1% (6.95 → 11.05 FPS)** and reducing p95
  latency **36.1% (207.14 → 132.35 ms)** across 1,782 measured frames per revision.
- Added offline regression tests, CI, per-stage profiling, SHA-256-bound experiment
  records and an ignore-aware accuracy audit for reproducible measurements.

Use the two metric bullets first when CV space is limited. The speed experiment
used the earlier ADE20K checkpoint; do not imply 11.05 FPS was measured with the
new fine-tuned checkpoint. These are pixel segmentation metrics, not object-level
precision/recall, lane F1 or pothole AP.

## A 45-second explanation

“I built a modular road-scene perception project combining object detection,
drivable-area segmentation and tracking. My main work was making it measurable:
I created drive-separated data splits, fine-tuned SegFormer-B0 for 50 epochs and
evaluated every official IDD Lite validation image. It reached 92.26% drivable IoU.
Separately, profiling showed that CPU surface filtering was the video bottleneck.
Restricting that operation to the supported-road region with a dependency-preserving
halo improved T4 pipeline throughput by 59%. I kept raw traces, source and input
hashes, tests and a model card. It is a research prototype, not a deployed vehicle
system; independent dashcam and dedicated lane/pothole validation are future work.”

## What to show in a recruiter walkthrough

| Order | Show | Explain |
|---|---|---|
| 1 | README result table and architecture | Which components are pretrained, what was fine-tuned, and what the project outputs |
| 2 | Validation chart and complete report | Global IoU, ignored pixels, full 204-image coverage and development-only selection |
| 3 | T4 before/after chart and traces | The bottleneck, 60-pixel halo, unchanged input resolution and recorded-field regression |
| 4 | Tests, source snapshot and model card | Reproduction, failure handling, evidence scope and remaining limitations |

## Questions worth preparing for

| Question | Concrete answer |
|---|---|
| Did you train the entire model from scratch? | No. ADE20K-pretrained SegFormer-B0 was adapted with a binary head and fine-tuned on IDD Lite. YOLO/optional OWLv2 remain pretrained. |
| How did you avoid split leakage? | Separate drive folders for fit/development/official validation, frozen hashes and exact encoded/decoded-image duplicate checks. Geographic and near-duplicate independence are not proven. |
| Why is best dev IoU higher than the headline number? | 94.18% is raw-model internal development IoU used for selection; 92.26% is separate official-validation output with deployed postprocessing. |
| What caused the speedup? | Less CPU morphology work on a bounded supported-road region, without frame skipping or reduced network input. |
| Is it real-time? | Not at the source's 59.94 FPS. The reported 11.05 FPS is an offline single-stream video benchmark with decode/rendering included. |
| Does it detect potholes and lanes accurately? | Those scores are not measured. Current surface anomalies are heuristics; road-region edges are not painted lane markings. |
| Can another person reproduce the exact accuracy today? | The repo contains the source/metadata/audit; exact inference also requires licensed IDD Lite data and the privately retained selected weights. |

## Evidence links

[Fine-tuning evidence](../benchmarks/idd_lite_finetuned_20261004/README.md) ·
[Video optimization evidence](../benchmarks/colab_t4_video_20261001/README.md) ·
[Model card](MODEL_CARD.md) · [Independent-test protocol](INDEPENDENT_ROAD_TEST.md)

Do not claim 95%+ IoU, industry certification, Level 4 autonomy, validated Indian
lane/pothole detection or improved tracking accuracy. This milestone supports a
solid personal-project claim in computer vision and performance engineering.
