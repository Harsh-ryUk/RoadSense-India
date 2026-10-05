# T4 moving-video benchmark: bounded-region surface filtering

Measured on 2026-10-01 in Google Colab using the supplied muted vertical road
clip. This is an offline research benchmark, not a real-time ADAS qualification.
The video is not redistributed. Detection, road and defect accuracy are not
measured by this unannotated clip.

## Results

![Paired throughput, p95 latency and processing-stage costs](comparison.png)

Regenerate the chart directly from the preserved JSON reports:

```bash
python scripts/plot_video_benchmark.py --directory benchmarks/colab_t4_video_20261001
```

| Measurement | Baseline quality pipeline | Bounded-region optimization |
|---|---:|---:|
| End-to-end throughput, FPS | 6.946 | 11.050 |
| Mean wall latency, ms | 143.953 | 90.474 |
| p50 wall latency, ms | 133.707 | 84.642 |
| p95 wall latency, ms | 207.140 | 132.349 |
| p99 wall latency, ms | 219.559 | 143.774 |
| Mean anomaly-stage latency, ms | 84.754 | 32.191 |
| Peak PyTorch allocated memory, MiB | 211.32 | 211.32 |
| Peak PyTorch reserved memory, MiB | 316.00 | 316.00 |
| Peak process RSS, reported MB | 2322.78 | 2319.23 |
| Measured frames | 1782 | 1782 |

The runs show **59.1% higher throughput**, **36.1% lower p95 wall latency** and
**62.0% lower mean anomaly-stage time**. Repeat FPS was 7.024/6.989/6.827 before
and 10.970/11.113/11.069 after. These are descriptive measurements from one
sequential before/after comparison on a shared Colab host, not confidence intervals
or guaranteed deployment performance.

The source is approximately 59.94 FPS. Neither run processed a measured frame
within its 16.68 ms frame budget. This offline service-time comparison does not
measure actual dropped frames, live queue latency or scheduling behavior.

## Optimization and regression evidence

The CPU surface-candidate stage previously applied a 61×61 black-hat operation
to the whole 720×1280 frame. It now works on the bounding region of supported
road pixels, with a 60-pixel halo, then maps the result back to the full frame.
Closing comprises dilation followed by erosion, each with a 30-pixel radius:
the halo preserves supported pixels' input dependencies. Image-edge clipping
retains original border behavior.

Kernel, thresholds, supported-road mask, object exclusions, morphology and
temporal confirmation are unchanged. There is no frame skipping or lower network
resolution. OWL inference was already disabled in both runs; this is not a
full-OWL speedup. Four generated fixtures compare the filter with full-frame
filtering pixel-for-pixel, covering interior, image-edge and whole-frame support;
another checks empty support. All 25 focused road-quality/metric tests passed
in Colab. Offline evidence tests also validate the archives and paired traces.

Every one of the 1782 paired trace rows agrees on source/repeat/frame IDs,
detection counts, track counts, anomaly counts, action, traffic signal,
road-mask coverage and road status. Both report 1740 observed-road frames,
42 unknown-road frames, 1236 cruise actions and 546 slow-down actions.
Traces contain counts and coverage, not full boxes or masks: this is not
complete prediction-array equivalence or accuracy validation.

## Reproduce the workload

- Native portrait input: 720×1280, 614 frames, approximately 10.24 s.
- Three complete replays: 20 warmup then 594 measured frames each.
- YOLOv8n at 640×640; ADE20K SegFormer-B0 at 512×512.
- `configs/dashcam_quality.yaml` retained without resolution overrides; clip-specific ROI.
- Decode and rendering included; model construction and warmup excluded from throughput.
- CUDA stage/completion synchronization enabled, preventing overlap and adding overhead.
- Two PyTorch CPU threads, one OpenCV thread; state retained between repeats;
  input reopened and warmup repeated. No concurrent accuracy inference during timing.

```bash
python scripts/benchmark_pipeline.py \
  --profile configured --config configs/dashcam_quality.yaml \
  --source /content/dashcam_muted.mp4 --native-resolution --device cuda:0 \
  --frames 594 --warmup 20 --repeats 3 --threads 2 \
  --output runs/t4_video_optimized/configured.json
```

Extract the desired revision's `measured_source.tar.gz` over a separate checkout
of `65c8b15c7b0eaed33820b8d873abb4e8e713a791`. Supply the same clip privately
and the same checkpoints. For baseline output use `runs/t4_video/configured.json`.
The commit alone is not the measured revision: both reports correctly record a
dirty source overlay. Both published archives are source-only, excluding model
weights, bytecode and AppleDouble metadata. The original baseline upload remains
preserved privately under `runs/t4_current_source.tar.gz`, with SHA-256
`eb4b1c72b2018e7badb477e37399a3a199c9c1f4c1b3d9cab71234da93efb7e3`.
Obtain YOLO weights separately and verify the checkpoint hash below before replay.
Raw reports retain original packaging-metadata hashes;
tests verify all executable Python hashes against their respective archives.
The optimized snapshot adds an unused evaluation module; the only changed runtime
Python file between the runs is `src/anomaly/event_detector.py`.

## Environment and artifact identity

Tesla T4, driver 580.82.07, CUDA runtime 12.8; Python 3.13.15,
PyTorch 2.11.0+cu128, Transformers 4.57.6, Ultralytics 8.4.170,
NumPy 2.1.3, OpenCV 5.0.0. Full records are in the JSON reports. The setup log
records conflicts with unused preinstalled Gradio/Diffusers packages; the pipeline
and focused tests completed successfully.

| Artifact | SHA-256 |
|---|---|
| Muted clip | `b1fa932d29dd88bd3eef541e7df38a9d8bf7f2abeffcfd3c39b2da60f8c14f58` |
| YOLO checkpoint | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Baseline JSON | `525f726e8c3bb21e5ec2cad097bbff7590c7eea94dad62ab15e050c8f903f549` |
| Optimized JSON | `5729e106aa124c7c054b8f041f00768465fa7c1256052d3dedc0de35c7f98289` |
| Baseline source-only archive | `847f758dc15a4e1060a8112a04e016434a7a711be62cbaff0c8d31d022657fcb` |
| Optimized source archive | `1a5d03effbe8ab2dbec7f95338731006ac4e36024c65d025a155ea2a04e0c91a` |

SegFormer revision: `489d5cd81a0b59fab9b7ea758d3548ebe99677da`.
Both configuration hashes: `870c10d409e8ca045f5c08aa5917b339dae8c2b1ac40728a4a6eee96dd97abc1`.

## Evidence, finding and next validation

| Evidence | Observation | Supported finding | Next validation |
|---|---|---|---|
| [Baseline JSON](baseline/configured.json), [CSV](baseline/configured.csv), [log](baseline/benchmark.log) | Anomaly mean 84.754 ms | Surface filtering dominates this workload | Profile bounded-region implementation on the same clip |
| [Optimized JSON](optimized/configured.json), [CSV](optimized/configured.csv), [log](optimized/benchmark.log) | 11.050 FPS; anomaly mean 32.191 ms | Reduced measured processing cost here | Repeat on diverse videos and target hardware |
| [Colab tests](optimized/tests.log), source archives and offline evidence tests | Filter fixtures and recorded paired fields match | Regression evidence without lowering inputs | Compare full prediction arrays and labelled defect accuracy |

Reproduction path: restore exact source → verify input/checkpoints → run the fixed
replay → inspect raw traces/stage costs → evaluate annotations separately.
These results support a portfolio profiling/optimization claim, not driving safety.
[Held-out evaluation](../../docs/ACCURACY_EVALUATION.md) is separate: no IDD score
is derived from this unannotated video. Separate [IDD Lite fine-tuning results](../idd_lite_finetuned_20261004/README.md)
now report drivable segmentation accuracy, using a different checkpoint.
Detection mAP needs bounding-box annotations. Historical repeated-image
M1/T4 records remain unchanged and are not comparable to this video as if paired.
