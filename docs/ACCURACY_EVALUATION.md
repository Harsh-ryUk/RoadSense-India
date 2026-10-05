# Measure held-out accuracy

The [completed binary fine-tuning run](../benchmarks/idd_lite_finetuned_20261004/README.md)
reports 92.26% drivable IoU, 97.47% pixel precision and 94.52% recall on all 204
official IDD Lite validation images. The [79.87% pretrained reference](../benchmarks/idd_lite_cpu_20261001/README.md)
remains unchanged. These do not establish lane/pothole or full-pipeline accuracy. Synthetic tests
validate metric code only. The default YOLO and
SegFormer demo checkpoints are COCO/ADE20K-pretrained. The measured binary
fine-tuned checkpoint is separate and must explicitly select road class 1.

## Start with IDD Lite

Obtain IDD Lite directly from the [official IDD portal](https://idd.insaan.iiit.ac.in/)
after accepting its research-use licence. Keep images, labels and download tokens
out of Git. Do not redistribute dataset imagery or annotation files in the project.

The [official label definitions](https://github.com/AutoNUE/public-code/blob/master/helpers/anue_labels.py)
map **level1Id 0** to drivable surfaces (road, parking and drivable fallback),
level1Ids 1–6 to other scene groups, and 255 to ignored pixels. Therefore this
binary target is **drivable-area IoU**, not pure road-only IoU. Never apply this
mapping to a raw `id`, `level3Id`, RGB colour mask or binary 0/255 mask. Inspect the
downloaded archive and select the matching explicit label schema first.

IDD Lite semantic masks alone cannot establish detection mAP. For that, obtain
an annotated validation subset from IDD Detection and preserve its object classes,
crowd/ignored regions and official split. Do not turn general traffic signs into
stop signs or fallback vehicles into cars. The existing training converter needs
a separate taxonomy review before use for published evaluation.

## Freeze a manifest

For the official `idd-lite.tar.gz` archive checked on 2026-10-01, use the included
adapter after extraction. Its `idd20k_lite` directory contains 204 validation pairs
named `leftImg8bit/val/<drive>/<frame>_image.jpg` and
`gtFine/val/<drive>/<frame>_label.png`. Instance masks are excluded. The adapter
validates dimensions/label values and freezes image/mask hashes before inference.

```bash
python scripts/prepare_idd_lite_eval.py --root runs/idd20k_lite --output runs/idd_lite_val_manifest.json
python scripts/evaluate_accuracy.py --manifest runs/idd_lite_val_manifest.json --config configs/idd_eval.yaml --device cuda --output runs/accuracy/result.json
```

The default selects all validation images, with no prediction-based selection.
`--limit 100 --seed 42` requests a smaller deterministic subset. IDD Lite images
and masks are low-resolution; scores are not IDD 20k high-resolution leaderboard
results.

The evaluator accepts JSON schema version 1. Paths are relative to the manifest
unless absolute. Each unique sample needs an image and a semantic mask, an
explicit box list (an empty list means an annotated negative), or both. A manifest
must declare `split` as `val` or `test`; the caller is responsible for verifying
that its data truly belongs to that official held-out split.

For level1 masks, the schema is:

```json
{
  "schema_version": 1,
  "dataset": "IDD Lite; obtained from https://idd.insaan.iiit.ac.in/",
  "split": "val",
  "label_schema": {
    "road_ids": [0],
    "valid_ids": [0, 1, 2, 3, 4, 5, 6],
    "ignore_ids": [255]
  },
  "samples": [
    {
      "id": "drive/frame",
      "image": "idd_lite/leftImg8bit/val/drive/frame_leftImg8bit.png",
      "mask": "idd_lite/gtFine/val/drive/frame_gtFine_labellevel1Ids.png"
    }
  ]
}
```

This illustrates the required schema, not a bundled dataset or a known archive
layout. Use the actual downloaded filenames. Keep the frozen manifest in the
ignored run directory. Select samples before running predictions; record the
selection policy, all sample IDs and hashes, and drive coverage. Never pick only
frames with attractive overlays or tune the configuration on this reported set.

For box evaluation, declare `categories` as unique exact class names and add
`boxes` to each fully annotated image. A box has `bbox_xyxy` in pixel coordinates,
`category`, and optional `iscrowd` (0/1). Categories with GT and no predictions
remain in the AP calculation; categories without GT return null per-class AP.
This generic COCO adapter is not an IDD leaderboard submission adapter. Do not
silently discard difficult/ignored annotation semantics when preparing a manifest.

## Run evaluation

```bash
python -m pip install -r requirements.txt -r requirements-eval.txt
python scripts/evaluate_accuracy.py --manifest runs/idd_val_manifest.json --config configs/idd_eval.yaml --device cuda --output runs/accuracy/result.json
python -m pytest -q tests/test_ground_truth.py
```

Audit saved identity/pixel-count arithmetic separately from model inference:

```bash
python scripts/audit_accuracy_report.py --report runs/accuracy/result.json --manifest runs/idd_val_manifest.json
```

An arithmetic audit is not a second model run or an annotation-quality review.

`configs/idd_eval.yaml` uses the complete image and no upper-image cut, unlike the
clip-specific Short crop. It retains the current semantic postprocessor and
runtime confidence thresholds. The report records configuration, image/mask and
source hashes, checkpoint revisions and environment versions.

Binary IoU aggregates TP/(TP+FP+FN) over all valid pixels, ignoring only the
explicit ignore labels. Empty/unknown road predictions still incur every missed
ground-truth road pixel. It is not an average of selected per-image IoUs or
multiclass mIoU. Unexpected label IDs and mismatched dimensions fail loudly.

Box AP uses `pycocotools` COCOeval at IoUs 0.50–0.95, with AP50/AP75 and per-class
AP. Because runtime confidence/category filters remain active, this measures
the deployed detector output, not the standard low-confidence raw-model benchmark.
Report the exact taxonomy and thresholds alongside scores.

## Benchmark the supplied moving video

```bash
python scripts/benchmark_pipeline.py --profile configured --config configs/dashcam_quality.yaml --source runs/youtube_0ziIYyosuuE/dashcam_muted.mp4 --native-resolution --device cuda --frames 594 --warmup 20 --repeats 3 --threads 2 --output runs/t4_video/configured.json
```

This specific source has 614 frames. Each repeat consumes its first 20 as warmup
and measures the remaining 594; no frames are sampled or dropped. Native video
dimensions preserve portrait framing. `configured` retains the requested model
choices rather than overriding them with the historical sample-image profiles.
Render work and decode time are included in wall latency. CUDA stage timers
synchronize completed work and add instrumentation overhead.

State is retained between repeats while the clip is reopened and warmup repeated.
This is an offline sequential replay, not a live camera scheduler test. Comparison
against the source-frame time budget does not measure actual dropped frames or
queue delay. Mask observation counts, rule actions and possible surface-defect
alerts are not accuracy scores. A ten-second Short does not establish robustness.
