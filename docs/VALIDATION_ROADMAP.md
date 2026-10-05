# Validate road regions, lanes and potholes separately

For the completed drivable fine-tuning run, the next task is
[independent testing of the frozen checkpoint](INDEPENDENT_ROAD_TEST.md).
The preparation workflow samples new road sessions before inference and requires
human-reviewed masks. No independent-session result is available yet. The
historical baseline and separate lane/pothole work below remain distinct.

The completed fine-tuned result is available:
[92.26% drivable-area IoU on all 204 IDD Lite validation images](../benchmarks/idd_lite_finetuned_20261004/README.md).
The [79.87% pretrained reference](../benchmarks/idd_lite_cpu_20261001/README.md) stays unchanged.
Lane-marking detection and pothole detection remain unvalidated. Do not relabel
road-mask boundaries or dark surface candidates to imply those capabilities.

The separate [drivable fine-tuning protocol](DRIVABLE_TRAINING.md) now freezes
1,213 fit / 190 development / 204 official-validation pairs. All 50 epochs completed,
with epoch 36 selected on development only. The 95/98/95% targets were not all met.
This milestone stops at the completed results and preparation tools; the remaining
research tasks are future work, not completed capabilities.

## Three independent tasks

| Task | Data / model route | Evidence required | Current status |
|---|---|---|---|
| Drivable region | IDD Lite; binary fine-tuned SegFormer-B0 | Global IoU, pixel precision/recall, ignore policy, every failed mask | Completed official validation; 92.26 / 97.47 / 94.52%; independent road test pending |
| Road damage / potholes | RDD2022 India; separately trained small detector | D40 AP50 and AP50–95, precision/recall, negative-scene false positives | Dataset access and model training pending; contrast heuristic is not a trained detector |
| Painted lane markings | Official UFLDv2 baseline and labelled lane data; separate Indian evaluation | Polyline matching precision/recall/F1 and scenario failures | No dedicated model/checkpoint or measured lane score integrated |

## Next: an Indian pothole detector

[RDD2022's official release](https://github.com/sekilab/RoadDamageDetector)
contains Indian images and Pascal VOC XML training annotations. Its four target
classes are D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack)
and D40 (pothole). The official challenge test images have **no released labels**.
Therefore an internal labelled holdout must be named a custom split of the
released training data, not the official challenge test result.

1. Obtain the official release and keep raw files under ignored `runs/` or an
   external dataset directory. The
   [Figshare record](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)
   specifies CC BY 4.0; retain the authors' citation and record the archive hash.
   On 2026-10-01 the original country-specific S3 link returned HTTP 403. The
   official public Figshare catalogue offered a 13,264,172,619-byte combined ZIP,
   not a separate India ZIP. That larger download has **not** been performed.
2. Audit the actual XML labels, image dimensions, coordinate convention,
   negatives and difficult/ignored flags before conversion. Preserve D40 as
   pothole; do not reuse the general COCO detector's categories as damage labels.
3. Freeze train/validation/test assignments before training. Group related
   drives/sequences where metadata permits; check exact and near-duplicate
   images across splits. A random image split alone does not establish
   independent-road generalization. If sequence metadata is unavailable,
   disclose that limitation and obtain a separate road-session test set.
4. Train a separate YOLOv8n road-damage baseline at 640-pixel input. Start with
   a short smoke run to check labels; then a planned 30-epoch experiment with
   a fixed seed, batch 8 and validation-only early stopping is a reasonable
   first experiment, **not a measured runtime or promised accuracy**. Record the
   actual GPU, epochs completed, learning settings and checkpoint hashes. Keep
   COCO road-user detection separate; replacing its weights with four damage
   classes would break existing class-dependent rules.
5. Select the checkpoint and operating threshold on validation only. Evaluate
   the untouched custom test once, retaining all negatives and missed potholes.
   Report D40 AP50–95/AP50, chosen-threshold precision/recall, per-class metrics,
   failure examples and false-positive counts. Use low-confidence predictions
   for standard AP; the existing deployed-threshold evaluator is not equivalent.
6. Integrate measured damage detections as a separate optional component; keep
   the old surface-contrast heuristic clearly named and run an ablation. Measure
   added latency on the same video/device protocol, rather than assuming the
   current 11.05 FPS is retained after adding a network.

The current `train_yolo_idd.py` and `train_segformer_idd.py` are legacy recipes,
not validated RDD2022/lane-training workflows. Do not run them unchanged and call
their outputs pothole or lane-marking accuracy. The segmentation recipe also
needs review of missing-mask handling, label reduction and ignored pixels before
it is used for a published IDD fine-tuning result.

## Then: a real lane-marking model

Use the authors' [Ultra-Fast-Lane-Detection-v2 implementation](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)
as a candidate baseline. They supply dedicated lane checkpoints/configurations
for CULane, TuSimple and CurveLanes, plus evaluation code. Their published scores
are **not this project's measured scores** and do not establish Indian accuracy.

- Review the model/code and dataset licences, pin the checkpoint/source revision
  and first reproduce the corresponding official evaluation protocol.
- Keep lane polylines separate from drivable-region masks. Validate coordinates,
  resizing/cropping and missing lanes; do not use a road contour as ground truth.
- For an Indian-road claim, annotate lane polylines on separately collected,
  permitted road sessions, including faded, absent, occluded and curved markings.
  Split by session, not adjacent frames. Keep a final session holdout untouched.
- Use a declared polyline matching protocol with precision, recall and F1;
  report unmarked-road false positives separately. Preserve unknown/absent-lane
  states instead of drawing confident lanes on every frame.
- Only consider lane-departure simulation after evaluating the dedicated lane
  model. No steering control or vehicle-safety claim follows from an F1 score.

## Protect the new road baseline

The current validation split is now a reported baseline. Reviewing its errors is
useful, but any configuration/model chosen using those errors must be disclosed
as validation-driven development. For an unbiased comparison after tuning,
reserve a different untouched test set. Do not erase the original result or
silently replace its source/configuration archive with the improved version.
