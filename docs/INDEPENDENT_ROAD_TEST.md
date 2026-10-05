# Test the frozen model on new Indian-road sessions

**Status: preparation workflow added; no independent-test accuracy has been measured.**
The completed Colab run reported 92.26% global drivable-area IoU, 97.47% pixel
precision and 94.52% pixel recall on the 204 official IDD Lite validation images.
Those images are not an untouched test after their results have been inspected.
Do not train further or tune postprocessing on the new test to reach a target.
This workflow covers drivable regions only, not painted lanes or potholes.

## Work through the four improvements in order

1. Preserve the completed checkpoint, configuration, training history and report;
   freeze new sessions and human-reviewed masks, then evaluate the frozen model.
2. Report scenario slices and session-level uncertainty without dropping difficult
   images. Missing rain/night data means **not evaluated**, not a passing result.
3. Benchmark the newly trained model in the complete pipeline on declared hardware,
   recording repetitions, warmup, p95 latency, resolution and frame deadline misses.
   The older model's 11.05 FPS is not the new checkpoint's measured performance.
4. Produce a labelled visual demo, baseline comparison and reproducible evidence.
   A visually convincing unannotated clip is a demo, not an accuracy measurement.

This change implements the preparation/gating part of step 1. The remaining steps
must not be marked complete until their actual measurements exist.

## Preserve the completed run first

Download `drivable_run_stable_20261004_results.tar.gz` from the existing Colab
results cell. Its expected SHA-256, observed in the completed job status, is:

```text
6c24be116ea1268265d78a012dda2b4564c3383956c253e2e2a6fb1c0f439591
```

Check the downloaded bytes with `shasum -a 256` on macOS or `sha256sum` on Colab.
Keep the archive and extracted weights under ignored `runs/`; never commit licensed
images, raw footage, account credentials or a Colab proxy URL. Do not modify the
original report, original cloud configuration or measured-source snapshot.
The recovered archive was verified locally against this hash on 2026-10-04.
Selected weights also match the training summary and official report. Public,
data-free metadata is in the [completed evidence bundle](../benchmarks/idd_lite_finetuned_20261004/README.md).
Recheck hashes on any newly transferred copy; a requested download alone is not verification.

Use `best/`, not the last epoch's weights. Check its `model.safetensors` hash against
`training_summary.json` and `official_val_result.json` before inference. The best
development checkpoint was epoch 36, not epoch 50. For a local configuration copy,
change only `segmentation.model_name` to the absolute extracted `best/` directory;
retain the measured class 1, 512-pixel input, full frame and postprocessing settings.
That copy will have a new file hash, explicitly frozen by this preparation script.

## Choose permitted, genuinely new footage

Start with 2–3 road sessions and 20 frames per session. This is a pilot test,
not evidence covering all Indian roads. Use longer clips where necessary: default
sampling requires at least two seconds between selected frames. More adjacent frames
do not create more independent roads. Different cuts of the same drive must not
be declared separate session groups. The earlier development YouTube Short is not
eligible for the independent test, although it can remain in the visual demo.

Spacing is computed from frame indices and reported FPS. Prefer constant-rate
clips; nominal times are not verified presentation timestamps for variable-rate
videos. This sampling assumption is recorded in the output limitations.

Create a private `sessions.json` next to your clips using this format:

```json
{
  "schema_version": 1,
  "sessions": [
    {"id": "city_day", "session_group": "drive_a", "video": "city_day.mp4",
     "permission_confirmed": true, "unused_for_development": true},
    {"id": "rural_day", "session_group": "drive_b", "video": "rural_day.mp4",
     "permission_confirmed": true, "unused_for_development": true}
  ]
}
```

Only make those declarations if they are true. IDs are examples, not downloaded
videos. No precise route, personal name or vehicle registration is required.
Programmatic exact-hash checks cannot establish licence rights, rule out near
duplicates or prove geographic independence; those limitations remain disclosed.

## Sample before seeing predictions

For preparation only, install `requirements-validation.txt`; add pytest to run
the generated-fixture tests. Actual model evaluation still needs the project's
model/evaluation dependencies from `requirements-eval.txt`.

```bash
python -m pip install -r requirements-validation.txt pytest
python -m pytest -q tests/test_independent_road_test.py
```

The command below assumes the exported run is extracted under `runs/` and its
local evaluation config points to that run's absolute checkpoint path:

```bash
python scripts/prepare_independent_road_test.py sample \
  --sessions runs/independent_inputs/sessions.json \
  --training-manifest runs/idd_drivable_training_20261001/splits.json \
  --checkpoint runs/drivable_run_stable_20261004/best \
  --config runs/drivable_run_stable_20261004/evaluation_config_local.yaml \
  --output runs/independent_road_test_01 \
  --frames-per-session 20 --min-gap-seconds 2
```

This produces native-resolution RGB PNG frames, empty mask directories,
`frame_plan.json`, a pending `annotation_reviews.json` and a frozen configuration
copy. It does not call the model or generate blank masks. It freezes source-video
hashes, frame indices, checkpoint/configuration hashes and the original training
exclusion manifest. Decode failures, duplicate sampled frames and exact training
overlaps fail rather than silently replacing frames. A failed attempt can leave a
partial output directory; inspect and retain it, then choose a new output name.

## Draw and review ground truth

Use the [local ground-truth studio](ROAD_ANNOTATION_STUDIO.md) to draw native-pixel
polygons and export review drafts without showing model predictions. Its importer
creates a fresh package and cannot bypass the final complete-human-review gate.

Annotate without displaying this model's predictions. The semantic task is the
visible drivable road surface, not "where my car should steer." Mark visible
obstacles, vehicles, pedestrians, sky, sidewalks and other non-road pixels as 0;
visible road pixels as 1. Use 255 only where the boundary genuinely cannot be
determined. Do not paint hidden road underneath vehicles or use ignored areas to
hide model errors. Clearly visible road under shadows remains road.

Save each expected mask filename from `frame_plan.json` as an **8-bit grayscale
PNG**, at exactly the source frame's dimensions. Values are 0 = non-drivable,
1 = drivable and 255 = uncertain/ignored. RGB/palette masks, resized masks,
unexpected IDs, missing masks and entirely ignored images are rejected. A genuine
all-non-road image is retained as a negative, not discarded.

For every frame, update its review entry with `human_reviewed: true`, nonempty
`annotator_alias` and `reviewer_alias`, the final mask's SHA-256 and these tags:
`shadows`, `heavy_traffic`, `poor_boundaries`, `rain`, `night`. Each tag is
`present`, `absent` or `uncertain`. Ideally use a second human reviewer. If the
same person draws and reviews a mask, disclose that rather than inventing an
independent reviewer. Hashes enforce reviewed-file identity, not annotation truth.

Freeze the complete annotation set:

```bash
python scripts/prepare_independent_road_test.py freeze \
  --plan runs/independent_road_test_01/frame_plan.json \
  --reviews runs/independent_road_test_01/annotation_reviews.json \
  --output runs/independent_road_test_01/manifest.json
```

Every sampled frame needs a review and a matching mask hash. The manifest refuses
overwrites and still reports `accuracy: null`: preparing labels is not measuring
accuracy. Keep the plan, reviews and manifest together with their source images.

## Evaluate once, retain all frames

```bash
python scripts/evaluate_accuracy.py \
  --manifest runs/independent_road_test_01/manifest.json \
  --config runs/independent_road_test_01/evaluation_config.yaml \
  --device cuda:0 --threads 2 \
  --output runs/independent_road_test_01/result.json
python scripts/audit_accuracy_report.py \
  --report runs/independent_road_test_01/result.json \
  --manifest runs/independent_road_test_01/manifest.json
```

The evaluator checks the frozen checkpoint and exact configuration before
inference, and retains failed/unknown predictions in the pixel counts. Report
global binary IoU, pixel precision/recall, all valid/ignored pixel counts, frame
and session counts and the checkpoint/configuration hashes. Not seven-class mIoU,
not detection precision/recall, and not vehicle safety. A single inspected test
becomes development data if used to tune the next model; reserve new sessions for
that next model's final evaluation.

## Evidence → finding → next action

| Evidence | Finding | Next action |
|---|---|---|
| Completed Colab status and per-image report: 50 epochs; 204 official-val images | 92.26/97.47/94.52% are validation results, not new-session results | Preserve/export and verify bytes before reusing the checkpoint |
| `frame_plan.json` and model/source/configuration hashes | Sampling is declared before inference; exact overlaps checked | Obtain honest session declarations and manually reviewed masks |
| Frozen manifest and annotation review hashes | Every sampled image has a bound, reviewed label file | Run the declared frozen evaluator once; disclose annotation limitations |
| Audited per-image evaluation report | Pixel arithmetic/identity can be independently checked | Only then add scenario slices, latency measurements and demo evidence |

The workflow contracts are tested using generated videos and masks, not as a
claim that real independent Indian-road footage has already been evaluated.
