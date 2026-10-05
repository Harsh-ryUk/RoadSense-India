# RoadSense India — system/model card

## Intended use

Road-scene research, reproducible latency experiments, annotated video demos and
simulation-oriented rules. This repository does not establish vehicle-level
safety or Level 4 autonomy. Legacy names remain for source/API compatibility.

## Components and evidence

| Component | Implementation | Validation here | Limitation |
|---|---|---|---|
| Detection | COCO YOLOv8n; actual labels; category thresholds | Real runtime; custom-taxonomy regression | India-specific fine-tuned weights not measured |
| Segmentation | Binary IDD Lite fine-tuned SegFormer-B0, drivable class 1; default demo still uses ADE20K class 6 | 92.26% global drivable IoU on all 204 IDD Lite validation pairs; completed 50-epoch run | Component/profile score only; selected weights retained privately; not lane or complete-system accuracy |
| Zero-shot | OWLv2 road-object text prompts | Real full benchmarks; scheduling/filter tests | Supplemental display; periodic slow inference |
| Tracking | Kalman + IoU + Hungarian | Lifecycle/association tests | No learned appearance; not full DeepSORT |
| Signals | HSV crop heuristic | Tests/orchestration; no lights in sample | No signal relevance reasoning |
| Anomalies | Image velocity/proximity/contrast rules | Synthetic scene tests | No metric depth or calibrated TTC |
| Decisions | Priority rules; simulation controls | Expected-action tests | Pixels are not physical stopping distances |

RGB preprocessing follows the official
[SegFormer input convention](https://huggingface.co/docs/transformers/model_doc/segformer).
Road-mask coverage is a geometric fraction, not calibrated confidence.
Historical benchmark code reduced capture input to 256×144 before network resizing.
Current code preserves source detail until the configured network resize and
supports an explicit crop. See [road-quality behaviour](ROAD_QUALITY.md).

## Data and measurement

The [completed IDD Lite fine-tuning experiment](../benchmarks/idd_lite_finetuned_20261004/README.md)
uses 1,213 fit images across 263 drive folders and 190 internal development images
across 46 separate drives. Official validation retains all 204 images across
61 drive folders. Fifty epochs completed; internal raw development selected epoch 36.
The selected checkpoint achieved **92.2577% drivable IoU, 97.4707% pixel precision
and 94.5205% recall** after deployed mask postprocessing, on a Colab Tesla T4.
All 204 masks were observed. It did not meet all 95/98/95% research targets.

The binary target maps raw level1Id 0 to class 1, raw 1–6 to class 0 and ignores 255.
The processor retains label zero. Official validation was not used for gradients
or checkpoint selection, but had already been inspected in the reference experiment.
Exact duplicate and drive-folder checks do not prove geographic independence.
The 40-frame new-session pilot still awaits human labels/review; no independent
dashcam accuracy or full-pipeline timing with the fine-tuned model is available.
Default runtime profiles are not silently switched to these privately retained weights.

The separate [IDD Lite accuracy experiment](../benchmarks/idd_lite_cpu_20261001/README.md)
evaluates all 204 official validation pairs (61 drive folders), at 320×227 source
resolution and 512×512 network input. Drivable-area IoU is 79.8684%, pixel
precision 94.9258% and recall 83.4304%. Four unknown masks are retained as misses.
The label target is level1Id 0, including parking/drivable fallback, with only
255 ignored. This is the SegFormer component plus mask postprocessing under
`Normal` preprocessing and a full-frame config, not the video crop, scene routing,
lane-marking F1, detection AP or pothole accuracy. The reference model was not fine-tuned
on IDD in this experiment. Dataset imagery/labels are not redistributed.

[CPU evidence](../benchmarks/mac_m1_cpu/README.md) and
[Tesla T4 evidence](../benchmarks/colab_t4_sample_20260930T202529Z/README.md) use one letterboxed, repeated
Ultralytics sample image at 1280×720. It exercises real inference but lacks motion,
diverse road scenes and annotations. Throughput includes per-frame input copying
and rendered processing; model construction and warmup are separate. Memory is
peak process RSS; GPU records additionally report post-warmup PyTorch allocated/
reserved memory. The T4 measured full512/full256/core256 at 8.05/8.06/19.65 FPS,
with p95 wall latency of 777.0/785.6/71.7 ms over 300 frames/profile. These are
synchronized single-stream sample measurements, not target-device deadlines.

The 512×512 profile produced no road mask on this sample. Smaller inputs changed
masks/boundaries. This is an observation, not an accuracy comparison. The sample
does not exercise every anomaly branch or traffic-light state.

## Failure modes

The current quality profile has a separate [moving-video T4 experiment](../benchmarks/colab_t4_video_20261001/README.md):
6.946 → 11.050 FPS, p95 wall latency 207.140 → 132.349 ms, over 1782 measured
frames per revision. Bounded-region surface filtering reduced CPU work with
matching recorded per-frame counts, coverage and actions. This profile disables
OWL and uses a clip-specific crop. Neither run met the source's 59.94 FPS budget.
The unannotated clip does not establish detection, road, tracking or defect accuracy.

Unknown road regions now prevent the pipeline's default cruise rule. This is an
experimental simulation policy, not a vehicle-safe fallback. Lane markings remain
unvalidated; road-region boundaries are not used as lane steering targets.
Surface contrast candidates need temporal confirmation and exclude tracked
objects, but shadows, reflections and camera motion can still fool the heuristic.
Wrong-side image-motion inference is disabled by default in the L4 pipeline;
ego-motion compensation and road-direction context are not implemented.

- COCO pretraining does not establish recognition of auto-rickshaws/overloaded vehicles.
- Road masks and fitted edges can fail under padding, lighting, occlusion or domain shift.
- IoU-only tracking can switch identities across occlusion/overlap.
- Cached OWLv2 boxes are older observations with exposed source age, without motion compensation.
- Pixel distance/velocity depends on camera geometry and ego motion.
- Missing masks reduce road-anomaly information; they do not establish clear road.
- Experimental rule commands have no complete vehicle safety envelope.

## Next validation

Retain the measured IDD Lite baseline and use separate held-out Indian-road data
for broader accuracy claims. See the [validation roadmap](VALIDATION_ROADMAP.md)
for dedicated damage/lane models and split isolation. Measure class-wise precision/recall,
road IoU, tracking ID switches/MOTA, event precision/recall and decision errors.
Compare resolution/scheduling profiles on identical examples. Then measure
deployment hardware with decode, queueing, cold start, peak/steady memory and
stale-observation rates. CARLA/ROS closed-loop behavior was not tested here.
