# Fine-tune drivable segmentation with separated training and evaluation

**95% IoU, 98% pixel precision and 95% recall are research targets, not achieved
results.** The completed 2026-10-04 experiment reached **92.26% drivable-area IoU,
97.47% pixel precision and 94.52% recall** on all 204 official validation images.
All 50 epochs completed; internal development selected epoch 36. See the
[unaltered reports, history and measured source](../benchmarks/idd_lite_finetuned_20261004/README.md).
The [79.87 / 94.93 / 83.43% reference](../benchmarks/idd_lite_cpu_20261001/README.md)
remains preserved separately. No further training is part of this milestone.

## Understand the targets

For the same global binary pixel counts:

```text
IoU = 1 / (1 / precision + 1 / recall - 1)
```

98% precision with 95% recall gives approximately **93.19% IoU**. At exactly 98%
precision, 95% IoU needs at least **96.8783% recall**. Increasing one score by
changing a threshold can reduce another; all three must be checked together on
the same images, ignore policy and output profile. Do not substitute pixel
accuracy or background-inclusive mIoU for drivable-class IoU.

## The first fixed experiment

| Choice | Protocol |
|---|---|
| Data | Licensed official IDD Lite; 1,403 train pairs and 204 official validation pairs |
| Fit | 1,213 images, 263 drive folders, selected before training |
| Internal development | 190 images, 46 separate drive folders drawn from official train |
| Official validation | All 204 images, 61 drive folders; never used for gradients/checkpoint selection |
| Split seed / grouping | 42 / drive folder; reject exact encoded/decoded-image duplicates across splits |
| Split-manifest SHA-256 | `1e30c0de0df837c0bb615b4b5ed7a42e5132e5c18a2733409bfe92b8e47433da` |
| Model | ADE20K SegFormer-B0 encoder/decoder; replace the 150-class classifier with a new 2-class head |
| Labels | Raw level1Id 0 → drivable class 1; 1–6 → non-drivable class 0; 255 stays ignored |
| Input / supervision | Resize RGB images to 512×512; keep native 320×227 labels and interpolate logits to labels |
| Augmentation | Training-only horizontal flips and bounded brightness/contrast changes |
| Optimizer / loss | AdamW, LR 6e-5, decay .01; ignore-aware cross entropy + .5 soft drivable Dice |
| Schedule | 50 planned epochs, 5% warmup then cosine; batch 8; seed 42 |
| Device | Explicit CUDA for full run; fail rather than silently falling back to CPU |
| Selection | Internal raw-model metrics only: prefer checkpoints meeting all three targets, then greatest global IoU |

The frozen split has no drive-folder overlap with official validation and no
exact encoded/decoded duplicate images across its three splits. Folder/duplicate
checks do not prove geographic independence or absence of similar neighbouring
frames. A separate road-session test set is still needed for broader claims.

The trainer uses the correct `do_reduce_labels=False` processor setting. IDD
class 0 is a real target, not background to be deleted. See the
[official SegFormer preprocessing notes](https://huggingface.co/docs/transformers/model_doc/segformer#notes).
Missing masks and unknown labels fail loudly; ignored pixels do not contribute
to CE/Dice or metric counts. The legacy `train_segformer_idd.py` is **not** this
workflow and must not be substituted.

## Run in Colab

Open [colab_drivable_training.ipynb](../notebooks/colab_drivable_training.ipynb)
in a GPU Colab runtime for a separately named reproduction. It is a clean workflow
template, not the original executed notebook output. The completed T4 run is
preserved separately; a new run is not guaranteed to reproduce identical weights
or scores. The current trainer includes the measured AMP overflow-recovery fix.

Upload your licensed `idd-lite.tar.gz`. The notebook clones the public repository,
checks the published frozen training-manifest hash, verifies the dataset archive,
safely extracts the licensed data, installs training dependencies, runs the experiment,
evaluates the selected checkpoint separately and exports results privately.
The published `training_splits.json` contains relative paths and hashes, not images
or annotation masks. No private source bundle or video is needed for this experiment.

For a directly prepared environment, the equivalent commands from the repository
root are:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_idd_lite_training.py --root runs/idd_lite_validation_20261001/idd20k_lite --output runs/new_drivable_splits.json --seed 42
python -u scripts/train_drivable_idd_lite.py --root runs/idd_lite_validation_20261001/idd20k_lite --manifest runs/new_drivable_splits.json --output runs/drivable_b0_512_seed42 --device cuda:0 --epochs 50 --batch 8 --input-size 512 --lr 0.00006 --seed 42
```

Use a new output directory for each experiment. The frozen manifest is never
overwritten. If a CUDA allocation fails, record a separately named smaller-batch
experiment rather than silently changing settings. The current trainer saves
the selected model and per-epoch history but does **not** resume optimizer state
after interruption; preserve artifacts and start a clearly named new run.
The completed run used FP16 forward passes, FP32 interpolation/loss and skipped
five unsafe AMP updates without advancing the scheduler; 7,595 updates completed.
The trainer records numerical events and stops after repeated nonfinite failures.
Colab storage is temporary. Download selected weights, protocol and history
before ending the session. No completion time or target score is guaranteed.

## Measure deployed output after training

Training history uses **raw binary argmax masks on internal development data**.
Those scores are not official-validation or postprocessed runtime accuracy.
The exported `evaluation_config.yaml` explicitly selects model class 1, not the
old ADE20K road class 6. It retains the deployed morphological/component/coverage
postprocessor for a separate official-validation run:

```bash
python scripts/prepare_idd_lite_eval.py --root runs/idd_lite_validation_20261001/idd20k_lite --output runs/drivable_b0_512_seed42/official_val_manifest.json
python scripts/evaluate_accuracy.py --manifest runs/drivable_b0_512_seed42/official_val_manifest.json --config runs/drivable_b0_512_seed42/evaluation_config.yaml --device cuda:0 --output runs/drivable_b0_512_seed42/official_val_result.json
python scripts/audit_accuracy_report.py --report runs/drivable_b0_512_seed42/official_val_result.json --manifest runs/drivable_b0_512_seed42/official_val_manifest.json
```

The evaluator now hashes local model weights/config/processor files and rejects
changes during inference. Compare `model.safetensors` with the selected training
checkpoint's hash. Keep every official validation frame, including empty masks
and unknown-road outcomes. Do not tune on the official-validation report and
then describe it as untouched testing. The existing baseline already makes this
a previously observed validation set, not a never-seen final test.

## If the first run misses the targets

Use the internal development split to separate raw-model errors from
postprocessing losses. Run declared ablations for component filtering, coverage
thresholds and probability thresholds; evaluate precision/recall together.
Review annotation boundaries and failures without removing difficult samples.

If domain errors remain, acquire more permitted, diverse labelled training data
(for example full-resolution IDD training data), audit its label schema and
split overlap, and consider a larger SegFormer backbone as a separately budgeted
experiment. More epochs, a larger model and GPU execution alone do not guarantee
95% IoU. Fine-tuning, deployment latency and broader Indian-road generalization
must each be measured. Do not overwrite the original baseline or publish a target
as a measured achievement.

## Evidence and next action

| Evidence | Finding | Next action |
|---|---|---|
| Frozen, hash-checked 1,213/190/204 split manifest and complete 50-epoch history | The fixed GPU experiment completed with development-only selection | Keep the completed run immutable; reproduce only as a new experiment |
| Ignore-gradient, missing-mask, leakage and identity regression tests | Training contracts are exercised; tests are not accuracy scores | Keep them passing when changing the training recipe |
| Separate complete official-validation report and selected model hash | 92.26 / 97.47 / 94.52% deployed drivable metrics | Independent human-reviewed road sessions; not lane/pothole or safety claims |

The path is frozen train/development assignment → binary-label fine-tuning →
development-only checkpoint selection → separately frozen deployed configuration
→ complete official-validation evaluation → independent arithmetic/hash audit.
