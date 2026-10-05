# Run the GPU benchmark in Colab

[A completed Tesla T4 experiment](../benchmarks/colab_t4_sample_20260930T202529Z/README.md)
measured 8.05/8.06/19.65 FPS for full512/full256/core256. This notebook runs real
pretrained models; it does not train on Indian-road data or measure accuracy
without annotations.
Colab GPU allocation, session duration and availability vary. Record the actual
GPU assigned to your session; do not assume a specific accelerator or FPS.

[Open the notebook](https://colab.research.google.com/github/Harsh-ryUk/Hybrid-Transformer-ADAS-India-Perspective/blob/main/notebooks/colab_benchmark.ipynb)

## Prepare the runtime

1. Open the notebook and select a GPU hardware accelerator in the runtime settings.
2. Run the setup cell. It clones this public repository, installs dependencies and
   prints the checked-out commit. An existing checkout is not silently overwritten.
3. If Colab requests a session restart after installation, restart and continue
   from the hardware-check cell. The project directory persists across a kernel
   restart, but not necessarily across a discarded VM.
4. Confirm that `torch.cuda.is_available()` is true and inspect `nvidia-smi`.
   A CPU session is rejected before GPU-labelled measurements can be produced.
5. Run the offline tests, then the three-profile experiment. First use downloads
   YOLOv8n, SegFormer-B0 and OWLv2 weights, so allow time and network access.
6. Generate the report, inspect the preview and download the evidence ZIP before
   the session ends. No repository credentials or Drive mount are needed.

The notebook installs `requirements-dev.txt`, including report-generation tools.
It does not force the Mac's Torch constraints onto Colab: the hosted Torch/CUDA
build must match the GPU runtime. Versions are recorded in each benchmark JSON;
the exported `pip-freeze.txt` preserves the installed environment.

The published run passed 70 tests on Python 3.13.15, Torch 2.11.0+cu128 and
Transformers 4.57.6. Installation downgraded Colab's preinstalled Transformers/
Hugging Face Hub and emitted conflicts for unused Gradio 6.26.0 and Diffusers
0.40.0. Those packages were not part of this benchmark; the warning and full
environment are preserved in its export. Do not treat that global environment
as a portable lock or combine unrelated Gradio/Diffusers work with it. Use a
fresh VM or an isolated project environment for another experiment and record
the actual versions and warnings.


## Run the sample experiment

The default notebook measures 100 frames × 3 repeats per profile, following 20
warmup frames per repeat. Each profile uses a fresh child process, while model
and tracking state persist across repeats inside that process. Video repeats
restart at the beginning; warmup is excluded but initializes that state. OWLv2
runs synchronously every 10 processed frames in the full profiles.

```bash
python scripts/benchmark_pipeline.py --device cuda --profile all --source sample --frames 100 --warmup 20 --repeats 3 --threads 2 --output runs/colab_sample/result.json
python scripts/summarize_benchmark.py --directory runs/colab_sample
```

| Output | What to inspect |
|---|---|
| `full512.json`, `full256.json`, `core256.json` | Device/GPU identity, versions, hashes, FPS, stage/tail timings, model revisions, OWL status |
| Corresponding CSV files | Every measured frame, action, road coverage, signal state and actual host thread count |
| Corresponding PNG previews | Qualitative detections/masks; not ground truth or accuracy proof |
| `README.md`, `comparison.png` | Human-readable profile comparison and chart |
| `pip-freeze.txt`, `nvidia-smi.txt`, `commit.txt` | Extra session/reproduction evidence exported by the notebook |

Sample input is Ultralytics' bundled `bus.jpg`, letterboxed to 1280×720 and
repeated. It has no motion, no held-out Indian-road coverage and no annotations.
The Mac measurements are already real CPU measurements; the new run supplies
additional GPU evidence, not a replacement for fabricated numbers.

## Measure your own footage

Upload a permitted dashcam clip through Colab's Files pane, then change the
notebook's `SOURCE` to its absolute path, for example `/content/dashcam.mp4`.
Change `OUTPUT_DIRECTORY` to `/content/Hybrid-Transformer-ADAS-India-Perspective/runs/colab_video`
to keep it separate from the sample experiment, then rerun measurement/report/export.
For the default protocol each repeat requires at least 120 decodable frames.
Each repeat reads the same first 120 frames; 300 distinct video frames are not
implied by 300 measured observations. Longer, varied workloads are needed to
study robustness and tail behavior.

Video frames are resized directly to the requested capture resolution rather
than letterboxed. Decode/resize time is included in `read_ms` and `wall_ms`.
No GUI display or video encoding is included. Rendering is on by default;
adding `--no-render` changes the workload and must be reported separately.
Do not publish footage containing faces/plates without appropriate permission.

## Understand the measurements

- FPS is measured frame count divided by measured elapsed wall time, not the
  average of per-frame reciprocal latency. It includes small trace/progress-log
  overhead; it excludes model construction, warmup and preview export.
- CUDA clocks synchronize each stage and frame completion. Times reflect finished
  work, including synchronization overhead; cross-stage overlap is intentionally
  absent. This is single-frame end-to-end latency, not batched GPU capacity.
- `gpu_peak_allocated_mib` and `gpu_peak_reserved_mib` are PyTorch allocator peaks
  reset after each warmup. They include already-resident models but omit driver
  memory, allocations by other libraries and pre-warmup transient peaks.
- `process_peak_rss_mb` retains its historical field name but the unit is MiB.
  It is whole-process peak host memory, including initialization, not GPU VRAM.
- `cpu_time_seconds` measures host process CPU time, not GPU execution time.
- Accuracy fields remain null. Faster inference does not establish better road
  recognition, safe braking, reliable tracking or Level 4 autonomy.

## Troubleshoot a failed run

| Symptom | Action |
|---|---|
| Requested CUDA unavailable | Select a GPU session and rerun the hardware check; do not relabel a CPU run |
| Import/version error after installation | Restart the session kernel and check versions; save the traceback if it persists |
| Download failure | Check Colab network access and retry; offline flags only work after all checkpoints are cached |
| CUDA out of memory | Stop other GPU jobs and restart a clean session; try `core256` separately, clearly labelling the reduced profile |
| OWLv2 benchmark fails | Inspect its completion/error status; never report an enabled full profile when inference did not complete |
| Source ended early | Supply more decodable frames or reduce warmup/frames; the runner intentionally fails rather than truncate silently |
| Report generator rejects protocols | All three files must share the same source, device, repeats, frame count and rendering setting |
| Slow full profile | Inspect periodic OWLv2 spikes; a fast median can coexist with poor p95. This is an observation, not necessarily an error |

## Save and publish evidence

Download the notebook's ZIP. Colab files can disappear when the VM is reclaimed.
Use [the results-update protocol](RESULTS.md) to add a separate benchmark directory
and update the README only after inspecting the raw records.

Official references: [Colab FAQ](https://research.google.com/colaboratory/faq.html)
and [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).
