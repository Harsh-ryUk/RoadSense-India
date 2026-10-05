# Draw and review road masks locally

This is a manual annotation tool, not an automatic labeller or accuracy result.
It serves only the frozen frames on `127.0.0.1`, shows no model predictions and
has no upload or server-side save endpoint. Real road masks are still pending.

## Open the studio

Run from the repository root with the validation dependencies installed:

```bash
python -m pip install -r requirements-validation.txt
python scripts/serve_road_annotation.py \
  --package runs/independent_road_test_20261004 --port 8765
```

Open `http://127.0.0.1:8765/`. Keep that terminal running while annotating; Ctrl+C
stops the server. Other hosts, arbitrary file paths, model weights and write
requests are not served. There is no public deployment or external font/CDN.

## Label one frame

1. Choose **Road** and click vertices around each visible drivable region. Use
   **Finish polygon** or Enter. Coordinates are stored as integer native pixels.
2. Choose **Non-road** and draw cutouts for vehicles, pedestrians, medians,
   sidewalks and other obstacles. Later polygons overwrite earlier polygons.
   Do not paint the hidden road under a vehicle. Road beneath shadows stays road.
3. Use **Genuinely uncertain** only for boundaries that cannot be determined,
   not to conceal difficult model errors. Every frame must retain valid pixels.
4. Inspect at increased canvas/browser zoom; toggle the drawn overlay to compare
   with the source. Concave polygons are supported; crossing edges, duplicate
   vertices, degenerate polygons and out-of-bounds coordinates fail import.
5. Enter your annotator alias and explicitly mark annotation complete. For a
   genuine no-visible-road frame, check the explicit negative declaration.
6. A human reviewer inspects the source and polygon layers, enters their actual
   alias and marks reviewed. The same person may review, but the importer records
   that as self-review. Do not invent another reviewer. Fill scenario tags honestly.

Enter finishes a polygon, Backspace removes its last vertex and Escape cancels
pending vertices while the canvas is focused. Numeric X/Y entry offers an
alternative to pointer placement. Drawing remains a visual task; a full screen-reader
or WCAG conformance audit has not been performed. Limits: 200 vertices per polygon,
100 polygons per frame, and drafts smaller than 20 MiB.

The preview is a coloured polygon overlay, not the exact exported raster boundary.
Pillow rasterization on import fills polygons at native resolution without
antialiasing. Review the resulting masks too before freezing final ground truth.

## Save and resume

**Export annotation draft** downloads JSON into the browser's chosen download
location. Confirm the file is saved. Browser autosave is only a convenience: it
does not save to the repository and can be lost. Export before closing or sharing
work with a reviewer. Use **Resume from exported JSON** with the same frozen plan.
Export before replacing unsaved work; mismatched plan/image identities are rejected.
Finish polygons before closing: active, unfinished vertices are not included in
browser autosave or exported drafts.

Partial drafts keep all frames. Missing annotations are not turned into blank
ground-truth masks. Do not use predictions as a shortcut for an unbiased test.

## Import without overwriting originals

The command below assumes the exported draft was saved to the stated input path.
Use a new output directory for every import; an existing output is rejected.

```bash
python scripts/import_road_annotations.py \
  --plan runs/independent_road_test_20261004/frame_plan.json \
  --annotations runs/annotation_input/roadsense_annotations.json \
  --output runs/independent_road_review_01
```

The importer verifies frame-plan, image, checkpoint and configuration hashes;
copies all source frames and the unchanged plan into the fresh package; and
creates native-size 8-bit grayscale PNGs only for explicitly completed annotations.
Values are 0 = non-road, 1 = visible drivable road and 255 = uncertain/ignored.
It preserves pending reviews and records the number of self-reviewed frames.
An entirely ignored mask is rejected; a genuine negative is retained.
Imports have `accuracy: null` and do not call the model. On failure, retain any
partial output for inspection and choose a new output directory after correcting
the input. The original frame set, weights and reports stay untouched.

## Freeze only after every mask is reviewed

Review native exported masks as well as the coloured drawing overlay. If you
change a mask after review, update its recorded final hash honestly before freezing.

```bash
python scripts/prepare_independent_road_test.py freeze \
  --plan runs/independent_road_review_01/frame_plan.json \
  --reviews runs/independent_road_review_01/annotation_reviews.json \
  --output runs/independent_road_review_01/manifest.json
```

Every selected frame needs a matching human-reviewed mask. Only then follow
[the frozen-model evaluation instructions](INDEPENDENT_ROAD_TEST.md#evaluate-once-retain-all-frames).
Do not tune on these test predictions. Keep self-review, polygon boundary
approximations, source resolution and correlated-session limitations in the report.

## Evidence → finding → next action

| Evidence | Finding | Next action |
|---|---|---|
| `frame_plan.json`, checkpoint/configuration hashes and source image hashes | The annotation inputs are bound before model predictions | Draw without seeing this model's outputs |
| Exported draft and `annotation_import.json` | Completion/review declarations and exact native masks are recorded; not independently certified | Inspect masks and obtain actual human review |
| Import/geometry/server tests on generated fixtures | Tool contracts can be checked without inventing real road labels | Retain those tests as engineering evidence, not accuracy evidence |
| Complete reviewed manifest | The original evaluator can enforce the frozen identities | Run once and report actual metrics, including failures |

```bash
python -m pytest -q tests/test_independent_road_test.py tests/test_road_annotation.py
node --check tools/road_annotation/app.js
```

The JavaScript command needs Node only for syntax checking. The studio itself
needs no Node runtime, frontend package install or additional model dependencies.
