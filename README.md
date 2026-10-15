# H&E ↔ OCT Registration Studio

Self-contained web app that registers 4× **H&E** whole-section scans (moving) onto **OCT en-face top-k
projections** (fixed, 2 µm/px) of the same specimen, and lets you inspect the result interactively.
Same look and file-browsing I/O as `oct_recon_AF_v2` (plain HTML/JS/CSS + Python standard-library server).

Pipeline (in `heoct/`, identical to `registration/v1`): H&E background removal + Macenko stain normalisation →
texture-based OCT background/debris removal + tile flat-field → GPU mirror/rotation/scale search → multi-scale affine →
5-level B-spline deformation (MIND similarity) → QC (lumen-landmark TRE, block-matching residual) →
`output/` (calibrated lossless TIFFs + one metadata JSON) → full-resolution figures.

## Start
```bash
./start_linux.sh                    # creates conda env "reg_env" on first use, opens http://127.0.0.1:8780/
# or, inside the env:
conda activate reg_env && python launch.py [--port 8780] [--no-browser]
```

## Using it
1. **Input** – *Single sample*: Browse… to a sample folder (H&E `*4x*.tif` and OCT `*recon_xy_topk*.tif` are detected), or pick each
   file; **Inspect** shows sizes, OCT calibration and thumbnails. *Batch*: Browse… to a data root, **Scan**, tick samples.
   The browser tags folders as **H&E + OCT pair** / **registered** and shows file sizes.
2. **Output** – output root + sample name → `<root>/<sample>/` (stages `01…06` and the final `output/`).
3. **Parameters** – deformation preset (fast / standard / accurate / affine-only), mirror search, rotation step, H&E pixel-size range,
   OCT tissue-mask controls, stage selection, **Save full-resolution previews** (stage 06, off by default); *Advanced* takes raw YAML merged over `configs/default.yaml`. Save/Load config.
4. **Run** – progress bar per sample and stage, live log, Cancel. Each run writes `<root>/<sample>/run_config.yaml`.
5. **Results & viewer** – table of registered samples with accuracy; **View** opens the interactive viewer:
   * modes: blend (opacity), swipe (slider or Shift-drag), checkerboard A/B (tile size, invert), flicker, false colour, side by side, H&E, OCT
   * deformable vs affine-only H&E, original colours, tissue outlines, 100 µm grid
   * wheel zoom to native pixels, drag pan, minimap, jump to QC ROIs, save view as PNG
   * click = pin: OCT px, µm, OCT scanner mm and the **H&E native pixel** under the cursor (from the dense transform map)
   * QC gallery (overview, accuracy, zoomed ROIs, preprocessing, full-res figure), output file links, QC page, report PDF

Keyboard (viewer): `+`/`−` zoom, arrows pan, `1`–`8` modes, `Space` swap/flicker, `I` invert checker, `F` fit.

## Deploy
The server binds to `127.0.0.1` by default (this machine only). Options:
* **SSH tunnel (recommended)**: on your laptop `ssh -L 8780:127.0.0.1:8780 user@server`, then open `http://127.0.0.1:8780/`.
* **Network**: `python launch.py --host 0.0.0.0 --token auto --allow-root /data/in --allow-root /data/out --no-browser`
  prints a URL with `?token=…`; the token is stored as a cookie after the first visit. `--allow-root` limits what can be browsed,
  read, written and served. Use HTTPS (reverse proxy, e.g. nginx/caddy) when exposing it beyond a trusted network.
* **Keep it running**: `deploy/regstudio.service` (systemd user service, instructions inside).

## Layout
```
launch.py, start_linux.sh      start the server
webapp/server.py               HTTP API (browse, detect, scan, inspect, run/job/cancel, results, tiles, coord, files)
webapp/static/                 index.html, app.js (I/O, browser, runs), viewer.js (interactive viewer), style.css
heoct/                         registration library (io, he_preproc, oct_preproc, features, register, qc, export, figures)
scripts/run_pipeline.py        CLI pipeline (the web app runs it as a subprocess)
scripts/make_report.py         Nature-Methods-style LaTeX/PDF report over a results root
scripts/epidermis_check.py     per-sample figure: epidermis kept in the OCT / registered-H&E / common masks
configs/default.yaml           all parameters (documented inline)
environment.yml                conda env "reg_env" (PyTorch CUDA 12.4 wheel)
deploy/regstudio.service       systemd unit
```
API reference: docstring at the top of `webapp/server.py`. Data and results are never committed (`.gitignore`).
