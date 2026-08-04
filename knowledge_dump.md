# Knowledge dump — OpenSfM (solar_panel_masking / ants-thermal-1)

A working reference of everything learned while running OpenSfM on the
`ants-thermal-1` thermal drone dataset. Covers bugs fixed, how matching/pair
selection really works, the reconstruction-fragmentation root cause,
georeferencing, the dense pipeline, config recommendations, and the workflow.

> Paths below are relative to `D:\S\ANTS\Repo\solar_panel_masking\code\third_party\OpenSfM`.
> Authoritative config source = `opensfm/config.py`. Companion docs: `doc/pipeline.md`,
> `doc/configuration.md`.

---

## 1. Environment & how to run things

- Conda env: **`opensfm`**. Python: `C:\Users\HSSL77\.conda\envs\opensfm\python.exe`
- OpenCV installed: **4.13.0** (has a `findFundamentalMat` RANSAC regression — see §3).
- Run a stage: `bin\opensfm.bat <stage> data\ants-thermal-1 [opts]`
- Run whole pipeline: `bin\opensfm_run_all.bat data\ants-thermal-1`
- Run lean pipeline (no `mesh`, georeferenced): `bin\opensfm_run_minimal.bat data\ants-thermal-1`

### Running Python directly (scripts/inspection)
- The `opensfm.bat` just calls `python opensfm_main.py` relying on the **activated** env.
- Calling `…\envs\opensfm\python.exe` **without activation crashes natively** with
  `0xC06D007F` (delay-load DLL not found) the first time a native extension method runs
  (e.g. `TopocentricConverter.to_topocentric` in `get_representative_points`). Symptom:
  silent hard crash (exit `-1066598273`), no Python traceback.
- Always invoke custom scripts via:
  ```
  cmd /c "conda run -n opensfm --no-capture-output python -u <script.py>"
  ```
- `conda run` prints a lot of vcvars/build-env noise; redirect to a file and grep.

### Available libs in the env
- `rasterio`: **NO**. `tifffile`: NO.
- `osgeo` (GDAL): **YES** ← use to read/write GeoTIFFs. `PIL`, `cv2`, `numpy`: YES.
- GDAL warns about `UseExceptions()` (harmless FutureWarning).
- PowerShell is the shell. Use `;` (not `&&`). Avoid `&&`/here-string terminators + redirection together.

### Important behavior
- OpenSfM commands **overwrite** outputs and **do not auto-skip** done work. Whole-run is
  always safe; partial runs save time by starting further down the chain.

---

## 2. The dataset — `data/ants-thermal-1`

- **1334 images** in `images/` (filename pattern `IRX_####.JPG`, sequential).
- Thermal camera: `640 x 512`, `perspective`, `focal_ratio = 1.5277`, make `AMODELX` / model `XL801`.
- **Metadata present**: GPS (lat/lon/alt), **OPK** (omega/phi/kappa), `capture_time` (unix epoch).
- **1 corrupt frame: `IRX_5490.JPG`** — no GPS, `focal_ratio = 0.0`, make/model `unknown`,
  swapped dims (`512 x 640`), `capture_time = 0.0`. It spawned a bogus camera
  `v2 unknown unknown 512 640 perspective 0.0` (focal fallback 0.85) in `camera_models.json`.
  → **Must be excluded** (see §4).
- Geometry (computed from GPS):
  - 335 m along flight line × **81 m across** (a wide 2D grid of many parallel lines, NOT a single line).
  - Median nearest-neighbour spacing **3.6 m**.
  - At ~46 m flight height + focal 1.5277 → each frame footprint ≈ **30 m × 24 m**, so
    neighbours a few metres apart overlap **70–90%**.
- `reference_lla.json`: lat `23.7859667`, lon `89.8240566`, **altitude 0** (topocentric origin).
  → DSM Z values are relative to this datum (mostly slightly negative; see §7).

---

## 3. Bug #1 — cv2 `findFundamentalMat` crash (match_features aborted)

- Symptom: `match_features` died with
  `cv2.error: (-215:Assertion failed) 0 <= _rowRange.start <= _rowRange.end <= m.rows in cv::Mat::Mat`
  at `opensfm/matching.py` `robust_match_fundamental` → `cv2.findFundamentalMat(p1,p2,FM_RANSAC,threshold,0.9999)`.
- Cause: **OpenCV 4.13.0 RANSAC regression** — asserts on degenerate point configs
  (duplicated/collinear/near-collinear points, common on low-texture thermal + flat panels).
- One bad pair aborted the whole matching stage. `opensfm_run_all.bat` uses bare `call`
  with no `errorlevel` check, so later steps kept running on **empty matches** →
  `create_tracks` 0 tracks → `reconstruct` 0 reconstructions → dense steps then
  `FileNotFoundError: …\undistorted\reconstruction.json`. (Those FileNotFoundErrors are
  fallout, not separate bugs.)
- **Fix applied** in `opensfm/matching.py` `robust_match_fundamental` (~line 919):
  wrapped the `cv2.findFundamentalMat` call in `try/except cv2.error` returning
  `(np.array([]), np.array([]))`, and guarded `F is None or mask is None or F.shape != (3,3)`.
  A degenerate pair now yields *no matches* instead of crashing the run.

---

## 4. Excluding the corrupt frame

- `DataSet` image list comes from `image_list.txt` if present, else `images/`.
- `image_list.txt` format: **one relative path per line**, e.g. `images/IRX_4549.JPG`
  (joined to dataset root; basename used as the image id). See `dataset.py` `_set_image_list`.
- Created `data/ants-thermal-1/image_list.txt` = all images **except `IRX_5490.JPG`** (1333 lines).
- After excluding it, **all** remaining images have GPS → the GPS-distance pair-selection
  strategy works cleanly (no "no-GPS images" fallback path). See `pairs_selection.py`
  `match_candidates_from_metadata` (splits gps / no-gps; only `reconstructions` with GPS run
  the GPS strategies).

---

## 5. Matching / pair selection — mechanics & speed

### Why it was slow
- Default `matching_gps_distance = 150` (and `matching_gps_neighbors = 0`).
- On this dense 3.6 m grid, **~1027 images fall within 150 m** of each image →
  ~685k pairs (≈ all-vs-all). Took "forever".
- If **all** of `matching_gps_distance`, `matching_gps_neighbors`, `matching_time_neighbors`,
  `matching_order_neighbors`, `matching_bow_neighbors`, `matching_vlad_neighbors`,
  `matching_graph_rounds` are `0` → OpenSfM matches **every** pair (`pairs_selection.py`
  `_run_matching_strategies`).

### How `matching_gps_neighbors` actually selects pairs
1. kD-tree over all image ground-points. With `matching_use_opk: true`, the point is
   **where the camera looks on the ground** (GPS + OPK projected), not the drone position —
   so "nearest" ≈ "footprints overlap most".
2. For each image, take the **k nearest** (`k = matching_gps_neighbors`).
3. These are only **candidates**; robust matching (fundamental/essential RANSAC) then keeps
   the ones that share real geometry. Non-overlaps are cheaply discarded.
- `matching_gps_neighbors = 16` → **~11,721 candidate pairs** (verified via dry-run). ~50–60× fewer than default.

### The 2D-grid insight (why you need >2–3 neighbours)
- "2–3 images on each side" is only **along-line (forward) overlap**. The survey is a 2D grid
  (81 m across → many parallel lines). An image's nearest neighbours span **several lines**;
  the **cross-line (side-overlap) neighbours are what stitch the lines together**.
- Match too few → lines disconnect → fragmentation (§6). 10 was too few; 16 (+ time-neighbours) is the fix.

### Strategies are a UNION
- `pairs = d | g | t | o | set(b) | set(v)` (distance | graph | time | order | bow | vlad).
- `matching_time_neighbors` matches consecutive frames by `capture_time` → guarantees the
  along-line chain (strongest overlap). `matching_order_neighbors` = by filename order.

---

## 6. Bug #2 — reconstruction fragmented → black patches in ortho (root cause)

- After the first good run: **37 disconnected components**, not 1:
  - component 0: **670 shots** (the only one dense-reconstructed)
  - component 1: **416 shots** (discarded for dense!)
  - 35 tiny components (2–49 shots); **86 images in no component at all**.
- OpenSfM dense-reconstructs **only `reconstructions[0]`**. So ~600 images' ground → holes.
- Result: **ortho.tif is ~33% black (value 0)**; the central dark patch is the seam between
  the two big unreconciled halves. DSM there is flat-filled at low altitude → the negative
  `(-2, -5, …)` pits (`-40.7` outlier = hole-fill pit).
- **Matching was NOT weak** (7,247 pairs, **1.82M track observations** ≈ 251 obs/pair). The
  problem was **graph connectivity** — too few bridging pairs (esp. cross-line) at
  `matching_gps_neighbors = 10`, plus low-texture thermal. Not a hole-fill bug.
- **Fix**: raise `matching_gps_neighbors` (16, or 20–24 if still split), add
  `matching_time_neighbors` (sequential bridging), and tune features for thermal (§8).
  After `reconstruct`, check the component count → should collapse to ~1.

### Diagnosing components
```python
import json
recs = json.load(open(r"data/ants-thermal-1/reconstruction.meshed.json"))
print(len(recs), "components")
for i,r in enumerate(recs): print(i, len(r["shots"]), "shots")
```

---

## 7. Georeferencing (DEM + ortho)

- `dense_merging` writes **topocentric (no CRS)** rasters by default. To georeference, pass
  **`--georeferenced`** (`commands/dense_merging.py` `--georeferenced` flag;
  `actions/dense_merging.py`: `output_crs = data.output_coordinate_system() if georeferenced else None`).
- `output_coordinate_system()` → `geo.decide_output_crs(gcp_crs, reference)` → auto-picks the
  **UTM zone from `reference_lla.json`**. For lon 89.824 / lat 23.786:
  zone = floor((89.824+180)/6)+1 = **45**, north → **EPSG:32645** (UTM 45N). Matches the desired CRS.
- The EO is stored/used in **EPSG:4326** internally; **no manual conversion needed** —
  `--georeferenced` handles the 4326 → UTM transform.
- `bin/opensfm_run_all.bat` and `bin/opensfm_run_minimal.bat` now pass `--georeferenced`.
- Outputs regenerated by `dense_merging --georeferenced`: `fused.las`, `fused.laz`, `fused.ply`,
  `dsm.tif`, `ortho.tif` (and `undistorted/point_cloud/`).
- **Vertical caveat**: DSM Z is relative to `reference_lla` altitude (0). Horizontal CRS (32645)
  is correct; if true orthometric/ellipsoid heights are needed, that's a separate datum adjustment.

### Negative DSM values
- The current (non-georeferenced) DSM is topocentric metres, origin at reference altitude 0.
  Range observed: `-40.7 … +6.75`, mean `0.05`, GSD `0.0256 m` (~2.56 cm/px), `DSM_NODATA = -9999`.
- Small negatives near the surface are normal relative heights; deep negatives are hole-fill pits.
  They become meaningful once georeferenced (after §6 is fixed so there are fewer holes).

---

## 8. Config recommendations for this dataset

`data/ants-thermal-1/config.yaml` (current recommended):
```yaml
matching_gps_neighbors: 16        # was 10 (too few -> fragmentation); bridges flight lines
matching_time_neighbors: 12       # sequential frames = strongest along-line links
matching_gps_distance: 0          # rely on neighbour count, not the 150 m radius
matching_use_opk: true            # neighbour = camera ground footprint, not drone pos

feature_type: SIFT                # more robust on low-contrast thermal than default HAHOG
sift_peak_threshold: 0.02         # default 0.1 -> lower detects more in low texture
feature_process_size: 2048
feature_min_frames: 4000

robust_matching_min_match: 16     # default 20; lower so sparse-but-good pairs survive

processes: 8                      # CPU core count

depthmap_num_matching_views: 6
depthmap_min_consistent_views: 2
```
- Full parameter reference: `doc/configuration.md` (but **`opensfm/config.py` is authoritative**).
- If still fragmented after re-run: raise `matching_gps_neighbors` to 20–24.

---

## 9. Dense pipeline — what's required vs skippable

Metashape-style goal → OpenSfM steps:

| Goal | Steps | Skippable? |
|---|---|---|
| Align/SfM | extract_metadata → detect_features → match_features → create_tracks → reconstruct | No |
| bridge to dense | undistort | No (dense runs in undistorted frame) |
| depth maps | dense_clustering → compute_depthmaps | No (clustering = required setup) |
| dense cloud | fuse_depthmaps (`fused.ply`) | No |
| DEM + ortho | dense_merging (`dsm.tif`, `ortho.tif`, `fused.las/.laz`) | No |

- **`mesh` is skippable** — only writes `reconstruction.meshed.json` (per-shot Delaunay meshes
  for the web viewer); **nothing in the dense chain reads it**. `undistort` and dense stages
  load plain `reconstruction.json`.
- **`dense_equalize` is optional** — produces `undistorted/equalization.json` for ortho
  colour/exposure balancing. Ortho is still produced without it. If you skip it, delete any
  stale `equalization.json` so dense_merging doesn't reuse it.
- Required-but-invisible intermediates: `exif/`, `features/`, `matches/`, `tracks.csv`,
  `undistorted/depthmaps/*.clean.npz` — leave them.
- To skip mesh, use `bin/opensfm_run_minimal.bat` (created).
- Optional config to reduce exports:
  ```yaml
  dense_pointcloud_export_las: false
  dense_pointcloud_export_laz: false   # keep only fused.ply
  ```
  `undistorted/point_cloud/` octree tiles = web viewer only; safe to ignore/delete.

---

## 10. Pipeline stages → files (read state from the folder)

| Stage | Produces |
|---|---|
| `extract_metadata` | `exif/*.exif`, `camera_models.json`, `reference_lla.json` |
| `detect_features` | `features/*.features.npz`, `reports/features.json` |
| `match_features` | `matches/*` (1 per image), `reports/matches.json` |
| `create_tracks` | `tracks.csv`, `reports/tracks.json` |
| `reconstruct` | `reconstruction.json`, `reports/reconstruction.json` |
| `mesh` | `reconstruction.meshed.json` (viewer) |
| `undistort` | `undistorted/` (`reconstruction.json`, `tracks.csv`, `images/`, `validity_masks/`) |
| `dense_equalize` | `undistorted/equalization.json` |
| `dense_clustering` | `undistorted/clusters.json`, `clusters_points.json`, `cluster_bboxes.json`, `neighbors_all.json`, `neighbors_best.json`, `depth_ranges.json`, `dense_crop_hull.json` |
| `compute_depthmaps` | `undistorted/depthmaps/*.clean.npz`, `reports/dense_depthmaps.json` |
| `fuse_depthmaps` | `undistorted/depthmaps/fused.ply`, `reports/dense_fusion.json` |
| `dense_merging` | `undistorted/depthmaps/dsm.tif`, `ortho.tif`, `fused.las/.laz`, `undistorted/point_cloud/` |

Every stage also writes `reports/<stage>.json`; newest one = last completed step.
Full workflow doc: `doc/pipeline.md`.

### "Changed X → re-run from Y" cheat sheet
- Added/removed images or edited `image_list.txt` → `extract_metadata` (everything)
- Feature settings (`feature_type`, `sift_peak_threshold`, `feature_process_size`) → `detect_features`
- Matching settings (`matching_gps_neighbors`, `matching_time_neighbors`, `robust_matching_min_match`) → `match_features`
- Reconstruction/BA settings → `reconstruct`
- Dense/depthmap settings (`depthmap_*`) → `compute_depthmaps`
- Need georeference only → `dense_merging --georeferenced` (minutes)

Work backwards from the goal: each stage needs the one above it. `match_features` saves only
at the very end (`save_matches` after `match_images`), so a mid-run crash leaves **no** partial
matches to clean up — safe to re-run.

---

## 11. Diagnostic snippets (run via `conda run -n opensfm`)

### Count components / coverage (do this after `reconstruct`)
```python
import json
recs = json.load(open(r"data/ants-thermal-1/reconstruction.meshed.json"))
print("components:", len(recs), " comp0 shots:", len(recs[0]["shots"]))
```

### Dry-run pair selection (verify pair count without full matching)
```python
from opensfm.dataset import DataSet
from opensfm import pairs_selection
d = DataSet('data/ants-thermal-1'); imgs = d.images()
exifs = {im: d.load_exif(im) for im in imgs}
pairs, rep = pairs_selection.match_candidates_from_metadata(imgs, imgs, exifs, d, {})
print("candidate pairs:", len(pairs), rep)
```

### Inspect GeoTIFFs (GDAL; rasterio not installed)
```python
from osgeo import gdal
import numpy as np
ds = gdal.Open(r"data/ants-thermal-1/undistorted/depthmaps/dsm.tif")
print("CRS:", ds.GetProjection() or "NONE")
print("GT:", ds.GetGeoTransform(), " nodata:", ds.GetRasterBand(1).GetNoDataValue())
a = ds.GetRasterBand(1).ReadAsArray()
print("range:", float(np.nanmin(a)), float(np.nanmax(a)))
```

### Image spacing / GPS extent
```python
import glob, json, numpy as np
from scipy.spatial import cKDTree
P=[]
for e in glob.glob(r'data/ants-thermal-1/exif/*.exif'):
    g=json.load(open(e,encoding='utf-8'))['gps']
    if 'latitude' in g:
        P.append([g['longitude']*111320*np.cos(np.radians(g['latitude'])), g['latitude']*111320])
P=np.array(P)
d,_=cKDTree(P).query(P,k=2); print("median spacing:", np.median(d[:,1]))
```

---

## 12. Small stuff / gotchas

- `opensfm_run_all.bat` uses bare `call` (no `errorlevel` check) → a failing stage is masked
  by cascading errors in later stages. Add `if errorlevel 1 exit /b 1` after each line if you
  want fail-fast.
- `extract_metadata` regenerates `camera_models.json` + `exif/` + `reference_lla.json` from
  images; excluding `IRX_5490` via `image_list.txt` means its bogus camera won't reappear.
- `reconstruction.json` (plain) = sparse SfM; `reconstruction.meshed.json` = same + per-shot
  meshes (viewer only).
- `DSM_NODATA = -9999.0`. Ortho is RGBA (4 bands uint8); holes show as 0.
- The dense stage depthmaps use **their own** neighbour selection
  (`depthmap_num_neighbors`, `depthmap_num_matching_views`) — independent of `matching_gps_neighbors`.
- `matching_use_opk: true` runs `find_best_altitude` (samples altitude every 25 m up to 8000 m)
  when no `relative_altitude`; fast. Works under activated env only (native).
- Created/modified files this session:
  - `opensfm/matching.py` — cv2 try/except fix.
  - `data/ants-thermal-1/config.yaml` — recommended config.
  - `data/ants-thermal-1/image_list.txt` — excludes `IRX_5490.JPG`.
  - `bin/opensfm_run_all.bat` — `dense_merging … --georeferenced`.
  - `bin/opensfm_run_minimal.bat` — lean pipeline (no `mesh`), georeferenced.
  - `doc/pipeline.md`, `doc/configuration.md` (existing), `knowledge_dump.md` (this file).
  - Removed erroneous duplicate `docs/configuration.md` (plural) — real docs live in `doc/`.
- Reference altitude = 0 means DSM elevations are relative; don't expect true ASL until a
  vertical datum is applied.
