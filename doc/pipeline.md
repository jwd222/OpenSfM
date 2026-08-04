# Pipeline stages

OpenSfM reconstruction is a fixed 12-stage chain. Each stage consumes the
previous stage's outputs and writes its own. This page tells you, for every
stage, **what it does, what files it reads, what files it writes, and how to
tell it has run** — so you can decide what to (re)run from the dataset folder
alone.

> Commands **overwrite** their outputs; they do **not** skip already-done work.
> "Run the whole thing" is always safe. Partial runs just save time by starting
> further down the chain.

## Running it

Whole pipeline (metadata → ortho/DSM/LAS), output georeferenced:

```
bin\opensfm_run_all.bat data\<dataset>
```

A single stage:

```
bin\opensfm.bat <stage> data\<dataset> [options]
```

## The chain

```
extract_metadata
      └─ detect_features
            └─ match_features
                  └─ create_tracks
                        └─ reconstruct
                              └─ mesh
                                    └─ undistort
                                          ├─ dense_equalize
                                          └─ dense_clustering
                                                └─ compute_depthmaps
                                                      └─ fuse_depthmaps
                                                            └─ dense_merging  (→ ortho / DSM / LAS)
```

Every stage also writes a small `reports/<stage>.json`; the newest of those is a
quick way to see the last completed step.

## Stage reference

| Stage | Does | Consumes | Produces (look for these to know it ran) |
|---|---|---|---|
| `extract_metadata` | Read EXIF, infer cameras + reference frame | `images/`, `image_list.txt` | `exif/*.exif`, `camera_models.json`, `reference_lla.json` |
| `detect_features` | Detect + describe keypoints | `images/`, `camera_models.json` | `features/*.features.npz`, `reports/features.json` |
| `match_features` | Pair selection + descriptor + robust matching | `features/`, `exif/` (GPS/OPK/time), `camera_models.json` | `matches/*` (1 file per image), `reports/matches.json` |
| `create_tracks` | Union-find tracks from matches | `matches/` | `tracks.csv`, `reports/tracks.json` |
| `reconstruct` | Incremental SfM (resection + BA) | `tracks.csv`, `features/`, `camera_models.json` | `reconstruction.json`, `reports/reconstruction.json` |
| `mesh` | Build shot visibility / surface mesh | `reconstruction.json` | `reconstruction.meshed.json` |
| `undistort` | Undistort images + recon into a clean frame | `reconstruction.meshed.json`, `images/`, `camera_models.json` | `undistorted/` (`reconstruction.json`, `tracks.csv`, `images/`, `validity_masks/`) |
| `dense_equalize` | Per-image exposure/white-balance/vignette | `undistorted/reconstruction.json`, `undistorted/tracks.csv` | `undistorted/equalization.json` |
| `dense_clustering` | Group shots into clusters; compute neighbours/ranges/hull | `undistorted/reconstruction.json`, `undistorted/tracks.csv` | `undistorted/clusters.json`, `clusters_points.json`, `cluster_bboxes.json`, `neighbors_all.json`, `neighbors_best.json`, `depth_ranges.json`, `dense_crop_hull.json` |
| `compute_depthmaps` | Per-view PatchMatch depthmaps + cleaning | clusters + `undistorted/reconstruction.json` + undistorted images | `undistorted/depthmaps/*.clean.npz`, `reports/dense_depthmaps.json` |
| `fuse_depthmaps` | SVO/TSDF fusion → dense cloud | `*.clean.npz`, clusters | `undistorted/depthmaps/fused.ply`, `reports/dense_fusion.json` |
| `dense_merging` | Merge cloud + render DSM/ortho + export | `fused.ply`, clusters, `equalization.json` | `undistorted/depthmaps/dsm.tif`, `ortho.tif`, `fused.las`/`fused.laz`, `undistorted/point_cloud/`, `reports/dense_merging.json` |

## Deciding what to (re)run

You can reason in either direction.

### Backward — from your goal

Each stage needs the one above it. If you only want the ortho and `fused.ply`
already exists, run just `dense_merging`. To redo geometry, go back to
`reconstruct` (and everything below it).

```
ortho / DSM / LAS  ← dense_merging ← fuse_depthmaps ← compute_depthmaps
                   ← dense_clustering ← undistort ← mesh ← reconstruct
                   ← create_tracks ← match_features ← detect_features
                   ← extract_metadata ← images/
```

### Forward — read the files

If a stage's "produces" files are present, that stage is done. Re-running from
a stage means: delete (mentally or actually) that stage's outputs and run that
line plus every line below it in `opensfm_run_all.bat`.

### Changed-something cheat sheet

| You changed… | Re-run from… |
|---|---|
| Added/removed images, edited `image_list.txt` | `extract_metadata` (everything) |
| `feature_type`, `sift_peak_threshold`, `feature_process_size`, … | `detect_features` |
| `matching_gps_neighbors`, `matching_time_neighbors`, `robust_matching_min_match`, … | `match_features` |
| Reconstruction / bundle-adjustment settings | `reconstruct` |
| Dense / depthmap settings (`depthmap_*`) | `compute_depthmaps` |
| Need the georeference only | `dense_merging --georeferenced` |

### Georeferencing

`dense_merging` writes **topocentric** (no CRS) rasters by default. Pass
`--georeferenced` to write `EPSG:<UTM zone>` (auto-selected from
`reference_lla.json`; e.g. lat 23.8 / lon 89.8 → `EPSG:32645`). The EO in
EPSG:4326 is used internally; no manual conversion is needed. `opensfm_run_all.bat`
already passes `--georeferenced`.

## Common pitfalls

- **Ortho has big black patches.** Usually means incomplete SfM coverage, not a
  hole-fill bug. Only `reconstructions[0]` is dense-reconstructed, so if the SfM
  splits into several disconnected components, the rest become holes. Check the
  component count in `reports/reconstruction.json` (or count keys in
  `reconstruction.json`); if it's fragmented, improve matching connectivity
  (raise `matching_gps_neighbors`, add `matching_time_neighbors`, tune features).
- **`reconstruction.json` missing → dense stages crash with `FileNotFoundError`.**
  A dense step ran before reconstruction succeeded (or reconstruction produced 0
  shots). Fix the SfM first.
- **`match_features` aborts mid-run.** An earlier stage (often a degenerate pair
  in robust matching, or an unactivated env) crashed; later stages then run on
  empty matches. Match errors must be fixed before tracks/reconstruction.
