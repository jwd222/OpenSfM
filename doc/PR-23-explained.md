# PR #23 — "Fix issue 22": In-Depth Walkthrough

> Branch: `fix-issue-22` · Target: `master` (OpenSfM/OpenSfM)
> Author: `DodgySpaniard` · Status: Open, awaiting review from `@YanNoun` and `@pierotofy`
> Commits (oldest → newest): `fd8654fc` → `34bae82b` → `1e8d2af2` → `4cb6427`
> Linked issue: [#22 — Orthophoto contains black patches and misaligned regions when processing thermal drone images](https://github.com/OpenSfM/OpenSfM/issues/22)

This document explains, in depth and from first principles, what PR #23 changes,
**why** each change exists, and how each change maps back to the symptoms reported
in issue #22. It includes worked examples, ASCII diagrams, and the actual code
snippets from the branch.

---

## 0. TL;DR

Issue #22 reports two artifacts in an orthophoto built from thermal drone imagery:

1. **Black patches** scattered across the output.
2. **Misaligned solar panels** where reconstruction clusters merge.

PR #23 is the proposed fix. It ships **four** changes, only one of which is the
"real" headline fix for the black patches:

| Commit | File | What it fixes | Symptom it targets |
|---|---|---|---|
| `fd8654fc` | `reconstruction_grower.cc` | Guards against `0` GPS STD in Bundle Adjustment | Misalignment (bad poses) |
| `34bae82b` | `camera_corrections.py` | Adds YPR correction for the **AModelX XL801** camera | Misalignment (wrong orientation metadata) |
| `1e8d2af2` | `matching.py` | Wraps `cv2.findFundamentalMat` in `try/except` | Crashes/instability on thermal pairs |
| `4cb6427` | `config.py`, `fusion.py`, `test_dense.py` | **Adaptive dense-fusion chunking** (opt-in flag) | **Black patches** (the core fix) |

---

## 1. Background: where "fusion" sits in the pipeline

When you run the dense part of OpenSfM, the typical chain is:

```
extract_metadata → detect_features → match_features → create_tracks
→ reconstruct → mesh → undistort
→ dense_equalize → dense_clustering → compute_depthmaps
→ fuse_depthmaps  ◄── this is where PR #23's core fix lives
→ dense_merging   ◄── produces the orthophoto / DSM
```

For each input image, `compute_depthmaps` produces a **depthmap** (a per-pixel
estimate of how far away the surface is, plus a confidence). `fuse_depthmaps`
then **fuses** those per-image depthmaps into a single, globally-consistent 3D
point cloud / surface. `dense_merging` then turns that surface into the
orthophoto (and DSM).

So: **if fusion leaves holes, the orthophoto ends up with holes** — which later
get inpainted (filled in) as something flat and dark. That is exactly the
"black patches" in issue #22.

### How fusion is chunked (the part that matters)

To keep memory and the GPU bounded, fusion does **not** process the whole scene
at once. Instead `opensfm/dense/fusion.py::fuse_chunks`:

1. Divides the scene's 3D space into a coarse grid of **voxel cells**.
2. Splits those cells into **chunks** using a KD-tree (a spatial partitioning
   tree), so each chunk has at most `depthmap_fusion_chunk_max_cells` cells.
3. For **each chunk**, picks a small **capped set of observer views** (images)
   — at most `depthmap_max_cluster_views` of them — that will contribute their
   depthmaps to fusing that chunk.
4. Fuses each chunk independently using only its chosen views.

The cap on views per chunk (`depthmap_max_cluster_views`) exists for performance:
you cannot afford to fuse a chunk against every image that sees it. But that cap
is exactly what can starve some cells — more on that in §4.

---

## 2. The issue (#22), in plain terms

Opened by `jwd222`. Dataset: a UAV **thermal** set from an **Autel EVO Max 4T**.
The thermal images are 8-bit JPGs with **three identical RGB channels** (each
channel carries the same thermal intensity). The full reconstruction pipeline
runs to completion, but the final orthophoto has:

- **Black patches** throughout.
- **Seam artifacts** + **misaligned solar-panel rows** where clusters merge.

Expected: a continuous orthophoto with no missing regions and aligned structures.
The reporter was unsure whether this is expected for thermal imagery or a real
pipeline bug.

---

## 3. How PR #23 relates to issue #22

The PR is the **fix under review** for that bug report. The GitHub linkage is
explicit and machine-actionable:

- The PR **title** is literally `Fix issue 22 - Orthophoto contains black
  patches and misaligned regions...`.
- The PR body's description functions as the `Fixes #22` trigger — that is why
  the PR sidebar shows *"Successfully merging this pull request may close these
  issues."* GitHub will auto-close #22 when (if) this PR merges into `master`.
- There is a bidirectional cross-reference: on issue #22 you can see
  *"DodgySpaniard mentioned this pull request"*, and the PR author `@`-mentioned
  maintainer `@YanNoun` and requested review from `@YanNoun` and `@pierotofy`.

In short: **#22 is the bug report; #23 is the proposed fix.** The four commits on
your `fix-issue-22` branch *are* this PR.

> ℹ️ Note on the camera: issue #22 reports an **Autel EVO Max 4T**, but that
> drone's thermal payload reports in EXIF as **AModelX XL801** (`make_model`
> → `amodelx_xl801`). So commit #2's YPR correction applies **directly** to the
> issue's images — it is the same sensor, not a different camera (see §8.1).

---

## 4. Root-cause deep dive (the "why")

### 4.1 The black patches = unfused "budget-limited" cells

This is the diagnosis in the PR description, stated precisely:

> The global fusion partitioner splits the scene into fixed-size KD-tree chunks
> and selects at most `depthmap_max_cluster_views` observer views per chunk.
> When a chunk's cells need more views than the budget allows, the leftover
> budget-limited cells go unfused and become holes that completion later
> inpaints (typically as dark, view-less ground).

Concretely, picture a chunk that spans a region **seen by many images**, but the
per-chunk view budget is small. Each image sees only part of the chunk. If the
budget is too small to "cover" every cell with at least one selected view, the
uncovered cells get **no depth data** → they are left as **holes** → the
completion step fills them with flat dark pixels → **black patch**.

### 4.2 The misalignment = a few independent culprits

The reporter also sees panels drifting apart across seams. The PR attributes that
to three separate, additive causes (each addressed by one of the smaller commits):

- **Zero GPS standard deviation** feeding Bundle Adjustment (BA), which
  over-weights a degenerate GPS prior and warps camera poses.
- **Wrong camera orientation (YPR) metadata** for a specific thermal camera,
  which orients depthmaps/poses incorrectly.
- **`cv2.findFundamentalMat` crashing** on degenerate thermal matches, which
  drops or destabilises image-pair matching.

---

## 5. The four commits, in depth

### 5.1 Commit `fd8654fc` — "Catching 0 values for gps STD in bundle"

**File:** `opensfm/src/lib/sfm/src/reconstruction_grower.cc` (around line 138)

**The problem (with the math).** In Bundle Adjustment, each camera's GPS position
is treated as a **prior**. The residual (cost) contributed by GPS is roughly:

```
cost_gps = Σ_dim  ( (gps_observed_dim − gps_estimated_dim) / std_dim )²
```

So the effective **weight** on each axis is `1 / std²`. If the EXIF/metadata
reports a standard deviation of **0** (degenerate/"null-island" values), that
weight goes to **infinity**:

| Reported `std` | Effective weight `1/std²` |
|---|---|
| `5.0` m (default) | `0.04` |
| `1.0` m | `1.0` |
| `1e-3` m | `1e6` |
| **`0`** | **`∞`** ← forces the optimizer to pin the camera exactly on the GPS point, ignoring all visual evidence |

That blows up the solve or skews poses, and skewed poses ⇒ misalignment at seams.

**The fix.** Introduce an epsilon and fall back to the defaults if any component
is effectively zero:

```cpp
constexpr double kEpsilon = 1e-8;
Vec3d gps_std;
if (gps.contains("latitude_std") && gps.contains("longitude_std") &&
    gps.contains("altitude_std")) {
  double longitude_std = gps["longitude_std"].cast<double>();
  double latitude_std  = gps["latitude_std"].cast<double>();
  double altitude_std  = gps["altitude_std"].cast<double>();
  if (latitude_std < kEpsilon || longitude_std < kEpsilon ||
      altitude_std < kEpsilon) {
    longitude_std = kDefaultGpsStd[0];
    latitude_std  = kDefaultGpsStd[1];
    altitude_std  = kDefaultGpsStd[2];
  }
  gps_std = Vec3d(longitude_std, latitude_std, altitude_std);
}
```

`kDefaultGpsStd` is `{5.0, 5.0, 15.0}` (metres), defined just above at
`reconstruction_grower.cc:122`. So a corrupt `std=0` now behaves like "we trust
GPS to about 5–15 m" instead of "trust it infinitely."

---

### 5.2 Commit `34bae82b` — "Adding camera OPK correction for AModelX XL801"

**File:** `opensfm/data/camera_corrections.py`

**Context.** Some cameras report yaw/pitch/roll (YPR) in their XMP metadata in a
way that needs per-camera fixing before OpenSfM can use the orientation. The file
holds a registry:

```python
# Each correction function takes (xmp_tags, geo) and returns a YPR array
# and a boolean indicating if pitch offset should be applied.
CorrectionFn = Callable[[Dict[str, Any],
                         Dict[str, Any]], Tuple[NDArray[Any], bool]]
```

The boolean means **"should the standard 90° pitch offset (that most DJI drones
need) be applied?"** The existing entry, `_fix_dji_fc7303`, returns `False`
*because the FC7303 does NOT need that offset* (and also uses `Flight*Degree`
fields because its gimbal reads are all zero):

```python
def _fix_dji_fc7303(xmp_tags, geo):
    if "latitude" in geo and "longitude" in geo:
        return np.array([
            float(xmp_tags["@drone-dji:FlightYawDegree"]),
            float(xmp_tags["@drone-dji:FlightPitchDegree"]),
            float(xmp_tags["@drone-dji:FlightRollDegree"]),
        ]), False          # ← no 90° pitch offset needed
    return np.array([None, None, None]), False
```

**The fix** adds a parallel entry for the **AModelX XL801**. It reads the
camera's own `@Camera:Yaw/Pitch/Roll` tags and returns `True`, meaning this
camera **does** need the standard 90° pitch offset applied:

```python
def _fix_amodelx_xl801(xmp_tags, geo):
    """
    The AModelX XL801 does require the 90 degree offset in pitch
    that other DJI drones do.
    """
    if "latitude" in geo and "longitude" in geo:
        return np.array([
            float(xmp_tags["@Camera:Yaw"]),
            float(xmp_tags["@Camera:Pitch"]),
            float(xmp_tags["@Camera:Roll"]),
        ]), True           # ← 90° pitch offset IS needed
    return np.array([None, None, None]), True

ypr_corrections: Dict[str, CorrectionFn] = {
    "dji_fc7303": _fix_dji_fc7303,
    "amodelx_xl801": _fix_amodelx_xl801,
}
```

The key is the lowercased `"{make}_{model}"` string, so any image whose EXIF
make/model resolve to `amodelx_xl801` will route through this correction.

**Why it matters for alignment.** A wrong 90° pitch means every shot's viewing
direction is rotated by 90° relative to truth → depthmaps project to the wrong
ground locations → panels that should align across neighbouring shots end up
offset. Correcting the per-camera YPR removes that systematic rotation.

> 🎯 This **is** issue #22's camera. The Autel EVO Max 4T's thermal sensor
> reports in EXIF as **AModelX XL801**, so its `make_model` key is exactly
> `amodelx_xl801` — the entry this commit adds. The 90°-pitch fix here is what
> removes the solar-panel misalignment reported in the issue.

> The branch's docstring has a small grammar slip ("XL801 **do** require" should
> read "**does** require"), but the **logic is consistent**: it returns `True` =
> apply the offset.

---

### 5.3 Commit `1e8d2af2` — "Fixing robust matching, adding try/catch"

**File:** `opensfm/matching.py` (around line 932)

**The problem.** During feature matching, OpenSfM estimates the **fundamental
matrix** between an image pair with RANSAC to reject outlier matches:

```python
F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, threshold, 0.9999)
```

With thermal imagery, matches can be sparse or geometrically degenerate, and
OpenCV can **raise** `cv2.error` here — which, unhandled, **crashes the entire
`match_features` step** (or at least that worker). A crash mid-pipeline means
missing correspondences and broken reconstruction.

**The fix** wraps the call and degrades gracefully:

```python
try:
    F, mask = cv2.findFundamentalMat(p1, p2, FM_RANSAC, threshold, 0.9999)
except cv2.error:
    logger.warning(
        "cv2.findFundamentalMat failed for {} matches".format(len(matches))
    )
    return None, np.array([])
```

Now a bad pair is **logged and skipped** (returns no inliers) instead of killing
the run. The downstream `if F is None ...` checks already handle the `None`
return, so the surrounding code is unchanged.

---

### 5.4 Commit `4cb6427` — "Added adaptative chunfing for dense"  ★ the core fix

**Files:** `opensfm/config.py`, `opensfm/dense/fusion.py`, `opensfm/test/test_dense.py`

This is the change that actually closes the black-patch hole. It is **opt-in**:

```python
# config.py (after depthmap_fusion_chunk_max_cells)
# Recursively split fusion chunks whose cells can't all be covered within the
# per-chunk view budget (depthmap_max_cluster_views), closing budget-driven
# fusion holes at the cost of more, smaller chunks.
depthmap_fusion_adaptive_chunking: bool = False
```

Because the default is `False`, **the existing/default behaviour is untouched.**
You opt in by setting `depthmap_fusion_adaptive_chunking: true` (the author
tested with `configs/aerial.yaml`).

`fuse_chunks` simply branches on the flag:

```python
if config["depthmap_fusion_adaptive_chunking"]:
    units, dsm_global_extent = _build_global_chunks_adaptive(...)
else:
    units, dsm_global_extent = _build_global_chunks(...)   # legacy path
```

The commit adds **two** new functions in `fusion.py`. Let's build up to them.

#### 5.4.1 The two view selectors, contrasted

Both pick a capped set of views for a chunk. The difference is the **order** in
which views are considered.

**Legacy `_select_chunk_views` (fusion.py:395)** — "keep if it helps":
iterate views **best-weight-first**; keep a view if it covers at least one
still-under-observed cell; stop when budget is spent or everything is covered.

**New `_select_chunk_views_greedy` (fusion.py:468)** — "maximum coverage": each
iteration pick the view (among **all** unselected) that covers the **most**
still-under-observed cells, tie-break by weight. Then a phase-2 quality fill.

**Worked example.** 6 cells in a row; three high-weight "narrow" views and one
low-weight "broad" view; budget = **1 view**.

```
Cells:      [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ]

View A (weight high): covers [0][1]
View B (weight high): covers [2][3]
View C (weight high): covers [4][5]
View D (weight LOW ): covers [0][1][2][3][4][5]   ← broad but "worst" weight

weight order: A, B, C, D        budget max_views = 1
```

| Selector | Walks in… | Picks | Cells left UNFUSED |
|---|---|---|---|
| Legacy `_select_chunk_views` | weight order → keeps **A** first | `["A"]` | **4** (cells 2,3,4,5) |
| New `_select_chunk_views_greedy` | max coverage → picks **D** (covers 6) | `["D"]` | **0** |

The legacy path spent its whole budget on a high-weight-but-narrow view *before*
it ever reached the low-weight broad view. The greedy path asks "which single
view covers the most ground?" first. With a bigger budget, greedy then fills the
remaining slots with the best skipped views (phase 2).

This is exactly the unit test `test_select_chunk_views_greedy_cover_prefers_broad_views`
in `test_dense.py`, which asserts the table above (budget 1 → `["A"]` vs `["D"]`;
budget 3 → `["D","A","B"]`; and with `min_obs=2`, budget 3 → `["D","A","B"]` but
2 cells still single-observer).

#### 5.4.2 `_build_global_chunks_adaptive` — the recursive partitioner

This is the adaptive counterpart of the legacy `_build_global_chunks`. Steps:

1. **Prescan** every fusable view once (subsampled), caching each view's
   `(cells, weights)` so depthmaps are not reloaded later.
2. **KD-tree split** the occupied cells into chunks of ≤ `max_chunk_cells`.
3. For each chunk, choose a **capped** view set via `_select_chunk_views_greedy`.
4. **The key recursion:** if a chunk's capped selection leaves **budget-limited**
   cells, **bisect the chunk along its longest splittable axis** and retry the
   capped selection on each half — repeat until every cell fits inside the
   budget, *or* the chunk can no longer be split geometrically.

In other words: **instead of accepting holes when a chunk is too big for its view
budget, keep splitting the chunk into smaller pieces until each piece's coverage
fits the budget.** The chunk size *adapts* to the view cap.

#### 5.4.3 Data-limited vs budget-limited (the crucial distinction)

The code distinguishes two failure kinds, because only one is repairable:

| Kind | Meaning | Can splitting help? |
|---|---|---|
| **DATA-limited** | A cell seen by **fewer than `min_observers` views in the whole prescan** — there simply aren't enough images of it. | ❌ No (more budget can't create images that don't exist) |
| **BUDGET-limited** | A cell that **does** have enough observers overall, but the per-chunk `max_views` cap ran out before reaching them. | ✅ Yes — split the chunk so each half needs fewer views |

The function reports both counts in a log line:

```python
logger.info(
    f"View selection: {n_partial_chunks}/{n_chunks} chunk(s) under "
    f"{min_observers} observer(s) — {n_data_limited} cell(s) DATA-limited "
    f"(< {min_observers} views see them in the prescan; more views "
    f"cannot help), {n_budget_limited} cell(s) BUDGET-limited (> "
    f"{max_views} views needed)"
)
```

#### 5.4.4 Worked example: the hole scenario (from the new test)

Setup: 6 cells in a row; views A, B, C each cover a 2-cell strip; budget
`max_views = 2`; `max_chunk_cells = 100` (so initially there is **one** chunk).

```
Cells:   [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ]
View A:   ◾◾
View B:         ◾◾
View C:               ◾◾
budget = 2 views per chunk
```

**Legacy path** (`_build_global_chunks`, one fixed chunk, budget 2):
greedy/weighted selection can afford only two of the three strips, so one strip
(2 cells) ends up with **no selected observer** → 2 holey cells.

```
one chunk (all 6 cells), budget 2  →  select A, B
                                        C's strip [4][5] has NO observer  ◼◼ = hole
```

**Adaptive path** (`_build_global_chunks_adaptive`):
the chunk is budget-limited (cells 4,5 had observers in the prescan — they're
not data-limited), so it **bisects along X** and retries each half:

```
                ┌─────────── split along longest axis (X) ───────────┐
                ▼                                                     ▼
  left half  cells {0,1,2}   right half  cells {3,4,5}
  observers: A {0,1}, B {2}   observers: B {3}, C {4,5}
  budget 2  → pick A, B        budget 2  → pick B, C
  all cells covered ✔          all cells covered ✔
                              ◼◼ holes = 0
```

Result: **2 chunks, 0 holes**, and the partition stays disjoint and complete
(every cell owned by exactly one chunk). This is precisely what
`test_adaptive_chunking_covers_cells_the_legacy_budget_left_as_holes` asserts:
legacy → 1 chunk with 2 holey cells; adaptive → 2 chunks with 0 holey cells and
all 6 cells accounted for exactly once.

---

## 6. Symptom → fix mapping

| Symptom from #22 | Root cause | Fixing commit |
|---|---|---|
| **Black patches** | Budget-limited fusion cells left unfused → inpainted as dark ground | **#4** adaptive chunking (recursive bisection so every cell fits the budget → nothing to inpaint) |
| Misalignment at seams | (a) `std=0` GPS prior in BA warps camera poses | **#1** GPS STD epsilon guard |
| Misalignment at seams | (b) wrong per-camera YPR orientation metadata | **#2** AModelX XL801 correction |
| Crashes / unstable pairs | `cv2.findFundamentalMat` throwing on degenerate thermal matches | **#3** `try/except cv2.error` |

The PR's before/after orthophoto images (in the PR description) show the result:
with `depthmap_fusion_adaptive_chunking: true`, the previously black/patchy
thermal orthophoto becomes continuous.

---

## 7. Tests added (in `opensfm/test/test_dense.py`)

1. **`test_select_chunk_views_greedy_cover_prefers_broad_views`**
   The 6-cell / A,B,C,D example from §5.4.1. Asserts the greedy selector picks
   the broad low-weight view the legacy weight-ordered pass misses, and behaves
   correctly at budgets 1 and 3 and with `min_obs=2`.

2. **`test_adaptive_chunking_covers_cells_the_legacy_budget_left_as_holes`**
   The 6-cell / A,B,C / budget-2 scenario from §5.4.4. Monkeypatches
   `_prescan_view_weights` with synthetic data and asserts:
   - Legacy builder → 1 chunk, **2** holey cells.
   - Adaptive builder → **2** chunks, **0** holey cells, all 6 cells assigned
     exactly once (disjoint + complete).

---

## 8. Caveats & things to be aware of

### 8.1 The "two cameras" are actually one
Issue #22 reports an **Autel EVO Max 4T**, but that drone's thermal sensor
reports in EXIF as **AModelX XL801** — which is exactly the `make_model` key
commit #2 targets (`amodelx_xl801`). So commit #2 applies **directly** to the
issue's images: it fixes the 90° pitch orientation that drives the panel
misalignment, and the PR's "Fix issue 22" title is fully consistent. (Earlier
drafts of this doc treated them as two different cameras — that was wrong.)

### 8.2 How the four commits relate to the issue
- **#4 (adaptive chunking)** is the headline fix for the black patches.
- **#2 (XL801 YPR correction)** directly targets the panel misalignment — it is
  the issue's own camera (see §8.1).
- **#1 (GPS std guard)** and **#3 (matching try/catch)** are defensive hardening
  for thermal pipelines; they only fire if the data hits those specific edge
  cases (zero GPS std / `findFundamentalMat` throwing). A reviewer might still
  prefer #1/#3 split into a separate hardening PR.

### 8.3 Default-off
`depthmap_fusion_adaptive_chunking` defaults to `False`. Existing users see **no
behaviour change** unless they opt in. Adaptive chunking trades "fewer, larger
chunks" for "more, smaller chunks" to guarantee coverage — so expect more fusion
chunks (and thus more DSM tiles / more seams) when enabled.

### 8.4 Adaptive chunking cannot fix everything
It only repairs **budget-limited** holes. **Data-limited** cells (genuinely
unobserved by enough images) are still reported and still get inpainted — there
is no algorithm that can fuse surface data that was never captured.

---

## 9. Local worktree state (this branch)

Your branch `fix-issue-22` matches `upstream/fix-issue-22` exactly and contains
exactly the four PR commits. The PR's changed-file set is:

```
opensfm/config.py
opensfm/data/camera_corrections.py
opensfm/dense/fusion.py
opensfm/matching.py
opensfm/src/lib/sfm/src/reconstruction_grower.cc
opensfm/test/test_dense.py
```

Your working tree also has **local-only changes that are NOT part of PR #23**:

- staged new file: `bin/opensfm_run_minimal.bat`
- unstaged modified:  `configs/object.yaml`

These do not appear in the PR diff, so treat them as your own local edits, not
part of what is under review.

---

## 10. How to try the fix

### 10.1 Install the worktree into your conda env (builds C++ + points env here)

This repo builds via **scikit-build-core** (see `pyproject.toml`), not the
legacy `setup.py` cmake path. The build command is therefore `pip install -e .`,
which uses the conda toolchain (CMake + MSVC + libs from `conda-win-64.lock`) —
**no vcpkg is needed**. Run it from inside the worktree:

```bat
conda activate opensfm
pip install -e .
```

> ⚠️ Do **not** use `python setup.py build_ext --inplace`. That bypasses
> scikit-build and runs the obsolete top of `setup.py`, which hardcodes a
> nonexistent vcpkg toolchain (`setup.py:40`) and fails with
> `Could not find toolchain file: ../vcpkg/...`. It is dead/legacy code on this
> repo; the real backend is scikit-build-core (the original install produced a
> `build/` dir with `.skbuild-info.json`, not a `cmake_build/` dir).

One `pip install -e .` from the worktree does three things at once:

1. **Builds the C++** via scikit-build → compiles the GPS guard from
   `reconstruction_grower.cc` (commit #1) into `pymap.pyd`.
2. **Re-points the env's `opensfm` to this worktree.** A common trap: if your
   conda env was installed editable from a *different* checkout (e.g. the main
   `OpenSfM` repo), then `import opensfm` resolves there and **none** of this
   PR's code runs. `pip install -e .` from the worktree fixes that — afterwards
   `opensfm.dense.fusion` resolves to the worktree. Verify:
   ```bat
   python -c "import opensfm.dense.fusion as f; print(f.__file__)"
   ```
   It should print a path inside `...\fix-issue-22\...`.
3. Keeps the editable install in scikit-build "redirect" mode (`.pyd` stay in
   `build/`).

### 10.2 Enable the flag and run

Use `configs/aerial.yaml` (a reasonable base for drone datasets) and add:

```yaml
depthmap_fusion_adaptive_chunking: true
```

Then re-run the dense stages (no need to redo SfM):

```bat
opensfm dense_equalize <dataset>
opensfm dense_clustering <dataset>
opensfm compute_depthmaps <dataset>
opensfm fuse_depthmaps <dataset>
opensfm dense_merging <dataset> --georeferenced
```

Compare the orthophoto before/after. With adaptive chunking on, the
budget-driven black patches should disappear; any remaining holes are
**data-limited** (not enough imagery of those areas) and will be logged.

### 10.3 Which fixes need the C++ build?

Since `pip install -e .` rebuilds the C++ anyway, this is mostly informational:

| Commit | Language | Needs C++ build to take effect? |
|---|---|---|
| `fd8654fc` — GPS `std=0` guard | **C++** (`reconstruction_grower.cc`) | ✅ Yes |
| `34bae82b` — XL801 YPR correction | Python | ❌ No |
| `1e8d2af2` — `findFundamentalMat` try/catch | Python | ❌ No |
| `4cb6427` — adaptive chunking (**the black-patch fix**) | Python | ❌ No |

So: for the black-patches fix alone you only need the Python code active
(step 10.1's env re-point). The C++ rebuild is only required for the optional
GPS guard — and that guard only matters if your EXIF actually reports
`*_std == 0`. `pip install -e .` from the worktree gives you both regardless.
