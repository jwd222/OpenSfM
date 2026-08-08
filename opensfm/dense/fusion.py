# pyre-strict
"""Per-cluster SVO fusion with sub-volume splitting (pipeline phase 3).

Each cluster is augmented with N-hop neighbours, pre-scanned on CPU, split into
GPU-budget-bounded sub-volumes, fused (TSDF) with optional photometric refine,
and extracted to a ``fused_batch_*.ply``.  When DSM is enabled it also rasters
the per-cluster DSM/ortho tile (``dsm_ortho_batch_*.npz``): Pass 1 builds the
DSM geometry, then the corrected DSM drives the Pass-2 ortho colour bake.
The interleaving (recursive split → refine → Pass 1 → Pass 2) is what keeps
peak GPU/CPU memory bounded.
"""

import gc
import logging
import os
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from opensfm import context, log, pydense, pymap, types
from opensfm.dataset import UndistortedDataSet

from .common import (
    _CLUSTER_COLORS,
    scale_down_image,
)
from .crop import compute_sfm_crop_hull, hull_contains
from .dsm_ortho import (
    _dsm_footprint,
    _dsm_point_normals,
    _fill_dsm_holes,
    _fill_ortho_holes,
    _inject_ortho_detail,
    _ortho_gated_median,
    _shock_dsm_edges,
)
from .equalize import apply_equalization

logger: logging.Logger = logging.getLogger(__name__)

# GPU memory model for SVO-fuser retention budgeting — mirrors the C++ layout in svo_opencl.{h,cc}.
# sizeof(GPUVoxelSlot); see svo_opencl.h static_assert
_SVO_SLOT_BYTES: int = 36
# Refine images kept resident for BakeColors, per pixel per view: color RGBA8 (4) + TSDF-rendered depth f32 (4) + clean depth f32 (4).
_SVO_REFINE_IMG_BYTES_PX_VIEW: int = 12
# Transient refine scratch, per slot: grad (4) + grad_w (4) + adam m/v (8).
_SVO_REFINE_SCRATCH_BYTES_SLOT: int = 16

# Coverage safety factor applied to the median per-pixel surface footprint when deriving the "fine" voxel size.
_SAMPLING_COVERAGE_FACTOR: float = 2.0

# SVO voxel resolution levels: voxel = fine_sampling * multiplier. "fine" tracks
# the depthmaps' median surface sampling; "half"/"quarter" coarsen it 2x/4x.
_VOXEL_LEVEL_MULTIPLIER: Dict[str, float] = {
    "fine": 1.0,
    "half": 2.0,
    "quarter": 4.0,
}


class _SubVolume(NamedTuple):
    """Axis-aligned sub-volume for bounded SVO fusion."""
    core_min: NDArray   # (3,) world-space core bbox min
    core_max: NDArray   # (3,) world-space core bbox max
    ext_min: NDArray    # (3,) extended bbox (core + margin)
    ext_max: NDArray    # (3,) extended bbox (core + margin)
    view_ids: Set[str]  # views contributing to this sub-volume


class _FuseUnit(NamedTuple):
    """One disjoint KD-tree chunk fed to the fusion body.

    A unit owns a set of coarse cells (its share of the surface, disjoint from
    every other unit by construction) and the views to fuse them with — both
    produced by ``_build_global_chunks``.
    """
    views: List[str]                  # best-observer views to fuse this chunk
    cells: Set[Tuple[int, int, int]]  # owned coarse cells (disjoint)
    # owned empty footprint columns (DSM fill)
    holes: Set[Tuple[int, int]]


def _load_prescan_depth(
    data: UndistortedDataSet,
    sid: str,
    clean_cache: Optional[
        Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]]
    ],
) -> Optional[NDArray]:
    """Fetch a shot's cleaned depth from the in-RAM cache or disk (or None)."""
    if clean_cache is not None:
        cached = clean_cache.get(sid)
        return None if cached is None else cached[0]
    if not data.clean_depthmap_exists(sid):
        return None
    depth, _normal, _score, _conf = data.load_clean_depthmap(sid)
    return depth


def _project_view_cells(
    depth: NDArray,
    shot: "pymap.Shot",
    inv_coarse: float,
    depth_factor: float,
    subsample: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Subsample a cleaned depthmap and back-project it to coarse-cell coords.

    Returns ``(cells, depths)`` — ``cells`` ``(M, 3)`` int coarse-cell
    coordinates, ``depths`` ``(M,)`` the per-sample camera depth (metres).  Returns ``None`` if
    nothing survives.  ``depth_factor`` (> 0) drops samples beyond that multiple
    of the view's median depth.  Shared by the coverage prescan and the weight prescan.
    """
    h, w = depth.shape
    rs = np.arange(0, h, subsample)
    cs = np.arange(0, w, subsample)
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()
    d = depth[rr, cc]
    valid = d > 0
    rr, cc, d = rr[valid], cc[valid], d[valid]
    if len(d) == 0:
        return None

    if depth_factor > 0.0:
        med = float(np.median(d))
        if med > 0.0:
            keep = d <= depth_factor * med
            rr, cc, d = rr[keep], cc[keep], d[keep]
            if len(d) == 0:
                return None

    K = shot.camera.get_K_in_pixel_coordinates(w, h)
    R = shot.pose.get_rotation_matrix()
    t_vec = shot.pose.translation
    Kinv = np.linalg.inv(K)
    pts_px = np.vstack(
        [cc.astype(np.float64), rr.astype(np.float64), np.ones(len(d))]
    )
    pts_cam = Kinv @ pts_px * d  # (3, M) camera-frame, camera at origin
    pts_world = R.T @ (pts_cam - t_vec[:, None])
    cells = np.floor(pts_world * inv_coarse).astype(np.int32).T  # (M, 3)
    return cells, d


def _prescan_view_weights(
    data: UndistortedDataSet,
    sid: str,
    shot: "pymap.Shot",
    inv_coarse: float,
    depth_factor: float,
    subsample: int,
) -> Optional[Tuple[str, NDArray, NDArray]]:
    """Load + project one view for the global chunk prescan (thread worker).

    Returns ``(sid, ukey, w)`` — the view's unique coarse cells ``(M, 3)`` and
    their summed observation weight ``(M,)`` — or ``None`` if the view has no
        usable depth.  The per-sample weight is ``(1/depth)``: closer surfaces weigh more.
        Pure per-view work (disk load + numpy, both GIL-releasing), so many views run concurrently in a thread pool.
    """
    if not data.clean_depthmap_exists(sid):
        return None
    depth, _, _, _ = data.load_clean_depthmap(sid)
    proj = _project_view_cells(
        depth, shot, inv_coarse, depth_factor, subsample)
    if proj is None:
        return None
    cells, d = proj
    wsample = 1.0 / d
    ukey, inverse = np.unique(cells, axis=0, return_inverse=True)
    w = np.bincount(inverse.ravel(), weights=wsample, minlength=len(ukey))
    return sid, ukey, w


def _prescan_coarse_grid(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    voxel_size: float,
    coarse_factor: int,
    subsample: int = 4,
    clean_cache: Optional[
        Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]]
    ] = None,
    depth_factor: float = 0.0,
) -> Dict[Tuple[int, int, int], Set[int]]:
    """CPU pre-scan: subsample cleaned depthmaps, project to world, bucket.

    For each view, loads the cleaned depth, subsamples, back-projects to world
    coordinates, and records which views contribute to each coarse grid cell.
    Used inside a chunk to assign views to its GPU sub-volumes.

    Args:
        data: dataset for loading cleaned depthmaps.
        view_ids: ordered list of shot IDs to scan.
        reconstruction: for camera parameters.
        voxel_size: SVO fine voxel size.
        coarse_factor: coarse cell = coarse_factor * voxel_size.
        subsample: take every Nth pixel in each dimension.

    Returns:
        Grid mapping coarse cell coordinate to set of view indices
        (indices into ``view_ids``).
    """
    inv_coarse = 1.0 / (voxel_size * coarse_factor)
    grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)

    for view_idx, sid in enumerate(view_ids):
        depth = _load_prescan_depth(data, sid, clean_cache)
        if depth is None:
            continue
        proj = _project_view_cells(
            depth, reconstruction.shots[sid], inv_coarse, depth_factor,
            subsample,
        )
        if proj is None:
            continue
        cells, _ = proj
        for ukey in np.unique(cells, axis=0):
            grid[(int(ukey[0]), int(ukey[1]), int(ukey[2]))].add(view_idx)

    return grid


def _split_into_subvolumes(
    grid: Dict[Tuple[int, int, int], Set[int]],
    view_ids: List[str],
    voxel_size: float,
    coarse_factor: int,
    max_voxels: int,
    trunc_factor: float,
) -> List[_SubVolume]:
    """Split occupied coarse cells into sub-volumes fitting GPU budget.

    Uses recursive KD-tree style median-split along the longest axis.
    Each sub-volume gets a margin of ``trunc_factor * voxel_size`` for
    correct TSDF at boundaries.  Views are assigned to a sub-volume if
    they have coarse cells inside its extended (core + margin) bbox.
    """
    if not grid:
        return []

    coarse_size = voxel_size * coarse_factor
    voxels_per_cell = coarse_factor ** 3
    margin_world = trunc_factor * voxel_size
    margin_cells = int(np.ceil(trunc_factor / coarse_factor))
    max_cells = max(1, max_voxels // voxels_per_cell)

    all_coords = np.array(list(grid.keys()), dtype=np.int32)  # (N, 3)
    all_keys = list(grid.keys())

    def _make_subvolume(cell_indices: NDArray) -> _SubVolume:
        """Build a SubVolume from a set of coarse cell indices."""
        cells = all_coords[cell_indices]
        cmin = cells.min(axis=0)
        cmax = cells.max(axis=0)

        core_min = cmin.astype(np.float64) * coarse_size
        core_max = (cmax + 1).astype(np.float64) * coarse_size
        ext_min = (core_min - margin_world).astype(np.float32)
        ext_max = (core_max + margin_world).astype(np.float32)
        core_min = core_min.astype(np.float32)
        core_max = core_max.astype(np.float32)

        # Gather views from all cells in the extended bbox (core + margin).
        ext_cmin = cmin - margin_cells
        ext_cmax = cmax + margin_cells
        sv_views: Set[str] = set()
        for idx in range(len(all_coords)):
            c = all_coords[idx]
            if np.all(c >= ext_cmin) and np.all(c <= ext_cmax):
                for vi in grid[all_keys[idx]]:
                    sv_views.add(view_ids[vi])

        return _SubVolume(
            core_min=core_min,
            core_max=core_max,
            ext_min=ext_min,
            ext_max=ext_max,
            view_ids=sv_views,
        )

    def _split(cell_indices: NDArray) -> List[_SubVolume]:
        if len(cell_indices) <= max_cells:
            return [_make_subvolume(cell_indices)]

        cells = all_coords[cell_indices]
        spans = cells.max(axis=0) - cells.min(axis=0)
        axis = int(np.argmax(spans))

        if spans[axis] == 0:
            # Can't split further — accept over-budget.
            return [_make_subvolume(cell_indices)]

        median = int(np.median(cells[:, axis]))
        left_mask = cells[:, axis] <= median
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            # All cells on one side; force split at midpoint.
            mid = (cells[:, axis].min() + cells[:, axis].max()) // 2
            left_mask = cells[:, axis] <= mid
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                return [_make_subvolume(cell_indices)]

        return (
            _split(cell_indices[left_mask])
            + _split(cell_indices[right_mask])
        )

    return _split(np.arange(len(all_coords)))


def _render_dsm_patch_from_sv(
    fuser: "pydense.SVOFuser",
    sv: "_SubVolume",
    dsm_grid: NDArray,
    owner_grid: NDArray,
    leaf_idx: int,
    origin_x: float,
    origin_y: float,
    gsd: float,
    grid_w: int,
    grid_h: int,
    z_min: float,
    z_max: float,
) -> None:
    """Render the DSM for a leaf's core XY footprint and composite (MAX-z).

    Records per-cell ownership (``owner_grid`` = ``leaf_idx`` where this leaf
    won the height) so Pass 2 can bake the ortho from the post-processed DSM
    using exactly the views that produced each cell.  No colour is baked here.
    """
    # Compute the sub-grid pixel range corresponding to this sub-volume's core.
    col_start = max(0, int(np.floor((sv.core_min[0] - origin_x) / gsd)))
    col_end = min(grid_w, int(np.ceil((sv.core_max[0] - origin_x) / gsd)))
    row_start = max(0, int(np.floor((sv.core_min[1] - origin_y) / gsd)))
    row_end = min(grid_h, int(np.ceil((sv.core_max[1] - origin_y) / gsd)))

    patch_w = col_end - col_start
    patch_h = row_end - row_start
    if patch_w <= 0 or patch_h <= 0:
        return

    patch_origin_x = origin_x + col_start * gsd
    patch_origin_y = origin_y + row_start * gsd

    # Render the DSM for this patch (colour is baked later, in Pass 2).
    dsm_patch, _hillshade, _normals = fuser.render_dsm_ortho(
        patch_origin_x, patch_origin_y, gsd,
        patch_w, patch_h, z_min, z_max,
    )
    dsm_patch = np.asarray(dsm_patch, dtype=np.float32)

    # Composite DSM: MAX-z (highest surface wins) and record ownership.
    valid = ~np.isnan(dsm_patch)
    dst_slice = dsm_grid[row_start:row_end, col_start:col_end]
    higher = valid & (
        np.isnan(dst_slice) | (dsm_patch > dst_slice)
    )
    dst_slice[higher] = dsm_patch[higher]
    owner_grid[row_start:row_end, col_start:col_end][higher] = leaf_idx

    logger.info(
        f"    DSM patch {patch_w}×{patch_h}: "
        f"{int(np.count_nonzero(valid))} valid, "
        f"{int(np.count_nonzero(higher))} updated"
    )


def _pack_coarse_cells(
    cells: NDArray, origin: NDArray, span: NDArray
) -> NDArray:
    """Pack integer coarse-cell coords into unique int64 keys for fast lookup.

    Coordinates are offset by ``origin`` into ``[0, span)`` per axis and folded
    into a single int64.  Cells outside that range get key ``-1`` (so they never
    match a real, non-negative owned key).
    """
    c = cells - origin
    in_range = np.all((c >= 0) & (c < span), axis=1)
    keys = np.full(len(c), -1, dtype=np.int64)
    if np.any(in_range):
        cc = c[in_range]
        keys[in_range] = (cc[:, 0] * span[1] + cc[:, 1]) * span[2] + cc[:, 2]
    return keys


def _select_chunk_views(
    chunk_packed_sorted: NDArray,
    weighted_sids: List[str],
    view_ukey: Dict[str, NDArray],
    origin: NDArray,
    span: NDArray,
    max_views: int,
    min_obs: int = 1,
) -> Tuple[List[str], int]:
    """Pick a chunk's fusion views COVERAGE-FIRST, then fill to ``max_views``.

    The naive "top-``max_views`` by inverse-depth weight" can leave chunk cells
    whose only observers fall below the cut unfused — they become holes the
    completion later inpaints (often as dark, view-less ground).  Instead:

    Phase 1 (coverage): walk the views best-weight-first and KEEP a view only if
    it brings a still-under-observed cell closer to ``min_obs`` observers, until
    every cell has ``min_obs`` observers or the budget is spent.  ``min_obs`` > 1
    guards against the SVO accepting nothing (it needs ``svo_min_count`` samples)
    and against single-view outliers — each cell ends up seen by at least N
    selected views.  Reaches a lower-weight view when it is the only remaining
    observer of some cell.

    Phase 2 (quality fill): once the observer floor is met, top up to
    ``max_views`` with the highest-weight views skipped as redundant.

    ``chunk_packed_sorted`` is the chunk's cells as sorted packed keys (used for
    membership + local index); ``weighted_sids`` the chunk's views sorted by
    weight desc; ``view_ukey`` maps sid → its ``(M, 3)`` coarse cells.  Returns
    ``(selected_sids, n_under_observed_cells)`` — coverage views first, then
    fill.  A cell with fewer than ``min_obs`` observers in the whole candidate
    set cannot be satisfied and is counted in the returned deficit.
    """
    n_cells = len(chunk_packed_sorted)
    min_obs = max(1, int(min_obs))
    count = np.zeros(n_cells, dtype=np.int32)
    n_sat = 0  # cells already seen by >= min_obs selected views
    selected: List[str] = []
    sel: Set[str] = set()

    # Phase 1 — coverage to >= min_obs observers per cell.
    for sid in weighted_sids:
        if len(selected) >= max_views or n_sat >= n_cells:
            break
        uk = view_ukey.get(sid)
        if uk is None:
            continue
        qp = _pack_coarse_cells(uk, origin, span)
        pos = np.clip(np.searchsorted(chunk_packed_sorted, qp), 0, n_cells - 1)
        hit = chunk_packed_sorted[pos] == qp
        if not hit.any():
            continue
        local = pos[hit]
        was_sat = count[local] >= min_obs
        if was_sat.all():
            continue  # every cell it sees already has enough observers
        count[local] += 1
        n_sat += int(np.count_nonzero((count[local] >= min_obs) & ~was_sat))
        selected.append(sid)
        sel.add(sid)

    # Phase 2 — quality fill with the best skipped views.
    if len(selected) < max_views:
        for sid in weighted_sids:
            if len(selected) >= max_views:
                break
            if sid not in sel:
                selected.append(sid)
                sel.add(sid)

    return selected, n_cells - n_sat


def _select_chunk_views_greedy(
    chunk_packed_sorted: NDArray,
    weighted_sids: List[str],
    view_ukey: Dict[str, NDArray],
    origin: NDArray,
    span: NDArray,
    max_views: int,
    min_obs: int = 1,
) -> Tuple[List[str], int]:
    """Pick a chunk's fusion views COVERAGE-FIRST, then fill to ``max_views``.

    The naive "top-``max_views`` by inverse-depth weight" can leave chunk cells
    whose only observers fall below the cut unfused — they become holes the
    completion later inpaints (often as dark, view-less ground).  Instead:

    Phase 1 (coverage): greedily select the view that covers the most
    still-under-observed cells, breaking ties by inverse-depth weight, until
    every cell has ``min_obs`` observers or the budget is spent.  ``min_obs`` > 1
    guards against the SVO accepting nothing (it needs ``svo_min_count`` samples)
    and against single-view outliers — each cell ends up seen by at least N
    selected views.  This avoids exhausting the cap on many high-weight but
    narrow views before considering a lower-weight view that covers a broad
    uncovered region.

    Phase 2 (quality fill): once the observer floor is met, top up to
    ``max_views`` with the highest-weight views skipped as redundant.

    ``chunk_packed_sorted`` is the chunk's cells as sorted packed keys (used for
    membership + local index); ``weighted_sids`` the chunk's views sorted by
    weight desc; ``view_ukey`` maps sid → its ``(M, 3)`` coarse cells.  Returns
    ``(selected_sids, n_under_observed_cells)`` — coverage views first, then
    fill.  A cell with fewer than ``min_obs`` observers in the whole candidate
    set cannot be satisfied and is counted in the returned deficit.

    Selected by ``_build_global_chunks_adaptive``
    (``depthmap_fusion_adaptive_chunking``).
    """
    n_cells = len(chunk_packed_sorted)
    min_obs = max(1, int(min_obs))
    count = np.zeros(n_cells, dtype=np.int32)
    n_sat = 0  # cells already seen by >= min_obs selected views
    selected: List[str] = []
    sel: Set[str] = set()

    # Phase 1 — coverage to >= min_obs observers per cell.  Pre-project every
    # candidate once, then solve the capped maximum-coverage problem greedily.
    candidate_cells: Dict[str, NDArray] = {}
    for sid in weighted_sids:
        uk = view_ukey.get(sid)
        if uk is None:
            continue
        qp = _pack_coarse_cells(uk, origin, span)
        pos = np.clip(np.searchsorted(chunk_packed_sorted, qp), 0, n_cells - 1)
        hit = chunk_packed_sorted[pos] == qp
        if hit.any():
            candidate_cells[sid] = pos[hit]

    while len(selected) < max_views and n_sat < n_cells:
        best_sid: Optional[str] = None
        best_gain = 0
        for sid in weighted_sids:
            if sid in sel:
                continue
            local = candidate_cells.get(sid)
            if local is None:
                continue
            gain = int(np.count_nonzero(count[local] < min_obs))
            if gain > best_gain:
                best_sid = sid
                best_gain = gain
        if best_sid is None:
            break
        local = candidate_cells[best_sid]
        was_sat = count[local] >= min_obs
        count[local] += 1
        n_sat += int(np.count_nonzero((count[local] >= min_obs) & ~was_sat))
        selected.append(best_sid)
        sel.add(best_sid)

    # Phase 2 — quality fill with the best skipped views.
    if len(selected) < max_views:
        for sid in weighted_sids:
            if len(selected) >= max_views:
                break
            if sid not in sel:
                selected.append(sid)
                sel.add(sid)

    return selected, n_cells - n_sat


def _estimate_sampling_voxel_size(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    subsample: int = 8,
    max_views: int = 200,
) -> float:
    """Median surface sampling distance of the cleaned depthmaps (world units).

    Each valid depth pixel back-projects to a surface footprint of about
    ``depth / focal_px`` — the finest detail the data actually resolves. We take
    each view's median footprint, then the median across views, giving a robust,
    redundancy-agnostic estimate of the surface sampling, scaled by
    ``_SAMPLING_COVERAGE_FACTOR`` to keep ray-driven integration hole-free. Views are
    scanned in parallel and only a subsampled, transient slice of each depthmap
    is held, so peak RAM stays bounded.

    At most ``max_views`` views are sampled (seeded, so the derived size is
    reproducible across runs); the median is stable on a sample this size, so a
    large scene's voxel size costs a bounded scan rather than one per view.
    """
    if len(view_ids) > max_views:
        idx = np.random.default_rng(0).choice(
            len(view_ids), size=max_views, replace=False
        )
        view_ids = [view_ids[i] for i in sorted(idx.tolist())]

    def _view_median(sid: str) -> Optional[float]:
        if not data.clean_depthmap_exists(sid):
            return None
        depth, _normal, _score, _conf = data.load_clean_depthmap(sid)
        h, w = depth.shape
        d = depth[::subsample, ::subsample].ravel()
        d = d[d > 0]
        if d.size == 0:
            return None
        K = reconstruction.shots[sid].camera.get_K_in_pixel_coordinates(w, h)
        focal_px = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
        if focal_px <= 0.0:
            return None
        return _SAMPLING_COVERAGE_FACTOR * float(np.median(d)) / focal_px

    with ThreadPoolExecutor(max_workers=data.config["io_processes"]) as pool:
        medians = [m for m in pool.map(
            _view_median, view_ids) if m is not None]

    if not medians:
        raise RuntimeError(
            "Cannot derive SVO voxel size: no cleaned depthmaps with valid "
            "depth were found."
        )
    return float(np.median(medians))


def _resolve_voxel_size(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    config: Dict[str, Any],
) -> float:
    """Resolve the SVO voxel size from ``depthmap_fusion_svo_voxel_level``.

    The "fine" sampling is measured once over all fusable cleaned depthmaps and
    scaled by the level multiplier (fine=1x, half=2x, quarter=4x). One global
    size keeps the cell-ownership grid, sub-volumes, and DSM raster consistent
    across clusters.
    """
    level = str(config["depthmap_fusion_svo_voxel_level"]).lower()
    if level not in _VOXEL_LEVEL_MULTIPLIER:
        raise ValueError(
            f"Unknown depthmap_fusion_svo_voxel_level '{level}'; expected one "
            f"of {sorted(_VOXEL_LEVEL_MULTIPLIER)}."
        )
    fine = _estimate_sampling_voxel_size(data, view_ids, reconstruction)
    voxel_size = fine * _VOXEL_LEVEL_MULTIPLIER[level]
    logger.info(
        "SVO voxel size: level=%s, fine sampling=%.4fm -> voxel=%.4fm",
        level, fine, voxel_size,
    )
    return voxel_size


def _dilate_core_xy(
    cells: "Any",
    core: Set[Tuple[int, int, int]],
    margin: int,
) -> Set[Tuple[int, int, int]]:
    """Return the subset of ``cells`` whose XY column is within ``margin`` coarse
    cells (Chebyshev / square halo) of a ``core`` cell's XY column.

    Used to grow a disjoint KD-tree chunk's DSM/ortho render footprint into a
    thin overlap with its neighbours: the core (owned, unique) cells are dilated
    in plan view and the result is intersected with ``cells`` — the cells the
    chunk's own views actually observed — so the margin only covers geometry the
    chunk can render.  Z is ignored (the DSM is a 2.5D heightfield).
    """
    cells_list = list(cells)
    if margin <= 0 or not core or not cells_list:
        return {c for c in cells_list if c in core}

    from scipy import ndimage as _ndi

    cells_arr = np.asarray(cells_list, dtype=np.int64)
    core_arr = np.asarray(list(core), dtype=np.int64)
    # 2D bbox spanning both the core columns and the candidate cells (the halo
    # may reach candidates outside the core's own XY extent).
    x0 = int(min(cells_arr[:, 0].min(), core_arr[:, 0].min())) - margin
    y0 = int(min(cells_arr[:, 1].min(), core_arr[:, 1].min())) - margin
    x1 = int(max(cells_arr[:, 0].max(), core_arr[:, 0].max())) + margin
    y1 = int(max(cells_arr[:, 1].max(), core_arr[:, 1].max())) + margin
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    plan = np.zeros((h, w), dtype=bool)
    plan[core_arr[:, 1] - y0, core_arr[:, 0] - x0] = True
    dilated = _ndi.binary_dilation(
        plan, structure=np.ones((3, 3), dtype=bool), iterations=int(margin)
    )
    keep = dilated[cells_arr[:, 1] - y0, cells_arr[:, 0] - x0]
    return {tuple(int(v) for v in row) for row in cells_arr[keep]}


def _trim_sparse_ends(counts: NDArray, frac: float) -> Tuple[int, int]:
    """Return the ``[lo, hi)`` range after cutting an axis's sparse EXTREMITIES.

    ``counts`` is the per-index occupied-cell count along one axis.  Leading and
    trailing entries whose count is below ``frac * max(counts)`` are trimmed
    (walking in from each end until a dense index is hit); the dense interior —
    including genuine low-count dips inside it — is preserved.  The threshold is
    relative to the densest slice (``max``), not the median, so the sparse fringe
    being trimmed cannot itself drag the reference down.  ``frac`` <= 0 disables.
    """
    n = len(counts)
    if frac <= 0.0:
        return 0, n
    peak = float(counts.max()) if n else 0.0
    if peak <= 0.0:
        return 0, n
    thr = frac * peak
    lo = 0
    while lo < n and counts[lo] < thr:
        lo += 1
    hi = n
    while hi > lo and counts[hi - 1] < thr:
        hi -= 1
    return lo, hi


def _assign_footprint_holes(
    cells_arr: NDArray,
    labels: NDArray,
    n_chunks: int,
    close_cells: int,
    coarse_size: float = 1.0,
    crop_hull: Optional[NDArray] = None,
    trim_fraction: float = 0.0,
) -> List[Set[Tuple[int, int]]]:
    """Hand every empty interior footprint column to its nearest chunk.

    The KD-tree partitions only OCCUPIED cells, so the empty interior of the
    scene (plazas, canopy gaps) belongs to no chunk and would never be hole-
    filled.  We mark which empty coarse columns are *completable* and assign each
    to the chunk of its nearest occupied cell, so each chunk owns — and completes
    — a disjoint share of the empty footprint (single owner per column: no
    double-invented fills, nothing to feather at merge).

    Completability is decided by the global PCA-trimmed SfM ``crop_hull`` when
    available: an empty column is completable iff its centre lies inside the hull
    — the true, oriented survey outline.  This catches voids that touch the
    ragged scene boundary (canopy edges), which ``binary_fill_holes`` treats as
    "exterior" and drops, WITHOUT bulging into the genuine no-data corners
    (outside the hull), and is consistent with the final SfM-hull crop.  When no
    hull is available we fall back to the morphological enclosure (close +
    fill-holes) of the occupied plan, which cannot complete boundary-adjacent
    voids.

    ``cells_arr`` is ``(N, 3)`` occupied coarse cells, ``labels`` the chunk id
    per cell, ``coarse_size`` the metric side of a coarse cell.  Returns one
    ``{(x, y)}`` set of owned hole columns per chunk.
    """
    from scipy import ndimage as _ndi
    from scipy.spatial import cKDTree

    holes: List[Set[Tuple[int, int]]] = [set() for _ in range(n_chunks)]
    occ_xy = np.unique(cells_arr[:, :2], axis=0)
    if len(occ_xy) == 0:
        return holes

    o0 = occ_xy.min(axis=0)
    pw = int(occ_xy[:, 0].max() - o0[0]) + 1
    ph = int(occ_xy[:, 1].max() - o0[1]) + 1
    plan = np.zeros((ph, pw), dtype=bool)
    plan[occ_xy[:, 1] - o0[1], occ_xy[:, 0] - o0[0]] = True

    if crop_hull is not None:
        # Completable = empty columns whose metric centre is inside the survey
        # hull.  Test every empty column in the occupied bbox; the corners
        # (inside the bbox, outside the hull) are correctly rejected.
        er, ec = np.where(~plan)
        hx = ec + int(o0[0])  # plan-frame → global coarse XY
        hy = er + int(o0[1])
        # coarse cell centres, topocentric metres
        cxm = (hx + 0.5) * coarse_size
        cym = (hy + 0.5) * coarse_size
        keep = hull_contains(crop_hull, cxm, cym)
        hx = hx[keep]
        hy = hy[keep]
        logger.info(
            f"Footprint territory (SfM hull): occupied={int(plan.sum())} "
            f"cells, {len(hx)} empty in-hull column(s) of {int((~plan).sum())} "
            f"empty in the {pw}x{ph} coarse bbox"
        )
    else:
        # Fallback: morphological enclosure of the occupied plan (legacy; cannot
        # reach boundary-adjacent voids, but keeps interior plazas/courtyards).
        cx0, cx1 = _trim_sparse_ends(plan.sum(axis=0), trim_fraction)
        ry0, ry1 = _trim_sparse_ends(plan.sum(axis=1), trim_fraction)
        core = plan[ry0:ry1, cx0:cx1]
        if not core.any():
            return holes
        if close_cells > 0:
            n = int(close_cells)
            padded = np.pad(core, n)
            closed = _ndi.binary_closing(
                padded, structure=np.ones((3, 3), dtype=bool), iterations=n
            )[n:-n, n:-n]
        else:
            closed = core
        footprint = _ndi.binary_fill_holes(closed)
        hole_plan = footprint & ~core
        logger.info(
            f"Footprint territory (morph): close_cells={close_cells}, "
            f"trim={trim_fraction} ({int(plan.sum()) - int(core.sum())} fringe "
            f"cell(s) cut), occupied={int(core.sum())} cells, "
            f"footprint={int(footprint.sum())}, interior holes="
            f"{int(hole_plan.sum())} (core bbox {cx1 - cx0}x{ry1 - ry0} of "
            f"{pw}x{ph} coarse cells)"
        )
        hr, hc = np.where(hole_plan)
        hx = hc + cx0 + int(o0[0])  # core-frame → global coarse XY
        hy = hr + ry0 + int(o0[1])

    if len(hx) == 0:
        return holes

    tree = cKDTree(cells_arr[:, :2].astype(np.float64))
    _, idx = tree.query(np.column_stack([hx, hy]).astype(np.float64))
    hole_labels = labels[idx]
    for x, y, lab in zip(hx, hy, hole_labels):
        holes[int(lab)].add((int(x), int(y)))
    return holes


def _kdtree_chunk_labels(cells: NDArray, max_cells: int) -> NDArray:
    """Median KD-tree split of ``(N, 3)`` int coarse cells into chunks.

    Recursively splits the longest axis at its median until every leaf holds
    ``<= max_cells`` cells — the same geometry-coherent, non-sparse partition the
    GPU sub-volume split uses, here serving as the disjoint *assignment* unit.
    Returns a chunk-id label per input cell.
    """
    labels = np.full(len(cells), -1, dtype=np.int64)
    next_id = [0]

    def rec(idx: NDArray) -> None:
        if len(idx) <= max_cells:
            labels[idx] = next_id[0]
            next_id[0] += 1
            return
        sub = cells[idx]
        spans = sub.max(axis=0) - sub.min(axis=0)
        axis = int(np.argmax(spans))
        if spans[axis] == 0:
            labels[idx] = next_id[0]
            next_id[0] += 1
            return
        med = int(np.median(sub[:, axis]))
        left = sub[:, axis] <= med
        if not left.any() or left.all():
            mid = (int(sub[:, axis].min()) + int(sub[:, axis].max())) // 2
            left = sub[:, axis] <= mid
            if not left.any() or left.all():
                labels[idx] = next_id[0]
                next_id[0] += 1
                return
        rec(idx[left])
        rec(idx[~left])

    rec(np.arange(len(cells)))
    return labels


def _build_global_chunks(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    fusable: List[str],
    voxel_size: float,
    coarse_factor: int,
    max_chunk_cells: int,
    max_views: int,
    depth_factor: float,
    footprint_close: int = 32,
    subsample: int = 4,
    min_observers: int = 1,
    footprint_trim: float = 0.0,
    crop_hull: Optional[NDArray] = None,
) -> Tuple[
    List[_FuseUnit], Optional[Tuple[float, float, float, float, float, float]]
]:
    """Partition the whole scene into disjoint KD-tree chunks (no graph clusters).

    Replaces the cluster + per-cell-ownership machinery with a single geometric
    assignment:

    1. Prescan every fusable view once (subsampled), accumulating per coarse cell
       the summed inverse depth, and caching each view's ``(cells, weights)`` so
       depthmaps are not reloaded for the view assignment.
    2. KD-tree split the occupied cells into chunks of ``<= max_chunk_cells``
       (dense, spatially coherent, disjoint by construction).
    3. Give each chunk its ``max_views`` best observers — the views with the
       highest accumulated inverse depth over the chunk's cells (closest /
       highest-resolution coverage).

    Chunks are disjoint, so the fusion body's core clip yields unique point and
    DSM assignment with no ownership/despeckle/outlier passes.  Returns the units
    and the global DSM extent (from all occupied cells).
    """
    coarse_size = voxel_size * coarse_factor
    inv_coarse = 1.0 / coarse_size

    context.log_memory("kdtree chunking: global prescan start")
    # (sid, cells, weights)
    view_cells: List[Tuple[str, NDArray, NDArray]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, int(data.config["processes"]))
    ) as pool:
        futures = [
            pool.submit(
                _prescan_view_weights, data, sid, reconstruction.shots[sid],
                inv_coarse, depth_factor, subsample,
            )
            for sid in fusable
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                view_cells.append(res)
    context.log_memory("kdtree chunking: global prescan done")

    if not view_cells:
        return [], None

    cells_arr = np.unique(
        np.concatenate([uk for _, uk, _ in view_cells]), axis=0
    ).astype(np.int64)  # (N, 3)
    labels = _kdtree_chunk_labels(cells_arr.astype(np.int32), max_chunk_cells)
    n_chunks = int(labels.max()) + 1

    # Per-chunk owned cell sets (for the body's ``cell in unit.cells`` filter).
    chunk_cells: List[Set[Tuple[int, int, int]]] = [
        set() for _ in range(n_chunks)
    ]
    for row, lab in zip(cells_arr, labels):
        chunk_cells[int(lab)].add((int(row[0]), int(row[1]), int(row[2])))

    # Fast cell → chunk lookup (packed int64 + searchsorted).
    origin = cells_arr.min(axis=0)
    span = cells_arr.max(axis=0) - origin + 1
    packed_sorted = _pack_coarse_cells(cells_arr, origin, span)
    order = np.argsort(packed_sorted)
    packed_sorted = packed_sorted[order]
    chunk_sorted = labels[order]
    n_cells = len(packed_sorted)

    # Per-chunk view weights → keep each chunk's best observers.  Also count, per
    # cell, how many views observe it AT ALL (prescan resolution) — the ceiling on
    # what view selection can achieve, used below to tell "data-limited" cells
    # (intrinsically too few observers) from "budget-limited" ones.
    chunk_view_w: List[Dict[str, float]] = [dict() for _ in range(n_chunks)]
    # observers per cells_arr cell
    cell_obs = np.zeros(n_cells, dtype=np.int32)
    for sid, ukey, w in view_cells:
        qp = _pack_coarse_cells(ukey, origin, span)
        pos = np.clip(np.searchsorted(packed_sorted, qp), 0, n_cells - 1)
        hit = packed_sorted[pos] == qp
        cid = np.where(hit, chunk_sorted[pos], -1)
        valid = cid >= 0
        if not valid.any():
            continue
        cell_obs[order[pos[valid]]] += 1  # g indices unique within a view
        cw = np.bincount(cid[valid], weights=w[valid], minlength=n_chunks)
        for c in np.nonzero(cw)[0]:
            chunk_view_w[int(c)][sid] = float(cw[int(c)])

    # Empty interior footprint → owned by the nearest chunk so it gets completed
    # (the KD-tree only partitions reconstructed cells).
    chunk_holes = _assign_footprint_holes(
        cells_arr, labels, n_chunks, footprint_close,
        coarse_size, crop_hull=crop_hull,
        trim_fraction=footprint_trim,
    )
    n_holes = sum(len(h) for h in chunk_holes)

    # Coverage-first view selection (replaces top-max_views by weight): ensure
    # every chunk cell has an observer before spending the budget on quality.
    view_ukey: Dict[str, NDArray] = {sid: uk for sid, uk, _ in view_cells}
    # Chunk cell groups (rows of cells_arr) via one argsort on labels.
    lab_order = np.argsort(labels, kind="stable")
    cbounds = np.searchsorted(labels[lab_order], np.arange(n_chunks + 1))

    units: List[_FuseUnit] = []
    n_data_limited = 0    # cells with < min_observers views in the WHOLE prescan
    n_budget_limited = 0  # cells that had enough views but the budget ran out
    n_partial_chunks = 0
    for c in range(n_chunks):
        vw = chunk_view_w[c]
        weighted = sorted(vw, key=vw.get, reverse=True)
        rows = lab_order[cbounds[c]:cbounds[c + 1]]
        cpk = np.sort(_pack_coarse_cells(cells_arr[rows], origin, span))
        views, n_unc = _select_chunk_views(
            cpk, weighted, view_ukey, origin, span, max_views,
            min_obs=min_observers,
        )
        if n_unc > 0:
            # Split the deficit: cells that simply have too few observers in the
            # prescan (raising the budget cannot help) vs cells that DO have
            # enough but the per-chunk budget ran out before reaching them.
            n_dl = int(np.count_nonzero(cell_obs[rows] < min_observers))
            n_data_limited += n_dl
            n_budget_limited += n_unc - n_dl
            n_partial_chunks += 1
        units.append(_FuseUnit(
            views=views, cells=chunk_cells[c], holes=chunk_holes[c],
        ))
    if n_partial_chunks:
        logger.info(
            f"View selection: {n_partial_chunks}/{n_chunks} chunk(s) under "
            f"{min_observers} observer(s) — {n_data_limited} cell(s) DATA-limited "
            f"(< {min_observers} views see them in the prescan; more views "
            f"cannot help), {n_budget_limited} cell(s) BUDGET-limited (> "
            f"{max_views} views needed)"
        )

    xy_min = cells_arr[:, :2].min(axis=0).astype(np.float64) * coarse_size
    xy_max = (cells_arr[:, :2].max(axis=0) +
              1).astype(np.float64) * coarse_size
    dsm_global_extent = (
        float(xy_min[0]), float(xy_min[1]), float(xy_max[0]), float(xy_max[1]),
        float(cells_arr[:, 2].min()) * coarse_size,
        float(cells_arr[:, 2].max() + 1) * coarse_size,
    )
    logger.info(
        f"Global KD-tree chunking: {n_cells} occupied cells → {n_chunks} "
        f"chunk(s) (<= {max_chunk_cells} cells each); "
        f"{n_holes} empty footprint cell(s) assigned for completion"
    )
    context.log_memory("kdtree chunking: done")
    return units, dsm_global_extent


def _build_global_chunks_adaptive(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    fusable: List[str],
    voxel_size: float,
    coarse_factor: int,
    max_chunk_cells: int,
    max_views: int,
    depth_factor: float,
    footprint_close: int = 32,
    subsample: int = 4,
    min_observers: int = 1,
    footprint_trim: float = 0.0,
    crop_hull: Optional[NDArray] = None,
) -> Tuple[
    List[_FuseUnit], Optional[Tuple[float, float, float, float, float, float]]
]:
    """Partition the whole scene into disjoint, coverage-complete KD-tree chunks.

    Adaptive variant of the global geometric assignment, selected by
    ``depthmap_fusion_adaptive_chunking``:

    1. Prescan every fusable view once (subsampled), accumulating per coarse cell
       the summed inverse depth, and caching each view's ``(cells, weights)`` so
       depthmaps are not reloaded for the view assignment.
    2. KD-tree split the occupied cells into chunks of ``<= max_chunk_cells``
       (dense, spatially coherent, disjoint by construction).
    3. Give each chunk a capped set of observer views, chosen coverage-first by
       greedy maximum coverage (``_select_chunk_views_greedy``).
    4. Where the ``max_views`` cap leaves budget-limited cells — cells with
       enough observers in the prescan but starved by the per-chunk budget —
       recursively bisect the chunk along its longest splittable axis and retry
       the capped selection on each half, until every cell can be covered within
       the budget or the chunk can no longer be split geometrically.  The cell
       size adapts to the view cap instead of accepting coverage holes.  Cells
       with too few observers in the whole prescan (data-limited) are reported
       but cannot be repaired by splitting.

    Chunks are disjoint, so the fusion body's core clip yields unique point and
    DSM assignment with no ownership/despeckle/outlier passes.  Returns the units
    and the global DSM extent (from all occupied cells).
    """
    coarse_size = voxel_size * coarse_factor
    inv_coarse = 1.0 / coarse_size

    context.log_memory("kdtree chunking: global prescan start")
    # (sid, cells, weights)
    view_cells: List[Tuple[str, NDArray, NDArray]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, int(data.config["processes"]))
    ) as pool:
        futures = [
            pool.submit(
                _prescan_view_weights, data, sid, reconstruction.shots[sid],
                inv_coarse, depth_factor, subsample,
            )
            for sid in fusable
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                view_cells.append(res)
    context.log_memory("kdtree chunking: global prescan done")

    if not view_cells:
        return [], None

    cells_arr = np.unique(
        np.concatenate([uk for _, uk, _ in view_cells]), axis=0
    ).astype(np.int64)  # (N, 3)

    # Fast global-cell lookup for observer counting and per-leaf coverage.
    origin = cells_arr.min(axis=0)
    span = cells_arr.max(axis=0) - origin + 1
    packed_cells = _pack_coarse_cells(cells_arr, origin, span)
    packed_order = np.argsort(packed_cells)
    packed_sorted = packed_cells[packed_order]
    n_cells = len(packed_cells)

    # Count, per cell, every view that observes it in the global prescan.  This
    # is the observer ceiling: spatial splitting cannot repair a cell with fewer
    # than ``min_observers`` source views.
    cell_obs = np.zeros(n_cells, dtype=np.int32)
    for sid, ukey, w in view_cells:
        qp = _pack_coarse_cells(ukey, origin, span)
        pos = np.clip(np.searchsorted(packed_sorted, qp), 0, n_cells - 1)
        hit = packed_sorted[pos] == qp
        cell_obs[packed_order[pos[hit]]] += 1

    view_ukey: Dict[str, NDArray] = {sid: uk for sid, uk, _ in view_cells}
    view_weights: Dict[str, NDArray] = {sid: w for sid, _uk, w in view_cells}

    def _select_leaf_views(rows: NDArray) -> Tuple[List[str], int]:
        """Choose the capped best cover for one prospective KD-tree leaf."""
        leaf_keys = np.sort(packed_cells[rows])
        weights: Dict[str, float] = {}
        for sid, ukey in view_ukey.items():
            qp = _pack_coarse_cells(ukey, origin, span)
            pos = np.clip(
                np.searchsorted(leaf_keys, qp), 0, len(leaf_keys) - 1
            )
            hit = leaf_keys[pos] == qp
            if hit.any():
                weights[sid] = float(view_weights[sid][hit].sum())
        weighted = sorted(weights, key=weights.get, reverse=True)
        return _select_chunk_views_greedy(
            leaf_keys, weighted, view_ukey, origin, span, max_views,
            min_obs=min_observers,
        )

    def _split_rows(rows: NDArray) -> Optional[Tuple[NDArray, NDArray]]:
        """Bisect rows along their longest splittable spatial axis."""
        subset = cells_arr[rows]
        spans = subset.max(axis=0) - subset.min(axis=0)
        axis = int(np.argmax(spans))
        if spans[axis] == 0:
            return None
        median = int(np.median(subset[:, axis]))
        left = subset[:, axis] <= median
        if not left.any() or left.all():
            midpoint = (
                int(subset[:, axis].min()) + int(subset[:, axis].max())
            ) // 2
            left = subset[:, axis] <= midpoint
        if not left.any() or left.all():
            return None
        return rows[left], rows[~left]

    # Start with bounded geometric KD-tree leaves, then recursively split a
    # leaf whenever its best capped view cover leaves budget-limited cells.
    initial_labels = _kdtree_chunk_labels(
        cells_arr.astype(np.int32), max_chunk_cells
    )
    initial_order = np.argsort(initial_labels, kind="stable")
    initial_bounds = np.searchsorted(
        initial_labels[initial_order],
        np.arange(int(initial_labels.max()) + 2),
    )

    leaves: List[Tuple[NDArray, List[str], int, int]] = []

    def _refine_leaf(rows: NDArray) -> None:
        views, n_unc = _select_leaf_views(rows)
        n_data_limited = int(
            np.count_nonzero(cell_obs[rows] < min_observers)
        )
        n_budget_limited = n_unc - n_data_limited
        split = _split_rows(rows) if n_budget_limited > 0 else None
        if split is not None:
            _refine_leaf(split[0])
            _refine_leaf(split[1])
            return
        leaves.append((rows, views, n_unc, n_data_limited))

    for chunk_idx in range(len(initial_bounds) - 1):
        _refine_leaf(
            initial_order[initial_bounds[chunk_idx]:initial_bounds[chunk_idx + 1]]
        )

    n_chunks = len(leaves)
    labels = np.full(n_cells, -1, dtype=np.int64)
    chunk_cells: List[Set[Tuple[int, int, int]]] = []
    for chunk_idx, (rows, _views, _n_unc, _n_dl) in enumerate(leaves):
        labels[rows] = chunk_idx
        chunk_cells.append({
            (int(cell[0]), int(cell[1]), int(cell[2]))
            for cell in cells_arr[rows]
        })

    # Empty interior footprint → owned by the nearest chunk so it gets completed
    # (the KD-tree only partitions reconstructed cells).
    chunk_holes = _assign_footprint_holes(
        cells_arr, labels, n_chunks, footprint_close,
        coarse_size, crop_hull=crop_hull,
        trim_fraction=footprint_trim,
    )
    n_holes = sum(len(h) for h in chunk_holes)

    units: List[_FuseUnit] = []
    n_data_limited = 0    # cells with < min_observers views in the WHOLE prescan
    n_budget_limited = 0  # cells that had enough views but the budget ran out
    n_partial_chunks = 0
    for c, (_rows, views, n_unc, n_dl) in enumerate(leaves):
        if n_unc > 0:
            # Split the deficit: cells that simply have too few observers in the
            # prescan (raising the budget cannot help) vs cells that DO have
            # enough but the per-chunk budget ran out before reaching them.
            n_data_limited += n_dl
            n_budget_limited += n_unc - n_dl
            n_partial_chunks += 1
        units.append(_FuseUnit(
            views=views, cells=chunk_cells[c], holes=chunk_holes[c],
        ))
    if n_partial_chunks:
        logger.info(
            f"View selection: {n_partial_chunks}/{n_chunks} chunk(s) under "
            f"{min_observers} observer(s) — {n_data_limited} cell(s) DATA-limited "
            f"(< {min_observers} views see them in the prescan; more views "
            f"cannot help), {n_budget_limited} cell(s) BUDGET-limited (> "
            f"{max_views} views needed)"
        )

    xy_min = cells_arr[:, :2].min(axis=0).astype(np.float64) * coarse_size
    xy_max = (cells_arr[:, :2].max(axis=0) +
              1).astype(np.float64) * coarse_size
    dsm_global_extent = (
        float(xy_min[0]), float(xy_min[1]), float(xy_max[0]), float(xy_max[1]),
        float(cells_arr[:, 2].min()) * coarse_size,
        float(cells_arr[:, 2].max() + 1) * coarse_size,
    )
    logger.info(
        f"Global KD-tree chunking: {n_cells} occupied cells → {n_chunks} "
        f"chunk(s) (<= {max_chunk_cells} cells each); "
        f"{n_holes} empty footprint cell(s) assigned for completion"
    )
    context.log_memory("kdtree chunking: done")
    return units, dsm_global_extent


def fuse_chunks(
    data: UndistortedDataSet,
    processable: List[str],
    config: Dict[str, Any],
    reconstruction: types.Reconstruction,
    depth_ranges: Dict[str, Tuple[float, float]],
    best_neighbors: Dict[str, List[pymap.Shot]],
) -> None:
    """Fuse the cleaned depthmaps into ``fused_batch_*.ply`` (+ DSM/ortho tiles).

    Space is partitioned ONCE into disjoint KD-tree chunks of the occupied coarse
    grid (``_build_global_chunks``); each chunk is fused by its best inverse-depth
    observers, its points clipped to the chunk's core bbox and its DSM rendered
    for the same footprint.  Points and raster therefore share one disjoint
    partition, so the merge needs no de-duplication.  Depthmaps + colour images
    are loaded per chunk and released after it is fused.
    """
    fusable = [
        sid for sid in processable if data.clean_depthmap_exists(sid)
    ]
    if not fusable:
        logger.warning("No cleaned views available for fusion.")
        return

    fusable_set: Set[str] = set(fusable)

    # Per-image radiometric equalization (gain + vignetting), if dense_equalize
    # was run: applied to each source colour image as it is loaded for the bake,
    # so every baked colour (ortho blend, sharp-detail source, point cloud, mesh)
    # is sampled from images on a common radiometric frame — removing the
    # exposure/vignette steps between source images.
    equalization: Optional[Dict[str, Any]] = None
    if config["equalize_apply_in_bake"] and data.equalization_exists():
        equalization = data.load_equalization()
        logger.info(
            "Loaded equalization for %d image(s); applying during colour bake",
            len(equalization),
        )

    # Global SVO voxel size from the depthmaps' surface sampling — one value
    # shared by every chunk (coarse grid, sub-volumes, DSM).
    voxel_size = _resolve_voxel_size(data, fusable, reconstruction, config)
    coarse_factor = config["depthmap_fusion_svo_coarse_factor"]
    logger.info(
        "Fusing %d views at %.4fm voxel size", len(fusable), voxel_size,
    )

    crop_hull: Optional[NDArray] = None
    if config["dsm_enabled"]:
        if data.crop_hull_exists():
            crop_hull = data.load_crop_hull()
        else:
            crop_hull = compute_sfm_crop_hull(
                reconstruction, config["dense_crop_percentile"]
            )
        if crop_hull is not None:
            logger.info(
                "Hole completion bounded by SfM hull (%d vertices)",
                len(crop_hull),
            )

    # One global geometric partition: KD-tree split of the occupied coarse grid
    # into disjoint chunks, each fused by its best inverse-depth observers.
    chunk_max_cells = config["depthmap_fusion_chunk_max_cells"]
    if chunk_max_cells <= 0:  # auto = one GPU sub-volume's worth of cells
        chunk_max_cells = max(
            1, config["depthmap_fusion_svo_max_voxels"] // (coarse_factor ** 3)
        )
    context.log_memory("fusion: chunking start")
    if config["depthmap_fusion_adaptive_chunking"]:
        units, dsm_global_extent = _build_global_chunks_adaptive(
            data, reconstruction, fusable,
            voxel_size, coarse_factor,
            chunk_max_cells, config["depthmap_max_cluster_views"],
            depth_factor=config["dsm_territory_depth_factor"],
            footprint_close=config["dsm_footprint_close_cells"],
            footprint_trim=config["dsm_footprint_trim_fraction"],
            min_observers=config["depthmap_fusion_min_cell_observers"],
            crop_hull=crop_hull,
        )
    else:
        units, dsm_global_extent = _build_global_chunks(
            data, reconstruction, fusable,
            voxel_size, coarse_factor,
            chunk_max_cells, config["depthmap_max_cluster_views"],
            depth_factor=config["dsm_territory_depth_factor"],
            footprint_close=config["dsm_footprint_close_cells"],
            footprint_trim=config["dsm_footprint_trim_fraction"],
            min_observers=config["depthmap_fusion_min_cell_observers"],
            crop_hull=crop_hull,
        )
    context.log_memory("fusion: chunking done")
    logger.info(f"Fusion: {len(units)} chunk(s)")

    def _fuse_unit(unit_idx: int) -> None:
        """Fuse one assignment unit (KD-tree chunk or cluster territory):
        load its views, fuse its owned cells, save the PLY + DSM tile."""
        log.setup()
        unit = units[unit_idx]

        # ── Batch-preload helper ──────────────────────────────────────
        def _prepare_views(
            view_ids: List[str],
            clean_cache: Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]],
        ) -> Dict[str, Tuple[
            NDArray, NDArray, NDArray, NDArray,
            NDArray, NDArray, NDArray, Optional[NDArray],
        ]]:
            """Batch-load cleaned depthmaps + color images in parallel.

            Returns a dict mapping shot id → (K, R, t, depth, normal,
            color, vmask, confidence) for every successfully loaded view.
            """
            loadable = [sid for sid in view_ids if sid in clean_cache]
            if not loadable:
                return {}

            # Depthmaps were already decompressed once (shared with the
            # pre-scan); reuse them here instead of loading a second time.
            loaded = [clean_cache[sid] for sid in loadable]

            # Load validity masks in parallel.
            vmask_map = data.load_undistorted_validity_masks_parallel(
                loadable,
            )

            sizes = {
                sid: (depth.shape[1], depth.shape[0])  # (width, height)
                for sid, (depth, _n, _c) in zip(loadable, loaded)
            }

            def _load_scaled_color(sid: str) -> Tuple[str, NDArray]:
                w, h = sizes[sid]
                color = scale_down_image(
                    data.load_undistorted_image(sid), w, h
                )
                if equalization is not None and sid in equalization:
                    color = apply_equalization(
                        color, equalization[sid],
                        highlight_knee=float(
                            config["equalize_highlight_knee"]),
                    )
                return sid, color

            with ThreadPoolExecutor(
                max_workers=data.config["io_processes"]
            ) as pool:
                color_map = dict(pool.map(_load_scaled_color, loadable))

            result: Dict[str, Tuple[
                NDArray, NDArray, NDArray, NDArray,
                NDArray, NDArray, NDArray, Optional[NDArray],
            ]] = {}
            for sid, (depth, normal, confidence) in zip(
                loadable, loaded,
            ):
                h, w = depth.shape[:2]

                shot = reconstruction.shots[sid]
                K = shot.camera.get_K_in_pixel_coordinates(w, h)
                R = shot.pose.get_rotation_matrix()
                t = shot.pose.translation

                combined = depth.ravel() > 0

                if sid in depth_ranges:
                    _, max_d = depth_ranges[sid]
                    extreme = depth.ravel() > max_d
                    n_ext = int(np.count_nonzero(extreme & combined))
                    if n_ext > 0:
                        logger.warning(
                            "Shot %s: masking %d pixels with depth > %.1f",
                            sid, n_ext, max_d,
                        )
                    combined &= ~extreme

                if sid in vmask_map:
                    vmask_raw = vmask_map[sid]
                    mh, mw = vmask_raw.shape[:2]
                    if (mh, mw) != (h, w):
                        vmask_raw = cv2.resize(
                            vmask_raw, (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    combined &= vmask_raw.ravel() > 0

                reject = ~combined.reshape(h, w)
                n_rejected = int(np.count_nonzero(reject & (depth > 0)))
                if n_rejected > 0:
                    logger.info(
                        "Shot %s: combined mask rejected %d pixels",
                        sid, n_rejected,
                    )
                depth[reject] = 0.0

                vmask = (combined.reshape(h, w).astype(np.uint8)) * 255

                color = color_map[sid]  # pre-scaled to (w, h) at load time

                result[sid] = (
                    K, R, t, depth, normal, color, vmask, confidence,
                )

            return result

        # ── SVO fusion with sub-volume splitting ─────────────────
        # voxel_size is resolved once for the whole run (see fuse_chunks top).
        trunc_factor_cfg = config["depthmap_fusion_svo_trunc_factor"]
        max_voxels = config["depthmap_fusion_svo_max_voxels"]
        coarse_factor = config["depthmap_fusion_svo_coarse_factor"]

        # Step 1: the unit's views are pre-selected (cluster augmentation or
        # per-chunk best-observer ranking) — fuse with exactly these.
        augmented = unit.views
        unit_cells = unit.cells
        logger.info(f"Unit {unit_idx}: {len(augmented)} views")

        # Decompress each augmented view's cleaned depthmap ONCE (parallel)
        loadable_aug = [s for s in augmented if data.clean_depthmap_exists(s)]
        _loaded_dm = data.load_clean_depthmaps_parallel(loadable_aug)
        clean_cache: Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]] = {
            sid: (d, n, c)
            for sid, (d, n, _s, c) in zip(loadable_aug, _loaded_dm)
        }
        del _loaded_dm
        context.log_memory(f"unit {unit_idx}: preloaded cleaned depthmaps")

        # Step 2: Pre-scan on CPU (subsampled depth projection).
        grid = _prescan_coarse_grid(
            data, augmented, reconstruction,
            voxel_size, coarse_factor,
            clean_cache=clean_cache,
            depth_factor=config["dsm_territory_depth_factor"],
        )
        coarse_size = voxel_size * coarse_factor
        logger.info(
            f"Unit {unit_idx}: pre-scan found {len(grid)} "
            f"coarse cells (cell_size={coarse_size:.3f}m)"
        )
        context.log_memory(f"unit {unit_idx}: pre-scan grid built")

        n_cells_all = len(grid)
        # Core = the cells this chunk OWNS (the disjoint KD-tree assignment).
        # Points and the mesh are clipped to it so the merge needs no dedup.
        core_grid = {
            cell: views for cell, views in grid.items()
            if cell in unit_cells
        }
        if not core_grid:
            logger.info(
                f"Unit {unit_idx}: owns no cells, nothing to fuse"
            )
            gc.collect()
            return

        # DSM/ortho render footprint = the core dilated in XY by the feather margin
        margin = int(config["dsm_feather_margin_cells"])
        if config["dsm_enabled"] and margin > 0:
            render_set = _dilate_core_xy(grid.keys(), unit_cells, margin)
            grid = {
                cell: views for cell, views in grid.items()
                if cell in render_set
            }
        else:
            grid = core_grid
        logger.info(
            f"Unit {unit_idx}: owns {len(core_grid)}/{n_cells_all} occupied "
            f"coarse cells (+{len(grid) - len(core_grid)} margin for DSM/ortho)"
        )

        inv_coarse = 1.0 / coarse_size

        # Owned-cell lookup for the point/mesh extraction clip (core only, so
        # chunks stay disjoint regardless of the DSM render margin).
        owned_cells = np.array(list(core_grid.keys()), dtype=np.int64)
        owned_origin = owned_cells.min(axis=0)
        owned_span = owned_cells.max(axis=0) - owned_origin + 1
        owned_keys = np.sort(
            _pack_coarse_cells(owned_cells, owned_origin, owned_span)
        )
        # Render-cell AABB (core + margin) sizes the DSM/ortho window below.
        render_cells = np.array(list(grid.keys()), dtype=np.int64)

        # Completion territory (XY columns) = this chunk's occupied columns plus the empty footprint columns it was assigned to fill.
        territory_xy = np.array(
            sorted(
                {(int(c[0]), int(c[1])) for c in core_grid} | set(unit.holes)
            ),
            dtype=np.int64,
        ).reshape(-1, 2)

        # Step 3: Split into sub-volumes (over the core + margin render set).
        subvolumes = _split_into_subvolumes(
            grid, augmented, voxel_size, coarse_factor,
            max_voxels, trunc_factor_cfg,
        )
        logger.info(
            f"Unit {unit_idx}: split into "
            f"{len(subvolumes)} sub-volume(s)"
        )

        # Step 4: Fuse each sub-volume independently.
        #         If the GPU counting pass reveals more voxels than
        #         the budget, recursively split along the longest
        #         core axis.
        all_points: List[NDArray] = []
        all_normals: List[NDArray] = []
        all_colors: List[NDArray] = []
        # Per-sub-volume mesh fragments (Surface Nets of the TSDF), accumulated
        # like the point cloud; faces are re-indexed at write time.
        mesh_enabled = config["depthmap_fusion_mesh_enabled"]
        all_mverts: List[NDArray] = []
        all_mnormals: List[NDArray] = []
        all_mcolors: List[NDArray] = []
        all_mfaces: List[NDArray] = []
        # Leaf sub-volumes actually fused (post-split), for the Pass 2 ortho
        # bake; owner_grid[cell] = index into `leaves` of the leaf that won the
        # cell's height (MAX-z), so each cell is re-coloured by the views that
        # produced it.

        leaves: List["_SubVolume"] = []
        # Pass-1 fusers kept alive (bounded) for the Pass-2 ortho bake.
        leaf_fusers: List[Optional["pydense.SVOFuser"]] = []
        max_reuse_fusers = config[
            "depthmap_fusion_svo_bake_reuse_max_fusers"
        ]
        # VRAM-aware retention budget: a retained fuser keeps its GPU hash table (capacity * slot) plus the refine images BakeColors needs resident
        reuse_vram_budget = int(
            pydense.DepthmapClusterEstimator.device_memory_bytes(0)
            * config["depthmap_fusion_svo_bake_reuse_vram_fraction"]
        )
        retained_gpu_bytes = 0  # running sum of retained fusers' resident bytes

        # --- DSM/ortho grid (populated per sub-volume if svo method) ---
        dsm_enabled = config["dsm_enabled"]
        dsm_gsd = voxel_size / _SAMPLING_COVERAGE_FACTOR
        dsm_grid: Optional[NDArray] = None
        ortho_grid: Optional[NDArray] = None
        owner_grid: Optional[NDArray] = None
        dsm_origin_x: float = 0.0
        dsm_origin_y: float = 0.0
        dsm_z_min: float = 0.0
        dsm_z_max: float = 0.0
        dsm_w: int = 0
        dsm_h: int = 0
        dsm_global_origin_x: float = 0.0
        dsm_global_origin_y: float = 0.0
        dsm_global_w: int = 0
        dsm_global_h: int = 0
        win_r0: int = 0
        win_c0: int = 0

        if dsm_enabled and dsm_global_extent is not None:
            # Shared global georeference, sized to the occupied grid (computed
            # once in fuse_chunks, identical across chunks so tiles align).
            g_x0, g_y0, g_x1, g_y1, g_z0, g_z1 = dsm_global_extent
            min_xyz = np.array([g_x0, g_y0, g_z0], dtype=np.float64)
            max_xyz = np.array([g_x1, g_y1, g_z1], dtype=np.float64)
            extent_xy = max_xyz[:2] - min_xyz[:2]
            margin_xy = np.maximum(extent_xy * 0.05, dsm_gsd * 2)
            dsm_global_origin_x = float(min_xyz[0] - margin_xy[0])
            dsm_global_origin_y = float(min_xyz[1] - margin_xy[1])
            max_x = float(max_xyz[0] + margin_xy[0])
            max_y = float(max_xyz[1] + margin_xy[1])
            dsm_global_w = int(
                np.ceil((max_x - dsm_global_origin_x) / dsm_gsd))
            dsm_global_h = int(
                np.ceil((max_y - dsm_global_origin_y) / dsm_gsd))
            z_margin = (max_xyz[2] - min_xyz[2]) * 0.1 + voxel_size * 5
            dsm_z_min = float(min_xyz[2] - z_margin)
            dsm_z_max = float(max_xyz[2] + z_margin)

            # Re-fit the raster to this chunk's RENDER footprint (core + feather
            # margin) UNION its completion territory (owned empty footprint cells
            # may sit far from any reconstructed cell): the XY AABB → a window
            # into the global grid (padded a few px so hole-filling has context
            # at the very edge).  Keeps the in-RAM rasters proportional to the
            # chunk, not the whole scene, while still covering everything it must
            # render and complete.
            win_xy = np.vstack([render_cells[:, :2], territory_xy])
            owned_xy_min = win_xy.min(axis=0).astype(
                np.float64) * coarse_size
            owned_xy_max = (
                win_xy.max(axis=0) + 1
            ).astype(np.float64) * coarse_size
            pad = 4
            win_c0 = max(0, int(np.floor(
                (owned_xy_min[0] - dsm_global_origin_x) / dsm_gsd)) - pad)
            win_r0 = max(0, int(np.floor(
                (owned_xy_min[1] - dsm_global_origin_y) / dsm_gsd)) - pad)
            win_c1 = min(dsm_global_w, int(np.ceil(
                (owned_xy_max[0] - dsm_global_origin_x) / dsm_gsd)) + pad)
            win_r1 = min(dsm_global_h, int(np.ceil(
                (owned_xy_max[1] - dsm_global_origin_y) / dsm_gsd)) + pad)
            dsm_w = max(0, win_c1 - win_c0)
            dsm_h = max(0, win_r1 - win_r0)
            dsm_origin_x = dsm_global_origin_x + win_c0 * dsm_gsd
            dsm_origin_y = dsm_global_origin_y + win_r0 * dsm_gsd

            if dsm_w > 0 and dsm_h > 0:
                dsm_grid = np.full((dsm_h, dsm_w), np.nan, dtype=np.float32)
                ortho_grid = np.zeros((dsm_h, dsm_w, 3), dtype=np.uint8)
                owner_grid = np.full((dsm_h, dsm_w), -1, dtype=np.int32)
                logger.info(
                    f"Unit {unit_idx}: DSM window {dsm_w}×{dsm_h} at "
                    f"(r{win_r0},c{win_c0}) of global {dsm_global_w}×"
                    f"{dsm_global_h}, GSD={dsm_gsd:.4f}, "
                    f"Z=[{dsm_z_min:.2f}, {dsm_z_max:.2f}]"
                )
            context.log_memory(f"unit {unit_idx}: DSM grid allocated")

        # Batch-load all views upfront (parallel I/O).
        view_cache = _prepare_views(augmented, clean_cache)
        logger.info(
            f"Unit {unit_idx}: loaded {len(view_cache)} views"
        )
        context.log_memory(f"unit {unit_idx}: views loaded")

        def _build_fuser(
            sv: "_SubVolume",
        ) -> Tuple["pydense.SVOFuser", int]:
            """Create a configured fuser for a sub-volume and load its views."""
            fuser = pydense.SVOFuser()
            fuser.set_voxel_size(voxel_size)
            fuser.set_trunc_factor(trunc_factor_cfg)
            fuser.set_min_weight(config["depthmap_fusion_svo_min_weight"])
            fuser.set_num_levels(config["depthmap_fusion_svo_num_levels"])
            fuser.set_decimate_flat(
                config["depthmap_fusion_svo_decimate_flat"])
            fuser.set_edge_threshold(
                config["depthmap_fusion_svo_edge_threshold"]
            )
            fuser.set_min_count(config["depthmap_fusion_svo_min_count"])
            fuser.set_relative_min_weight(
                config["depthmap_fusion_svo_relative_min_weight"]
            )
            fuser.set_dsm_wall_cull_nz(config["dsm_wall_cull_nz"])
            fuser.set_device(0)
            fuser.set_bbox(sv.ext_min, sv.ext_max)
            n_loaded = 0
            for sid in sorted(sv.view_ids):
                result = view_cache.get(sid)
                if result is None:
                    continue
                K, R, t, depth, normal, color, vmask, conf = result
                fuser.add_view(
                    K, R, t, depth, normal, color, vmask,
                    confidence=conf, name=sid,
                )
                n_loaded += 1
            return fuser, n_loaded

        def _fuse_subvolume(sv: _SubVolume) -> None:
            """Fuse a sub-volume (geometry only), splitting if over budget.

            Renders the DSM patch and records per-cell ownership; the ortho is
            baked later (Pass 2) from the post-processed DSM.
            """
            nonlocal retained_gpu_bytes
            fuser, n_loaded = _build_fuser(sv)

            logger.info(
                f"  Sub-volume ({n_loaded} views), core "
                f"[{sv.core_min[0]:.1f},{sv.core_min[1]:.1f},"
                f"{sv.core_min[2]:.1f}]-"
                f"[{sv.core_max[0]:.1f},{sv.core_max[1]:.1f},"
                f"{sv.core_max[2]:.1f}]"
            )

            if n_loaded == 0:
                return

            # Count actual voxels on GPU.
            n_voxels = fuser.count_voxels()
            if n_voxels > max_voxels:
                logger.info(
                    f"  → {n_voxels} voxels > budget "
                    f"{max_voxels}, splitting …"
                )
                del fuser
                # Binary split along longest core axis.
                spans = sv.core_max - sv.core_min
                axis = int(np.argmax(spans))
                mid = (sv.core_min[axis] + sv.core_max[axis]) / 2.0
                margin = trunc_factor_cfg * voxel_size

                # Left half.
                left_core_max = sv.core_max.copy()
                left_core_max[axis] = mid
                left_ext_min = (
                    sv.core_min - margin
                ).astype(np.float32)
                left_ext_max = (
                    left_core_max + margin
                ).astype(np.float32)
                left_views = {
                    sid for sid in sv.view_ids
                    if sid in fusable_set
                }
                left_sv = _SubVolume(
                    core_min=sv.core_min.copy(),
                    core_max=left_core_max.astype(np.float32),
                    ext_min=left_ext_min,
                    ext_max=left_ext_max,
                    view_ids=left_views,
                )

                # Right half.
                right_core_min = sv.core_min.copy()
                right_core_min[axis] = mid
                right_ext_min = (
                    right_core_min - margin
                ).astype(np.float32)
                right_ext_max = (
                    sv.core_max + margin
                ).astype(np.float32)
                right_sv = _SubVolume(
                    core_min=right_core_min.astype(np.float32),
                    core_max=sv.core_max.copy(),
                    ext_min=right_ext_min,
                    ext_max=right_ext_max,
                    view_ids=left_views,  # same views
                )

                _fuse_subvolume(left_sv)
                _fuse_subvolume(right_sv)
                return

            # Voxels within budget — proceed with integration.
            refine_enabled = config[
                "depthmap_fusion_svo_refine_enabled"
            ]
            if refine_enabled:
                fuser.fuse_only()
                refine_nbrs = {
                    sid: [s.id for s in best_neighbors.get(sid, [])]
                    for sid in sorted(sv.view_ids)
                }
                fuser.refine_geometry(
                    iters=config[
                        "depthmap_fusion_svo_refine_iters"
                    ],
                    lambda_reg=config[
                        "depthmap_fusion_svo_refine_lambda_reg"
                    ],
                    neighbors=refine_nbrs,
                    lambda_anchor=config[
                        "depthmap_fusion_svo_refine_lambda_anchor"
                    ],
                    early_stop_rel=config[
                        "depthmap_fusion_svo_refine_early_stop_rel"
                    ],
                )
            else:
                fuser.fuse_only()

            # --- Render DSM patch (geometry) for this leaf's core footprint;
            #     the ortho is baked in Pass 2 from the corrected DSM. ---
            keep_for_bake = False
            if dsm_grid is not None and owner_grid is not None:
                leaf_idx = len(leaves)
                leaves.append(sv)
                _render_dsm_patch_from_sv(
                    fuser, sv, dsm_grid, owner_grid, leaf_idx,
                    dsm_origin_x, dsm_origin_y, dsm_gsd,
                    dsm_w, dsm_h, dsm_z_min, dsm_z_max,
                )
                # Retain this leaf's fuser for the Pass-2 ortho bake, bounded by BOTH a hard count cap and the live VRAM budget.
                cap = fuser.capacity()
                # Refine images are at depth resolution, uniform across a leaf's views; read W*H off any one of them.
                sample = next(
                    (view_cache[s] for s in sv.view_ids if s in view_cache),
                    None,
                )
                img_px = (
                    int(sample[3].shape[0]) * int(sample[3].shape[1])
                    if sample is not None
                    else 0
                )
                table_b = cap * _SVO_SLOT_BYTES
                img_b = n_loaded * img_px * _SVO_REFINE_IMG_BYTES_PX_VIEW
                scratch_b = cap * _SVO_REFINE_SCRATCH_BYTES_SLOT
                retained_cost = table_b  # table only after release
                next_active_peak = table_b + img_b + scratch_b
                n_retained = sum(f is not None for f in leaf_fusers)
                fits_vram = (
                    retained_gpu_bytes + retained_cost + next_active_peak
                    <= reuse_vram_budget
                )
                keep_for_bake = n_retained < max_reuse_fusers and fits_vram
                if keep_for_bake:
                    retained_gpu_bytes += retained_cost
                elif n_retained < max_reuse_fusers and not fits_vram:
                    logger.info(
                        f"  Pass-2 reuse throttled by VRAM: "
                        f"{retained_gpu_bytes / 1e9:.1f} GB tables retained + "
                        f"{retained_cost / 1e9:.1f} GB this table + "
                        f"{next_active_peak / 1e9:.1f} GB next-leaf peak > "
                        f"{reuse_vram_budget / 1e9:.1f} GB budget → "
                        f"rebuilding this leaf's fuser in Pass 2"
                    )
                leaf_fusers.append(fuser if keep_for_bake else None)

            # Always bake colors (voxels no longer store color).
            pts_arr, nrm_arr, clr_arr = fuser.extract_and_bake()
            pts = np.asarray(pts_arr, dtype=np.float32)
            nrm = np.asarray(nrm_arr, dtype=np.float32)
            clr = np.asarray(clr_arr, dtype=np.uint8)

            # Drop the TSDF "skirt" extracted just past the validity-mask edge the bake emits exactly (0,0,0) for points no view could colour
            if len(pts) > 0:
                colored = clr.any(axis=1)
                n_black = int((~colored).sum())
                if n_black:
                    pts = pts[colored]
                    nrm = nrm[colored]
                    clr = clr[colored]
                    logger.info(
                        f"    dropped {n_black} uncoloured (mask-edge) point(s)"
                    )

            if len(pts) > 0:
                inside_core = np.all(
                    (pts >= sv.core_min) & (pts <= sv.core_max),
                    axis=1,
                )
                pts = pts[inside_core]
                nrm = nrm[inside_core]
                clr = clr[inside_core]

            if len(pts) > 0:
                # Restrict to cells this cluster actually owns (exact, jagged-
                # boundary-safe dedup across clusters; the AABB core clip above
                # already separates sibling sub-volumes within this cluster).
                pcells = np.floor(
                    pts.astype(np.float64) * inv_coarse
                ).astype(np.int64)
                owned_mask = np.isin(
                    _pack_coarse_cells(pcells, owned_origin, owned_span),
                    owned_keys,
                )
                pts = pts[owned_mask]
                nrm = nrm[owned_mask]
                clr = clr[owned_mask]

            if len(pts) > 0:
                all_points.append(pts)
                all_normals.append(nrm)
                all_colors.append(clr)
                logger.info(
                    f"    → {len(pts)} points extracted"
                )

            # --- Surface Nets mesh (TSDF zero-set) for this sub-volume. ---
            # Must run while `fuser` is still alive (before release/del below):
            # extract the dual-contour mesh, keep the triangles this sub-volume
            # owns, then bake per-vertex colour with the same multi-view IRLS
            # consensus used for the point cloud.
            if mesh_enabled:
                mv_arr, mn_arr, mf_arr = fuser.extract_mesh()
                mverts = np.asarray(mv_arr, dtype=np.float32)
                mnrm = np.asarray(mn_arr, dtype=np.float32)
                mfaces = np.asarray(mf_arr, dtype=np.int64)
                if len(mfaces) > 0 and len(mverts) > 0:
                    # Keep a triangle when its centroid falls in this leaf's
                    # core AND an owned coarse cell — the same jagged-boundary
                    # dedup as the point cloud, but on faces (dropping a shared
                    # vertex would tear the surface).
                    cent = mverts[mfaces].mean(axis=1)
                    inside = np.all(
                        (cent >= sv.core_min) & (cent <= sv.core_max), axis=1
                    )
                    ccells = np.floor(
                        cent.astype(np.float64) * inv_coarse
                    ).astype(np.int64)
                    owned = np.isin(
                        _pack_coarse_cells(ccells, owned_origin, owned_span),
                        owned_keys,
                    )
                    mfaces = mfaces[inside & owned]
                if len(mfaces) > 0:
                    # Compact to referenced vertices and re-index the faces.
                    used = np.unique(mfaces)
                    remap = np.full(len(mverts), -1, dtype=np.int64)
                    remap[used] = np.arange(len(used))
                    mverts = mverts[used]
                    mnrm = mnrm[used]
                    mfaces = remap[mfaces].astype(np.int32)
                    mclr = np.asarray(
                        fuser.bake_colors(mverts, mnrm), dtype=np.uint8
                    )
                    all_mverts.append(mverts)
                    all_mnormals.append(mnrm)
                    all_mcolors.append(mclr)
                    all_mfaces.append(mfaces)
                    logger.info(
                        f"    → mesh: {len(mverts)} verts, "
                        f"{len(mfaces)} faces"
                    )

            if not keep_for_bake:
                del fuser
            else:
                # Strip the retained fuser to its hash table only — the refine images + grad/adam are dead weight between passes
                fuser.release_refine_buffers()

        # --- Pass 1: build the global DSM geometry (no colour yet). ---
        for sv in subvolumes:
            _fuse_subvolume(sv)
            gc.collect()

        # --- Finalize DSM, then bake the ortho from the CORRECTED DSM. ---
        # Pass 1 produced the raw composited DSM + per-cell ownership.  We first
        # post-process the DSM (fill no-data holes, then sharpen building edges
        # with the shock filter), and only THEN bake the ortho (Pass 2): each
        # leaf re-fuses its views and colours the cells it owns by projecting
        # the FINAL heights into its images.  This keeps DSM↔ortho consistent
        # and gives the ortho the sharpened edges (no bake↔DSM chicken-and-egg).
        if dsm_grid is not None:
            valid_count = int(np.count_nonzero(~np.isnan(dsm_grid)))
            logger.info(
                f"Unit {unit_idx}: DSM mesh raster, "
                f"{valid_count}/{dsm_grid.size} valid cells"
            )

            # debug: which cells were reconstructed BEFORE the hole-fill (vs
            # hole-filled afterwards) — used by the bake-category raster below.
            orig_valid = ~np.isnan(dsm_grid)

            # Territory mask at DSM resolution: the coarse columns this chunk owns (occupied + assigned empty footprint), upsampled to the window.
            terr_o = territory_xy.min(axis=0)
            terr_w = int(territory_xy[:, 0].max() - terr_o[0]) + 1
            terr_h = int(territory_xy[:, 1].max() - terr_o[1]) + 1
            terr_plan = np.zeros((terr_h, terr_w), dtype=bool)
            terr_plan[territory_xy[:, 1] - terr_o[1],
                      territory_xy[:, 0] - terr_o[0]] = True
            col_c = np.floor(
                (dsm_origin_x + (np.arange(dsm_w) + 0.5) * dsm_gsd) * inv_coarse
            ).astype(np.int64) - int(terr_o[0])
            row_c = np.floor(
                (dsm_origin_y + (np.arange(dsm_h) + 0.5) * dsm_gsd) * inv_coarse
            ).astype(np.int64) - int(terr_o[1])
            col_ok = (col_c >= 0) & (col_c < terr_w)
            row_ok = (row_c >= 0) & (row_c < terr_h)
            terr_mask = (
                terr_plan[np.clip(row_c, 0, terr_h - 1)][
                    :, np.clip(col_c, 0, terr_w - 1)]
                & row_ok[:, None] & col_ok[None, :]
            )

            if crop_hull is not None:
                xs = dsm_origin_x + (np.arange(dsm_w) + 0.5) * dsm_gsd
                ys = dsm_origin_y + (np.arange(dsm_h) + 0.5) * dsm_gsd
                terr_mask = terr_mask | hull_contains(
                    crop_hull, xs[None, :], ys[:, None]
                )

            dsm_grid, dsm_extrap = _fill_dsm_holes(
                dsm_grid, config, footprint=terr_mask,
            )
            logger.info(
                "  Filled DSM territory holes (diffuse tiny + flat-low pockets)"
            )

            dsm_grid = _shock_dsm_edges(dsm_grid, dsm_gsd, config)
            logger.info("  Sharpened DSM edges (shock filter)")

            # --- Pass 2: bake the ortho from the corrected DSM. ---
            # Every cell that carries a height — reconstructed OR hole-filled — is
            # coloured by REPROJECTING its final DSM height into source images and
            # taking the best-resolution inlier views (real multi-view baking, NOT
            # colour interpolation).
            if ortho_grid is not None and owner_grid is not None:
                n_final = int(config["ortho_bake_n_final_views"])
                irls_iters = int(config["ortho_bake_irls_iterations"])
                valid = ~np.isnan(dsm_grid)
                # DSM heightfield for per-view horizon occlusion of filled cells:
                # the bake marches it from each cell toward the camera and drops
                # views a taller surface (e.g. a roof) blocks — so occluded ground
                # gets real colour from clear nadir views, not a roof ghost.
                dsm_occ_arr = None
                dsm_max_z = 0.0
                if config["ortho_bake_dsm_occlusion"]:
                    dsm_occ_arr = np.ascontiguousarray(
                        dsm_grid, dtype=np.float32)
                    dsm_max_z = float(np.nanmax(dsm_grid))
                # Detail injection: the bake also returns, per cell, the single
                # sharpest inlier's raw colour; after all leaves we add its high-
                # frequency texture back onto the (smooth) blend (no seams, full
                # detail).  Accumulate it over the chunk window like ortho_grid.
                detail_on = bool(config["ortho_detail_injection"])
                sharp_grid: Optional[NDArray] = (
                    np.zeros((dsm_h, dsm_w, 3), dtype=np.uint8)
                    if detail_on else None
                )
                # Cells that received a REAL (non-black) bake.  A cell is baked by exactly one leaf
                baked_mask = np.zeros((dsm_h, dsm_w), dtype=bool)
                n_filled_tot = 0  # diagnostics: filled cells + how many baked
                n_filled_black = 0  # filled cells the bake still left black
                n_owned_black = 0  # reconstructed cells left black
                for leaf_idx, leaf_sv in enumerate(leaves):
                    # This leaf's core cell footprint (same formula as Pass 1).
                    cs = max(0, int(np.floor(
                        (leaf_sv.core_min[0] - dsm_origin_x) / dsm_gsd)))
                    ce = min(dsm_w, int(np.ceil(
                        (leaf_sv.core_max[0] - dsm_origin_x) / dsm_gsd)))
                    rs = max(0, int(np.floor(
                        (leaf_sv.core_min[1] - dsm_origin_y) / dsm_gsd)))
                    re_ = min(dsm_h, int(np.ceil(
                        (leaf_sv.core_max[1] - dsm_origin_y) / dsm_gsd)))
                    if ce <= cs or re_ <= rs:
                        continue
                    # view → writes back
                    sub_owner = owner_grid[rs:re_, cs:ce]
                    sub_valid = valid[rs:re_, cs:ce]
                    # Cells this leaf bakes: the ones it won, plus still-unowned
                    # filled holes inside its footprint (owner < 0 & has height).
                    fill_local = (sub_owner < 0) & sub_valid
                    sub_mask = ((sub_owner == leaf_idx)
                                | fill_local) & sub_valid
                    if not sub_mask.any():
                        continue
                    lr, lc = np.where(sub_mask)
                    rows_idx = lr + rs
                    cols_idx = lc + cs
                    pts = np.empty((len(rows_idx), 3), dtype=np.float32)
                    pts[:, 0] = dsm_origin_x + (cols_idx + 0.5) * dsm_gsd
                    pts[:, 1] = dsm_origin_y + (rows_idx + 0.5) * dsm_gsd
                    pts[:, 2] = dsm_grid[rows_idx, cols_idx]
                    nrm = _dsm_point_normals(
                        dsm_grid, rows_idx, cols_idx, dsm_gsd
                    )
                    # Filled holes carry interpolated geometry: bake them with a flat up-normal (no false grazing rejects)
                    is_filled = fill_local[lr, lc]
                    if is_filled.any():
                        nrm[is_filled] = (0.0, 0.0, 1.0)
                    # Mask-relax filled cells so they sample real colour where no
                    # view has depth — EXCEPT cells flagged no-relax: extrapolated
                    # (linear) or OCCLUDED ground under a tall roof (low_flat), to
                    # avoid ghosting roof colour onto occluded ground.
                    relax = (
                        is_filled & ~dsm_extrap[rows_idx, cols_idx]
                    ).astype(np.uint8)
                    baker = leaf_fusers[leaf_idx]
                    if baker is None:
                        # Not retained from Pass 1 (over the GPU cap) — rebuild and re-fuse, as before.
                        baker, n_loaded = _build_fuser(leaf_sv)
                        if n_loaded == 0:
                            del baker
                            continue
                        baker.count_voxels()
                        baker.fuse_only()
                    bake_out = baker.bake_colors(
                        pts, nrm, n_final=n_final, irls_iters=irls_iters,
                        relax_occlusion=relax,
                        dsm_occ=dsm_occ_arr,
                        dsm_origin_x=dsm_origin_x,
                        dsm_origin_y=dsm_origin_y,
                        dsm_gsd=dsm_gsd,
                        dsm_max_z=dsm_max_z,
                        with_sharp=detail_on,
                    )
                    if detail_on:
                        colors_raw, sharp_raw = bake_out
                        sharp_grid[rows_idx, cols_idx, :] = np.asarray(
                            sharp_raw, dtype=np.uint8)
                    else:
                        colors_raw = bake_out
                    colors = np.asarray(colors_raw, dtype=np.uint8)
                    ortho_grid[rows_idx, cols_idx, :] = colors
                    # A cell is "really baked" iff the kernel found >=1 view
                    # (n_valid==0 emits pure black); record that, not sum>0 on
                    # the final ortho (which a legitimately-dark bake would fail).
                    ok = colors.any(axis=1)
                    baked_mask[rows_idx, cols_idx] = ok
                    n_filled_tot += int(is_filled.sum())
                    n_filled_black += int((is_filled & ~ok).sum())
                    n_owned_black += int((~is_filled & ~ok).sum())
                    # Claim the filled holes so a neighbour leaf (footprints can
                    # overlap by ~1 cell) does not re-bake them.
                    sub_owner[fill_local] = leaf_idx
                    del baker
                    # release the retained GPU fuser
                    leaf_fusers[leaf_idx] = None
                    gc.collect()
                n_unbaked = int((valid & ~baked_mask).sum())
                logger.info(
                    f"  Baked ortho from corrected DSM ({len(leaves)} leaves); "
                    f"{n_unbaked} cell(s) saw no view → residual fill"
                )
                logger.info(
                    f"    of which filled-cell black: {n_filled_black}/"
                    f"{n_filled_tot} filled, reconstructed black: "
                    f"{n_owned_black}"
                )

                # Detail injection: add the sharpest-view high-frequency texture
                # back onto the smooth blend (before residual hole-fill / median,
                # which both assume the bake's colours).
                if sharp_grid is not None:
                    ortho_grid = _inject_ortho_detail(
                        ortho_grid, sharp_grid, baked_mask, config
                    )
                    logger.info(
                        "  Injected sharp-view detail (sigma=%.1f, strength=%.2f)",
                        float(config["ortho_detail_sigma"]),
                        float(config["ortho_detail_strength"]),
                    )

                ortho_grid = _fill_ortho_holes(
                    ortho_grid, dsm_grid, config, baked_mask=baked_mask
                )
                logger.info(
                    "  Filled residual ortho holes (occluded in every view)")

                ortho_grid = _ortho_gated_median(ortho_grid, dsm_grid, config)
                logger.info("  Despeckled ortho (gated 3x3 median)")

            # Release any Pass-1 fusers still held (skipped leaves, or ortho
            # disabled) before clearing the view cache they borrow from.
            leaf_fusers = []
            view_cache.clear()

            if not config["dsm_merge_feather"]:
                owned_xy = np.unique(owned_cells[:, :2], axis=0)
                oxy_origin = owned_xy.min(axis=0)
                oxy_span = owned_xy.max(axis=0) - oxy_origin + 1
                owned_xy_keys = np.sort(
                    (owned_xy[:, 0] - oxy_origin[0]) * oxy_span[1]
                    + (owned_xy[:, 1] - oxy_origin[1])
                )
                vr, vc = np.where(~np.isnan(dsm_grid))
                cx = np.floor(
                    (dsm_origin_x + (vc + 0.5) * dsm_gsd) * inv_coarse
                ).astype(np.int64) - oxy_origin[0]
                cy = np.floor(
                    (dsm_origin_y + (vr + 0.5) * dsm_gsd) * inv_coarse
                ).astype(np.int64) - oxy_origin[1]
                in_range = (
                    (cx >= 0) & (cx < oxy_span[0])
                    & (cy >= 0) & (cy < oxy_span[1])
                )
                col_keys = np.where(in_range, cx * oxy_span[1] + cy, -1)
                spill = ~np.isin(col_keys, owned_xy_keys)
                if spill.any():
                    dsm_grid[vr[spill], vc[spill]] = np.nan
                    if ortho_grid is not None:
                        ortho_grid[vr[spill], vc[spill]] = 0
                    logger.info(
                        f"Unit {unit_idx}: masked {int(spill.sum())} "
                        f"DSM/ortho spill cell(s) (hard territory mask)"
                    )

            # Persist this cluster's finished DSM+ortho as a compact tile;
            ortho_tile = (
                ortho_grid if ortho_grid is not None
                else np.zeros((dsm_h, dsm_w, 3), dtype=np.uint8)
            )
            # The tile carries the GLOBAL georeference + shape and its window
            # offset, so the merge places this territory-sized raster correctly.
            data.save_dsm_ortho_batch(
                unit_idx, dsm_grid, ortho_tile,
                dsm_global_origin_x, dsm_global_origin_y, dsm_gsd,
                base_offset=(win_r0, win_c0),
                global_shape=(dsm_global_h, dsm_global_w),
                confidence=orig_valid,
            )
            logger.info(
                f"Unit {unit_idx}: DSM/ortho tile saved → "
                f"{data.dsm_ortho_batch_file(unit_idx)}"
            )

            # Debug: also dump this cluster's own DSM+ortho window as standalone
            # (topocentric, untagged) GeoTIFFs that overlay the final raster, to
            # isolate per-cluster boundary / grazing-bake artefacts.
            if config["dsm_save_cluster_tiles"]:
                data.save_dsm(
                    dsm_grid, dsm_origin_x, dsm_origin_y, dsm_gsd,
                    path=data.dsm_cluster_file(unit_idx),
                )
                data.save_ortho(
                    ortho_tile, dsm_origin_x, dsm_origin_y, dsm_gsd,
                    nodata_mask=np.isnan(dsm_grid),
                    path=data.ortho_cluster_file(unit_idx),
                )
                logger.info(
                    f"Unit {unit_idx}: debug DSM/ortho GeoTIFFs → "
                    f"{data.ortho_cluster_file(unit_idx)}"
                )
        else:
            view_cache.clear()

        # Concatenate all sub-volume results.
        if all_points:
            points = np.concatenate(all_points)
            normals = np.concatenate(all_normals)
            colors = np.concatenate(all_colors)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            normals = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)
        logger.info(
            f"Unit {unit_idx}: {len(points)} total fused points "
            f"from {len(subvolumes)} sub-volume(s)"
        )
        context.log_memory("fused cluster data in memory")
        logger.info("Batch %d: %d fused points", unit_idx, len(points))

        if len(points) > 0:
            labels = np.zeros(len(points), dtype=np.uint8)
            filename = f"fused_batch_{unit_idx:04d}.ply"
            data.save_point_cloud(
                points, normals, colors, labels, filename=filename
            )

        # Concatenate the per-sub-volume mesh fragments into this cluster's
        # mesh batch, offsetting each fragment's face indices by the running
        # vertex count (vertices are simply stacked).
        if mesh_enabled and all_mverts:
            mfaces_offset: List[NDArray] = []
            vbase = 0
            for mv, mf in zip(all_mverts, all_mfaces):
                mfaces_offset.append(mf.astype(np.int64) + vbase)
                vbase += len(mv)
            mverts_all = np.concatenate(all_mverts)
            mnorm_all = np.concatenate(all_mnormals)
            mcolor_all = np.concatenate(all_mcolors)
            mfaces_all = np.concatenate(mfaces_offset).astype(np.int32)
            data.save_mesh(
                mverts_all, mnorm_all, mcolor_all, mfaces_all,
                filename=f"mesh_batch_{unit_idx:04d}.ply",
            )
            logger.info(
                f"Unit {unit_idx}: mesh batch saved "
                f"({len(mverts_all)} verts, {len(mfaces_all)} faces)"
            )

        if config["depthmap_save_debug_ply"] and len(points) > 0:
            # Recolour this chunk's points by a per-chunk colour so the disjoint
            # KD-tree partition is visible in a viewer.
            cr, cg, cb = _CLUSTER_COLORS[unit_idx % len(_CLUSTER_COLORS)]
            dbg_colors = np.full(
                (len(points), 3), (cr, cg, cb), dtype=np.uint8)
            dbg_labels = np.zeros(len(points), dtype=np.uint8)
            data.save_point_cloud(
                points, normals, dbg_colors, dbg_labels,
                filename=f"fused_batch_{unit_idx:04d}_debug.ply",
            )

        gc.collect()

    # Run unit fusions (one batch per unit).
    n_fusion_workers = 1  # min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(
        max_workers=n_fusion_workers, thread_name_prefix="fuse"
    ) as pool:
        futures: Dict[Future, int] = {}
        for unit_idx in range(len(units)):
            fut = pool.submit(_fuse_unit, unit_idx)
            futures[fut] = unit_idx

        for fut in as_completed(futures):
            unit_idx = futures[fut]
            try:
                fut.result()
                logger.info(f"Fusion unit {unit_idx} complete")
            except Exception as exc:
                logger.error(f"Fusion unit {unit_idx} failed: {exc}")
