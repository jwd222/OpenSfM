#!/usr/bin/env python3
"""Export optimized camera poses (position + orientation) to a CSV.

Standalone, decoupled helper: it imports opensfm as a library, reads a
reconstruction, and writes one CSV per run with the camera positions in a
projected CRS (default EPSG:32645) and in EPSG:4326, plus aerospace
Yaw/Pitch/Roll expressed in the projected CRS local grid frame.  An optional
geoid grid (GeoTIFF, e.g. EGM2008) adds an orthometric height column.

Usage:
    python export_camera_poses.py <dataset>
        [--proj EPSG:32645] [--geoid geoid.tif] [--output camera_poses.csv]
        [--reconstruction reconstruction.json]

Columns: name, X, Y, Z, lon, lat, Yaw, Pitch, Roll [, Z_orthometric]
    X, Y        projected easting / northing (default EPSG:32645)
    Z           ellipsoidal height (m)
    lon, lat    EPSG:4326
    Yaw/Pitch/Roll  degrees, see convention below
    Z_orthometric  Z - geoid_undulation  (only with --geoid)

Orientation convention (ZYX yaw-pitch-roll), body frame
    Xb = forward (optical axis == OpenSfM cam +Z)
    Yb = right   (== OpenSfM cam +X)
    Zb = down    (== OpenSfM cam +Y)
expressed in the local NED of the projected CRS (grid north):
    Yaw   = heading from grid North, clockwise +
    Pitch = + optical axis up        (a nadir camera reads ~ -90)
    Roll  = + right side down
The true->grid-North rotation (UTM meridian convergence) is applied per camera.
"""
import argparse
import csv
import logging
import math
import os
import sys
from os.path import abspath, dirname, join
from typing import List, Optional, Tuple

import numpy as np
import pyproj

# Make the opensfm package importable regardless of cwd (this script is in bin/).
sys.path.insert(0, abspath(join(dirname(__file__), "..")))

from opensfm.dataset import DataSet
from opensfm.geo import TopocentricConverter

logger = logging.getLogger(__name__)

# Body (Xb=forward, Yb=right, Zb=down) <- OpenSfM cam (Xc=right, Yc=down, Zc=forward)
R_BC: np.ndarray = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
# Projected map (Xm=east, Ym=north(grid), Zm=up) -> NED-grid (N, E, D)
R_MG: np.ndarray = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", help="OpenSfM dataset path")
    p.add_argument(
        "--proj",
        default="EPSG:32645",
        help="Projected CRS for X/Y (PROJ string or EPSG). Default EPSG:32645.",
    )
    p.add_argument(
        "--geoid",
        default=None,
        help="Path to a geoid grid (GeoTIFF, e.g. EGM2008) for orthometric height.",
    )
    p.add_argument(
        "--output", default=None, help="Output CSV path (default <dataset>/camera_poses.csv)."
    )
    p.add_argument(
        "--reconstruction",
        default=None,
        help="Reconstruction JSON filename (default reconstruction.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
    )

    data = DataSet(args.dataset)
    if not data.reference_exists():
        logger.error("No reference_lla.json in dataset; run reconstruction first.")
        sys.exit(1)
    reference = data.load_reference()
    reconstructions = data.load_reconstruction(args.reconstruction)
    if not reconstructions:
        logger.error("No reconstruction found.")
        sys.exit(1)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", args.proj, always_xy=True)
    geoid: Optional[GeoidGrid] = GeoidGrid(args.geoid) if args.geoid else None

    header = ["name", "X", "Y", "Z", "lon", "lat", "Yaw", "Pitch", "Roll"]
    if geoid is not None:
        header.append("Z_orthometric")

    rows: List[List[str]] = []
    for rec in reconstructions:
        for shot in rec.shots.values():
            rows.append(_shot_row(shot, reference, transformer, geoid))
    rows.sort(key=lambda r: r[0])

    output = args.output or os.path.join(data.data_path, "camera_poses.csv")
    with open(output, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("Wrote %d camera poses to %s", len(rows), output)


def _shot_row(
    shot, reference: TopocentricConverter, transformer: pyproj.Transformer, geoid: Optional["GeoidGrid"]
) -> List[str]:
    east, north, up = np.asarray(shot.pose.get_origin(), dtype=float).tolist()
    lat, lon, alt_ell = reference.to_lla(east, north, up)
    x, y = transformer.transform(lon, lat)  # always_xy: lon,lat -> easting,northing

    R_cw = np.asarray(shot.pose.get_R_cam_to_world(), dtype=float)
    R_tm = _topo_to_map_rotation(east, north, up, reference, transformer)
    C_nb = R_MG @ (R_tm @ R_cw) @ R_BC  # body -> NED-grid
    yaw, pitch, roll = _yaw_pitch_roll(C_nb)

    row = [
        shot.id,
        _fmt(x), _fmt(y), _fmt(alt_ell),
        _fmt(lon), _fmt(lat),
        _fmt(math.degrees(yaw)),
        _fmt(math.degrees(pitch)),
        _fmt(math.degrees(roll)),
    ]
    if geoid is not None:
        row.append(_fmt(alt_ell - geoid.undulation(lon, lat)))
    return row


def _map_point(p: np.ndarray, reference: TopocentricConverter, transformer: pyproj.Transformer) -> np.ndarray:
    lat, lon, alt = reference.to_lla(float(p[0]), float(p[1]), float(p[2]))
    x, y = transformer.transform(lon, lat)
    return np.array([x, y, alt], dtype=float)


def _topo_to_map_rotation(
    e: float, n: float, u: float, reference: TopocentricConverter, transformer: pyproj.Transformer
) -> np.ndarray:
    """Rotation mapping topocentric ENU axes -> projected map axes at p (convergence-aware).

    For a conformal projection (e.g. UTM) the local Jacobian is an isotropic
    scale times a rotation; modified Gram-Schmidt on its first two columns
    recovers that rotation and the cross product enforces a right-handed frame.
    Only elementwise numpy ops are used (no LAPACK).
    """
    eps = 1e-3
    p = np.array([e, n, u], dtype=float)

    def col(delta: np.ndarray) -> np.ndarray:
        return (_map_point(p + delta, reference, transformer) - _map_point(
            p - delta, reference, transformer)) / (2 * eps)

    c0 = col(np.array([eps, 0.0, 0.0]))  # topo East axis in map coords
    c1 = col(np.array([0.0, eps, 0.0]))  # topo North axis in map coords

    def _norm(v: np.ndarray) -> np.ndarray:
        return v / float(np.sqrt((v * v).sum()))

    u0 = _norm(c0)
    u1 = _norm(c1 - float((c1 * u0).sum()) * u0)
    u2 = np.cross(u0, u1)
    return np.column_stack((u0, u1, u2))


def _yaw_pitch_roll(R: np.ndarray) -> Tuple[float, float, float]:
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def _fmt(v) -> str:
    return f"{float(v):.9g}"


class GeoidGrid:
    """Bilinear interpolator over a geoid GeoTIFF (undulation N, metres)."""

    def __init__(self, path: str) -> None:
        try:
            from osgeo import gdal
        except ImportError as e:
            raise ImportError(
                "GDAL Python bindings (osgeo) are required to read the geoid grid."
            ) from e
        ds = gdal.Open(path)
        if ds is None:
            raise IOError(f"Could not open geoid grid: {path}")
        self.ds = ds  # keep the dataset alive so the band stays valid
        self.band = ds.GetRasterBand(1)
        self.gt = ds.GetGeoTransform()
        self.nodata = self.band.GetNoDataValue()
        self.nx = ds.RasterXSize
        self.ny = ds.RasterYSize
        logger.info("Loaded geoid grid %s (%dx%d)", path, self.nx, self.ny)

    def undulation(self, lon: float, lat: float) -> float:
        gt = self.gt
        col = (lon - gt[0]) / gt[1]
        row = (lat - gt[3]) / gt[5]
        c0 = min(max(int(math.floor(col)), 0), self.nx - 2)
        r0 = min(max(int(math.floor(row)), 0), self.ny - 2)
        fx, fy = col - c0, row - r0
        arr = self.band.ReadAsArray(c0, r0, 2, 2).astype(float)
        if self.nodata is not None and bool(np.any(arr == self.nodata)):
            logger.warning("Geoid nodata at (%.6f, %.6f); using N=0", lon, lat)
            return 0.0
        return float(
            arr[0, 0] * (1 - fx) * (1 - fy)
            + arr[0, 1] * fx * (1 - fy)
            + arr[1, 0] * (1 - fx) * fy
            + arr[1, 1] * fx * fy
        )


if __name__ == "__main__":
    main()
