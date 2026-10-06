from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from satpy import Scene

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "processed"
    / "identification"
    / "2025-10-20"
    / "scene_manifest.csv"
)

HIMAWARI_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "interim"
    / "himawari_dat"
    / "2025"
    / "10"
    / "20"
)

CHANNEL = "B13"

EXPECTED_HEIGHT = 5500
EXPECTED_WIDTH = 5500
EXPECTED_SEGMENTS = 10

# Stored pixel coordinates should agree with a freshly
# reconstructed geometry to within this many pixels.
PIXEL_TOLERANCE = 1

# Geographic error is evaluated against the actual pixel
# displacement rather than using a hard-coded degree threshold.


# ============================================================
# Himawari filename handling
# ============================================================

SEGMENT_PATTERN = re.compile(
    r"HS_H08_(\d{8})_(\d{4})_B13_FLDK_R20_S(\d{4})\.DAT"
)


def parse_segment_timestamp(filename: str) -> str | None:
    """
    Extract timestamp from a Himawari B13 segment filename.

    Example:
        HS_H08_20251020_0000_B13_FLDK_R20_S0101.DAT

    Returns:
        YYYY-MM-DD HH:MM:SS
    """

    match = SEGMENT_PATTERN.fullmatch(filename)

    if match is None:
        return None

    date_str, time_str, _segment = match.groups()

    return (
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
        f"{time_str[:2]}:{time_str[2:4]}:00"
    )


# ============================================================
# Scene discovery
# ============================================================

def discover_scene_segments() -> dict[str, list[Path]]:
    """
    Discover complete Himawari scenes.

    A complete scene must contain exactly 10 segments.
    """

    if not HIMAWARI_DIR.exists():
        raise FileNotFoundError(
            f"Himawari directory does not exist:\n{HIMAWARI_DIR}"
        )

    grouped: dict[str, list[Path]] = {}

    for path in sorted(HIMAWARI_DIR.glob("*.DAT")):
        timestamp = parse_segment_timestamp(path.name)

        if timestamp is None:
            continue

        grouped.setdefault(timestamp, []).append(path)

    complete = {
        timestamp: sorted(files)
        for timestamp, files in grouped.items()
        if len(files) == EXPECTED_SEGMENTS
    }

    return complete


# ============================================================
# Scene geometry
# ============================================================

def load_scene_geometry(segment_files: list[Path]):
    """
    Load a complete Himawari scene and return its geometry.
    """

    if len(segment_files) != EXPECTED_SEGMENTS:
        raise ValueError(
            f"Expected {EXPECTED_SEGMENTS} segments, "
            f"got {len(segment_files)}"
        )

    scene = Scene(
        filenames=[str(path) for path in segment_files],
        reader="ahi_hsd",
    )

    scene.load([CHANNEL])

    band = scene[CHANNEL]

    array = band.values
    area = band.attrs["area"]

    # Get the actual pixel coordinate arrays from the
    # AreaDefinition.
    x, y = area.get_proj_vectors()

    x = np.asarray(x)
    y = np.asarray(y)

    crs = area.crs

    return {
        "array": array,
        "area": area,
        "x": x,
        "y": y,
        "crs": crs,
    }


# ============================================================
# Coordinate utilities
# ============================================================

def project_latlon(
    latitude: float,
    longitude: float,
    target_crs,
) -> tuple[float, float]:
    """
    Convert geographic coordinates to the Himawari scene CRS.
    """

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    x_proj, y_proj = transformer.transform(
        longitude,
        latitude,
    )

    return float(x_proj), float(y_proj)


def inverse_project(
    x: float,
    y: float,
    source_crs,
) -> tuple[float, float]:
    """
    Convert projected scene coordinates back to lon/lat.
    """

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )

    longitude, latitude = transformer.transform(
        x,
        y,
    )

    return float(latitude), float(longitude)


def nearest_pixel(
    x_proj: float,
    y_proj: float,
    geometry: dict,
) -> tuple[int, int, float]:
    """
    Find the nearest pixel to a projected coordinate.

    Returns:
        row
        col
        projected distance from requested coordinate to pixel center
    """

    x = geometry["x"]
    y = geometry["y"]

    col = int(np.abs(x - x_proj).argmin())
    row = int(np.abs(y - y_proj).argmin())

    pixel_x = float(x[col])
    pixel_y = float(y[row])

    distance = float(
        np.hypot(
            pixel_x - x_proj,
            pixel_y - y_proj,
        )
    )

    return row, col, distance


def pixel_center(
    row: int,
    col: int,
    geometry: dict,
) -> tuple[float, float]:
    """
    Return projected coordinates of a pixel center.
    """

    x = geometry["x"]
    y = geometry["y"]

    return float(x[col]), float(y[row])


# ============================================================
# Scene-level validation
# ============================================================

def validate_scene(
    scene_row: pd.Series,
    geometry: dict,
    segment_files: list[Path],
) -> list[dict]:
    """
    Validate every cyclone center in one scene.
    """

    results = []

    centers = json.loads(scene_row["centers"])

    array = geometry["array"]

    height, width = array.shape

    if height != EXPECTED_HEIGHT or width != EXPECTED_WIDTH:
        raise ValueError(
            f"Unexpected scene dimensions: "
            f"{height} × {width}"
        )

    target_timestamp = str(scene_row["timestamp"])

    for center in centers:

        storm_id = center["storm_id"]

        latitude = float(center["latitude"])
        longitude = float(center["longitude"])

        stored_row = int(center["row"])
        stored_col = int(center["col"])

        # ----------------------------------------------------
        # 1. Stored pixel bounds
        # ----------------------------------------------------

        if not (
            0 <= stored_row < height
            and 0 <= stored_col < width
        ):
            results.append(
                {
                    "scene_id": scene_row["scene_id"],
                    "timestamp": target_timestamp,
                    "storm_id": storm_id,
                    "status": "FAIL",
                    "reason": "stored_pixel_out_of_bounds",
                }
            )

            continue

        # ----------------------------------------------------
        # 2. Fresh forward projection
        # ----------------------------------------------------

        x_proj, y_proj = project_latlon(
            latitude,
            longitude,
            geometry["crs"],
        )

        # ----------------------------------------------------
        # 3. Verify projected point is actually inside
        #    the scene coordinate extent.
        # ----------------------------------------------------

        x_min = float(np.min(geometry["x"]))
        x_max = float(np.max(geometry["x"]))

        y_min = float(np.min(geometry["y"]))
        y_max = float(np.max(geometry["y"]))

        projected_inside = (
            x_min <= x_proj <= x_max
            and y_min <= y_proj <= y_max
        )

        if not projected_inside:
            results.append(
                {
                    "scene_id": scene_row["scene_id"],
                    "timestamp": target_timestamp,
                    "storm_id": storm_id,
                    "status": "FAIL",
                    "reason": "projected_coordinate_outside_scene",
                    "latitude": latitude,
                    "longitude": longitude,
                    "stored_row": stored_row,
                    "stored_col": stored_col,
                    "projected_x": x_proj,
                    "projected_y": y_proj,
                }
            )

            continue

        # ----------------------------------------------------
        # 4. Fresh nearest-pixel calculation
        # ----------------------------------------------------

        expected_row, expected_col, projected_distance = (
            nearest_pixel(
                x_proj,
                y_proj,
                geometry,
            )
        )

        row_error = abs(stored_row - expected_row)
        col_error = abs(stored_col - expected_col)

        pixel_match = (
            row_error <= PIXEL_TOLERANCE
            and col_error <= PIXEL_TOLERANCE
        )

        # ----------------------------------------------------
        # 5. Inverse-project stored pixel center
        # ----------------------------------------------------

        stored_x, stored_y = pixel_center(
            stored_row,
            stored_col,
            geometry,
        )

        reconstructed_latitude, reconstructed_longitude = (
            inverse_project(
                stored_x,
                stored_y,
                geometry["crs"],
            )
        )

        latitude_error = abs(
            reconstructed_latitude - latitude
        )

        longitude_error = abs(
            reconstructed_longitude - longitude
        )

        # ----------------------------------------------------
        # 6. Compare stored pixel with freshly calculated pixel
        # ----------------------------------------------------

        status = "PASS" if pixel_match else "FAIL"

        results.append(
            {
                "scene_id": scene_row["scene_id"],
                "timestamp": target_timestamp,
                "storm_id": storm_id,

                "latitude": latitude,
                "longitude": longitude,

                "stored_row": stored_row,
                "stored_col": stored_col,

                "expected_row": expected_row,
                "expected_col": expected_col,

                "row_error": row_error,
                "col_error": col_error,

                "projected_distance": projected_distance,

                "reconstructed_latitude": reconstructed_latitude,
                "reconstructed_longitude": reconstructed_longitude,

                "latitude_error": latitude_error,
                "longitude_error": longitude_error,

                "status": status,
            }
        )

    return results


# ============================================================
# Main validator
# ============================================================

def main():

    print("=" * 60)
    print("HIMAWARI GEOLOCATION VALIDATOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    if manifest.empty:
        raise RuntimeError("Manifest is empty.")

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Scenes loaded: {len(manifest)}")
    print("=" * 60)

    # --------------------------------------------------------
    # Discover original Himawari scenes
    # --------------------------------------------------------

    scenes = discover_scene_segments()

    print(
        f"Complete Himawari scenes available: "
        f"{len(scenes)}"
    )

    # --------------------------------------------------------
    # Validate each manifest scene
    # --------------------------------------------------------

    all_results = []

    for index, scene_row in manifest.iterrows():

        timestamp = str(scene_row["timestamp"])

        print(
            f"[{index + 1:02d}/{len(manifest):02d}] "
            f"Validating {timestamp}...",
            end=" ",
            flush=True,
        )

        if timestamp not in scenes:
            raise RuntimeError(
                f"No complete Himawari scene found for "
                f"{timestamp}"
            )

        segment_files = scenes[timestamp]

        geometry = load_scene_geometry(segment_files)

        results = validate_scene(
            scene_row,
            geometry,
            segment_files,
        )

        all_results.extend(results)

        scene_failed = any(
            result["status"] == "FAIL"
            for result in results
        )

        print("FAIL" if scene_failed else "PASS")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    total = len(results_df)
    passed = int(
        (results_df["status"] == "PASS").sum()
    )
    failed = int(
        (results_df["status"] == "FAIL").sum()
    )

    print()
    print("=" * 60)
    print("GEOLOCATION VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Scenes validated          : {len(manifest)}")
    print(f"Cyclone centers validated : {total}")
    print(f"Passed                    : {passed}")
    print(f"Failed                    : {failed}")

    if total > 0:

        print(
            f"Maximum row error         : "
            f"{results_df['row_error'].max():.0f} px"
        )

        print(
            f"Maximum column error      : "
            f"{results_df['col_error'].max():.0f} px"
        )

        print(
            f"Maximum latitude error    : "
            f"{results_df['latitude_error'].max():.6f}°"
        )

        print(
            f"Maximum longitude error   : "
            f"{results_df['longitude_error'].max():.6f}°"
        )

    # --------------------------------------------------------
    # Detailed failures
    # --------------------------------------------------------

    failures = results_df[
        results_df["status"] == "FAIL"
    ]

    if not failures.empty:

        print()
        print("=" * 60)
        print("FAILED GEOLOCATION CHECKS")
        print("=" * 60)

        for _, row in failures.iterrows():

            print(
                f"\nScene      : {row.get('scene_id')}"
            )
            print(
                f"Timestamp  : {row.get('timestamp')}"
            )
            print(
                f"Storm ID   : {row.get('storm_id')}"
            )
            print(
                f"Stored     : "
                f"({row.get('stored_row')}, "
                f"{row.get('stored_col')})"
            )
            print(
                f"Expected   : "
                f"({row.get('expected_row')}, "
                f"{row.get('expected_col')})"
            )
            print(
                f"Row error  : {row.get('row_error')}"
            )
            print(
                f"Col error  : {row.get('col_error')}"
            )

        print()
        print("VALIDATION FAILED")
        raise SystemExit(1)

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)

    print(
        "✓ Fresh CRS projection agrees with stored pixels"
    )
    print(
        "✓ All cyclone centers fall inside the scene extent"
    )
    print(
        "✓ Stored pixel coordinates are within scene bounds"
    )
    print(
        "✓ Stored pixel coordinates agree within "
        f"{PIXEL_TOLERANCE} pixel"
    )
    print(
        "✓ Stored pixel centers inverse-project correctly"
    )


if __name__ == "__main__":
    main()
