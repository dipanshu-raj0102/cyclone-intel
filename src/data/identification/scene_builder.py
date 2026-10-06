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

DATE = "2025-10-20"
CHANNEL = "B13"

HIMAWARI_DIR = (
    Path("../")
    / "interim"
    / "himawari_dat"
    / "2025"
    / "10"
    / "20"
)

IBTRACS_PATH = (
    Path("../raw/ibtracs")
    / "IBTrACS.ALL.v04r01.nc"
)

OUTPUT_DIR = (
    Path("../processed")
    / "identification"
    / DATE
)

MANIFEST_PATH = OUTPUT_DIR / "scene_manifest.csv"


# ============================================================
# Himawari file pattern
# ============================================================

FILE_PATTERN = re.compile(
    r"HS_H08_(\d{8})_(\d{4})_B13_FLDK_R20_S(\d{4})\.DAT$"
)


# ============================================================
# Discover Himawari scenes
# ============================================================

def discover_scenes() -> dict[pd.Timestamp, list[Path]]:
    """
    Discover all complete Himawari B13 scenes.

    Returns
    -------
    dict
        timestamp -> list of 10 segment files
    """

    scenes: dict[pd.Timestamp, list[Path]] = {}

    for path in sorted(HIMAWARI_DIR.glob("*.DAT")):
        match = FILE_PATTERN.match(path.name)

        if not match:
            continue

        date_str, time_str, _segment = match.groups()

        timestamp = pd.to_datetime(
            f"{date_str}{time_str}",
            format="%Y%m%d%H%M",
        )

        scenes.setdefault(timestamp, []).append(path)

    # Keep only complete 10-segment scenes.
    complete_scenes = {
        timestamp: sorted(files)
        for timestamp, files in scenes.items()
        if len(files) == 10
    }

    print(f"Discovered timestamps : {len(scenes)}")
    print(f"Complete scenes       : {len(complete_scenes)}")

    incomplete = {
        timestamp: files
        for timestamp, files in scenes.items()
        if len(files) != 10
    }

    if incomplete:
        print(f"Incomplete scenes     : {len(incomplete)}")

        for timestamp, files in sorted(incomplete.items()):
            print(
                f"  {timestamp} -> "
                f"{len(files)}/10 segments"
            )

    return complete_scenes


# ============================================================
# Load IBTrACS
# ============================================================

def load_ibtracs(
    target_timestamps: set[pd.Timestamp],
) -> pd.DataFrame:
    """
    Load only IBTrACS observations whose timestamps match
    the requested Himawari timestamps.

    Uses vectorized NumPy operations instead of iterating
    through individual xarray elements.
    """

    import xarray as xr

    print("Loading IBTrACS...")

    ds = xr.open_dataset(
        IBTRACS_PATH,
        engine="h5netcdf",
    )

    target_ns = np.array(
        [
            np.datetime64(
                pd.Timestamp(t).floor("min"),
                "m",
            )
            for t in target_timestamps
        ],
        dtype="datetime64[m]",
    )

    # --------------------------------------------------------
    # Extract arrays once
    # --------------------------------------------------------

    times = ds["time"].values.astype("datetime64[m]")
    lats = ds["lat"].values
    lons = ds["lon"].values
    winds = ds["usa_wind"].values
    pressures = ds["usa_pres"].values

    # storm × observation
    print(
        f"IBTrACS time array shape : {times.shape}"
    )

    # --------------------------------------------------------
    # Find matching timestamps vectorially
    # --------------------------------------------------------

    valid_time = ~np.isnat(times)

    matching_time = np.isin(
        times,
        target_ns,
    )

    valid_position = (
        valid_time
        & matching_time
        & np.isfinite(lats)
        & np.isfinite(lons)
    )

    storm_indices, time_indices = np.where(
        valid_position
    )

    print(
        f"Matching observations : "
        f"{len(storm_indices)}"
    )

    # --------------------------------------------------------
    # Storm IDs
    # --------------------------------------------------------

    storm_ids = ds["sid"].values

    selected_storm_ids = storm_ids[
        storm_indices
    ]

    selected_storm_ids = np.array(
        [
            sid.decode("utf-8")
            if isinstance(sid, bytes)
            else str(sid)
            for sid in selected_storm_ids
        ]
    )

    # --------------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        {
            "storm_id": selected_storm_ids,
            "timestamp": pd.to_datetime(
                times[
                    storm_indices,
                    time_indices,
                ]
            ).floor("min"),
            "latitude": lats[
                storm_indices,
                time_indices,
            ].astype(float),
            "longitude": lons[
                storm_indices,
                time_indices,
            ].astype(float),
            "wind_kt": winds[
                storm_indices,
                time_indices,
            ].astype(float),
            "pressure_mb": pressures[
                storm_indices,
                time_indices,
            ].astype(float),
        }
    )

    # --------------------------------------------------------
    # Clean invalid intensity values
    # --------------------------------------------------------

    df.loc[
        ~np.isfinite(df["wind_kt"]),
        "wind_kt",
    ] = np.nan

    df.loc[
        ~np.isfinite(df["pressure_mb"]),
        "pressure_mb",
    ] = np.nan

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                "storm_id",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "timestamp",
                "storm_id",
            ]
        )
        .reset_index(drop=True)
    )

    ds.close()

    print(
        f"Matching IBTrACS observations : "
        f"{len(df)}"
    )

    print(
        f"Unique storms                 : "
        f"{df['storm_id'].nunique()}"
    )

    return df# ============================================================
# Geolocation
# ============================================================

def get_scene_geometry(
    segment_files: list[Path],
):
    """
    Load a complete Himawari B13 scene and return its geometry.
    """

    scene = Scene(
        filenames=[str(path) for path in segment_files],
        reader="ahi_hsd",
    )

    scene.load([CHANNEL])

    band = scene[CHANNEL]

    array = band.values.astype(np.float32)

    x = band.coords["x"].values
    y = band.coords["y"].values

    area = band.attrs["area"]

    return {
        "array": array,
        "x": x,
        "y": y,
        "area": area,
        "crs": area.crs,
    }


def latlon_to_pixel(
    latitude: float,
    longitude: float,
    geometry,
) -> tuple[int, int] | None:
    """
    Convert latitude/longitude to nearest Himawari pixel.
    """

    transformer = Transformer.from_crs(
        "EPSG:4326",
        geometry["crs"],
        always_xy=True,
    )

    x_proj, y_proj = transformer.transform(
        longitude,
        latitude,
    )

    x = geometry["x"]
    y = geometry["y"]

    col = int(np.abs(x - x_proj).argmin())
    row = int(np.abs(y - y_proj).argmin())

    height, width = geometry["array"].shape

    if not (
        0 <= row < height
        and 0 <= col < width
    ):
        return None

    return row, col


# ============================================================
# Scene labeling
# ============================================================

def build_scene_record(
    timestamp: pd.Timestamp,
    segment_files: list[Path],
    ibtracs: pd.DataFrame,
):
    """
    Create one scene-level identification label.
    """

    geometry = get_scene_geometry(segment_files)

    array = geometry["array"]

    height, width = array.shape

    observations = ibtracs[
        ibtracs["timestamp"] == timestamp
    ]

    centers = []

    for _, obs in observations.iterrows():

        pixel = latlon_to_pixel(
            latitude=obs["latitude"],
            longitude=obs["longitude"],
            geometry=geometry,
        )

        if pixel is None:
            continue

        row, col = pixel

        centers.append(
            {
                "storm_id": obs["storm_id"],
                "latitude": float(obs["latitude"]),
                "longitude": float(obs["longitude"]),
                "row": row,
                "col": col,
                "wind_kt": (
                    None
                    if pd.isna(obs["wind_kt"])
                    else float(obs["wind_kt"])
                ),
                "pressure_mb": (
                    None
                    if pd.isna(obs["pressure_mb"])
                    else float(obs["pressure_mb"])
                ),
            }
        )

    record = {
        "scene_id": timestamp.strftime(
            "H08_%Y%m%d_%H%M"
        ),
        "timestamp": timestamp,
        "channel": CHANNEL,
        "height": height,
        "width": width,
        "n_segments": len(segment_files),
        "cyclone_present": int(len(centers) > 0),
        "num_cyclones": len(centers),
        "centers": json.dumps(
            centers,
            separators=(",", ":"),
        ),
        "segment_1": str(segment_files[0]),
        "segment_2": str(segment_files[1]),
        "segment_3": str(segment_files[2]),
        "segment_4": str(segment_files[3]),
        "segment_5": str(segment_files[4]),
        "segment_6": str(segment_files[5]),
        "segment_7": str(segment_files[6]),
        "segment_8": str(segment_files[7]),
        "segment_9": str(segment_files[8]),
        "segment_10": str(segment_files[9]),
    }

    return record


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("HIMAWARI IDENTIFICATION DATASET BUILDER")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Discover scenes
    # --------------------------------------------------------

    scenes = discover_scenes()

    if not scenes:
        raise RuntimeError(
            "No complete Himawari scenes found."
        )

    # --------------------------------------------------------
    # 2. Load IBTrACS
    # --------------------------------------------------------

    ibtracs = load_ibtracs(
    target_timestamps=set(scenes.keys())
)

    ibtracs["timestamp"] = pd.to_datetime(
        ibtracs["timestamp"]
    ).dt.floor("min")

    # --------------------------------------------------------
    # 3. Determine exact Himawari ↔ IBTrACS timestamp matches
    # --------------------------------------------------------
    #
    # Identification uses observed IBTrACS timestamps only.
    # Unmatched Himawari timestamps are UNKNOWN, not negative.
    # Therefore they are excluded from the manifest.
    #
    # No temporal interpolation is performed here.
    # --------------------------------------------------------

    himawari_timestamps = set(scenes.keys())
    ibtracs_timestamps = set(
        ibtracs["timestamp"].dropna().unique()
    )

    matched_timestamps = sorted(
        himawari_timestamps & ibtracs_timestamps
    )

    unknown_count = (
        len(himawari_timestamps) - len(matched_timestamps)
    )

    print(
        f"Exact timestamp matches : {len(matched_timestamps)}"
    )
    print(
        f"Unknown Himawari scenes  : {unknown_count}"
    )

    if not matched_timestamps:
        raise RuntimeError(
            "No exact Himawari ↔ IBTrACS timestamp matches found."
        )

    records = []

    for index, timestamp in enumerate(
        matched_timestamps,
        start=1,
    ):

        files = scenes[timestamp]

        print(
            f"\n[{index}/{len(matched_timestamps)}] "
            f"{timestamp}"
        )

        try:

            record = build_scene_record(
                timestamp=timestamp,
                segment_files=files,
                ibtracs=ibtracs,
            )

            records.append(record)

            print(
                f"  Shape       : "
                f"{record['height']} × "
                f"{record['width']}"
            )

            print(
                f"  Cyclone     : "
                f"{record['cyclone_present']}"
            )

            print(
                f"  Centers     : "
                f"{record['num_cyclones']}"
            )

        except Exception as exc:

            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # 4. Save manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(records)

    manifest = manifest.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------

    positives = int(
        manifest["cyclone_present"].sum()
    )

    # Unmatched Himawari scenes are UNKNOWN and were excluded.
    # They are intentionally not counted as negative examples.

    print("\\n" + "=" * 60)
    print("IDENTIFICATION DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Complete Himawari scenes : {len(scenes)}"
    )

    print(
        f"Exact-match scenes       : {len(manifest)}"
    )

    print(
        f"Unknown scenes excluded  : "
        f"{len(scenes) - len(manifest)}"
    )

    print(
        f"Positive scenes          : {positives}"
    )

    print(
        f"Output                   : {MANIFEST_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
