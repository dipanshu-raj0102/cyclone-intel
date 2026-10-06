from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IBTRACS_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "raw"
    / "ibtracs"
    / "IBTrACS.ALL.v04r01.nc"
)

HIMAWARI_MANIFEST = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "processed"
    / "identification"
    / "2025-10-20"
    / "scene_manifest.csv"
)

# We inspect a wider window around the Himawari observation day.
WINDOW_BEFORE_HOURS = 24
WINDOW_AFTER_HOURS = 24


# ============================================================
# IBTrACS loading
# ============================================================

def load_ibtracs() -> pd.DataFrame:

    print("Loading IBTrACS...")

    ds = xr.open_dataset(
        IBTRACS_PATH,
        engine="h5netcdf",
    )

    try:
        time = ds["time"].values
        lat = ds["lat"].values
        lon = ds["lon"].values
        sid = ds["sid"].values

        records = []

        for storm_index in range(ds.sizes["storm"]):

            storm_id_raw = sid[storm_index]

            if isinstance(storm_id_raw, bytes):
                storm_id = storm_id_raw.decode(
                    "utf-8",
                    errors="ignore",
                ).strip()
            else:
                storm_id = str(storm_id_raw).strip()

            for observation_index in range(
                ds.sizes["date_time"]
            ):

                timestamp = time[
                    storm_index,
                    observation_index,
                ]

                latitude = lat[
                    storm_index,
                    observation_index,
                ]

                longitude = lon[
                    storm_index,
                    observation_index,
                ]

                if pd.isna(timestamp):
                    continue

                if not np.isfinite(latitude):
                    continue

                if not np.isfinite(longitude):
                    continue

                records.append(
                    {
                        "storm_id": storm_id,
                        "timestamp": pd.Timestamp(timestamp),
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                    }
                )

    finally:
        ds.close()

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "No valid IBTrACS observations were found."
        )

    df = df.sort_values(
        ["timestamp", "storm_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("IBTRACS TEMPORAL COVERAGE INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Himawari manifest
    # --------------------------------------------------------

    if not HIMAWARI_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{HIMAWARI_MANIFEST}"
        )

    manifest = pd.read_csv(HIMAWARI_MANIFEST)

    if manifest.empty:
        raise RuntimeError("Himawari manifest is empty.")

    himawari_times = pd.to_datetime(
        manifest["timestamp"]
    )

    observation_start = (
        himawari_times.min()
        - pd.Timedelta(hours=WINDOW_BEFORE_HOURS)
    )

    observation_end = (
        himawari_times.max()
        + pd.Timedelta(hours=WINDOW_AFTER_HOURS)
    )

    print(
        f"Himawari period : "
        f"{himawari_times.min()} → {himawari_times.max()}"
    )

    print(
        f"Inspection window: "
        f"{observation_start} → {observation_end}"
    )

    # --------------------------------------------------------
    # Load IBTrACS
    # --------------------------------------------------------

    ibtracs = load_ibtracs()

    window = ibtracs[
        (ibtracs["timestamp"] >= observation_start)
        & (ibtracs["timestamp"] <= observation_end)
    ].copy()

    print()
    print(
        f"IBTrACS observations in window: "
        f"{len(window)}"
    )

    print(
        f"Unique storms in window: "
        f"{window['storm_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Storm summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STORM COVERAGE")
    print("=" * 70)

    storm_summary = (
        window
        .groupby("storm_id")
        .agg(
            first_observation=("timestamp", "min"),
            last_observation=("timestamp", "max"),
            observations=("timestamp", "count"),
        )
        .sort_values("first_observation")
    )

    if storm_summary.empty:
        print("No IBTrACS storms found in inspection window.")
    else:
        print(storm_summary.to_string())

    # --------------------------------------------------------
    # Unique IBTrACS timestamps
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IBTRACS OBSERVATION TIMESTAMPS")
    print("=" * 70)

    unique_times = (
        window["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    for timestamp in unique_times:

        storms = (
            window[
                window["timestamp"] == timestamp
            ]["storm_id"]
            .drop_duplicates()
            .tolist()
        )

        print(
            f"{timestamp}  "
            f"storms={len(storms)}  "
            f"{', '.join(storms)}"
        )

    # --------------------------------------------------------
    # Himawari timestamps and nearest IBTrACS observations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HIMAWARI → NEAREST IBTRACS OBSERVATION")
    print("=" * 70)

    for timestamp in himawari_times.sort_values():

        before = window[
            window["timestamp"] <= timestamp
        ]

        after = window[
            window["timestamp"] >= timestamp
        ]

        previous_time = (
            before["timestamp"].max()
            if not before.empty
            else pd.NaT
        )

        next_time = (
            after["timestamp"].min()
            if not after.empty
            else pd.NaT
        )

        previous_gap = (
            timestamp - previous_time
            if pd.notna(previous_time)
            else pd.NaT
        )

        next_gap = (
            next_time - timestamp
            if pd.notna(next_time)
            else pd.NaT
        )

        print(
            f"{timestamp} | "
            f"prev={previous_time} "
            f"({previous_gap}) | "
            f"next={next_time} "
            f"({next_gap})"
        )

    # --------------------------------------------------------
    # Gaps between consecutive IBTrACS observations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IBTRACS TEMPORAL GAPS")
    print("=" * 70)

    if len(unique_times) > 1:

        gaps = unique_times.diff()

        gap_df = pd.DataFrame(
            {
                "previous": unique_times,
                "next": unique_times,
                "gap": gaps,
            }
        )

        gap_df = gap_df.iloc[1:]

        print(
            gap_df.to_string(index=False)
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Himawari scenes              : "
        f"{len(manifest)}"
    )

    print(
        f"Himawari time range          : "
        f"{himawari_times.min()} → "
        f"{himawari_times.max()}"
    )

    print(
        f"IBTrACS observations         : "
        f"{len(window)}"
    )

    print(
        f"IBTrACS unique timestamps    : "
        f"{len(unique_times)}"
    )

    print(
        f"IBTrACS unique storms        : "
        f"{window['storm_id'].nunique()}"
    )

    print()
    print(
        "No negative labels have been generated."
    )
    print(
        "This script is diagnostic only."
    )


if __name__ == "__main__":
    main()
