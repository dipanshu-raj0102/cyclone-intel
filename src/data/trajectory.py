from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

IBTRACS_PATH = Path(
    "./raw/ibtracs/IBTrACS.ALL.v04r01.nc"
)

STORM_ID = "2025289N13140"

START_TIME = np.datetime64("2025-10-20T00:00")
END_TIME = np.datetime64("2025-10-20T21:00")

INTERVAL = "10min"


def load_ibtracs_track(
    path: Path,
    storm_id: str,
) -> pd.DataFrame:

    ds = xr.open_dataset(path)

    sids = ds["sid"].values.astype(str)

    matches = np.where(sids == storm_id)[0]

    if len(matches) == 0:
        raise ValueError(
            f"Storm {storm_id} not found"
        )

    storm_idx = matches[0]

    times = ds["time"].isel(
        storm=storm_idx
    ).values

    lat = ds["lat"].isel(
        storm=storm_idx
    ).values

    lon = ds["lon"].isel(
        storm=storm_idx
    ).values

    wind = ds["usa_wind"].isel(
        storm=storm_idx
    ).values

    pressure = ds["usa_pres"].isel(
        storm=storm_idx
    ).values

    valid = (
        ~np.isnat(times)
        & ~np.isnan(lat)
        & ~np.isnan(lon)
    )

    # --------------------------------------------------------
    # Normalize IBTrACS timestamps to minute precision
    # --------------------------------------------------------

    times = times[valid].astype(
        "datetime64[m]"
    )

    return pd.DataFrame({
        "timestamp": times,
        "latitude": lat[valid],
        "longitude": lon[valid],
        "wind_kt": wind[valid],
        "pressure_mb": pressure[valid],
    })


def interpolate_track(
    track: pd.DataFrame,
    start_time: np.datetime64,
    end_time: np.datetime64,
) -> pd.DataFrame:

    # Restrict to requested period
    mask = (
        (track["timestamp"] >= start_time)
        & (track["timestamp"] <= end_time)
    )

    track = track.loc[mask].copy()

    track = track.sort_values(
        "timestamp"
    )

    track = track.drop_duplicates(
        subset="timestamp"
    )

    # Convert timestamps to nanoseconds
    track["timestamp"] = pd.to_datetime(
        track["timestamp"]
    )

    # Make timestamp the index
    track = track.set_index("timestamp")

    # Generate every 10 minutes
    target_times = pd.date_range(
        start=pd.Timestamp(start_time),
        end=pd.Timestamp(end_time),
        freq=INTERVAL,
    )

    # Add target timestamps temporarily
    combined = track.reindex(
        track.index.union(target_times)
    ).sort_index()

    # Interpolate only position
    combined["latitude"] = (
        combined["latitude"]
        .interpolate(method="time")
    )

    combined["longitude"] = (
        combined["longitude"]
        .interpolate(method="time")
    )

    # Select only Himawari timestamps
    result = combined.loc[
        target_times
    ].copy()

    result = result.reset_index()

    result = result.rename(
        columns={"index": "timestamp"}
    )

    return result


def main():

    print("Loading IBTrACS...")

    track = load_ibtracs_track(
        IBTRACS_PATH,
        STORM_ID,
    )

    print(
        f"IBTrACS observations: {len(track)}"
    )

    print("\nOriginal IBTrACS track:")
    print(
        track[
            [
                "timestamp",
                "latitude",
                "longitude",
            ]
        ].to_string(index=False)
    )

    interpolated = interpolate_track(
        track,
        START_TIME,
        END_TIME,
    )

    print(
        "\nInterpolated 10-minute track:"
    )

    print(
        interpolated[
            [
                "timestamp",
                "latitude",
                "longitude",
            ]
        ].to_string(index=False)
    )

    output = Path(
        "processed/"
        "himawari_ibtracs/"
        "2025-10-20/"
        f"{STORM_ID}/"
        "interpolated_track.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    interpolated.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )

    print(
        f"Total interpolated positions: "
        f"{len(interpolated)}"
    )


if __name__ == "__main__":
    main()
