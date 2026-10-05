import numpy as np
import xarray as xr

IBTRACS_PATH = "./raw/ibtracs/IBTrACS.ALL.v04r01.nc"

ds = xr.open_dataset(IBTRACS_PATH)

target_date = np.datetime64("2025-10-20")

mask = (
    (ds["time"].dt.floor("D") == target_date)
    & ds["lat"].notnull()
    & ds["lon"].notnull()
)

storm_indices, time_indices = np.where(mask.values)

unique_storms = np.unique(storm_indices)

for storm_idx in unique_storms:

    print("\n" + "=" * 80)
    print(f"STORM INDEX: {storm_idx}")

    sid = ds["sid"].isel(storm=storm_idx).item()
    season = ds["season"].isel(storm=storm_idx).item()

    print(f"SID: {sid}")
    print(f"Season: {season}")
    print("=" * 80)

    # Select only observations for this storm on target date
    storm_mask = mask.isel(storm=storm_idx)

    times = ds["time"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    lats = ds["lat"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    lons = ds["lon"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    winds = ds["usa_wind"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    pressures = ds["usa_pres"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    speeds = ds["storm_speed"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    directions = ds["storm_dir"].isel(storm=storm_idx).where(
        storm_mask,
        drop=True
    )

    for i in range(len(times)):
        print(
            f"{times.values[i]} | "
            f"lat={lats.values[i]:7.2f} | "
            f"lon={lons.values[i]:7.2f} | "
            f"wind={winds.values[i]:6.1f} kt | "
            f"pressure={pressures.values[i]:7.1f} mb | "
            f"speed={speeds.values[i]:6.1f} kt | "
            f"direction={directions.values[i]:6.1f}°"
        )
