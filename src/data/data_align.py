import re
from pathlib import Path

import numpy as np
import xarray as xr

IBTRACS_PATH = "./raw/ibtracs/IBTrACS.ALL.v04r01.nc"
HIMAWARI_DIR = Path("./interim/himawari_dat/2025/10/20")

TARGET_STORM = "2025289N13140"


# --------------------------------------------------
# 1. Load IBTrACS
# --------------------------------------------------

ds = xr.open_dataset(IBTRACS_PATH)


# --------------------------------------------------
# 2. Find target storm
# --------------------------------------------------

storm_sids = ds["sid"].values.astype(str)

matches = np.where(storm_sids == TARGET_STORM)[0]

if len(matches) == 0:
    raise ValueError(f"Storm {TARGET_STORM} not found")

storm_idx = matches[0]

print(f"Storm index: {storm_idx}")
print(f"SID: {TARGET_STORM}")


# --------------------------------------------------
# 3. Extract valid IBTrACS observations
# --------------------------------------------------

times = ds["time"].isel(storm=storm_idx).values
lats = ds["lat"].isel(storm=storm_idx).values
lons = ds["lon"].isel(storm=storm_idx).values

valid = (
    ~np.isnat(times)
    & ~np.isnan(lats)
    & ~np.isnan(lons)
)

ib_times = times[valid].astype("datetime64[m]")
ib_lats = lats[valid]
ib_lons = lons[valid]


# --------------------------------------------------
# 4. Restrict to 2025-10-20
# --------------------------------------------------

target_day = np.datetime64("2025-10-20")

day_mask = (
    ib_times.astype("datetime64[D]") == target_day
)

ib_times = ib_times[day_mask]
ib_lats = ib_lats[day_mask]
ib_lons = ib_lons[day_mask]


# --------------------------------------------------
# 5. Extract Himawari timestamps
# --------------------------------------------------

himawari_times = set()

pattern = re.compile(
    r"HS_H08_(\d{8})_(\d{4})_"
)

for file in HIMAWARI_DIR.glob("*.DAT"):

    match = pattern.search(file.name)

    if not match:
        continue

    date = match.group(1)
    time = match.group(2)

    timestamp = np.datetime64(
        f"{date[:4]}-{date[4:6]}-{date[6:8]}T"
        f"{time[:2]}:{time[2:]}"
    )

    himawari_times.add(timestamp)


himawari_times = sorted(himawari_times)


# --------------------------------------------------
# 6. Find exact timestamp matches
# --------------------------------------------------

himawari_times = [
    t.astype("datetime64[m]")
    for t in himawari_times
]

himawari_set = set(himawari_times)

print("\nExact timestamp matches")
print("-" * 80)

match_count = 0

for time, lat, lon in zip(
    ib_times,
    ib_lats,
    ib_lons
):

    if time in himawari_set:

        print(
            f"{time} | "
            f"lat={lat:.2f} | "
            f"lon={lon:.2f}"
        )

        match_count += 1


print("-" * 80)
print(f"IBTrACS observations: {len(ib_times)}")
print(f"Himawari timestamps: {len(himawari_times)}")
print(f"Exact matches: {match_count}")
