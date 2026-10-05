import re
from pathlib import Path

import pandas as pd

TRACK_PATH = Path(
    "processed/himawari_ibtracs/"
    "2025-10-20/2025289N13140/"
    "interpolated_track.csv"
)

HIMAWARI_DIR = Path(
    "./interim/himawari_dat/"
    "2025/10/20"
)

OUTPUT_PATH = Path(
    "processed/himawari_ibtracs/"
    "2025-10-20/2025289N13140/"
    "aligned_track.csv"
)


# ------------------------------------------------------------
# Load interpolated trajectory
# ------------------------------------------------------------

track = pd.read_csv(
    TRACK_PATH,
    parse_dates=["timestamp"],
)

track = track.sort_values(
    "timestamp"
)

print(
    f"Trajectory positions: {len(track)}"
)


# ------------------------------------------------------------
# Extract Himawari timestamps
# ------------------------------------------------------------

pattern = re.compile(
    r"HS_H08_(\d{8})_(\d{4})_"
)

himawari_timestamps = set()

for file in HIMAWARI_DIR.glob("*.DAT"):

    match = pattern.search(file.name)

    if not match:
        continue

    date = match.group(1)
    clock = match.group(2)

    timestamp = pd.Timestamp(
        f"{date[:4]}-{date[4:6]}-{date[6:8]} "
        f"{clock[:2]}:{clock[2:]}"
    )

    himawari_timestamps.add(timestamp)


himawari_timestamps = sorted(
    himawari_timestamps
)

print(
    f"Himawari timestamps: {len(himawari_timestamps)}"
)


# ------------------------------------------------------------
# Restrict Himawari to trajectory period
# ------------------------------------------------------------

start = track["timestamp"].min()
end = track["timestamp"].max()

himawari_in_range = [
    t
    for t in himawari_timestamps
    if start <= t <= end
]

print(
    f"Himawari timestamps within "
    f"trajectory period: {len(himawari_in_range)}"
)


# ------------------------------------------------------------
# Find missing Himawari timestamps
# ------------------------------------------------------------

track_timestamps = set(
    track["timestamp"]
)

himawari_set = set(
    himawari_in_range
)

missing_himawari = sorted(
    track_timestamps - himawari_set
)

print(
    "\nTrajectory timestamps without "
    "Himawari data:"
)

for timestamp in missing_himawari:
    print(
        timestamp.strftime(
            "%Y-%m-%d %H:%M"
        )
    )


# ------------------------------------------------------------
# Keep only timestamps with both datasets
# ------------------------------------------------------------

aligned = track[
    track["timestamp"].isin(
        himawari_set
    )
].copy()

aligned = aligned.sort_values(
    "timestamp"
)

# ------------------------------------------------------------
# Add alignment metadata
# ------------------------------------------------------------

aligned["has_himawari"] = True

aligned["n_segments"] = 10


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

aligned.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    "\n" + "=" * 70
)

print(
    f"Aligned samples: {len(aligned)}"
)

print(
    f"Output: {OUTPUT_PATH}"
)

print(
    "=" * 70
)
