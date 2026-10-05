from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

STORM_ID = "2025289N13140"
DATE = "2025-10-20"

BASE_DIR = (
    Path("processed")
    / "himawari_ibtracs"
    / DATE
    / STORM_ID
)

CROP_DIR = BASE_DIR / "crops"
METADATA_PATH = BASE_DIR / "metadata.csv"
ALIGNED_TRACK_PATH = BASE_DIR / "aligned_track.csv"


EXPECTED_SHAPE = (256, 256)
EXPECTED_DTYPE = np.float32


# ============================================================
# Helpers
# ============================================================

def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")

    if not condition:
        raise RuntimeError(message)


# ============================================================
# Main validation
# ============================================================

def main():

    print("=" * 70)
    print("CYCLONE CROP DATASET VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check paths
    # --------------------------------------------------------

    check(BASE_DIR.exists(), f"Dataset directory exists: {BASE_DIR}")
    check(CROP_DIR.exists(), f"Crop directory exists: {CROP_DIR}")
    check(
        METADATA_PATH.exists(),
        f"Metadata exists: {METADATA_PATH}",
    )
    check(
        ALIGNED_TRACK_PATH.exists(),
        f"Aligned track exists: {ALIGNED_TRACK_PATH}",
    )

    # --------------------------------------------------------
    # 2. Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)

    print()
    print(f"Metadata rows: {len(metadata)}")

    check(
        len(metadata) == 125,
        "Metadata contains exactly 125 samples",
    )

    # --------------------------------------------------------
    # 3. Required metadata columns
    # --------------------------------------------------------

    required_columns = [
        "sample_id",
        "storm_id",
        "timestamp",
        "latitude",
        "longitude",
        "wind_kt",
        "pressure_mb",
        "pixel_row",
        "pixel_col",
        "crop_size",
        "channel",
        "n_segments",
        "nan_pixels",
        "finite_pixels",
        "min_temperature_k",
        "max_temperature_k",
        "mean_temperature_k",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    check(
        len(missing_columns) == 0,
        "Required metadata columns present",
    )

    # --------------------------------------------------------
    # 4. Timestamp validation
    # --------------------------------------------------------

    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"]
    )

    duplicate_timestamps = metadata["timestamp"].duplicated().sum()

    check(
        duplicate_timestamps == 0,
        f"No duplicate timestamps ({duplicate_timestamps})",
    )

    timestamps_sorted = metadata["timestamp"].is_monotonic_increasing

    check(
        timestamps_sorted,
        "Metadata timestamps are sorted",
    )

    # --------------------------------------------------------
    # 5. Storm ID validation
    # --------------------------------------------------------

    unique_storms = metadata["storm_id"].unique()

    check(
        len(unique_storms) == 1
        and unique_storms[0] == STORM_ID,
        f"All samples belong to {STORM_ID}",
    )

    # --------------------------------------------------------
    # 6. Crop-level validation
    # --------------------------------------------------------

    crop_files = sorted(CROP_DIR.glob("*.npy"))

    print()
    print(f"Crop files found: {len(crop_files)}")

    check(
        len(crop_files) == 125,
        "Exactly 125 crop files exist",
    )

    shape_errors = []
    dtype_errors = []
    nan_errors = []
    finite_errors = []
    corrupted_files = []

    global_min = np.inf
    global_max = -np.inf

    for index, crop_path in enumerate(crop_files, start=1):

        try:
            crop = np.load(crop_path)

            if crop.shape != EXPECTED_SHAPE:
                shape_errors.append(
                    (crop_path.name, crop.shape)
                )

            if crop.dtype != EXPECTED_DTYPE:
                dtype_errors.append(
                    (crop_path.name, crop.dtype)
                )

            nan_count = np.isnan(crop).sum()

            if nan_count != 0:
                nan_errors.append(
                    (crop_path.name, int(nan_count))
                )

            finite_count = np.isfinite(crop).sum()

            if finite_count != crop.size:
                finite_errors.append(
                    (crop_path.name, int(finite_count))
                )

            finite_values = crop[np.isfinite(crop)]

            if finite_values.size > 0:
                global_min = min(
                    global_min,
                    float(finite_values.min()),
                )

                global_max = max(
                    global_max,
                    float(finite_values.max()),
                )

        except Exception as exc:
            corrupted_files.append(
                (crop_path.name, str(exc))
            )

    check(
        len(corrupted_files) == 0,
        f"No corrupted crop files ({len(corrupted_files)})",
    )

    check(
        len(shape_errors) == 0,
        f"All crops have shape {EXPECTED_SHAPE}",
    )

    check(
        len(dtype_errors) == 0,
        f"All crops use dtype {EXPECTED_DTYPE}",
    )

    check(
        len(nan_errors) == 0,
        "All crops contain zero NaN pixels",
    )

    check(
        len(finite_errors) == 0,
        "All crop pixels are finite",
    )

    # --------------------------------------------------------
    # 7. Metadata ↔ crop correspondence
    # --------------------------------------------------------

    metadata_sample_ids = set(
        metadata["sample_id"].astype(str)
    )

    crop_sample_ids = {
        crop_path.stem
        for crop_path in crop_files
    }

    missing_crops = metadata_sample_ids - crop_sample_ids
    extra_crops = crop_sample_ids - metadata_sample_ids

    check(
        len(missing_crops) == 0,
        "No metadata entries missing corresponding crops",
    )

    check(
        len(extra_crops) == 0,
        "No extra crops without metadata",
    )

    # --------------------------------------------------------
    # 8. Pixel coordinate validation
    # --------------------------------------------------------

    invalid_rows = metadata[
        (metadata["pixel_row"] < 0)
        | (metadata["pixel_row"] >= 5500)
    ]

    invalid_cols = metadata[
        (metadata["pixel_col"] < 0)
        | (metadata["pixel_col"] >= 5500)
    ]

    check(
        len(invalid_rows) == 0,
        "All pixel row coordinates are valid",
    )

    check(
        len(invalid_cols) == 0,
        "All pixel column coordinates are valid",
    )

    # --------------------------------------------------------
    # 9. Crop metadata consistency
    # --------------------------------------------------------

    check(
        (metadata["crop_size"] == 256).all(),
        "All metadata entries specify crop_size=256",
    )

    check(
        (metadata["channel"] == "B13").all(),
        "All samples use Himawari B13",
    )

    check(
        (metadata["n_segments"] == 10).all(),
        "All samples contain 10 Himawari segments",
    )

    check(
        (metadata["nan_pixels"] == 0).all(),
        "Metadata reports zero NaN pixels for every crop",
    )

    # --------------------------------------------------------
    # 10. Temperature statistics
    # --------------------------------------------------------

    check(
        metadata["min_temperature_k"].notna().all(),
        "All samples have minimum temperature",
    )

    check(
        metadata["max_temperature_k"].notna().all(),
        "All samples have maximum temperature",
    )

    check(
        metadata["mean_temperature_k"].notna().all(),
        "All samples have mean temperature",
    )

    check(
        (
            metadata["min_temperature_k"]
            <= metadata["mean_temperature_k"]
        ).all(),
        "Minimum temperature <= mean temperature",
    )

    check(
        (
            metadata["mean_temperature_k"]
            <= metadata["max_temperature_k"]
        ).all(),
        "Mean temperature <= maximum temperature",
    )

    # --------------------------------------------------------
    # 11. Track validation
    # --------------------------------------------------------

    aligned = pd.read_csv(ALIGNED_TRACK_PATH)

    aligned["timestamp"] = pd.to_datetime(
        aligned["timestamp"]
    )

    check(
        len(aligned) == 125,
        "Aligned track contains 125 observations",
    )

    check(
        aligned["timestamp"].is_unique,
        "Aligned track timestamps are unique",
    )

    # --------------------------------------------------------
    # 12. Metadata ↔ aligned track timestamps
    # --------------------------------------------------------

    metadata_times = set(metadata["timestamp"])
    aligned_times = set(aligned["timestamp"])

    missing_alignment = aligned_times - metadata_times
    extra_metadata = metadata_times - aligned_times

    check(
        len(missing_alignment) == 0,
        "Every aligned timestamp has a crop",
    )

    check(
        len(extra_metadata) == 0,
        "Every crop corresponds to an aligned timestamp",
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Storm ID             : {STORM_ID}")
    print(f"Samples              : {len(crop_files)}")
    print(f"Crop shape           : {EXPECTED_SHAPE}")
    print(f"Dtype                : {EXPECTED_DTYPE}")
    print("Channel              : B13")
    print(f"Total NaN pixels     : {metadata['nan_pixels'].sum()}")
    print(f"Global minimum       : {global_min:.2f} K")
    print(f"Global maximum       : {global_max:.2f} K")

    print()
    print("Dataset validation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
