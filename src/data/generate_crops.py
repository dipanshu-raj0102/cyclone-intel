import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from satpy import Scene

# ============================================================
# Configuration
# ============================================================

STORM_ID = "2025289N13140"

DATE = "2025-10-20"

CHANNEL = "B13"

CROP_SIZE = 256
HALF = CROP_SIZE // 2


# ------------------------------------------------------------
# Input paths
# ------------------------------------------------------------

HIMAWARI_DIR = Path(
    "./interim/himawari_dat/2025/10/20"
)

ALIGNED_TRACK = Path(
    "processed/himawari_ibtracs/"
    "2025-10-20/"
    f"{STORM_ID}/"
    "aligned_track.csv"
)


# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "processed/himawari_ibtracs/"
    "2025-10-20/"
    f"{STORM_ID}"
)

CROP_DIR = OUTPUT_DIR / "crops"

METADATA_PATH = OUTPUT_DIR / "metadata.csv"


# ============================================================
# Himawari filename pattern
# ============================================================

FILENAME_PATTERN = re.compile(
    r"HS_H08_(\d{8})_(\d{4})_"
)


# ============================================================
# Build Himawari timestamp index
# ============================================================

def build_himawari_index():

    print("=" * 70)
    print("INDEXING HIMAWARI FILES")
    print("=" * 70)

    timestamp_index = {}

    files = list(
        HIMAWARI_DIR.glob("*.DAT")
    )

    print(
        f"DAT files found: {len(files)}"
    )

    for file in files:

        match = FILENAME_PATTERN.search(
            file.name
        )

        if not match:
            continue

        date = match.group(1)
        clock = match.group(2)

        timestamp = pd.Timestamp(
            f"{date[:4]}-{date[4:6]}-{date[6:8]} "
            f"{clock[:2]}:{clock[2:]}"
        )

        timestamp_index.setdefault(
            timestamp,
            []
        ).append(file)

    # Sort segments
    for timestamp in timestamp_index:
        timestamp_index[timestamp].sort()

    print(
        f"Unique Himawari timestamps: "
        f"{len(timestamp_index)}"
    )

    # Validate number of segments
    invalid = {
        timestamp: files
        for timestamp, files
        in timestamp_index.items()
        if len(files) != 10
    }

    if invalid:

        print(
            "\nWARNING:"
            f" {len(invalid)} timestamps "
            "do not contain exactly 10 segments."
        )

        for timestamp, files in invalid.items():

            print(
                f"  {timestamp}: "
                f"{len(files)} segments"
            )

    else:

        print(
            "All timestamps contain exactly "
            "10 segments."
        )

    return timestamp_index


# ============================================================
# Load aligned trajectory
# ============================================================

def load_aligned_track():

    print("\n" + "=" * 70)
    print("LOADING ALIGNED TRACK")
    print("=" * 70)

    if not ALIGNED_TRACK.exists():

        raise FileNotFoundError(
            f"Aligned track not found:\n"
            f"{ALIGNED_TRACK}"
        )

    track = pd.read_csv(
        ALIGNED_TRACK,
        parse_dates=["timestamp"],
    )

    track = track.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Aligned samples: {len(track)}"
    )

    print(
        f"First timestamp: "
        f"{track['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp: "
        f"{track['timestamp'].iloc[-1]}"
    )

    required_columns = [
        "timestamp",
        "latitude",
        "longitude",
    ]

    missing = [
        column
        for column in required_columns
        if column not in track.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return track


# ============================================================
# Load one Himawari image
# ============================================================

def load_himawari_b13(files):

    scene = Scene(
        filenames=[
            str(file)
            for file in files
        ],
        reader="ahi_hsd",
    )

    scene.load([CHANNEL])

    band = scene[CHANNEL]

    image = band.values.astype(
        np.float32
    )

    return band, image


# ============================================================
# Geographic coordinate → pixel
# ============================================================

def latlon_to_pixel(
    band,
    latitude,
    longitude,
):

    area = band.attrs["area"]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        area.crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        float(longitude),
        float(latitude),
    )

    x_coords = band.coords["x"].values
    y_coords = band.coords["y"].values

    column = int(
        np.abs(x_coords - x).argmin()
    )

    row = int(
        np.abs(y_coords - y).argmin()
    )

    return row, column, x, y


# ============================================================
# Extract crop
# ============================================================

def extract_crop(
    image,
    row,
    column,
):

    row_start = row - HALF
    row_end = row + HALF

    col_start = column - HALF
    col_end = column + HALF

    # Check image boundaries
    if row_start < 0:
        raise ValueError(
            f"Crop extends above image: "
            f"row_start={row_start}"
        )

    if row_end > image.shape[0]:
        raise ValueError(
            f"Crop extends below image: "
            f"row_end={row_end}, "
            f"image_height={image.shape[0]}"
        )

    if col_start < 0:
        raise ValueError(
            f"Crop extends left of image: "
            f"col_start={col_start}"
        )

    if col_end > image.shape[1]:
        raise ValueError(
            f"Crop extends right of image: "
            f"col_end={col_end}, "
            f"image_width={image.shape[1]}"
        )

    crop = image[
        row_start:row_end,
        col_start:col_end,
    ]

    if crop.shape != (
        CROP_SIZE,
        CROP_SIZE,
    ):

        raise ValueError(
            f"Unexpected crop shape: "
            f"{crop.shape}"
        )

    return crop


# ============================================================
# Main generation
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load track
    # --------------------------------------------------------

    track = load_aligned_track()

    # --------------------------------------------------------
    # Build Himawari index
    # --------------------------------------------------------

    himawari_index = (
        build_himawari_index()
    )

    # --------------------------------------------------------
    # Metadata fields
    # --------------------------------------------------------

    metadata_fields = [
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

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    successful = 0
    failed = 0

    failures = []

    # --------------------------------------------------------
    # Metadata CSV
    # --------------------------------------------------------

    with METADATA_PATH.open(
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=metadata_fields,
        )

        writer.writeheader()

        # ----------------------------------------------------
        # Process each aligned observation
        # ----------------------------------------------------

        for index, observation in track.iterrows():

            timestamp = observation[
                "timestamp"
            ]

            latitude = float(
                observation["latitude"]
            )

            longitude = float(
                observation["longitude"]
            )

            timestamp = pd.Timestamp(
                timestamp
            )

            timestamp_str = (
                timestamp.strftime(
                    "%Y-%m-%dT%H%M"
                )
            )

            print(
                "\n" + "=" * 70
            )

            print(
                f"[{index + 1}/{len(track)}] "
                f"{timestamp_str}"
            )

            print(
                f"Lat/Lon: "
                f"{latitude:.6f}, "
                f"{longitude:.6f}"
            )

            # ------------------------------------------------
            # Find Himawari segments
            # ------------------------------------------------

            files = himawari_index.get(
                timestamp,
                [],
            )

            if len(files) != 10:

                message = (
                    f"Expected 10 segments, "
                    f"found {len(files)}"
                )

                print(
                    f"FAILED: {message}"
                )

                failures.append({
                    "timestamp": timestamp_str,
                    "reason": message,
                })

                failed += 1

                continue

            # ------------------------------------------------
            # Load satellite data
            # ------------------------------------------------

            try:

                band, image = (
                    load_himawari_b13(files)
                )

            except Exception as exc:

                message = (
                    f"Satpy loading failed: "
                    f"{exc}"
                )

                print(
                    f"FAILED: {message}"
                )

                failures.append({
                    "timestamp": timestamp_str,
                    "reason": message,
                })

                failed += 1

                continue

            # ------------------------------------------------
            # Convert lat/lon → pixel
            # ------------------------------------------------

            try:

                (
                    pixel_row,
                    pixel_col,
                    projection_x,
                    projection_y,
                ) = latlon_to_pixel(
                    band,
                    latitude,
                    longitude,
                )

            except Exception as exc:

                message = (
                    f"Geolocation failed: "
                    f"{exc}"
                )

                print(
                    f"FAILED: {message}"
                )

                failures.append({
                    "timestamp": timestamp_str,
                    "reason": message,
                })

                failed += 1

                continue

            print(
                f"Pixel: "
                f"row={pixel_row}, "
                f"col={pixel_col}"
            )

            # ------------------------------------------------
            # Extract crop
            # ------------------------------------------------

            try:

                crop = extract_crop(
                    image,
                    pixel_row,
                    pixel_col,
                )

            except Exception as exc:

                message = (
                    f"Crop extraction failed: "
                    f"{exc}"
                )

                print(
                    f"FAILED: {message}"
                )

                failures.append({
                    "timestamp": timestamp_str,
                    "reason": message,
                })

                failed += 1

                continue

            # ------------------------------------------------
            # Validate numerical values
            # ------------------------------------------------

            nan_pixels = int(
                np.isnan(crop).sum()
            )

            finite_pixels = int(
                np.isfinite(crop).sum()
            )

            if finite_pixels == 0:

                message = (
                    "Crop contains no finite "
                    "pixels"
                )

                print(
                    f"FAILED: {message}"
                )

                failures.append({
                    "timestamp": timestamp_str,
                    "reason": message,
                })

                failed += 1

                continue

            min_temperature = float(
                np.nanmin(crop)
            )

            max_temperature = float(
                np.nanmax(crop)
            )

            mean_temperature = float(
                np.nanmean(crop)
            )

            print(
                f"Crop shape: {crop.shape}"
            )

            print(
                f"NaN pixels: {nan_pixels}"
            )

            print(
                f"Temperature: "
                f"{min_temperature:.2f} K → "
                f"{max_temperature:.2f} K"
            )

            # ------------------------------------------------
            # Sample ID
            # ------------------------------------------------

            sample_id = (
                f"{STORM_ID}_"
                f"{timestamp_str}"
            )

            output_path = (
                CROP_DIR
                / f"{sample_id}.npy"
            )

            # ------------------------------------------------
            # Save NumPy crop
            # ------------------------------------------------

            np.save(
                output_path,
                crop,
            )

            # ------------------------------------------------
            # Read actual IBTrACS labels if present
            #
            # For interpolated timestamps these are NaN.
            # We deliberately DO NOT interpolate them.
            # ------------------------------------------------

            wind = observation.get(
                "wind_kt",
                np.nan,
            )

            pressure = observation.get(
                "pressure_mb",
                np.nan,
            )

            if pd.isna(wind):
                wind_value = ""
            else:
                wind_value = float(wind)

            if pd.isna(pressure):
                pressure_value = ""
            else:
                pressure_value = float(pressure)

            # ------------------------------------------------
            # Write metadata
            # ------------------------------------------------

            writer.writerow({
                "sample_id": sample_id,
                "storm_id": STORM_ID,
                "timestamp": timestamp_str,
                "latitude": latitude,
                "longitude": longitude,
                "wind_kt": wind_value,
                "pressure_mb": pressure_value,
                "pixel_row": pixel_row,
                "pixel_col": pixel_col,
                "crop_size": CROP_SIZE,
                "channel": CHANNEL,
                "n_segments": len(files),
                "nan_pixels": nan_pixels,
                "finite_pixels": finite_pixels,
                "min_temperature_k": min_temperature,
                "max_temperature_k": max_temperature,
                "mean_temperature_k": mean_temperature,
            })

            successful += 1

            print(
                f"Saved: {output_path}"
            )

    # ========================================================
    # Save failure report
    # ========================================================

    failure_path = (
        OUTPUT_DIR
        / "generation_failures.csv"
    )

    if failures:

        pd.DataFrame(
            failures
        ).to_csv(
            failure_path,
            index=False,
        )

    # ========================================================
    # Final report
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Aligned observations : {len(track)}"
    )

    print(
        f"Successful crops     : {successful}"
    )

    print(
        f"Failed crops         : {failed}"
    )

    print(
        f"Success rate         : "
        f"{successful / len(track) * 100:.2f}%"
    )

    print(
        "\nCrop directory:"
    )

    print(
        CROP_DIR
    )

    print(
        "\nMetadata:"
    )

    print(
        METADATA_PATH
    )

    if failures:

        print(
            "\nFailure report:"
        )

        print(
            failure_path
        )


if __name__ == "__main__":
    main()
