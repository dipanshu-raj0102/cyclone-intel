from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================

DATE = "2025-10-20"

MANIFEST_PATH = (
    Path("../processed")
    / "identification"
    / DATE
    / "scene_manifest.csv"
)

EXPECTED_CHANNEL = "B13"
EXPECTED_HEIGHT = 5500
EXPECTED_WIDTH = 5500
EXPECTED_SEGMENTS = 10


# ============================================================
# Validation helpers
# ============================================================

def fail(message: str) -> None:
    """Raise a validation error with a clear message."""
    raise ValueError(f"\nVALIDATION FAILED:\n{message}")


def check_required_columns(df: pd.DataFrame) -> None:
    """Verify that all required manifest columns exist."""

    required = {
        "scene_id",
        "timestamp",
        "channel",
        "height",
        "width",
        "n_segments",
        "cyclone_present",
        "num_cyclones",
        "centers",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        fail(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def check_manifest_not_empty(df: pd.DataFrame) -> None:
    """Ensure that the manifest contains scenes."""

    if df.empty:
        fail("Manifest contains zero scenes.")


def check_unique_scene_ids(df: pd.DataFrame) -> None:
    """Ensure scene IDs are unique."""

    duplicates = df[df["scene_id"].duplicated(keep=False)]

    if not duplicates.empty:
        fail(
            "Duplicate scene_id values found:\n"
            + duplicates["scene_id"].to_string(index=False)
        )


def check_unique_timestamps(df: pd.DataFrame) -> None:
    """Ensure there is only one scene per timestamp."""

    duplicates = df[df["timestamp"].duplicated(keep=False)]

    if not duplicates.empty:
        fail(
            "Duplicate timestamps found:\n"
            + duplicates["timestamp"].to_string(index=False)
        )


def check_channel(df: pd.DataFrame) -> None:
    """Verify the expected Himawari channel."""

    invalid = df[df["channel"] != EXPECTED_CHANNEL]

    if not invalid.empty:
        fail(
            f"Unexpected channel values found. "
            f"Expected {EXPECTED_CHANNEL}.\n"
            f"{invalid[['scene_id', 'channel']].to_string(index=False)}"
        )


def check_scene_dimensions(df: pd.DataFrame) -> None:
    """Verify complete full-disk scene dimensions."""

    invalid = df[
        (df["height"] != EXPECTED_HEIGHT)
        | (df["width"] != EXPECTED_WIDTH)
    ]

    if not invalid.empty:
        fail(
            "Unexpected scene dimensions found. "
            f"Expected {EXPECTED_HEIGHT} × {EXPECTED_WIDTH}.\n"
            f"{invalid[['scene_id', 'height', 'width']].to_string(index=False)}"
        )


def check_segment_count(df: pd.DataFrame) -> None:
    """Verify that every scene contains exactly 10 segments."""

    invalid = df[df["n_segments"] != EXPECTED_SEGMENTS]

    if not invalid.empty:
        fail(
            f"Invalid segment count. "
            f"Expected {EXPECTED_SEGMENTS} per scene.\n"
            f"{invalid[['scene_id', 'n_segments']].to_string(index=False)}"
        )


def parse_centers(value, scene_id: str) -> list[dict]:
    """Parse the JSON encoded cyclone centers."""

    if pd.isna(value):
        fail(f"{scene_id}: centers is NaN.")

    try:
        centers = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        fail(
            f"{scene_id}: invalid centers JSON.\n"
            f"Error: {exc}"
        )

    if not isinstance(centers, list):
        fail(
            f"{scene_id}: centers must decode to a list, "
            f"got {type(centers).__name__}."
        )

    return centers


def check_cyclone_labels(df: pd.DataFrame) -> None:
    """
    Validate the relationship between cyclone_present,
    num_cyclones and centers.
    """

    for _, row in df.iterrows():

        scene_id = row["scene_id"]

        present = row["cyclone_present"]
        num_cyclones = row["num_cyclones"]

        if present not in (0, 1):
            fail(
                f"{scene_id}: cyclone_present must be 0 or 1, "
                f"got {present!r}."
            )

        if not isinstance(num_cyclones, (int, np.integer)):
            if not float(num_cyclones).is_integer():
                fail(
                    f"{scene_id}: num_cyclones must be an integer, "
                    f"got {num_cyclones!r}."
                )

        num_cyclones = int(num_cyclones)

        if num_cyclones < 0:
            fail(
                f"{scene_id}: num_cyclones cannot be negative."
            )

        centers = parse_centers(
            row["centers"],
            scene_id,
        )

        if len(centers) != num_cyclones:
            fail(
                f"{scene_id}: num_cyclones={num_cyclones}, "
                f"but centers contains {len(centers)} entries."
            )

        expected_present = int(num_cyclones > 0)

        if int(present) != expected_present:
            fail(
                f"{scene_id}: inconsistent cyclone label. "
                f"cyclone_present={present}, "
                f"num_cyclones={num_cyclones}."
            )


def check_centers(df: pd.DataFrame) -> None:
    """Validate every stored cyclone center."""

    required_center_fields = {
        "storm_id",
        "latitude",
        "longitude",
        "row",
        "col",
    }

    seen_pairs: set[tuple[str, pd.Timestamp]] = set()

    for _, scene in df.iterrows():

        scene_id = scene["scene_id"]
        timestamp = pd.Timestamp(scene["timestamp"])

        centers = parse_centers(
            scene["centers"],
            scene_id,
        )

        for center_index, center in enumerate(centers):

            prefix = (
                f"{scene_id}, center #{center_index + 1}"
            )

            if not isinstance(center, dict):
                fail(
                    f"{prefix}: center must be a JSON object."
                )

            missing = required_center_fields - set(center)

            if missing:
                fail(
                    f"{prefix}: missing fields: "
                    f"{sorted(missing)}"
                )

            storm_id = center["storm_id"]

            if not isinstance(storm_id, str) or not storm_id:
                fail(
                    f"{prefix}: invalid storm_id."
                )

            latitude = center["latitude"]
            longitude = center["longitude"]

            if not np.isfinite(latitude):
                fail(
                    f"{prefix}: latitude is not finite."
                )

            if not np.isfinite(longitude):
                fail(
                    f"{prefix}: longitude is not finite."
                )

            if not -90.0 <= float(latitude) <= 90.0:
                fail(
                    f"{prefix}: latitude out of range: "
                    f"{latitude}"
                )

            if not -180.0 <= float(longitude) <= 180.0:
                fail(
                    f"{prefix}: longitude out of range: "
                    f"{longitude}"
                )

            row = center["row"]
            col = center["col"]

            if not isinstance(row, (int, np.integer)):
                if not float(row).is_integer():
                    fail(
                        f"{prefix}: row must be an integer."
                    )

            if not isinstance(col, (int, np.integer)):
                if not float(col).is_integer():
                    fail(
                        f"{prefix}: col must be an integer."
                    )

            row = int(row)
            col = int(col)

            if not (
                0 <= row < EXPECTED_HEIGHT
                and 0 <= col < EXPECTED_WIDTH
            ):
                fail(
                    f"{prefix}: pixel outside scene bounds: "
                    f"row={row}, col={col}"
                )

            pair = (storm_id, timestamp)

            if pair in seen_pairs:
                fail(
                    f"Duplicate storm/timestamp annotation: "
                    f"{storm_id} @ {timestamp}"
                )

            seen_pairs.add(pair)


def check_unknowns(df: pd.DataFrame) -> None:
    """
    The identification manifest should contain only verified
    exact-match scenes.

    Unmatched Himawari timestamps were deliberately excluded
    during dataset construction.
    """

    if df["cyclone_present"].isna().any():
        fail(
            "Manifest contains scenes with unknown "
            "cyclone_present labels."
        )


# ============================================================
# Main validation
# ============================================================

def main() -> None:

    print("=" * 60)
    print("HIMAWARI IDENTIFICATION DATASET VALIDATOR")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Locate manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    print(f"\nManifest: {MANIFEST_PATH}")

    # --------------------------------------------------------
    # 2. Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Scenes loaded: {len(df)}")

    # --------------------------------------------------------
    # 3. Structural validation
    # --------------------------------------------------------

    check_manifest_not_empty(df)
    check_required_columns(df)

    # Normalize timestamp.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        fail("One or more timestamps could not be parsed.")

    # --------------------------------------------------------
    # 4. Scene-level validation
    # --------------------------------------------------------

    check_unique_scene_ids(df)
    check_unique_timestamps(df)
    check_channel(df)
    check_scene_dimensions(df)
    check_segment_count(df)

    # --------------------------------------------------------
    # 5. Label validation
    # --------------------------------------------------------

    check_cyclone_labels(df)
    check_centers(df)
    check_unknowns(df)

    # --------------------------------------------------------
    # 6. Dataset statistics
    # --------------------------------------------------------

    total_scenes = len(df)

    positive_scenes = int(
        (df["cyclone_present"] == 1).sum()
    )

    negative_scenes = int(
        (df["cyclone_present"] == 0).sum()
    )

    total_centers = int(
        df["num_cyclones"].sum()
    )

    unique_storms = set()

    for centers_json in df["centers"]:
        centers = json.loads(centers_json)

        for center in centers:
            unique_storms.add(center["storm_id"])

    # --------------------------------------------------------
    # 7. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION PASSED")
    print("=" * 60)

    print(f"Total scenes              : {total_scenes}")
    print(f"Positive scenes           : {positive_scenes}")
    print(f"Negative scenes           : {negative_scenes}")
    print(f"Total cyclone annotations : {total_centers}")
    print(f"Unique storms             : {len(unique_storms)}")
    print(f"Channel                   : {EXPECTED_CHANNEL}")
    print(
        f"Scene dimensions          : "
        f"{EXPECTED_HEIGHT} × {EXPECTED_WIDTH}"
    )
    print(f"Segments per scene        : {EXPECTED_SEGMENTS}")

    print("\nDataset integrity checks:")
    print("  ✓ Required columns")
    print("  ✓ Non-empty manifest")
    print("  ✓ Unique scene IDs")
    print("  ✓ Unique timestamps")
    print("  ✓ Correct channel")
    print("  ✓ Correct scene dimensions")
    print("  ✓ Correct segment count")
    print("  ✓ Binary cyclone labels")
    print("  ✓ cyclone_present ↔ num_cyclones")
    print("  ✓ centers ↔ num_cyclones")
    print("  ✓ Valid geographic coordinates")
    print("  ✓ Valid pixel coordinates")
    print("  ✓ Unique storm/timestamp annotations")
    print("  ✓ No unknown labels")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
