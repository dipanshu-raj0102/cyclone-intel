from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from satpy import Scene

DATA_DIR = Path(
    "./interim/himawari_dat/2025/10/20"
)

# --------------------------------------------------
# 1. Load Himawari image
# --------------------------------------------------

files = sorted(
    DATA_DIR.glob(
        "HS_H08_20251020_0000_B13_FLDK_R20_S*.DAT"
    )
)

scene = Scene(
    filenames=[str(f) for f in files],
    reader="ahi_hsd",
)

scene.load(["B13"])

band = scene["B13"]
area = band.attrs["area"]


# --------------------------------------------------
# 2. Get Himawari projection
# --------------------------------------------------

projection = area.crs

print("Himawari CRS:")
print(projection)


# --------------------------------------------------
# 3. Create transformer
# --------------------------------------------------

transformer = Transformer.from_crs(
    "EPSG:4326",
    projection,
    always_xy=True,
)


# --------------------------------------------------
# 4. Cyclone position from IBTrACS
# --------------------------------------------------

latitude = 18.00
longitude = 116.50


# --------------------------------------------------
# 5. Geographic → projection coordinates
# --------------------------------------------------

x, y = transformer.transform(
    longitude,
    latitude,
)

# --------------------------------------------------
# 6. Convert projection coordinates to pixel indices
# --------------------------------------------------

x_coords = band.coords["x"].values
y_coords = band.coords["y"].values


# Find nearest pixel
column = np.abs(x_coords - x).argmin()
row = np.abs(y_coords - y).argmin()


# --------------------------------------------------
# 7. Visual verification
# --------------------------------------------------

image = band.values

plt.figure(figsize=(10, 10))

plt.imshow(
    image,
    cmap="gray_r",
    vmin=np.nanpercentile(image, 1),
    vmax=np.nanpercentile(image, 99),
)

plt.scatter(
    column,
    row,
    marker="x",
    s=150,
    linewidths=3,
)

plt.xlim(column - 500, column + 500)
plt.ylim(row + 500, row - 500)

plt.xlabel("Column")
plt.ylabel("Row")
plt.title(
    "Himawari-8 B13 — IBTrACS Cyclone Center\n"
    "2025-10-20 00:00 UTC"
)

plt.tight_layout()

plt.savefig(
    "cyclone_center_check.png",
    dpi=150,
)
plt.close()

# --------------------------------------------------
# 8. Extract cyclone-centered crop
# --------------------------------------------------

CROP_SIZE = 256
HALF = CROP_SIZE // 2

row_start = row - HALF
row_end = row + HALF

col_start = column - HALF
col_end = column + HALF


# Check that crop stays inside image
if (
    row_start < 0
    or row_end > band.shape[0]
    or col_start < 0
    or col_end > band.shape[1]
):
    raise ValueError(
        "Cyclone crop extends outside the Himawari image"
    )


crop = band.values[
    row_start:row_end,
    col_start:col_end,
]

# --------------------------------------------------
# 9. Visualize cyclone-centered crop
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(
    crop,
    cmap="gray_r",
    vmin=np.nanpercentile(crop, 1),
    vmax=np.nanpercentile(crop, 99),
)

# Crop center
plt.scatter(
    HALF,
    HALF,
    marker="x",
    s=150,
    linewidths=3,
)

plt.xlabel("Crop Column")
plt.ylabel("Crop Row")

plt.title(
    "Himawari-8 B13 Cyclone-Centered Crop\n"
    "2025-10-20 00:00 UTC"
)

plt.tight_layout()

plt.savefig(
    "cyclone_crop_20251020_0000.png",
    dpi=150,
)

plt.close()

print(
    "\nSaved: cyclone_crop_20251020_0000.png"
)

print("\nCrop:")
print(f"shape = {crop.shape}")
print(f"row range = {row_start}:{row_end}")
print(f"column range = {col_start}:{col_end}")

print("\nCrop statistics:")
print(f"NaN count = {np.isnan(crop).sum()}")
print(f"finite pixels = {np.isfinite(crop).sum()}")

print(
    f"min = {np.nanmin(crop):.2f}"
)

print(
    f"max = {np.nanmax(crop):.2f}"
)

print(
    f"mean = {np.nanmean(crop):.2f}"
)

print("\nSaved: cyclone_center_check.png")

print("\nPixel coordinates:")
print(f"column = {column}")
print(f"row    = {row}")

print("\nPixel coordinate values:")
print(f"x[{column}] = {x_coords[column]:.2f} m")
print(f"y[{row}]    = {y_coords[row]:.2f} m")

print("\nImage bounds:")
print(f"rows    : 0 → {band.shape[0] - 1}")
print(f"columns : 0 → {band.shape[1] - 1}")

print("\nCyclone:")
print(f"Latitude : {latitude}")
print(f"Longitude: {longitude}")

print("\nProjection coordinates:")
print(f"x = {x:.2f} m")
print(f"y = {y:.2f} m")
