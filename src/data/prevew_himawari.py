from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ahi_reader import AHIReader

# Load all 10 segments for one timestamp
files = sorted(
    Path("interim/himawari_dat/2025/10/20")
    .glob("*20251020_0000_B13*.DAT")
)

reader = AHIReader()
result = reader.load_band13(files)

img = result["array"]

# Stretch only valid pixels for better contrast
vmin = np.nanpercentile(img, 1)
vmax = np.nanpercentile(img, 99)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap="gray_r", vmin=vmin, vmax=vmax)
plt.title("Himawari-8 Band 13 • 2025-10-20 00:00 UTC")
plt.axis("off")
plt.tight_layout()
plt.show()
