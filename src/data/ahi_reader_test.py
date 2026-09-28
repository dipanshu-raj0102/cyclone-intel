from pathlib import Path

import numpy as np
from ahi_reader import AHIReader

files = sorted(
    Path("interim/himawari_dat/2025/10/20")
    .glob("*20251020_0000_B13*.DAT")
)

result = AHIReader().load_band13(files)
arr = result["array"]

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)

print("NaN count:", np.isnan(arr).sum())
print("Finite count:", np.isfinite(arr).sum())

finite = arr[np.isfinite(arr)]

print("Min:", finite.min())
print("Max:", finite.max())
print("Mean:", finite.mean())
