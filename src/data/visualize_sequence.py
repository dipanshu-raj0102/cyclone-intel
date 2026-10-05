from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(
    "processed/himawari_ibtracs/2025-10-20/2025289N13140"
)

timestamps = [
    "0000",
    "0300",
    "0600",
    "0900",
    "1200",
    "1500",
    "1800",
    "2100",
]


fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8),
)


for ax, timestamp in zip(
    axes.flat,
    timestamps,
):

    file = (
        DATA_DIR
        / f"2025289N13140_2025-10-20T{timestamp}.npy"
    )

    crop = np.load(file)

    vmin = np.nanpercentile(crop, 1)
    vmax = np.nanpercentile(crop, 99)

    ax.imshow(
        crop,
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.scatter(
        128,
        128,
        marker="x",
        s=80,
        linewidths=2,
    )

    ax.set_title(
        f"2025-10-20 {timestamp[:2]}:{timestamp[2:]} UTC"
    )

    ax.set_xticks([])
    ax.set_yticks([])


fig.suptitle(
    "Himawari-8 B13 — Cyclone 2025289N13140",
    fontsize=16,
)

plt.tight_layout()

output = DATA_DIR / "temporal_sequence.png"

plt.savefig(
    output,
    dpi=150,
)

plt.close()

print(f"Saved: {output}")
