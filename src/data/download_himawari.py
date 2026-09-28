import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

YEAR, MONTH, DAY = "2025", "10", "21"

SAVE_DIR = Path(f"raw/himawari/{YEAR}/{MONTH}/{DAY}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

times = [f"{h:02d}{m:02d}" for h in range(24) for m in (0,10,20,30,40,50)]

def download_timestamp(t):
    remote = f"s3://noaa-himawari8/AHI-L1b-FLDK/{YEAR}/{MONTH}/{DAY}/{t}/"

    cmd = [
        "aws","s3","cp",
        remote,
        str(SAVE_DIR),
        "--recursive",
        "--no-sign-request",
        "--exclude","*",
        "--include","*B13*DAT.bz2",
        "--only-show-errors"
    ]

    subprocess.run(cmd, check=True)
    return t

with ThreadPoolExecutor(max_workers=8) as pool:
    for i, t in enumerate(pool.map(download_timestamp, times), 1):
        print(f"[{i:3}/144] {t} ✓")

print("Download completed.")
