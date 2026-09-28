from pathlib import Path

from reader import HimawariReader

file = sorted(
    Path("interim/himawari_dat/2025/10/20").glob("*S0110.DAT")
)[0]

header = HimawariReader().parse_header(file)
print(header)
