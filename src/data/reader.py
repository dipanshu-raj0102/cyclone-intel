from dataclasses import dataclass
from pathlib import Path


@dataclass
class HimawariHeader:
    satellite: str
    center: str
    observation_type: str
    segment: int
    total_segments: int
    file_size: int


class HimawariReader:

    def parse_header(self, file_path: str):
        path = Path(file_path)

        with open(path, "rb") as f:
            header = f.read(64)

        name = path.stem

        # S0110 → segment 01 / total 10
        seg = name.split("_")[-1]
        current = int(seg[1:3])
        total = int(seg[3:5])

        return HimawariHeader(
            satellite="Himawari-8",
            center="MSC",
            observation_type="FLDK",
            segment=current,
            total_segments=total,
            file_size=path.stat().st_size,
        )
