import bz2
import shutil
from pathlib import Path


class HimawariDecompressor:
    def __init__(
        self,
        raw_dir="raw/himawari",
        output_dir="interim/himawari_dat",
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

    def decompress_file(self, file_path: Path):
        relative = file_path.relative_to(self.raw_dir)
        output_path = (self.output_dir / relative).with_suffix("")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            return output_path

        with bz2.open(file_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        return output_path

    def run(self):
        files = sorted(self.raw_dir.rglob("*.DAT.bz2"))

        print(f"Found {len(files)} compressed files")

        for i, file in enumerate(files, 1):
            self.decompress_file(file)

            if i % 50 == 0 or i == len(files):
                print(f"{i}/{len(files)} completed")


if __name__ == "__main__":
    HimawariDecompressor().run()
