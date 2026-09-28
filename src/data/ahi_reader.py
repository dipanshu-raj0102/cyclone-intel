import numpy as np
from satpy import Scene


class AHIReader:
    def load_band13(self, files):
        scn = Scene(
            filenames=[str(f) for f in files],
            reader="ahi_hsd"
        )

        scn.load(["B13"])

        band = scn["B13"]

        return {
            "array": band.values.astype(np.float32),
            "attrs": band.attrs,
            "x": band.coords["x"].values,
            "y": band.coords["y"].values,
        }
