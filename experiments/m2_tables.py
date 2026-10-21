"""M2 derived table: each method's PSNR gain over its own input (noisy / zero-filled / masked image).

Raw PSNR on a shifted test set mixes method behaviour with image content (on IXI even the noisy input
scores higher than on MSD), so the shift comparison uses the gain over the input instead.

  python -m experiments.m2_tables      writes results/full/m2_classical/<dataset>/gain_over_input.csv
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT = {"denoise": "noisy", "mri": "zero_filled", "inpaint": "masked"}


def gain_table(summary):
    rows = []
    for (split, task), g in summary.groupby(["split", "task"], sort=False):
        base = g[g.method == INPUT[g["task"].iloc[0].split("_")[0]]]["psnr"].iloc[0]
        for _, r in g.iterrows():
            if r.method != INPUT[task.split("_")[0]]:
                rows.append({"split": split, "task": task, "method": r.method, "input_psnr": base,
                             "psnr": r.psnr, "gain_db": r.psnr - base})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for p in sorted((ROOT / "results" / "full" / "m2_classical").glob("*/summary.csv")):
        out = p.parent / "gain_over_input.csv"
        gain_table(pd.read_csv(p)).to_csv(out, index=False, float_format="%.4f")
        print("wrote", out.relative_to(ROOT))
