"""M3 derived tables: learned restorers against the M2 classical baselines on identical test degradations.

  python -m experiments.m3_tables                  full profile, every dataset with both M2 and M3 results
  python -m experiments.m3_tables --profile smoke
  python -m experiments.m3_tables --stage m4_gen  the same tables for the M4 generative restorers

M2 and M3 draw their test degradations from the same streams (task_generator(task, split, dseed)), so each
test image is degraded identically for every method and the comparison can be paired per patient.

Writes, per dataset, into results/<profile>/m3_nets/<dataset>/:
  comparison.csv   per (split, task, method): PSNR and SSIM with 95 % patient CIs, gain over the input,
                   and the paired PSNR difference to the reference classical method (patient bootstrap)
  uncertainty.csv  per (split, task, method with a spread): 90 % coverage, mean spread, spread/|error| Spearman
  shift.csv        (only with an OOD split) gain over the input in distribution vs OOD, per method
and prints the same tables as markdown for LOG.md.

Reference classical method per task: the classical method with the highest in-distribution test PSNR in M2
(BM3D for denoising, TV for MRI, biharmonic for inpainting in the M2 runs). It is chosen on the M2 test
summary, i.e. it is a fixed, strongest-available comparison target, not a tuned quantity. The same
reference is used on the OOD split.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from src.eval.stats import paired_bootstrap

ROOT = pathlib.Path(__file__).resolve().parents[1]

INPUT = {"denoise": "noisy", "mri": "zero_filled", "inpaint": "masked"}
ORDER = ["unet", "mcdropout", "hetero", "ensemble", "quantile", "unrolled", "diffusion", "diffusion_sample", "gan"]


def _ttype(task):
    return task.split("_")[0]


def per_image(df, metric="psnr"):
    """One value per (split, task, method, idx): mean over degradation seeds (and model seeds for M3)."""
    g = df.groupby(["split", "task", "method", "idx"], sort=False)
    return g.agg(**{metric: (metric, "mean"), "pid": ("pid", "first")}).reset_index()


def reference_methods(m2_summary):
    ref = {}
    test = m2_summary[m2_summary.split == "test"]
    for task, g in test.groupby("task", sort=False):
        g = g[g.method != INPUT[_ttype(task)]]
        ref[task] = g.sort_values("psnr", ascending=False).method.iloc[0]
    return ref


def check(m3_summary, m3_img, m2_img, cfg_methods=None):
    """Consistency checks; returns a list of problems (empty = fine)."""
    problems = []
    if m3_summary["psnr"].isna().any():
        problems.append("NaN PSNR rows in M3 summary")
    for split, g in m3_summary.groupby("split"):
        if g["n_patients"].nunique() != 1:
            problems.append(f"{split}: methods evaluated on different patient counts {sorted(g.n_patients.unique())}")
    key = ["split", "task", "idx"]
    a = m3_img.drop_duplicates(key)[key + ["pid"]]
    b = m2_img.drop_duplicates(key)[key + ["pid"]]
    j = a.merge(b, on=key, suffixes=("_m3", "_m2"))
    if len(j) == 0:
        problems.append("no (split, task, idx) overlap between M2 and M3")
    elif (j.pid_m3 != j.pid_m2).any():
        problems.append(f"{int((j.pid_m3 != j.pid_m2).sum())} images have different patient ids in M2 and M3 "
                        "(different test sets): paired comparison invalid")
    return problems


def comparison(m3_summary, m2_summary, m3_img, m2_img, ref):
    rows = []
    for (split, task), g in m3_summary.groupby(["split", "task"], sort=False):
        c2 = m2_summary[(m2_summary.split == split) & (m2_summary.task == task)]
        inp = c2[c2.method == INPUT[_ttype(task)]]
        inp_psnr = float(inp.psnr.iloc[0]) if len(inp) else np.nan
        r = ref.get(task)
        rimg = m2_img[(m2_img.split == split) & (m2_img.task == task) & (m2_img.method == r)].set_index("idx")
        rsum = c2[c2.method == r]
        if len(rsum):
            rr = rsum.iloc[0]
            rows.append({"split": split, "task": task, "method": f"{r} (M2)", "psnr": rr.psnr, "psnr_lo": rr.psnr_lo,
                         "psnr_hi": rr.psnr_hi, "ssim": rr.ssim, "gain_db": rr.psnr - inp_psnr,
                         "input_psnr": inp_psnr, "reference": r})
        for m in [x for x in ORDER if x in set(g.method)]:
            s = g[g.method == m].iloc[0]
            row = {"split": split, "task": task, "method": m, "psnr": s.psnr, "psnr_lo": s.psnr_lo,
                   "psnr_hi": s.psnr_hi, "psnr_sd_over_model_seeds": s.get("psnr_sd_over_model_seeds", np.nan),
                   "ssim": s.ssim, "gain_db": s.psnr - inp_psnr, "input_psnr": inp_psnr, "reference": r}
            mimg = m3_img[(m3_img.split == split) & (m3_img.task == task) & (m3_img.method == m)].set_index("idx")
            common = mimg.index.intersection(rimg.index)
            if len(common):
                d, lo, hi, p = paired_bootstrap(mimg.loc[common, "psnr"].values, rimg.loc[common, "psnr"].values,
                                                mimg.loc[common, "pid"].values)
                row.update({"diff_vs_ref_db": d, "diff_lo": lo, "diff_hi": hi, "diff_p": p,
                            "n_paired_images": len(common)})
            rows.append(row)
    return pd.DataFrame(rows)


def uncertainty(m3_summary):
    cols = [c for c in ("cov90", "mean_spread", "spread_err_spearman") if c in m3_summary.columns]
    u = m3_summary[m3_summary[cols[0]].notna()] if cols else m3_summary.iloc[:0]
    return u[["split", "task", "method"] + cols].reset_index(drop=True)


def shift(comp, ood_split="ood"):
    if ood_split not in set(comp.split):
        return pd.DataFrame()
    a = comp[comp.split == "test"][["task", "method", "gain_db"]]
    b = comp[comp.split == ood_split][["task", "method", "gain_db"]]
    j = a.merge(b, on=["task", "method"], suffixes=("_test", "_ood"))
    j["gain_change_db"] = j.gain_db_ood - j.gain_db_test
    return j


def _md(df, cols, fmt="{:.2f}"):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else
                         ("" if isinstance(v, float) and np.isnan(v) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def markdown(comp, unc, sh):
    c = comp.copy()
    c["PSNR [95% CI]"] = [f"{p:.2f} [{lo:.2f}, {hi:.2f}]" for p, lo, hi in zip(c.psnr, c.psnr_lo, c.psnr_hi)]
    c["vs ref (paired)"] = [f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]" if not np.isnan(d) else ""
                            for d, lo, hi in zip(c.get("diff_vs_ref_db", np.nan), c.get("diff_lo", np.nan),
                                                 c.get("diff_hi", np.nan))]
    out = []
    for split, g in c.groupby("split", sort=False):
        out.append(f"PSNR ({split}), dB; 'vs ref' = paired per-patient difference to the M2 reference method\n")
        out.append(_md(g, ["task", "method", "PSNR [95% CI]", "gain_db", "vs ref (paired)", "ssim"]))
    if len(unc):
        out.append("\nUncertainty (raw, before any conformal step; nominal 90 %)\n")
        out.append(_md(unc, list(unc.columns), "{:.3f}"))
    if len(sh):
        out.append("\nGain over input, in distribution vs OOD (dB)\n")
        out.append(_md(sh, ["task", "method", "gain_db_test", "gain_db_ood", "gain_change_db"]))
    return "\n".join(out)


def run(profile, stage="m3_nets"):
    done = []
    for m3dir in sorted((ROOT / "results" / profile / stage).glob("*")):
        ds = m3dir.name
        m2dir = ROOT / "results" / profile / "m2_classical" / ds
        need = [m3dir / "summary.csv", m3dir / "per_image.csv.gz", m2dir / "summary.csv", m2dir / "per_image.csv.gz"]
        if not all(p.exists() for p in need):
            print(f"[{ds}] skipped: missing {[str(p.relative_to(ROOT)) for p in need if not p.exists()]}")
            continue
        m3s, m2s = pd.read_csv(need[0]), pd.read_csv(need[2])
        m3i, m2i = per_image(pd.read_csv(need[1])), per_image(pd.read_csv(need[3]))
        problems = check(m3s, m3i, m2i)
        for p in problems:
            print(f"[{ds}] CHECK FAILED: {p}")
        ref = reference_methods(m2s)
        comp = comparison(m3s, m2s, m3i, m2i, ref)
        unc, sh = uncertainty(m3s), shift(comp)
        comp.to_csv(m3dir / "comparison.csv", index=False, float_format="%.4f")
        unc.to_csv(m3dir / "uncertainty.csv", index=False, float_format="%.4f")
        if len(sh):
            sh.to_csv(m3dir / "shift.csv", index=False, float_format="%.4f")
        print(f"\n## {ds} ({profile})  reference methods: {ref}\n")
        print(markdown(comp, unc, sh))
        done.append((ds, problems))
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--stage", default="m3_nets", help="m3_nets or m4_gen: which learned restorers to compare")
    a = ap.parse_args()
    res = run(a.profile, a.stage)
    if any(p for _, p in res):
        sys.exit(1)
