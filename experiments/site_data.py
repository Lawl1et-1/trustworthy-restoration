"""Project page data: collects the result files into docs/data/results.json for docs/index.html.

  python -m experiments.site_data [--profile full]

Nothing on the page is typed in by hand: every number comes from results/<profile>/... through this script.
Datasets whose M3 tables are missing are written as {"status": "pending"} so the page says so.
M4 (generative restorers) rows are merged into the M3 comparison / uncertainty / shift tables; M5 (lesions),
M5b (real tumours), M6 (conformal) and M7 (shift study) are added when their result files exist.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASETS = ["phantom", "msd"]
VERDICT = {"lo": 0.87, "hi": 0.93}  # 90 % nominal: within 3 points holds, below is overconfident, above conservative


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _records(df):
    return json.loads(df.to_json(orient="records", double_precision=4))


def collect(profile):
    out = {"generated": dt.date.today().isoformat(), "git": _git(), "profile": profile, "verdict_band": VERDICT,
           "datasets": {}}
    for ds in DATASETS:
        m3 = ROOT / "results" / profile / "m3_nets" / ds
        m2 = ROOT / "results" / profile / "m2_classical" / ds
        entry = {}
        if (m2 / "summary.csv").exists():
            s2 = pd.read_csv(m2 / "summary.csv")
            entry["classical"] = _records(s2[["split", "task", "method", "psnr", "psnr_lo", "psnr_hi", "ssim",
                                              "n_patients"]])
        if (m3 / "comparison.csv").exists():
            entry["comparison"] = _records(pd.read_csv(m3 / "comparison.csv"))
            entry["uncertainty"] = _records(pd.read_csv(m3 / "uncertainty.csv"))
            if (m3 / "shift.csv").exists():
                entry["shift"] = _records(pd.read_csv(m3 / "shift.csv"))
            meta = json.loads((m3 / "meta.json").read_text()) if (m3 / "meta.json").exists() else {}
            summ = pd.read_csv(m3 / "summary.csv")
            entry["n_patients"] = {sp: int(g.n_patients.max()) for sp, g in summ.groupby("split")}
            entry["n_slices"] = meta.get("n_slices")
            entry["device"] = meta.get("device")
            if (m3 / "train_logs.json").exists():
                entry["n_networks"] = len(json.loads((m3 / "train_logs.json").read_text()))
            m4 = ROOT / "results" / profile / "m4_gen" / ds
            if (m4 / "comparison.csv").exists():  # M4 rows next to M3's; the M2 reference rows are already there
                c4 = pd.read_csv(m4 / "comparison.csv")
                entry["comparison"] += _records(c4[~c4.method.str.endswith("(M2)")])
                entry["uncertainty"] += _records(pd.read_csv(m4 / "uncertainty.csv"))
                if "shift" in entry and (m4 / "shift.csv").exists():
                    s4 = pd.read_csv(m4 / "shift.csv")
                    entry["shift"] += _records(s4[~s4.method.str.endswith("(M2)")])
                if (m4 / "train_logs.json").exists():
                    entry["n_networks"] = entry.get("n_networks", 0) + len(json.loads((m4 / "train_logs.json").read_text()))
                # diffusion with data consistency: from the M4 run itself when it has those rows, else from the
                # seed-0 DC evaluation (results/<profile>/m4_dc)
                m4dc = ROOT / "results" / profile / "m4_dc" / ds
                if not c4.method.str.startswith("diffusion_dc").any() and (m4dc / "comparison.csv").exists():
                    cd = pd.read_csv(m4dc / "comparison.csv")
                    entry["comparison"] += _records(cd[cd.method.str.startswith("diffusion_dc")])
                    ud = pd.read_csv(m4dc / "uncertainty.csv")
                    entry["uncertainty"] += _records(ud[ud.method.str.startswith("diffusion_dc")])
                    if "shift" in entry and (m4dc / "shift.csv").exists():
                        sd = pd.read_csv(m4dc / "shift.csv")
                        entry["shift"] += _records(sd[sd.method.str.startswith("diffusion_dc")])
            m5 = ROOT / "results" / profile / "m5_halluc" / ds / "summary.csv"
            if m5.exists():
                l5 = pd.read_csv(m5)
                entry["lesions"] = _records(l5[[c for c in ["task", "method", "n_lesions", "recovery", "erased_frac",
                                                             "detect_auc", "fab_absent", "core_cov90", "erased_flag_auroc"]
                                                if c in l5.columns]])
                m5meta = m5.parent / "meta.json"
                if m5meta.exists():
                    mm = json.loads(m5meta.read_text())
                    entry["n_lesions"], entry["n_lesion_images"] = mm["n_lesions"], mm["n_images"]
            m5t = ROOT / "results" / profile / "m5_tumour" / ds / "summary.csv"  # real tumours (MSD only)
            if m5t.exists():
                entry["tumours"] = _records(pd.read_csv(m5t)[["task", "method", "size", "n_slices", "n_patients",
                                                              "contrast_kept", "err_ratio", "cov_tumour", "cov_rest",
                                                              "spread_ratio"]])
                mt = json.loads((m5t.parent / "meta.json").read_text())
                entry["n_tumour_slices"], entry["n_tumour_patients"] = mt["n_slices"], mt["n_patients"]
            m6 = ROOT / "results" / profile / "m6_conformal" / ds / "summary.csv"
            if m6.exists():
                entry["conformal"] = _records(pd.read_csv(m6)[["split", "task", "method", "rule", "coverage", "width",
                                                               "frac_images_over_alpha"]])
            entry["status"] = "done"
        else:
            entry["status"] = "pending"
        demo = ROOT / "docs" / "demo" / ds / "manifest.json"
        if demo.exists():
            entry["demo"] = f"demo/{ds}/manifest.json"
        out["datasets"][ds] = entry
    m7 = ROOT / "results" / profile / "m7_shift"
    if (m7 / "region.csv").exists():
        out["m7"] = {"region": _records(pd.read_csv(m7 / "region.csv")),
                     "recal": _records(pd.read_csv(m7 / "recal.csv")),
                     "site": _records(pd.read_csv(m7 / "site.csv")),
                     "noise": json.loads((m7 / "noise.json").read_text())}
    if (ROOT / "docs" / "demo" / "enhance_example.png").exists():  # written by experiments/enhance_example.py
        out["enhance_example"] = {"src": "demo/enhance_example.png",
                                  "label": "Example: IXI002, Guys Hospital, a site the network never saw (256 × 256)",
                                  "label_fr": "Exemple : IXI002, Guys Hospital, un site que le réseau n'a jamais vu (256 × 256)"}
    # in-browser networks for "try your own image": the MSD (real MRI) set when exported, else the phantom set
    for ds in ("msd", "phantom"):
        if (ROOT / "docs" / "models" / ds / "models.json").exists():
            out["models"] = f"models/{ds}/models.json"
            break
    # example image for that demo: the first IXI gallery scan (a hospital the models never saw), else a phantom
    from PIL import Image
    for ds, split, label, label_fr in (
            ("msd", "ood", "Example: IXI test scan, {site}, a hospital the models never saw",
             "Exemple : examen de test IXI, {site}, un hôpital que les modèles n'ont jamais vu"),
            ("phantom", "test", "Example: a test phantom", "Exemple : un fantôme de test")):
        mf = ROOT / "docs" / "demo" / ds / "manifest.json"
        if mf.exists():
            th = json.loads(mf.read_text()).get("thumbs", {}).get(split)
            if th:
                im = Image.open(mf.parent / th["file"])
                t = im.size[1]
                im.crop((0, 0, t, t)).save(ROOT / "docs" / "demo" / "example.png")
                out["example"] = {"src": "demo/example.png", "label": label.format(site=th["site"][0]),
                                  "label_fr": label_fr.format(site=th["site"][0])}
                break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    a = ap.parse_args()
    data = collect(a.profile)
    p = ROOT / "docs" / "data" / "results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    print("wrote", p.relative_to(ROOT), {k: v["status"] for k, v in data["datasets"].items()})
