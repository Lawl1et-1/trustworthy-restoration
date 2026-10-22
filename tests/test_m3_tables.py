"""experiments.m3_tables: pairing with M2 and its consistency checks."""
import numpy as np
import pandas as pd

from experiments.m3_tables import check, comparison, per_image, reference_methods, shift


def _frames(shift_db=1.0, n=6):
    pids = [f"p{i // 2}" for i in range(n)]  # 2 slices per patient
    base = np.linspace(20, 25, n)
    m2 = []
    for split in ("test", "ood"):
        for meth, off in (("noisy", -5.0), ("tv", 0.0), ("bm3d", 0.5)):
            for i in range(n):
                m2.append({"split": split, "task": "denoise_s0.06", "method": meth, "seed": 0, "idx": i,
                           "pid": pids[i], "psnr": base[i] + off})
    m3 = []
    for split in ("test", "ood"):
        for ms in (0, 1):
            for i in range(n):
                m3.append({"split": split, "task": "denoise_s0.06", "method": "unet", "seed": 0, "model_seed": ms,
                           "idx": i, "pid": pids[i], "psnr": base[i] + 0.5 + shift_db + (0.1 if ms else -0.1)})
    m2, m3 = pd.DataFrame(m2), pd.DataFrame(m3)

    def summ(df):
        g = df.groupby(["split", "task", "method"], sort=False).psnr.mean().reset_index()
        return g.assign(psnr_lo=g.psnr - 1, psnr_hi=g.psnr + 1, ssim=0.5, n_patients=n // 2)
    return m2, m3, summ(m2), summ(m3)


def test_reference_is_best_classical_and_paired_difference():
    m2, m3, s2, s3 = _frames(shift_db=1.0)
    ref = reference_methods(s2)
    assert ref == {"denoise_s0.06": "bm3d"}
    i2, i3 = per_image(m2), per_image(m3)
    assert check(s3, i3, i2) == []
    c = comparison(s3, s2, i3, i2, ref)
    u = c[(c.split == "test") & (c.method == "unet")].iloc[0]
    assert abs(u.diff_vs_ref_db - 1.0) < 1e-9  # model seeds average out, so exactly +1 dB over bm3d
    assert abs(u.gain_db - 6.5) < 1e-9  # input is 5 dB below tv, unet is 1.5 dB above tv
    sh = shift(c)
    assert set(sh.method) == {"unet", "bm3d (M2)"}


def test_check_flags_different_test_sets():
    m2, m3, s2, s3 = _frames()
    m2.loc[m2.idx == 0, "pid"] = "other"
    assert any("different patient ids" in p for p in check(s3, per_image(m3), per_image(m2)))
