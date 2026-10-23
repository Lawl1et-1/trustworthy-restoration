# Trustworthy Restoration

Can learned medical image restoration be trusted? This project restores damaged brain MRI (noise,
accelerated acquisition, missing regions) with classical methods, discriminative networks and generative
models, and measures three things for each method:

1. **Accuracy**: how close the restoration is to the truth, paired against the strongest classical method.
2. **Hallucination**: whether it erases real lesions or invents new ones (exact ground truth on phantoms).
3. **Honest uncertainty**: whether a stated 90 % interval covers 90 % of pixels, before and after
   conformal calibration, at the training hospital and at three others.

Project page with interactive results: `docs/index.html` (GitHub Pages from `/docs`).

## Main findings

- **Accuracy is not trust.** On phantoms, the networks with the best PSNR erase many small lesions
  (MRI at R = 2: the U-Net erases 73 % of lesions, TV 2 %, the unrolled network that keeps the measured
  k-space 4 %), and their uncertainty is *more* confident on the lesions they erase.
- **Generative models are no free lunch.** Diffusion is the strongest learned denoiser on phantoms and,
  for denoising, degrades least at new hospitals, but a single diffusion sample invents lesion-like structure, and
  without data consistency both diffusion and the GAN lose to classical TV in accelerated MRI.
- **Calibration fixes stated uncertainty at home only.** Conformal risk control brings every method to
  89-91 % coverage on new MSD patients (MC-dropout starts at 20-28 %), but the same calibration covers
  only 69-90 % of pixels at the IXI hospitals. A constant-width band around a good network is about as
  narrow as the per-pixel uncertainty methods.
- **The shift is content, not noise, and the uncertainty is cheap to repair.** Adding IXI's acquisition
  noise to MSD changes almost nothing; on IXI the denoising networks lose most of their gain in the skull
  and scalp (skull-stripped training data), and in MRI every method loses on the finer anatomy. Recalibrating
  on 3-5 patients of a new hospital brings coverage back to 92-94 %.

Every number here and on the page comes from the result files in `results/` (summary tables, per-image
results and run metadata for each milestone); the commit messages describe each step.

## Data

All public and open: synthetic phantoms (built in), MSD Task01 BrainTumour T2 (CC BY-SA 4.0, 484
patients, split by patient 60/10/15/15 into train/val/calibration/test) and IXI T2 (CC BY-SA 3.0, 120
healthy subjects from three hospitals, used only as an untouched out-of-distribution test set).
Sources, checksums and preparation: [DATA.md](DATA.md).

## Methods

| Group | Methods |
|---|---|
| Classical (M2) | TV, wavelet-L1, BM3D, biharmonic inpainting; weights tuned on validation patients |
| Discriminative (M3) | U-Net, MC-dropout, heteroscedastic (Stirn et al. 2023), deep ensemble, quantile, unrolled (MRI) |
| Generative (M4) | conditional diffusion (SR3-style, DDIM), pix2pix GAN |
| Analyses | paired lesion test (M5), conformal risk control and RCPS (M6), shift study (M7) |

One network per degradation type covers all its severities. Test, calibration and IXI data are never used
for training or selection; confidence intervals resample patients.

## Reproduce

Windows or Linux, Python 3.12, one GPU (a 6 GB laptop GPU was used; the full profile trains 96 networks
in about a day).

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on Linux
python -m experiments.pipeline test        # unit tests
python -m experiments.pipeline data        # download and prepare MSD + IXI (see DATA.md)
python -m experiments.pipeline reproduce   # M2 classical + M3 networks + tables
python -m experiments.m3_nets --config configs/m4_full_phantom.yaml   # M4 (also m4_full_msd.yaml)
python -m experiments.m3_tables --stage m4_gen
python -m experiments.m5_hallucination --config configs/m5_full_phantom.yaml
python -m experiments.m6_conformal --config configs/m6_full_msd.yaml  # also m6_full_phantom.yaml
python -m experiments.m7_shift --config configs/m7_full.yaml
python -m experiments.site_data            # rebuilds docs/data/results.json
```

Each experiment has a `*_smoke.yaml` config that runs on a CPU in under a minute.

## Layout

```
src/            phantoms, degradations, models (classical, networks, diffusion), metrics, conformal
experiments/    one script per milestone; pipeline.py runs them in order
configs/        smoke, pilot and full settings (pilots score on validation patients only)
results/        summary tables, per-image results, metadata (git hash, device, timing) per run
docs/           project page, its data, demo images and in-browser ONNX models
tests/          unit tests (python -m pytest)
```

## Limits

MRI k-space is simulated single-coil from magnitude images. The lesion test uses synthetic phantoms.
One diffusion model and three GANs per degradation type were trained, without a data-consistency step.
Conformal guarantees are marginal (on average over pixels and patients) and assume exchangeable data.
This is research code with no clinical or diagnostic value.

## Licence of the data

MSD: CC BY-SA 4.0 (Antonelli et al. 2022). IXI: CC BY-SA 3.0 (brain-development.org).
