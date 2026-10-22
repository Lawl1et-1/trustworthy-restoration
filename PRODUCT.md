# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS served by GitHub Pages from `docs/` (user's choice). No build step; data is a JSON file
generated from `results/` by a script in the repo, demo images are PNGs exported from the trained checkpoints.

## Users

Supervisors and admissions readers evaluating Rayan (GitHub `Lawl1et-1`) for graduate study. They skim a project
page in a few minutes on a laptop, want to judge research judgment and rigour, and may open the repo to check.

## Product Purpose

`trustworthy-restoration` asks whether learned medical image restoration (denoising, accelerated MRI,
inpainting) can be trusted: how accurate it is against strong classical baselines, and whether each method's
stated uncertainty holds (does a nominal 90 % interval cover 90 % of pixels), in distribution and under a
scanner/site shift. The site shows the results and lets a reader inspect real test outputs and uncertainty maps.

## Positioning

Every comparison is paired on identical test degradations, every CI is a patient-level bootstrap, test data is
never used for training or selection, and negative findings (overconfident MC-dropout, the classical MRI gain
collapsing under shift) are reported as found.

## Operating Context

Milestones M0-M3 done (phantoms, operators, patient splits, classical baselines on phantoms + MSD Task01 + IXI,
learned restorers). Conformal calibration (M6), hallucination analysis on phantom lesions, and shift experiments
(M7) are planned, not done. MRI k-space is simulated single-coil from magnitude images; no fastMRI.

## Capabilities and Constraints

- Datasets: synthetic phantoms (exact lesion ground truth), MSD Task01 BrainTumour T2 (CC BY-SA 4.0), IXI T2
  (CC BY-SA 3.0, test-only OOD). Licences and citations must be shown.
- Methods: classical TV / wavelet / BM3D / biharmonic; learned U-Net, MC-dropout, heteroscedastic (faithful
  loss), deep ensemble, quantile, unrolled (MRI).
- Numbers on the site come only from `results/` files; nothing is typed in by hand.

## Evidence on Hand

`results/full/m2_classical/*`, `results/full/m3_nets/*` (summary, per-image, comparison, uncertainty, shift),
checkpoints in `checkpoints/full/` (gitignored), LOG.md (local). No publications, no external validation, no
clinical claims. Must not claim clinical use or diagnostic value.

## Product Principles

1. Report what was measured, with its uncertainty and its caveats.
2. A negative result is a result.
3. Every number traceable to a file in the repo.
4. Clear about what is planned versus done.
