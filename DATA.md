# Data

All data used here is public and open. Nothing that needs registration (BraTS, fastMRI, Mayo) is
downloaded or bypassed; see "Restricted datasets" for where to add them.

Raw archives go in `data/raw/`, processed slice sets in `data/processed/`. Both are gitignored.

## Synthetic phantoms (built in, no download)

`src/phantoms/phantom.py`: modified Shepp-Logan geometry (Toft 1996), randomised, with smooth texture and
0 to 4 bright lesions of known position, radius (1.5 to 4 px) and contrast (0.1 to 0.3). Each image is its
own "patient". This is the only dataset with exact ground truth for erased and fabricated lesions.

## Medical Segmentation Decathlon, Task01 BrainTumour (primary)

| | |
|---|---|
| Source | AWS Open Data registry, <https://registry.opendata.aws/msd/> |
| File | `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar` |
| Size | 7,608,266,240 bytes (HTTP `Content-Length`, checked 2026-10-21; `Last-Modified` 2020-08-14) |
| Licence | CC BY-SA 4.0 (AWS registry entry; Simpson et al. 2019, arXiv:1902.09063; `dataset.json` in the archive says `CC-BY-SA 4.0`, release 2.0 04/05/2018) |
| Content | 484 labelled training volumes, 4 MRI channels (FLAIR, T1w, T1gd, T2w), tumour labels; derived from BraTS 2016/2017 |
| Used | T2w channel (`dataset.json` modality 3), 8 axial slices per volume, 128x128, per-volume 99.8th-percentile scaling |
| Download date | 2026-10-21, to `data/raw/msd/` |
| Integrity | resumed download; the S3 multipart ETag `dfd1b5a5df62dadb48cad3d347148c70-907` (MD5 of 907 MD5s of 8 MiB parts) was recomputed locally and matches |
| Contents | 484 `imagesTr` with 484 `labelsTr`, 266 unlabelled `imagesTs` (not used) |

Citation: Antonelli M. et al., "The Medical Segmentation Decathlon", *Nature Communications* 13, 4128
(2022), doi:10.1038/s41467-022-30695-9 (verified 2026-10-21).

Caveat: the images come from BraTS. MSD redistributes them under CC BY-SA 4.0; check the original BraTS
terms before any commercial use.

## IXI T2 (out-of-distribution test set only)

| | |
|---|---|
| Source | <https://brain-development.org/ixi-dataset/> |
| File | `https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar` |
| Size | 3,853,445,120 bytes (HTTP `Content-Length`, checked 2026-10-21; `Last-Modified` 2014-10-07) |
| Licence | CC BY-SA 3.0, stated on the IXI page (read 2026-10-21); acknowledge the IXI website |
| Sites | Guy's Hospital (Guys), Hammersmith Hospital (HH), Institute of Psychiatry (IOP); per-site field strength is reported inconsistently by secondary sources, `TODO[verify]` against the IXI demographics sheet |
| Used | `--ixi-per-site 40` subjects per site (first 40 by file name), 8 axial slices each, never trained on or tuned on |
| Download date | 2026-10-21, to `data/raw/ixi/` |
| Integrity | resumed download (the server dropped connections often); no checksum is published, so: exact byte size, the full tar index reads cleanly, and gzip CRCs are checked on every file used |
| Contents | 578 T2 volumes: Guys 319, HH 185, IOP 74 |

IXI images are healthy subjects and are not skull-stripped, so the evaluation region is a head mask
(threshold at half Otsu, closing, hole filling and largest component, slice by slice;
`experiments/prepare_data.py`). Only brain-level slices are used: the lowest 45 % of the head's axial
extent (jaw, neck, skull base) is skipped, then the same >= 50 %-of-max-area rule as MSD applies.
Label masks are all zero. MSD is skull-stripped and IXI is not, so the IXI shift includes the skull and scalp
as well as site/scanner; keep that in mind when reading OOD results. Anatomy coverage also differs a
little: the lowest MSD slices include the cerebellum, while IXI slices start around the midbrain.

Prepared sets (2026-10-21): `msd_t2_128.npz` has 3,872 slices from 484 patients (69 % of slices contain
tumour); `ixi_t2_128.npz` has 960 slices from 120 subjects (40 per site).

## Preparing the slice datasets

```bash
python -m experiments.pipeline data
```

This reads the two `.tar` archives in place (no extraction) and writes `data/processed/msd_t2_128.npz`
and `data/processed/ixi_t2_128.npz`.

## Splits

`src/data/splits.py` splits by **patient**: 60 % train, 10 % val, 15 % cal (conformal calibration),
15 % test, from split seed 0. Each split is saved to `results/splits/<dataset>.json` with a hash per
split. Every experiment reloads that file, and loading fails if it was edited.
`tests/test_splits.py` checks that no patient is in two splits and every slice of a patient stays together.
No intensity normalisation is fitted across patients.

## Restricted datasets (not used; add them here if you obtain access)

- **fastMRI** (real single- and multi-coil k-space): put it in `data/raw/fastmri/` and tell me the path.
  All MRI results in this repo use **simulated** single-coil Cartesian k-space from magnitude images.
- **BraTS** (official challenge release) and **Mayo low-dose CT** need registration and are not used.
- **LoDoPaB-CT** (Zenodo, open) is a "could" extension for D4 and is not downloaded yet.
