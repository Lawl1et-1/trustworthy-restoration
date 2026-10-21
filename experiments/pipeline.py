"""Cross-platform entry point behind the Makefile targets (Windows machines often have no `make`).

  python -m experiments.pipeline test        unit tests
  python -m experiments.pipeline smoke       tests + every milestone at smoke scale (CPU, < 10 min)
  python -m experiments.pipeline reproduce   full profile: every table and figure in results/ and figures/
  python -m experiments.pipeline data        MSD / IXI slice datasets from the raw archives (see DATA.md)

Each stage is appended here as its milestone lands, so `smoke` and `reproduce` always cover all of them.
"""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(*args):
    print(">>", " ".join(args), flush=True)
    t = time.time()
    subprocess.run([PY, *args], cwd=ROOT, check=True)
    print(f"<< {time.time() - t:.0f} s", flush=True)


STAGES = {
    "smoke": [
        ("-m", "pytest", "-q", "tests"),
        ("-m", "experiments.run_task", "--config", "configs/smoke.yaml"),
    ],
    "reproduce": [
        ("-m", "pytest", "-q", "tests"),
        ("-m", "experiments.run_task", "--config", "configs/full_phantom.yaml"),
        ("-m", "experiments.run_task", "--config", "configs/full_msd.yaml"),
        ("-m", "experiments.m2_tables"),
    ],
    "test": [("-m", "pytest", "-q", "tests")],
    "data": [("-m", "experiments.prepare_data", "--msd", "data/raw/msd/Task01_BrainTumour.tar",
              "--ixi", "data/raw/ixi/IXI-T2.tar")],
}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    t0 = time.time()
    for st in STAGES[target]:
        _run(*st)
    print(f"{target}: OK in {time.time() - t0:.0f} s")
