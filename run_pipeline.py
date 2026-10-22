"""Run the data and training pipeline in order (Windows / macOS / Linux).

    python run_pipeline.py                    # download -> parse -> weather -> features -> validate -> train
    python run_pipeline.py --from features    # resume from a step

With DVC installed, `dvc repro` runs the same stages and skips unchanged ones.
"""
import argparse
import subprocess
import sys

STEPS = [
    ("download", [sys.executable, "-m", "src.dcs", "download"]),
    ("parse", [sys.executable, "-m", "src.dcs", "parse"]),
    ("weather", [sys.executable, "-m", "src.weather"]),
    ("features", [sys.executable, "-m", "src.features"]),
    ("validate", [sys.executable, "-m", "src.contract"]),
    ("train", [sys.executable, "-m", "src.train", "--n-iter", "30"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", default=None, choices=[n for n, _ in STEPS])
a = ap.parse_args()
names = [n for n, _ in STEPS]
steps = STEPS[names.index(a.start):] if a.start else STEPS
for name, cmd in steps:
    print(f"\n=== {name}: {' '.join(cmd[1:])} ===", flush=True)
    if subprocess.call(cmd) != 0:
        sys.exit(f"Step '{name}' failed - fix it, then rerun with --from {name}")
print("\nDone. Next: run the notebooks (01 -> 03), then `streamlit run app/app.py`.")
