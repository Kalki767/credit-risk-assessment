"""Assemble the Hugging Face Space in build/space: app + the SAME src modules + model + contract (no data).
  python scripts/build_space.py                      # build only (test locally: cd build/space && python app.py)
  python scripts/build_space.py --deploy USER/NAME   # build and upload to the Space"""
import argparse
import shutil
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"
SRC_FILES = ["cleaning.py", "features.py", "model.py", "data.py", "scoring.py", "validation.py", "predict.py"]
ARTIFACTS = ["models/scorecard.joblib", "models/metadata.json", "contract/transform.joblib", "contract/data_spec.json",
             "examples/low_risk.json", "examples/borderline.json", "examples/high_risk.json"]
PACKAGES = ["gradio", "pandas", "numpy", "scipy", "scikit-learn", "factor_analyzer", "joblib", "pyyaml"]
DATA_LIKE = (".csv", ".parquet", "reference_transforms", "data/")


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    init = ROOT / "src" / "__init__.py"
    shutil.copy2(init, OUT / "src" / "__init__.py") if init.exists() else (OUT / "src" / "__init__.py").touch()
    for f in SRC_FILES:
        shutil.copy2(ROOT / "src" / f, OUT / "src" / f)
    for a in ARTIFACTS:
        (OUT / a).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / a, OUT / a)
    shutil.copy2(ROOT / "space" / "app.py", OUT / "app.py")
    readme = (ROOT / "space" / "README.md").read_text().replace("{gradio_version}", version("gradio"))
    (OUT / "README.md").write_text(readme)
    (OUT / "requirements.txt").write_text("\n".join(f"{p}=={version(p)}" for p in PACKAGES) + "\n")
    files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    bad = [f for f in files if any(s in f for s in DATA_LIKE)]
    if bad:
        raise SystemExit(f"refusing: data-like files in the Space: {bad}")
    print("built", OUT, "\n  " + "\n  ".join(files))


def deploy(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space")
    print(f"uploaded: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", metavar="USER/NAME")
    args = ap.parse_args()
    build()
    if args.deploy:
        deploy(args.deploy)