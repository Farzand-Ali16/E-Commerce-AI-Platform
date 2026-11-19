"""
Read-only environment check for the saved production artifacts.

Run from the project root, with the SAME python that runs the app / pytest:
    python scripts/diagnose_environment.py

It prints the ACTUAL installed versions, tries to load each artifact on its own (with the complete traceback
if one fails), checks the SHA-256 of the files, and prints the exact fix. It never writes or changes any file.
Exit code 0 = everything loads and predicts correctly, 1 = a problem was found.
"""
import hashlib
import json
import platform
import sys
import traceback
import warnings
from importlib import metadata
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))      # the saved preprocessor is a src.preprocessing... class

MODELS = ROOT / "models"
FILES = {"model": MODELS / "customer_segmentation_kmeans_core5.joblib",
         "preprocessor": MODELS / "cluster_preprocessor_core5.joblib",
         "metadata": MODELS / "segment_metadata_core5.json"}

# Versions the artifacts were created and verified with
VERIFIED = {"pandas": "3.0.2", "numpy": "2.4.4", "scikit-learn": "1.8.0", "joblib": "1.5.3"}
VERIFIED_SHA256 = {
    "customer_segmentation_kmeans_core5.joblib": "3a4aabf254db2a06ad418a9408fedaa385953bd0778c1b0068366828236cc528",
    "cluster_preprocessor_core5.joblib": "4560ca853fb95b6dc694cb3a4affd7812ec218c4c60381570d97dcc4d1fec7b4",
    "segment_metadata_core5.json": "be1af52622f5335bc8cf759dac752885fdbb2445a44cf3b488c02d2fba7b4afa",
}
FIX = "python -m pip install -r requirements.txt"


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def installed(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    problems = []

    section("1. INTERPRETER AND ACTUAL INSTALLED VERSIONS")
    print("python executable :", sys.executable)
    print("python version    :", platform.python_version())
    if sys.version_info < (3, 11):
        problems.append("Python 3.11 or newer is required.")
    print(f"\n{'package':<14}{'installed':<14}{'verified with':<16}status")
    for pkg, want in VERIFIED.items():
        have = installed(pkg)
        status = "OK" if have == want else ("MISSING" if have is None else "DIFFERENT")
        print(f"{pkg:<14}{str(have):<14}{want:<16}{status}")
        if have != want:
            problems.append(f"{pkg} is {have}, artifacts were verified with {want}.")

    section("2. ARTIFACT FILES (SHA-256)")
    for name, path in FILES.items():
        if not path.exists():
            print(f"{path.name:<45} MISSING")
            problems.append(f"{path.name} is missing.")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        same = digest == VERIFIED_SHA256[path.name]
        print(f"{path.name:<45} {digest[:16]}...  {'UNCHANGED' if same else 'DIFFERENT from the verified file'}")
        if not same:
            problems.append(f"{path.name} differs from the verified artifact (re-download it; do not regenerate it).")

    section("3. LOAD EACH ARTIFACT ON ITS OWN")
    loaded = {}
    import joblib
    for name, path in FILES.items():
        if not path.exists():
            continue
        try:
            loaded[name] = json.loads(path.read_text()) if name == "metadata" else joblib.load(path)
            print(f"{name:<13} OK")
        except Exception as exc:
            print(f"{name:<13} FAILED: {type(exc).__name__}: {exc}\n")
            traceback.print_exc()
            problems.append(f"{name} could not be loaded ({type(exc).__name__}).")

    section("4. PREDICTION CHECK (needs model + preprocessor)")
    if "model" in loaded and "preprocessor" in loaded:
        try:
            import pandas as pd
            cust = pd.read_csv(ROOT / "data" / "processed" / "customer_features.csv")
            seg = pd.read_csv(ROOT / "data" / "processed" / "customer_segments_core5.csv", usecols=["Cluster"])
            pred = loaded["model"].predict(loaded["preprocessor"].transform(cust).to_numpy())
            ok = int((pred == seg["Cluster"].to_numpy()).sum())
            print(f"{ok}/{len(seg)} customers get the same segment as the saved Stage 5.1 result")
            if ok != len(seg):
                problems.append("Predictions differ from the saved segments.")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
            problems.append("Prediction check failed.")
    else:
        print("skipped (model or preprocessor did not load)")

    section("RESULT")
    if not problems:
        print("ALL GOOD: both artifacts load and reproduce the saved segments.")
        return 0
    for p in problems:
        print(" -", p)
    pre_failed = any(p.startswith("preprocessor") for p in problems)
    pandas_ver = installed("pandas") or "?"
    if pre_failed and not pandas_ver.startswith("3."):
        print(f"\nCAUSE: pandas {pandas_ver} is installed, but the saved preprocessor was created with pandas 3.x "
              "(it contains a pandas 3 string dtype that pandas 2.x cannot read).")
    print(f"\nFIX (no model changes needed):  {sys.executable} -m pip install -r requirements.txt")
    print("Then run this script again. If versions still do not change, the app/pytest is using a different "
          "python than the one shown above (activate the right virtual environment first).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
