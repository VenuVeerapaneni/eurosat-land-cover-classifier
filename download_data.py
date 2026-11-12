"""Download and verify the official EuroSAT RGB archive from Zenodo."""
from pathlib import Path
import hashlib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARCHIVE = DATA / "EuroSAT_RGB.zip"
URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"

def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading official EuroSAT RGB archive to {ARCHIVE} ...")
        request = urllib.request.Request(URL, headers={"User-Agent": "EuroSAT-course-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, ARCHIVE.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    actual = md5(ARCHIVE)
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"Archive checksum mismatch: expected {EXPECTED_MD5}, got {actual}")
    print("Archive checksum verified.")
    target = DATA / "EuroSAT_RGB"
    if not target.is_dir():
        with zipfile.ZipFile(ARCHIVE) as zf:
            zf.extractall(DATA)
    candidates = [target, DATA / "2750", DATA]
    for root in candidates:
        if all((root / name).is_dir() for name in ("AnnualCrop", "Forest", "River")):
            print(f"Dataset ready at {root}")
            return
    raise FileNotFoundError("Archive extracted, but the expected EuroSAT class folders were not found under data/.")

if __name__ == "__main__":
    main()
