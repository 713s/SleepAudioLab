"""Build a local manifest from separately downloaded Khan Snoring Dataset.

Expected layout: dataset_root/0/*.wav and dataset_root/1/*.wav.
The dataset may have reuse restrictions: do not push its audio to this repo.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/processed/khan_manifest.csv"))
    p.add_argument("--source-map", type=Path, default=None,
                   help="CSV with filename,source_id (original recording group)")
    args = p.parse_args()
    root = args.dataset_root.resolve()
    mapping = {}
    if args.source_map:
        with args.source_map.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mapping[row["filename"]] = row["source_id"]
    rows = []
    for label in ("0", "1"):
        folder = root / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected {folder}")
        for audio in sorted(folder.rglob("*")):
            if audio.suffix.lower() in {".wav", ".flac", ".ogg"}:
                rows.append({"path": str(audio.resolve()), "label": label,
                             "source_id": mapping.get(audio.name, "")})
    if not rows:
        raise ValueError("No audio files in 0/ and 1/")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest: {args.out}; {len(rows)} files; "
          f"{sum(bool(r['source_id']) for r in rows)} files with known source groups")


if __name__ == "__main__":
    main()
