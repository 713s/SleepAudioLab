"""ICSD TRAIN/VALIDATION conversion; never opens ICSD's official TEST files.

All original/derived ICSD content is restricted: keep outputs local and private.
Strong event annotations only. Weak labels are deliberately not time-expanded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
LAYOUT = {
    "train_real": ("audio/train/real_strong_train", "metadata/train/real_strong_train.tsv"),
    "train_synth": ("audio/train/synth_strong_train", "metadata/train/synth_strong_train.tsv"),
    "validation": ("audio/validation/strong_val", "metadata/validation/strong_val.tsv"),
}
REQUIRED = {"filename", "onset", "offset", "event_label"}
CLASSES = {"Snoring", "Infantcry"}


def read_strong_tsv(path: Path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing official strong label fields in {path.name}")
        for row in reader:
            name = row["filename"].strip()
            if not name or name != Path(name).name:
                raise ValueError("Unsafe audio filename")
            tag = row["event_label"].strip()
            if tag not in CLASSES:
                raise ValueError(f"Unsupported class {tag!r}")
            start, end = float(row["onset"]), float(row["offset"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f"Invalid interval in {name}")
            grouped[name].append((start, end, tag))
    if not grouped:
        raise ValueError(f"No annotations in {path}")
    return grouped


def union_coverage(intervals, left: float, right: float) -> float:
    """Length of union of clipped intervals, avoiding double counting."""
    pieces = sorted(
        (max(left, a), min(right, b))
        for a, b in intervals if min(right, b) > max(left, a)
    )
    if not pieces:
        return 0.0
    total, a, b = 0.0, *pieces[0]
    for c, d in pieces[1:]:
        if c > b:
            total += b - a
            a, b = c, d
        else:
            b = max(b, d)
    return total + b - a


def _audio(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    x = values.mean(axis=1).astype("float64")
    if rate != SAMPLE_RATE:
        common = math.gcd(int(rate), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // common, int(rate) // common)
    return x


def convert_icsd_train_validation(root: str | Path, destination: str | Path, *,
                                  assume_exhaustive: bool = False,
                                  minimum_snore_seconds: float = 0.5,
                                  include_synthetic: bool = True):
    """Create disjoint local TRAIN and VALIDATION manifests, never reading TEST.

    Strong snoring overlap >= 0.5 s: positive if no Infantcry intersects.
    Otherwise unannotated windows become negative ONLY if assume_exhaustive.
    Weak-label recordings are excluded, not assigned fabricated event timestamps.
    """
    if not 0 < minimum_snore_seconds <= 1:
        raise ValueError("minimum_snore_seconds must be within (0, 1]")
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents:
        raise ValueError("Output must be outside the ICSD original dataset root")
    if (destination / "train_manifest.csv").exists() or (destination / "validation_manifest.csv").exists():
        raise FileExistsError("Existing frozen manifest: choose a new output directory")

    subsets = ("train_real", "train_synth", "validation") if include_synthetic else (
        "train_real", "validation")
    records = {"train": [], "validation": []}
    summary = {}
    source_hashes = {"train": set(), "validation": set()}
    for subset in subsets:
        relative_audio, relative_tsv = LAYOUT[subset]
        audio_folder, tsv = root / relative_audio, root / relative_tsv
        if not audio_folder.is_dir() or not tsv.is_file():
            raise FileNotFoundError(f"ICSD subset incomplete: {subset}: {audio_folder}, {tsv}")
        annotations = read_strong_tsv(tsv)
        label_split = "validation" if subset == "validation" else "train"
        counts = {"recordings": 0, "positive": 0, "negative": 0,
                  "ambiguous": 0, "short_tail_discarded": 0}
        for filename in sorted(annotations):
            original = audio_folder / filename
            if not original.is_file():
                raise FileNotFoundError(original)
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            source_hashes[label_split].add(digest)
            x = _audio(original)
            duration = x.size / SAMPLE_RATE
            if any(b > duration + 0.02 for _, b, _ in annotations[filename]):
                raise ValueError(f"Annotation beyond recording: {subset}/{filename}")
            counts["recordings"] += 1
            if x.size % SAMPLE_RATE:
                counts["short_tail_discarded"] += 1
            for index in range(x.size // SAMPLE_RATE):
                left, right = float(index), float(index + 1)
                snore = union_coverage(
                    [(a, b) for a, b, tag in annotations[filename] if tag == "Snoring"],
                    left, right)
                cry = union_coverage(
                    [(a, b) for a, b, tag in annotations[filename] if tag == "Infantcry"],
                    left, right)
                if cry > 0 or (0 < snore < minimum_snore_seconds):
                    counts["ambiguous"] += 1
                    continue
                if snore >= minimum_snore_seconds:
                    label = 1
                    counts["positive"] += 1
                elif assume_exhaustive and snore == 0:
                    label = 0
                    counts["negative"] += 1
                else:
                    counts["ambiguous"] += 1
                    continue
                clip_name = f"{subset}_{digest[:16]}_{index:05d}.wav"
                out = destination / "clips" / label_split / clip_name
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, x[index*SAMPLE_RATE:(index+1)*SAMPLE_RATE],
                         SAMPLE_RATE, subtype="PCM_16")
                records[label_split].append({
                    "path": str(out),
                    "label": label,
                    "source_id": f"{subset}:{digest}",
                    "source_split": label_split,
                    "subset": subset,
                })
        summary[subset] = counts

    if source_hashes["train"].intersection(source_hashes["validation"]):
        raise ValueError("Exact original-audio overlap across train and validation")
    for split in ("train", "validation"):
        csv_path = destination / f"{split}_manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "path", "label", "source_id", "source_split", "subset"])
            writer.writeheader()
            writer.writerows(records[split])
    audit = {
        "dataset": "ICSD", "included_subsets": list(subsets),
        "weak_label_policy": "excluded; never treated as time-localized",
        "official_test_policy": "never opened",
        "positive_overlap_seconds": minimum_snore_seconds,
        "assume_exhaustive": assume_exhaustive,
        "subsets": summary,
        "train_samples": len(records["train"]),
        "validation_samples": len(records["validation"]),
        "note": "Keep all ICSD-derived files private; check strong label exhaustiveness before creating negatives."
    }
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/processed/icsd/strong-v1"))
    parser.add_argument("--assume-exhaustive", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert_icsd_train_validation(
        args.root, args.out, assume_exhaustive=args.assume_exhaustive,
        include_synthetic=not args.no_synthetic), indent=2))


if __name__ == "__main__":
    main()
