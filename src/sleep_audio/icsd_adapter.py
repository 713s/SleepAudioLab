"""Adapt authorized ICSD real strong TEST annotations to 1-second binary clips.

ICSD access terms prohibit redistribution. Outputs are PRIVATE and ignored by Git.
Train/validation/weak/synthetic data are never read or imported by this entrypoint.
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
WINDOW_SECONDS = 1.0
REQUIRED_METADATA = Path("metadata/test/real_strong_test.tsv")
REQUIRED_AUDIO = Path("audio/test/real_strong_test")
FIELDS = ("filename", "onset", "offset", "event_label")
LABELS = {"Snoring", "Infantcry"}


def _within(child: Path, parent: Path) -> bool:
    return child.resolve().is_relative_to(parent.resolve())


def _read_annotations(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    records = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Expected official ICSD strong TSV columns: " + ", ".join(FIELDS))
        for row in reader:
            filename = row["filename"].strip()
            if not filename or Path(filename).name != filename:
                raise ValueError("Unsafe or empty audio filename in TSV")
            event = row["event_label"].strip()
            if event not in LABELS:
                raise ValueError(f"Unknown event label {event!r}; refusing to infer a negative")
            onset, offset = float(row["onset"]), float(row["offset"])
            if not (math.isfinite(onset) and math.isfinite(offset) and 0 <= onset < offset):
                raise ValueError(f"Invalid event interval for {filename}")
            records[filename].append((onset, offset, event))
    if not records:
        raise ValueError("No real strong annotations found")
    return dict(records)


def _read_resample(path: Path) -> np.ndarray:
    data, original_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(np.float64)
    if original_rate != SAMPLE_RATE:
        common = math.gcd(int(original_rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // common, int(original_rate) // common)
    return mono


def convert_icsd_test(dataset_root: str | Path, out_dir: str | Path, *,
                      min_positive_overlap: float = .50,
                      assume_exhaustive: bool = False):
    """Convert only the official real_strong_test split to local 1-second WAVs.

    Snoring overlap >= threshold => positive, unless Infantcry also overlaps.
    All remaining overlaps (including partial snoring) => exclude as ambiguous.
    Pure background windows are labeled negative ONLY with --assume-exhaustive.
    """
    if not (0 < min_positive_overlap <= 1):
        raise ValueError("Overlap threshold must be in (0,1]")
    root = Path(dataset_root).resolve()
    metadata = root / REQUIRED_METADATA
    audio_dir = root / REQUIRED_AUDIO
    if not metadata.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError("Expected official test split metadata and audio under Dataset root")
    out = Path(out_dir).resolve()
    if _within(out, root):
        raise ValueError("Output must be outside ICSD dataset root to preserve original split")
    annotations = _read_annotations(metadata)
    all_audio = {p.name: p for p in audio_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg"}}
    missing = sorted(set(annotations) - set(all_audio))
    if missing:
        raise FileNotFoundError(f"Annotated audio not found: {missing[:5]}")
    if len(set(all_audio)) != len(all_audio):
        raise ValueError("Duplicate file basenames")

    # fail-closed: do not silently disregard unannotated test recordings
    unannotated = sorted(set(all_audio) - set(annotations))
    if unannotated:
        raise ValueError(f"Test audio without TSV annotation: {unannotated[:5]}")
    rows = []
    counts = {"positive": 0, "negative": 0, "ambiguous": 0,
              "unused_tail": 0, "files": 0}
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for filename in sorted(annotations):
        source = all_audio[filename]
        samples = _read_resample(source)
        duration = len(samples) / SAMPLE_RATE
        if any(end > duration + .02 for _, end, _ in annotations[filename]):
            raise ValueError(f"Annotation extends beyond audio duration: {filename}")
        windows = len(samples) // SAMPLE_RATE
        counts["files"] += 1
        if len(samples) % SAMPLE_RATE:
            counts["unused_tail"] += 1
        origin = hashlib.sha256(source.read_bytes()).hexdigest()
        for idx in range(windows):
            start, stop = idx * WINDOW_SECONDS, (idx + 1) * WINDOW_SECONDS
            overlap = {"Snoring": 0., "Infantcry": 0.}
            for begin, end, event in annotations[filename]:
                overlap[event] += max(0., min(stop, end) - max(start, begin))
            # A window with any infant cry is considered ambiguous rather than
            # a negative example for snoring, avoiding a different-class shortcut.
            if overlap["Infantcry"] > 0 or 0 < overlap["Snoring"] < min_positive_overlap:
                counts["ambiguous"] += 1
                continue
            if overlap["Snoring"] >= min_positive_overlap:
                label = 1
                counts["positive"] += 1
            elif assume_exhaustive and overlap["Snoring"] == 0:
                label = 0
                counts["negative"] += 1
            else:
                counts["ambiguous"] += 1
                continue
            clipname = f"{origin[:20]}_{idx:05d}.wav"
            target = clips_dir / clipname
            sf.write(target, samples[idx*SAMPLE_RATE:(idx+1)*SAMPLE_RATE],
                     SAMPLE_RATE, subtype="PCM_16")
            rows.append({"path": str(target), "label": label,
                         "source_id": origin})
    manifest = out / "test_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "dataset": "ICSD", "official_split": "test",
        "subset": "real_strong_test", "label": "Snoring",
        "window_seconds": 1.0,
        "min_positive_overlap_seconds": min_positive_overlap,
        "assume_exhaustive": assume_exhaustive,
        "manifest": str(manifest),
        "counts": counts,
        "warning": ("Restricted ICSD-derived clips and manifest: keep private. "
                    "No clinical diagnosis. Negative windows require explicit "
                    "exhaustiveness assumption; never relabel weak data."),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n",
                                    encoding="utf-8")
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_root", type=Path,
                   help="Authorized extracted ICSD Dataset root (contains audio/, metadata/)")
    p.add_argument("--out", type=Path, default=Path("data/processed/icsd/test"))
    p.add_argument("--positive-overlap", type=float, default=.5)
    p.add_argument("--assume-exhaustive", action="store_true",
                   help="Treat unannotated background in REAL STRONG test as non-snore")
    a = p.parse_args()
    print(json.dumps(convert_icsd_test(a.dataset_root, a.out,
         min_positive_overlap=a.positive_overlap,
         assume_exhaustive=a.assume_exhaustive), indent=2))


if __name__ == "__main__":
    main()
