"""Download a freely redistributable CC0 snoring recording from Wikimedia Commons.

Run: python scripts/fetch_public_snore_sample.py

This script uses only Python's standard library. The source recording is public,
but the recording is NOT medical ground truth for sleep quality or apnea events.
"""
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE = "https://commons.wikimedia.org/wiki/Special:Redirect/file/Snoring_of_old_man.ogg"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "snoring_cc0.ogg"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        SOURCE,
        headers={"User-Agent": "SleepAudioLab/0.2 (+https://github.com/713s/SleepAudioLab)"},
    )
    with urlopen(request, timeout=90) as response:
        data = response.read()
    if not data.startswith(b"OggS"):
        raise RuntimeError("The retrieved content is not an Ogg container")
    if not 5_000 <= len(data) <= 10_000_000:
        raise RuntimeError(f"Unexpected download size: {len(data)} bytes")
    OUTPUT.write_bytes(data)
    print(f"Saved {len(data)} bytes to {OUTPUT}")


if __name__ == "__main__":
    main()
