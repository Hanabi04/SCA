"""Verify all files listed in the release checksum manifest."""

import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    rows = (root / "SHA256SUMS").read_text(encoding="utf8").splitlines()
    for line in rows:
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checksum manifest path escapes the release directory")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Verified {len(rows)} release files")


if __name__ == "__main__":
    main()
