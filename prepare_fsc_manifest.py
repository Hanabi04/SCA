"""Create a local inference manifest from the official FSC CSV."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, help="Optional local list of recording IDs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    excluded = set(args.exclude_ids.read_text().splitlines()) if args.exclude_ids else set()
    records = []
    seen = set()
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "object", "action"} <= set(reader.fieldnames or []):
            parser.error("Expected official CSV columns: path, object, action")
        for row in reader:
            audio = Path(row["path"].replace(chr(92), "/")).name
            uid = Path(audio).stem
            if uid in excluded:
                continue
            if uid in seen:
                raise ValueError("Duplicate recording ID")
            seen.add(uid)
            records.append({"sample_id": uid, "audio_file": audio,
                            "gold": [row["object"], row["action"]]})
    if not records:
        parser.error("No recordings selected")
    data = "".join(json.dumps(row) + "\n" for row in records).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(data, mtime=0) if args.output.suffix == ".gz" else data)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
