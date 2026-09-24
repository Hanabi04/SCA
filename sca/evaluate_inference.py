"""Evaluate newly generated decisions and export Excel-readable tables."""

import argparse
import json
from pathlib import Path
from .core import decision_key, evaluate_rows
from .export import export_tables
from .reproduce import LABELS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset", choices=["fsc", "slurp"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    raw = [
        json.loads(l)
        for l in args.input.read_text(encoding="utf8").splitlines()
        if l.strip()
    ]
    if not raw:
        raise ValueError("Empty predictions")
    rows = [
        {
            "sample_id": r["sample_id"],
            "gold": r["gold"],
            "margins": r["margins"],
            "predictions": {
                m: {"decision": decision_key(args.dataset, v)}
                for m, v in r["predictions"].items()
            },
        }
        for r in raw
    ]
    arrays, transitions, _ = evaluate_rows(
        rows, list(rows[0]["predictions"]), list(raw[0]["selected_methods"])
    )
    table = [["Dataset", "Method", "N", "Correct", "Accuracy (%)"]]
    table.extend(
        [args.dataset, LABELS.get(m, m), len(a), int(a.sum()), float(a.mean() * 100)]
        for m, a in arrays.items()
    )
    export_tables(
        args.output,
        {
            "Fresh inference": table,
            "Transitions": [
                ["Metric", "Count"],
                *[[k, v] for k, v in transitions.items()],
            ],
        },
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
