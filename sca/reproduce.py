"""Recompute paper metrics from compact, sample-level prediction archives."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform

from .core import evaluate_rows, paired_interval
from .export import export_tables

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "one_best": "1-best",
    "all5": "All-5",
    "top2": "Top-2",
    "asr_mass": "ASR-Mass (90%)",
    "coarse_decision_dedup": "Decision deduplication",
    "scoreaware_all5": "Score-aware All-5",
    "repeated_1best5": "Repeated 1-best x5",
    "sca": "SCA",
    "sa_sca": "SA-Cand-SCA",
    "h1_only": "SCA with 1-best-only judge",
    "orientation_1": "SCA orientation 1",
    "orientation_2": "SCA orientation 2",
    "pairwise_oracle": "Pairwise oracle (1-best, All-5)",
}
COMPARATORS = {
    "fsc_validation": "asr_mass",
    "slurp_validation": "all5",
    "fsc_retained_test": "coarse_decision_dedup",
    "slurp_retained_test": "all5",
    "slurp_synthetic": "all5",
}
PAPER_CI = {
    "fsc_validation": (1.48, 2.82),
    "slurp_validation": (1.74, 2.46),
    "fsc_retained_test": (0.97, 2.06),
    "slurp_retained_test": (1.87, 2.46),
    "slurp_synthetic": (0.80, 3.10),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--experiment",
        action="append",
        help="Run only the named experiment; repeat to select several",
    )
    args = parser.parse_args(argv)
    import numpy as np

    output = args.output or ROOT / "reproduction_results" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    records = json.loads((ROOT / "data/experiments.json").read_text())
    names = {r["name"] for r in records}
    if args.experiment and not set(args.experiment) <= names:
        raise ValueError("Unknown experiment name")
    tables = {
        "Table 1": [
            [
                "Experiment",
                "Dataset",
                "Model",
                "N",
                "1-best (%)",
                "All-5 (%)",
                "Score-aware (%)",
                "SCA (%)",
                "Comparator",
                "Comparator (%)",
                "Gain (pp)",
                "CI low (pp)",
                "CI high (pp)",
            ]
        ],
        "All methods": [
            ["Experiment", "Paper location", "Method", "N", "Correct", "Accuracy (%)"]
        ],
        "Ablations": [
            [
                "Experiment",
                "N",
                "Orientation 1 (%)",
                "Orientation 2 (%)",
                "SCA (%)",
                "SA-Cand-SCA (%)",
                "1-best-only judge (%)",
            ]
        ],
        "Transitions": [
            [
                "Experiment",
                "N",
                "Disagreements",
                "Disagreement (%)",
                "Corrections",
                "Recovered",
                "Recovery (%)",
                "Corruptions",
                "Rejected",
                "Rejection (%)",
                "Order disagreements",
            ]
        ],
        "Comparisons": [
            [
                "Experiment",
                "Method",
                "Comparator",
                "Gain (pp)",
                "CI low (pp)",
                "CI high (pp)",
                "McNemar p",
                "Resamples",
                "Seed",
            ]
        ],
        "Full outputs": [["Experiment", "Method", "Metric", "N", "Value", "Unit"]],
        "Sampling summary": [
            [
                "Dataset",
                "Method",
                "Runs",
                "Mean accuracy (%)",
                "Sample SD (pp)",
                "Min (%)",
                "Max (%)",
            ]
        ],
        "Cost": [
            [
                "Experiment",
                "Task generations per utterance",
                "Judge calls per utterance",
                "Judge-call reduction (%)",
            ]
        ],
        "Archived latency": [
            [
                "Dataset",
                "Hardware",
                "Disagreements",
                "Repeats",
                "Median (s)",
                "IQR (s)",
                "Scope",
            ]
        ],
        "Checks": [
            ["Experiment", "N", "Unique IDs", "Input SHA256", "Paper rounded values"]
        ],
        "Parse fallback": [
            [
                "Experiment",
                "Method",
                "Archived policy (%)",
                "Valid-candidate policy (%)",
                "Changed outcomes",
                "N",
            ]
        ],
        "Definitions": [
            ["Field", "Meaning"],
            [
                "Accuracy",
                "Exact object-action (FSC) or scenario-action (SLURP), in percent.",
            ],
            ["Gain", "Accuracy difference in percentage points; higher is better."],
            ["CI", "Paired percentile 95% interval; 10000 resamples in the paper."],
            ["Blank", "Method or metric was not evaluated for this experiment."],
            [
                "Pairwise oracle",
                "Correct when either proposal is correct; uses labels, not an executable baseline.",
            ],
            [
                "Full outputs",
                "Selected candidate field scores from the original evaluation, not a new entity reannotation.",
            ],
            [
                "Cached reproduction",
                "Recomputes decisions and statistics; does not run speech or language models.",
            ],
            [
                "Sampling seeds",
                "Three 500-sample generation runs; separate from the full-validation single-run table.",
            ],
            [
                "Archived latency",
                "Summary recomputed from saved hardware measurements; not a timing of this CPU run.",
            ],
        ],
    }
    expected = json.loads((ROOT / "data/expected.json").read_text())
    summary = {}
    for cfg in records:
        name = cfg["name"]
        if args.experiment and name not in args.experiment:
            continue
        path = ROOT / cfg["path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != cfg["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {name}")
        rows = [json.loads(x) for x in gzip.decompress(data).decode().splitlines()]
        if len(rows) != cfg["n"]:
            raise ValueError(f"Incorrect denominator: {name}")
        arrays, t, choices = evaluate_rows(
            rows,
            cfg["methods"],
            cfg["judges"],
            cfg.get("parse_policy", "valid_candidate"),
        )
        if cfg.get("parse_policy") == "retain_first":
            corrected, _, _ = evaluate_rows(rows, cfg["methods"], cfg["judges"])
            for method in cfg["judges"]:
                tables["Parse fallback"].append(
                    [
                        name,
                        LABELS[method],
                        float(arrays[method].mean() * 100),
                        float(corrected[method].mean() * 100),
                        int((arrays[method] != corrected[method]).sum()),
                        len(rows),
                    ]
                )
        acc = {m: float(v.mean() * 100) for m, v in arrays.items()}
        for m, v in acc.items():
            if m in expected.get(name, {}) and abs(v - expected[name][m]) > 0.0051:
                raise ValueError(f"Paper value differs: {name}/{m}: {v}")
            tables["All methods"].append(
                [
                    name,
                    cfg["paper_location"],
                    LABELS.get(m, m),
                    len(rows),
                    int(arrays[m].sum()),
                    v,
                ]
            )
        tables["Checks"].append(
            [
                name,
                len(rows),
                len({r["sample_id"] for r in rows}),
                cfg["sha256"],
                "PASS",
            ]
        )
        n = len(rows)
        tables["Transitions"].append(
            [
                name,
                n,
                t["disagreements"],
                100 * t["disagreements"] / n,
                t["corrections"],
                t["recovered"],
                100 * t["recovered"] / t["corrections"] if t["corrections"] else None,
                t["corruptions"],
                t["rejected"],
                100 * t["rejected"] / t["corruptions"] if t["corruptions"] else None,
                t["order_disagreements"],
            ]
        )
        tables["Ablations"].append(
            [
                name,
                n,
                acc["orientation_1"],
                acc["orientation_2"],
                acc["sca"],
                acc.get("sa_sca"),
                acc.get("h1_only"),
            ]
        )
        tables["Cost"].append(
            [name, 2, 2 * t["disagreements"] / n, 100 * (1 - t["disagreements"] / n)]
        )
        comps = {}
        pairs = [
            ("sca", m)
            for m in [
                "one_best",
                "all5",
                "scoreaware_all5",
                "asr_mass",
                "coarse_decision_dedup",
                "orientation_1",
                "orientation_2",
            ]
            if m in arrays
        ]
        if "sa_sca" in arrays:
            pairs.append(("sa_sca", "sca"))
        for a, b in pairs:
            result = paired_interval(
                arrays[a], arrays[b], cfg["bootstrap_seed"], args.bootstrap
            )
            comps[f"{a}/{b}"] = result
            tables["Comparisons"].append(
                [
                    name,
                    LABELS[a],
                    LABELS[b],
                    *result,
                    args.bootstrap,
                    cfg["bootstrap_seed"],
                ]
            )
        if name in COMPARATORS:
            comparator = COMPARATORS[name]
            difference, lo, hi, _ = comps[f"sca/{comparator}"]
            if args.bootstrap == 10000 and any(
                abs(x - y) > 0.0051 for x, y in zip((lo, hi), PAPER_CI[name])
            ):
                raise ValueError(f"Paper confidence interval differs: {name}")
            tables["Table 1"].append(
                [
                    name,
                    cfg["dataset"],
                    cfg["model"],
                    n,
                    acc["one_best"],
                    acc["all5"],
                    acc.get("scoreaware_all5"),
                    acc["sca"],
                    LABELS[comparator],
                    acc[comparator],
                    difference,
                    lo,
                    hi,
                ]
            )
        for m in [*cfg["methods"], *cfg["judges"]]:
            scores = [
                r["predictions"][choices[m][i] if m in choices else m].get("scores", {})
                for i, r in enumerate(rows)
            ]
            for field in [
                "full_command_exact",
                "full_frame_exact",
                "location_ok",
                "entity_micro_f1",
            ]:
                if all(field in s for s in scores):
                    scale = 1 if field == "entity_micro_f1" else 100
                    tables["Full outputs"].append(
                        [
                            name,
                            LABELS.get(m, m),
                            field,
                            n,
                            float(np.mean([s[field] for s in scores]) * scale),
                            "ratio" if scale == 1 else "%",
                        ]
                    )
        summary[name] = {
            "n": n,
            "accuracy_percent": acc,
            "transitions": t,
            "comparisons": comps,
        }
        print(f'{name}: n={n}, SCA={acc["sca"]:.2f}%', flush=True)
    for dataset in ["fsc", "slurp"]:
        runs = [
            v["accuracy_percent"]
            for k, v in summary.items()
            if k.startswith(dataset + "_sampling_seed")
        ]
        if len(runs) == 3:
            for method in ["one_best", "all5", "scoreaware_all5", "sca", "sa_sca"]:
                values = [r[method] for r in runs]
                tables["Sampling summary"].append(
                    [
                        dataset,
                        LABELS[method],
                        3,
                        float(np.mean(values)),
                        float(np.std(values, ddof=1)),
                        min(values),
                        max(values),
                    ]
                )
    latency = json.loads((ROOT / "data/latency.json").read_text())
    for ds, values in latency["datasets"].items():
        q = np.quantile(values, [0.25, 0.75])
        tables["Archived latency"].append(
            [
                ds,
                latency["hardware"],
                latency["n_disagreements"],
                len(values),
                float(np.median(values)),
                float(q[1] - q[0]),
                "Saved judge-only timing; both orientations",
            ]
        )
    export_tables(output, tables)
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf8"
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "mode": "cached_prediction_recomputation",
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "bootstrap": args.bootstrap,
                "experiments": list(summary),
                "status": "passed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
