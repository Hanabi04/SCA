"""Run ASR, task baselines, SCA, and result export on locally supplied audio."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment",
        required=True,
        choices=[
            "fsc_validation",
            "slurp_validation",
            "fsc_retained_test",
            "slurp_retained_test",
            "slurp_synthetic",
        ],
    )
    p.add_argument("--audio-root", required=True, type=Path)
    p.add_argument("--manifest", type=Path, help="Locally prepared manifest, required for FSC")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--asr-python",
        default=sys.executable,
        help="Python executable in the NeMo environment",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    manifest = args.manifest or ROOT / f"data/manifests/{args.experiment}.jsonl.gz"
    if not manifest.is_file():
        p.error("Manifest not found. For FSC, run prepare_fsc_manifest.py and pass --manifest.")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    ds = args.experiment.split("_")[0]
    cmds = [
        [
            args.asr_python,
            "-m",
            "sca.decode",
            "--manifest",
            str(manifest.resolve()),
            "--audio-root",
            str(args.audio_root.resolve()),
            "--output",
            str(out / "nbest.jsonl"),
            "--device",
            args.device,
        ],
        [
            sys.executable,
            "-m",
            "sca.infer",
            "--dataset",
            ds,
            "--input",
            str(out / "nbest.jsonl"),
            "--output",
            str(out / "predictions.jsonl"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--h1-only",
            "--methods",
            "one_best",
            "all5",
            "top2",
            "asr_mass",
            "scoreaware_all5",
            "coarse_decision_dedup",
            "repeated_1best5",
        ],
        [
            sys.executable,
            "-m",
            "sca.evaluate_inference",
            "--dataset",
            ds,
            "--input",
            str(out / "predictions.jsonl"),
            "--output",
            str(out / "tables"),
        ],
    ]
    if args.limit:
        cmds[0] += ["--limit", str(args.limit)]
    for i, cmd in enumerate(cmds, 1):
        print(f"Stage {i}/3: {cmd[2]}", flush=True)
        with (out / f"stage_{i}.log").open("w", encoding="utf8") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Stage {i} failed. See {out/f"stage_{i}.log"}')
    print(f'Completed: {out/"tables/reproduction_results.xlsx"}')


if __name__ == "__main__":
    main()
