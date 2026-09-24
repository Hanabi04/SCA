"""Decode the release manifest with the pinned Parakeet checkpoint."""

import argparse
import gzip
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new output filename")
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    model_id = "nvidia/parakeet-rnnt-0.6b"
    revision = "1b6b548f70b93d2410c3d13cc0654cab300f06ef"
    checkpoint = hf_hub_download(model_id, "parakeet-rnnt-0.6b.nemo", revision=revision)
    model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location=args.device)
    model.to(args.device).eval()
    config = OmegaConf.merge(
        model.cfg.decoding,
        OmegaConf.create(
            {
                "strategy": "malsd_batch",
                "beam": {
                    "beam_size": 10,
                    "return_best_hypothesis": False,
                    "score_norm": True,
                    "tsd_max_sym_exp": 50,
                    "alsd_max_target_len": 2.0,
                },
                "greedy": {"max_symbols": 10},
            }
        ),
    )
    model.change_decoding_strategy(config)
    raw = args.manifest.read_bytes()
    if args.manifest.suffix == ".gz":
        raw = gzip.decompress(raw)
    records = [json.loads(line) for line in raw.decode().splitlines()]
    if args.limit:
        records = records[: args.limit]
    wanted = {r["audio_file"] for r in records}
    index = {}
    for f in args.audio_root.rglob("*"):
        if f.is_file() and f.name in wanted:
            if f.name in index:
                raise ValueError(f"Ambiguous audio basename: {f.name}")
            index[f.name] = f
    missing = wanted - index.keys()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files missing, for example {sorted(missing)[:3]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as out:
        for r in records:
            result = model.transcribe(
                [str(index[r["audio_file"]])], return_hypotheses=True
            )
            obj = result[0]
            while isinstance(obj, (list, tuple)) and len(obj) == 1:
                obj = obj[0]
            if hasattr(obj, "n_best_hypotheses"):
                obj = obj.n_best_hypotheses
            if not isinstance(obj, (list, tuple)):
                obj = [obj]
            obj = list(obj)[:10]
            if not obj or any(not hasattr(h, "score") for h in obj):
                raise ValueError("Decoder did not return scored hypotheses")
            out.write(
                json.dumps(
                    {
                        "sample_id": r["sample_id"],
                        "gold": r["gold"],
                        "raw_hypotheses": [h.text for h in obj],
                        "raw_scores": [float(h.score) for h in obj],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "revision": revision,
                "decoding": OmegaConf.to_container(config, resolve=True),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
