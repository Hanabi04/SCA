"""Generate task proposals and apply SCA to a JSONL file of ASR hypotheses."""

import argparse
import hashlib
import json
from pathlib import Path
import random

from .core import decision_key, select
from .nbest import exact_unique_hypotheses, normalize_decoding_weights, asr_mass_cover
from .prompts import (
    build_fsc_agent_prompt,
    build_slurp_agent_prompt,
    build_scoreaware_fsc_prompt,
    build_scoreaware_slurp_prompt,
    build_sca_user_prompt,
    build_h1only_user_prompt,
    format_decision,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen3": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "phi4": (
        "microsoft/Phi-4-mini-instruct",
        "cfbefacb99257ffa30c83adab238a50856ac3083",
    ),
}


def semantic_dedup(hypotheses, weights, labels, rho=0.95):
    """Cover decision mass, retaining the highest-weight text in each group."""
    groups = {}
    for i, z in enumerate(labels):
        groups.setdefault(z, []).append(i)
    masses = {z: sum(weights[i] for i in members) for z, members in groups.items()}
    selected = []
    total = 0.0
    for z in sorted(groups, key=lambda z: (-masses[z], z)):
        i = min(groups[z], key=lambda i: (-weights[i], i))
        selected.append(hypotheses[i])
        total += masses[z]
        if total + 1e-12 >= rho:
            break
    return selected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=["fsc", "slurp"], required=True)
    p.add_argument("--model", choices=MODELS, default="qwen3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["one_best", "all5", "top2", "asr_mass", "scoreaware_all5"],
        choices=[
            "one_best",
            "all5",
            "top2",
            "asr_mass",
            "scoreaware_all5",
            "coarse_decision_dedup",
            "repeated_1best5",
        ],
    )
    p.add_argument(
        "--h1-only",
        action="store_true",
        help="Also evaluate the judge with only the top transcript",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new output filename")
    if not {"one_best", "all5"} <= set(args.methods):
        raise ValueError("SCA requires one_best and all5 proposals")
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from outlines import Generator, from_transformers
    from outlines.types import JsonSchema

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_id, revision = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=args.offline,
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        local_files_only=args.offline,
        trust_remote_code=False,
    )
    model.eval()
    schema = json.loads((ROOT / f"configs/{args.dataset}_schema.json").read_text())
    generator = Generator(from_transformers(model, tok), JsonSchema(schema))
    ids = [tok.encode(s, add_special_tokens=False) for s in ["A", "B"]]
    if any(len(i) != 1 for i in ids):
        raise ValueError("A and B must be single-token labels")
    a_id, b_id = ids[0][0], ids[1][0]

    def template(text):
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def predict(text):
        result = generator(
            template(text),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            num_beams=1,
        )
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        text = str(result).split("</think>")[-1].strip()
        try:
            value = json.loads(text[text.index("{") : text.rindex("}") + 1])
            return value if isinstance(value, dict) else None
        except (ValueError, TypeError):
            return None

    @torch.inference_mode()
    def judge(h, w, first, alternative, h1_only=False):
        y1 = format_decision(args.dataset, first)
        yn = format_decision(args.dataset, alternative)
        builder = build_h1only_user_prompt if h1_only else build_sca_user_prompt
        texts = [template(builder(h, w, y1, yn)), template(builder(h, w, yn, y1))]
        encoded = [tok.encode(text, add_special_tokens=False) for text in texts]
        # Score each orientation without padding; logits are promoted before log-softmax.
        lp = []
        for token_ids in encoded:
            x = torch.tensor([token_ids], dtype=torch.long, device=model.device)
            logits = (
                model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
                .logits[0, -1]
                .float()
            )
            probs = torch.log_softmax(logits, dim=-1)
            lp.append((float(probs[a_id]), float(probs[b_id])))
        return [lp[0][1] - lp[0][0], lp[1][0] - lp[1][1]]

    sketcher = None
    if "coarse_decision_dedup" in args.methods:
        from sentence_transformers import SentenceTransformer

        metadata = json.loads((ROOT / f"configs/{args.dataset}_dedup.json").read_text())
        sketcher = SentenceTransformer(
            metadata["embedding_model_id"],
            revision=metadata["embedding_revision"],
            device="cpu",
            local_files_only=args.offline,
        )
        linear = np.load(
            ROOT / f"data/models/{args.dataset}_dedup.npz", allow_pickle=False
        )
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate input sample IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as out:
        for row in records:
            h, s, _ = exact_unique_hypotheses(
                row["raw_hypotheses"], row["raw_scores"], 5
            )
            if not h:
                raise ValueError(f'Empty ASR output: {row["sample_id"]}')
            w = normalize_decoding_weights(s)
            inputs = {
                "one_best": h[:1],
                "all5": h,
                "top2": h[:2],
                "asr_mass": asr_mass_cover(h, w, 0.9)[0],
                "scoreaware_all5": h,
                "repeated_1best5": [h[0]] * 5,
            }
            if sketcher is not None:
                emb = sketcher.encode(
                    h,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
                indices = (emb @ linear["coef"].T + linear["intercept"]).argmax(axis=1)
                labels = [
                    metadata["id_to_label"][str(int(linear["classes"][i]))]
                    for i in indices
                ]
                inputs["coarse_decision_dedup"] = semantic_dedup(h, w, labels)
            predictions = {}
            for method in args.methods:
                if method == "scoreaware_all5":
                    prompt = (
                        build_scoreaware_fsc_prompt
                        if args.dataset == "fsc"
                        else build_scoreaware_slurp_prompt
                    )(h, w)
                else:
                    prompt = (
                        build_fsc_agent_prompt
                        if args.dataset == "fsc"
                        else build_slurp_agent_prompt
                    )(inputs[method])
                predictions[method] = predict(prompt)
            chosen = {}
            margin_map = {}
            variants = [("sca", "all5", h, w)]
            if "scoreaware_all5" in predictions:
                variants.append(("sa_sca", "scoreaware_all5", h, w))
            if args.h1_only:
                variants.append(("h1_only", "all5", h[:1], [1.0]))
            for method, alternative, evidence, weights in variants:
                p1, pn = predictions["one_best"], predictions[alternative]
                k1, kn = decision_key(args.dataset, p1), decision_key(args.dataset, pn)
                margins = (
                    judge(evidence, weights, p1, pn, method == "h1_only")
                    if k1 is not None and kn is not None and k1 != kn
                    else None
                )
                if margins is not None:
                    margin_map[method] = margins
                chosen[method] = alternative if select(k1, kn, margins) else "one_best"
            result = {
                "sample_id": row["sample_id"],
                "predictions": predictions,
                "margins": margin_map,
                "selected_methods": chosen,
            }
            # Labels are copied only after inference and never enter a prompt.
            if "gold" in row:
                result["gold"] = row["gold"]
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
    metadata = {
        "model_id": model_id,
        "revision": revision,
        "seed": args.seed,
        "n": len(records),
        "methods": args.methods,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 256,
        "dtype": "bfloat16",
        "device": args.device,
        "torch": torch.__version__,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
