"""Task and adjudication prompts used in the experiments."""

import json


def build_slurp_agent_prompt(hypotheses: list[str]) -> str:
    """Fixed SLURP semantic parsing prompt. Hypotheses in ASR rank order."""
    lines = [
        "You are a spoken language understanding module.",
        "Given candidate ASR transcriptions of one user utterance (best-first order),",
        "predict the user's intent as structured JSON.",
        "",
        "ASR candidates:",
    ]
    for i, h in enumerate(hypotheses, start=1):
        lines.append(f"{i}. {h}")
    lines.extend(
        [
            "",
            "Return JSON with fields: scenario, action, entities.",
            "entities is a list of {type, value} objects (may be empty).",
            "Use only labels allowed by the response schema.",
            "Do not include explanations.",
        ]
    )
    return "\n".join(lines)


def build_fsc_agent_prompt(
    hypotheses: list[str], weights: list[float] | None = None
) -> str:
    """FSC prompt. Primary path ignores weights (kept for API compatibility)."""
    _ = weights
    lines = [
        "You are the decision module of a task-oriented voice assistant.",
        "Given candidate ASR transcriptions of one user utterance (best-first order),",
        "predict the structured command as JSON.",
        "",
        "ASR candidates:",
    ]
    for i, h in enumerate(hypotheses, start=1):
        lines.append(f"{i}. {h}")
    lines.extend(
        [
            "",
            "Return JSON with fields: object, action, location.",
            "Use only labels allowed by the response schema.",
            "Do not include explanations.",
        ]
    )
    return "\n".join(lines)


def build_chat_messages(
    user_prompt: str, system: str | None = None
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def build_scoreaware_slurp_prompt(hyps: list[str], weights: list[float]) -> str:
    lines = [
        "You are a spoken language understanding module.",
        "Given candidate ASR transcriptions of one user utterance (best-first order),",
        "predict the user's intent as structured JSON.",
        "",
        "ASR candidates:",
    ]
    for i, (h, w) in enumerate(zip(hyps, weights), start=1):
        lines.append(f"H{i} [p={float(w):.6f}]: {h}")
    lines.extend(
        [
            "",
            "Return JSON with fields: scenario, action, entities.",
            "entities is a list of {type, value} objects (may be empty).",
            "Use only labels allowed by the response schema.",
            "Do not include explanations.",
        ]
    )
    return "\n".join(lines)


def build_scoreaware_fsc_prompt(hyps: list[str], weights: list[float]) -> str:
    lines = [
        "You are the decision module of a task-oriented voice assistant.",
        "Given candidate ASR transcriptions of one user utterance (best-first order),",
        "predict the structured command as JSON.",
        "",
        "ASR candidates:",
    ]
    for i, (h, w) in enumerate(zip(hyps, weights), start=1):
        lines.append(f"H{i} [p={float(w):.6f}]: {h}")
    lines.extend(
        [
            "",
            "Return JSON with fields: object, action, location.",
            "Use only labels allowed by the response schema.",
            "Do not include explanations.",
        ]
    )
    return "\n".join(lines)


SCA_INSTRUCTION = """The hypotheses below are alternative ASR transcriptions of ONE spoken utterance.
They are mutually exclusive alternatives, not independent observations or votes.

Repeated or shared wording across hypotheses must NOT be treated as repeated independent evidence.

Use the ASR hypothesis probabilities together with the linguistic evidence to determine which of the two candidate task decisions is better supported by the spoken utterance.

Choose exactly one candidate."""


def format_decision(dataset: str, pred: dict[str, Any]) -> str:
    if dataset == "fsc":
        return (
            f'{{"object": {json.dumps(pred.get("object", ""), ensure_ascii=False)}, '
            f'"action": {json.dumps(pred.get("action", ""), ensure_ascii=False)}}}'
        )
    return (
        f'{{"scenario": {json.dumps(pred.get("scenario", ""), ensure_ascii=False)}, '
        f'"action": {json.dumps(pred.get("action", ""), ensure_ascii=False)}}}'
    )


def build_sca_user_prompt(
    hyps: list[str],
    weights: list[float],
    cand_a: str,
    cand_b: str,
) -> str:
    lines = [SCA_INSTRUCTION, ""]
    for i, (h, w) in enumerate(zip(hyps, weights), start=1):
        lines.append(f"H{i} [p={w:.6f}]: {h}")
    lines.extend(
        [
            "",
            f"Candidate A: {cand_a}",
            f"Candidate B: {cand_b}",
            "",
            "Answer with a single character (A or B):",
        ]
    )
    return "\n".join(lines)


H1ONLY_INSTRUCTION = """The hypothesis below is an ASR transcription of ONE spoken utterance.

Use the linguistic evidence to determine which of the two candidate task decisions is better supported by the spoken utterance.

Choose exactly one candidate."""


def build_h1only_user_prompt(
    hyps: list[str],
    weights: list[float],
    cand_a: str,
    cand_b: str,
) -> str:
    _ = weights  # intentionally unused — no posteriors
    h1 = hyps[0] if hyps else ""
    return "\n".join(
        [
            H1ONLY_INSTRUCTION,
            "",
            f"H1: {h1}",
            "",
            f"Candidate A: {cand_a}",
            f"Candidate B: {cand_b}",
            "",
            "Answer with a single character (A or B):",
        ]
    )
