"""ASR sequence-score normalization and exact transcript deduplication."""

import math


def normalize_decoding_weights(
    scores: list[float], temperature: float = 1.0
) -> list[float]:
    """Convert sequence scores s_i into normalized decoding weights.

    w_i = exp(s_i / T) / sum_j exp(s_j / T)

    Terminology: normalized decoding weight (not calibrated posterior).
    """
    if not scores:
        return []
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    z = sum(exps)
    if z == 0:
        # Degenerate numerical case: uniform fallback with explicit equal weights.
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in exps]


def normalize_text_for_dedup(text: str, lowercase: bool = True) -> str:
    """Minimal normalization for exact-duplicate detection only."""
    t = " ".join(text.strip().split())
    if lowercase:
        t = t.lower()
    return t


def exact_unique_hypotheses(
    hypotheses: list[str],
    scores: list[float],
    max_unique: int | None = 5,
    lowercase: bool = True,
) -> tuple[list[str], list[float], list[str]]:
    """Keep the first occurrence of each normalized transcript, up to max_unique."""
    if len(hypotheses) != len(scores):
        raise ValueError("hypotheses and scores must have the same length")

    seen: set[str] = set()
    uniq_h: list[str] = []
    uniq_s: list[float] = []
    norms: list[str] = []
    for h, s in zip(hypotheses, scores):
        norm = normalize_text_for_dedup(h, lowercase=lowercase)
        if norm in seen:
            continue
        seen.add(norm)
        uniq_h.append(h)
        uniq_s.append(float(s))
        norms.append(norm)
        if max_unique is not None and len(uniq_h) >= max_unique:
            break
    return uniq_h, uniq_s, norms


def asr_mass_cover(
    hypotheses: list[str],
    weights: list[float],
    rho: float,
    scores: list[float] | None = None,
) -> tuple[list[str], list[float], list[float], float]:
    """Select hypotheses by descending weight until cumulative mass >= rho.

    Ties are resolved by original index.
    """
    if not 0.0 < rho <= 1.0 + 1e-12:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    if scores is None:
        scores = [0.0] * len(hypotheses)
    if not hypotheses:
        return [], [], [], 0.0

    order = sorted(range(len(hypotheses)), key=lambda i: (-float(weights[i]), i))
    retained_h: list[str] = []
    retained_s: list[float] = []
    retained_w: list[float] = []
    cum = 0.0
    for i in order:
        retained_h.append(hypotheses[i])
        retained_s.append(float(scores[i]))
        retained_w.append(float(weights[i]))
        cum += float(weights[i])
        if cum + 1e-12 >= rho:
            break
    return retained_h, retained_s, retained_w, cum
