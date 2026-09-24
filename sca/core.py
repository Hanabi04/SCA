"""Decision selection and paired evaluation, independent of model inference."""

import math


def decision_key(dataset, prediction):
    if not isinstance(prediction, dict):
        return None
    fields = ("object", "action") if dataset == "fsc" else ("scenario", "action")
    return [
        " ".join(str(prediction.get(k, "")).strip().lower().split()) for k in fields
    ]


def select(first, alternative, margins=None, orientation=None):
    """Return candidate index; ties and semantic agreement retain candidate 0."""
    if first is None:
        return 1 if alternative is not None else 0
    if alternative is None or first == alternative:
        return 0
    if (
        margins is None
        or len(margins) != 2
        or not all(math.isfinite(x) for x in margins)
    ):
        raise ValueError("A valid disagreement requires two finite signed margins")
    margin = margins[orientation] if orientation is not None else sum(margins) / 2
    return int(margin > 0)


def evaluate_rows(rows, methods, judges, parse_policy="valid_candidate"):
    """Compute correctness from categorical predictions and labels."""
    import numpy as np

    if parse_policy not in ("valid_candidate", "retain_first"):
        raise ValueError("Unknown parse-failure policy")

    ids = [r["sample_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be nonempty and unique")
    arrays = {m: [] for m in methods}
    for judge in judges:
        arrays[judge] = []
    arrays.update({"orientation_1": [], "orientation_2": [], "pairwise_oracle": []})
    chosen = {m: [] for m in judges}
    transition = dict(
        disagreements=0,
        corrections=0,
        recovered=0,
        corruptions=0,
        rejected=0,
        order_disagreements=0,
    )
    for r in rows:
        preds = r["predictions"]
        for m in methods:
            if m not in preds:
                raise ValueError(f'Missing {m} for {r["sample_id"]}')
            correct = int(
                preds[m]["decision"] is not None and preds[m]["decision"] == r["gold"]
            )
            saved = preds[m].get("scores", {}).get("scenario_action_ok")
            if saved is not None and int(saved) != correct:
                raise ValueError(f'Correctness mismatch: {r["sample_id"]}/{m}')
            arrays[m].append(correct)
        p1, pn = preds["one_best"]["decision"], preds["all5"]["decision"]
        c1, cn = arrays["one_best"][-1], arrays["all5"][-1]
        arrays["pairwise_oracle"].append(max(c1, cn))
        for j in judges:
            alt = "scoreaware_all5" if j == "sa_sca" else "all5"
            selected = select(p1, preds[alt]["decision"], r["margins"].get(j))
            if parse_policy == "retain_first" and p1 is None:
                selected = 0
            method = alt if selected else "one_best"
            chosen[j].append(method)
            arrays[j].append(arrays[method][-1])
        for o in range(2):
            selected = select(p1, pn, r["margins"].get("sca"), orientation=o)
            if parse_policy == "retain_first" and p1 is None:
                selected = 0
            arrays[f"orientation_{o+1}"].append(cn if selected else c1)
        if p1 is not None and pn is not None and p1 != pn:
            transition["disagreements"] += 1
            m = r["margins"]["sca"]
            transition["order_disagreements"] += int((m[0] > 0) != (m[1] > 0))
            transition["corrections"] += int(not c1 and cn)
            transition["recovered"] += int(not c1 and cn and arrays["sca"][-1])
            transition["corruptions"] += int(c1 and not cn)
            transition["rejected"] += int(c1 and not cn and arrays["sca"][-1])
    return (
        {k: np.asarray(v, dtype=np.int8) for k, v in arrays.items()},
        transition,
        chosen,
    )


def paired_interval(a, b, seed=20260920, repeats=10000):
    """Paired percentile bootstrap in percentage points, with original ID order."""
    import numpy as np

    if len(a) != len(b) or len(a) == 0 or repeats < 1:
        raise ValueError(
            "Paired nonempty arrays and a positive repeat count are required"
        )
    rng = np.random.default_rng(seed)
    delta = a.astype(float) - b.astype(float)
    draws = np.empty(repeats)
    for start in range(0, repeats, 64):
        count = min(64, repeats - start)
        indices = rng.integers(0, len(a), size=(count, len(a)))
        draws[start : start + count] = delta[indices].mean(axis=1) * 100
    low, high = np.quantile(draws, [0.025, 0.975])
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    p = (
        min(
            1.0,
            2 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / (2**n),
        )
        if n
        else 1.0
    )
    return float(delta.mean() * 100), float(low), float(high), p
