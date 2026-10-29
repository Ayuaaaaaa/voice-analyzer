"""Compare predictions with expected labels from a manifest."""
from .schema import FIELDS

SCORED_FIELDS = FIELDS[:-1]
TONES = ["neutral", "satisfied", "frustrated", "upset", "distressed"]


def _norm(v):
    return str(v).strip().lower()


def field_matches(pred: dict, label: dict) -> dict:
    return {k: _norm(pred.get(k)) == _norm(label.get(k)) for k in SCORED_FIELDS if k in label}


def summarize(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs: (prediction, label). Returns per-field accuracy and the emotional-tone confusion matrix."""
    per_field = {k: [0, 0] for k in SCORED_FIELDS}
    cm = {t: {p: 0 for p in TONES} for t in TONES}
    for pred, lab in pairs:
        for k, ok in field_matches(pred, lab).items():
            per_field[k][0] += int(ok)
            per_field[k][1] += 1
        t, p = _norm(lab.get("emotional_tone")), _norm(pred.get("emotional_tone"))
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return {
        "n": len(pairs),
        "per_field": {k: {"correct": c, "total": n, "accuracy": round(c / n, 3) if n else None}
                      for k, (c, n) in per_field.items()},
        "tone_confusion": cm,
    }
