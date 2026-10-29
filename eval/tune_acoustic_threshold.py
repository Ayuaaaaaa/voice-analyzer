"""Dev-only check of the acoustic override thresholds in rules.py for the chosen v2 heads (grouped CV)."""
import json

from eval_emotion_combined import CHOICE_FILE, cv_predictions, load_rows, macro_f1

from pipeline import rules

rows = [r for r in load_rows() if not r["test"]]
cfg = json.load(open(CHOICE_FILE))
for t in (0.4, 0.5, 0.6, 0.7):
    rules.ACOUSTIC_MIN = t
    preds = cv_predictions(rows, cfg)
    res = {"ACOUSTIC_MIN": t}
    for ds in ("ravdess", "cremad"):
        idx = [i for i, r in enumerate(rows) if r["dataset"] == ds]
        y, p = [rows[i]["label"] for i in idx], [preds[i][0] for i in idx]
        res[ds] = round(macro_f1(y, p), 3)
        res[ds + "_neutral_pred_rate"] = round(sum(x == "neutral" for x in p) / len(p), 3)
        res[ds + "_neutral_true_rate"] = round(sum(x == "neutral" for x in y) / len(y), 3)
    print(json.dumps(res), flush=True)
