import json
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

HUMAN_INPUT = """
| #  | ID   | Groundedness | Factual Correctness | Tone / Empathy | Actionability | Conciseness | Overall |
| -- | ---- | -----------: | ------------------: | -------------: | ------------: | ----------: | ------: |
| 1  | E002 |            5 |                   5 |              5 |             4 |           5 |     4.8 |
| 2  | E007 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 3  | E008 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 4  | E009 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 5  | E023 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 6  | E024 |            5 |                   5 |              5 |             4 |           5 |     4.8 |
| 7  | E025 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 8  | E027 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 9  | E029 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 10 | E036 |            4 |                   4 |              5 |             2 |           5 |     4.0 |
| 11 | E040 |            3 |                   4 |              5 |             2 |           5 |     3.8 |
| 12 | E041 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 13 | E051 |            4 |                   4 |              5 |             4 |           5 |     4.4 |
| 14 | E056 |            2 |                   2 |              5 |             3 |           5 |     3.4 |
| 15 | E057 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 16 | E058 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 17 | E060 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 18 | E063 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 19 | E071 |            4 |                   4 |              5 |             3 |           5 |     4.2 |
| 20 | E072 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 21 | E087 |            4 |                   4 |              5 |             4 |           5 |     4.4 |
| 22 | E088 |            4 |                   4 |              5 |             3 |           5 |     4.2 |
| 23 | E098 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 24 | E108 |            5 |                   5 |              5 |             4 |           5 |     4.8 |
| 25 | E109 |            4 |                   4 |              5 |             4 |           5 |     4.4 |
| 26 | E115 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 27 | E130 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 28 | E140 |            2 |                   2 |              5 |             1 |           5 |     3.0 |
| 29 | E144 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 30 | E151 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 31 | E152 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 32 | E153 |            5 |                   5 |              5 |             4 |           5 |     4.8 |
| 33 | E155 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 34 | E164 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 35 | E165 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 36 | E166 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 37 | E169 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 38 | E172 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 39 | E173 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
| 40 | E178 |            5 |                   5 |              5 |             5 |           5 |     5.0 |
"""

def parse_table(text):
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|") or line.startswith("| --") or "Groundedness" in line:
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 8:
            sample_num = int(parts[0])
            ex_id = parts[1]
            g = int(parts[2])
            f = int(parts[3])
            t = int(parts[4])
            a = int(parts[5])
            c = int(parts[6])
            overall = float(parts[7].replace("*", ""))
            rows.append({
                "sample_number": sample_num,
                "example_id": ex_id,
                "groundedness": g,
                "factual_correctness": f,
                "tone_empathy": t,
                "actionability": a,
                "conciseness": c,
                "overall_average": overall
            })
    return pd.DataFrame(rows)

df_human = parse_table(HUMAN_INPUT)
print(f"Parsed {len(df_human)} human scores.")

# Join with candidate metadata
df_candidates = pd.read_csv("data/human_judge_samples_40.csv")
df_merged = pd.merge(df_candidates[["sample_number", "example_id", "customer_message", "predicted_intent", "drafted_reply"]], df_human, on=["sample_number", "example_id"])
df_merged["human_notes"] = ""

HUMAN_SCORES_FILE = os.path.join("data", "human_judge_scores_40.csv")
df_merged.to_csv(HUMAN_SCORES_FILE, index=False, encoding="utf-8")
print(f"Saved human scores to {HUMAN_SCORES_FILE}")

# Load LLM judge scores
df_judge_all = pd.read_csv("data/judge_scores_180.csv")
df_judge_40 = pd.merge(df_human[["example_id"]], df_judge_all, on="example_id")

dimensions = ["groundedness", "factual_correctness", "tone_empathy", "actionability", "conciseness", "overall_average"]

print("\n" + "=" * 80)
print("HUMAN VS. LLM-JUDGE AGREEMENT METRICS (N=40)")
print("=" * 80)

metrics = []
for dim in dimensions:
    h_vals = df_merged[dim].values
    j_vals = df_judge_40[dim].values
    
    mae = mean_absolute_error(h_vals, j_vals)
    rmse = np.sqrt(mean_squared_error(h_vals, j_vals))
    
    # Exact and within-1 agreement
    diffs = np.abs(h_vals - j_vals)
    exact_match = (diffs == 0).mean() * 100
    within_1 = (diffs <= 1.0).mean() * 100
    
    # Correlation (handle constant variance case if any)
    if np.std(h_vals) > 1e-6 and np.std(j_vals) > 1e-6:
        p_r, p_val = pearsonr(h_vals, j_vals)
        s_r, s_val = spearmanr(h_vals, j_vals)
    else:
        p_r, s_r = 0.0, 0.0
        
    metrics.append({
        "dimension": dim,
        "human_mean": float(np.mean(h_vals)),
        "judge_mean": float(np.mean(j_vals)),
        "mae": float(mae),
        "rmse": float(rmse),
        "exact_agreement_pct": float(exact_match),
        "within_1_pct": float(within_1),
        "pearson_r": float(p_r),
        "spearman_rho": float(s_r)
    })
    
    print(f"[{dim}]")
    print(f"  Human Mean: {np.mean(h_vals):.2f} | Judge Mean: {np.mean(j_vals):.2f}")
    print(f"  MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    print(f"  Exact Match: {exact_match:.1f}% | Within +/-1: {within_1:.1f}%")
    print(f"  Pearson r: {p_r:.3f} | Spearman rho: {s_r:.3f}")
    print("-" * 50)

# Save metrics summary to json for report inclusion
with open("data/human_judge_agreement_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Agreement metrics saved to data/human_judge_agreement_metrics.json")
