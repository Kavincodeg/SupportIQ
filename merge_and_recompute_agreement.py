import os
import sys
import pandas as pd
import numpy as np

# UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

SCORES_38_FILE = os.path.join("data", "human_rescoring_scores_38.csv")
ORIGINAL_40_FILE = os.path.join("data", "human_judge_scores_40.csv")
SAMPLES_40_FILE = os.path.join("data", "human_judge_samples_40.csv")
FINAL_SCORES_40_FILE = os.path.join("data", "human_judge_scores_40.csv")
JUDGE_180_FILE = os.path.join("data", "judge_scores_180.csv")
AGREEMENT_JSON = os.path.join("data", "human_judge_agreement_metrics.json")

NON_STALE_IDS = ["E063", "E130"]

def main():
    if not os.path.exists(SCORES_38_FILE):
        print(f"Error: {SCORES_38_FILE} does not exist yet. Please finish re-scoring first.")
        return

    df_38 = pd.read_csv(SCORES_38_FILE)
    print(f"Loaded {len(df_38)} re-scored items from {SCORES_38_FILE}.")
    if len(df_38) < 38:
        print(f"Warning: Only {len(df_38)} out of 38 items have been scored so far.")

    df_orig = pd.read_csv(ORIGINAL_40_FILE)
    df_non_stale = df_orig[df_orig["example_id"].isin(NON_STALE_IDS)].copy()
    print(f"Retained {len(df_non_stale)} non-stale original scores ({', '.join(NON_STALE_IDS)}).")

    # Ensure updated drafted replies for non-stale rows
    if os.path.exists(SAMPLES_40_FILE):
        df_s40 = pd.read_csv(SAMPLES_40_FILE)
        reply_map = dict(zip(df_s40["example_id"], df_s40["drafted_reply"]))
        for idx, row in df_non_stale.iterrows():
            eid = row["example_id"]
            if eid in reply_map:
                df_non_stale.at[idx, "drafted_reply"] = reply_map[eid]

    # Combine into 40 samples
    combined = pd.concat([df_38, df_non_stale], ignore_index=True)
    
    # Sort by sample_number
    if "sample_number" in combined.columns:
        combined["sample_number"] = pd.to_numeric(combined["sample_number"], errors="coerce")
        combined = combined.sort_values(by="sample_number").reset_index(drop=True)

    combined.to_csv(FINAL_SCORES_40_FILE, index=False, encoding="utf-8")
    print(f"Saved complete 40-sample human score dataset to {FINAL_SCORES_40_FILE}")

    # Now execute compute_agreement.py
    print("\n" + "=" * 80)
    print("RECOMPUTING REAL AGREEMENT METRICS VIA compute_agreement.py")
    print("=" * 80)
    import subprocess
    res = subprocess.run([sys.executable, "compute_agreement.py"], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

if __name__ == "__main__":
    main()
