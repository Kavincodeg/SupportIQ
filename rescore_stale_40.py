import os
import sys
import csv
import pandas as pd

# UTF-8 encoding support on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

RESCORING_NEEDED_FILE = os.path.join("data", "human_rescoring_needed.csv")
SAMPLES_40_FILE = os.path.join("data", "human_judge_samples_40.csv")
OUTPUT_SCORES_38_FILE = os.path.join("data", "human_rescoring_scores_38.csv")

SCORE_COLUMNS = [
    "sample_number",
    "example_id",
    "customer_message",
    "predicted_intent",
    "drafted_reply",
    "groundedness",
    "factual_correctness",
    "tone_empathy",
    "actionability",
    "conciseness",
    "overall_average",
    "human_notes"
]

def load_already_scored_ids():
    if not os.path.exists(OUTPUT_SCORES_38_FILE):
        return set()
    try:
        df = pd.read_csv(OUTPUT_SCORES_38_FILE)
        return set(df["example_id"].astype(str).tolist())
    except Exception:
        return set()

def save_score_record(rec):
    file_exists = os.path.exists(OUTPUT_SCORES_38_FILE)
    with open(OUTPUT_SCORES_38_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(rec)

def get_valid_score(dimension_name):
    while True:
        val = input(f"  {dimension_name} (1-5, 'q' to save & quit): ").strip()
        if val.lower() in ["q", "quit"]:
            return "quit"
        if val in ["1", "2", "3", "4", "5"]:
            return int(val)
        print("    [!] Please enter an integer from 1 to 5 (or 'q' to save and exit).")

def main():
    if not os.path.exists(RESCORING_NEEDED_FILE):
        print(f"Error: Required candidate file not found at {RESCORING_NEEDED_FILE}")
        return

    df_needed = pd.read_csv(RESCORING_NEEDED_FILE)
    
    # Enrich with metadata from samples_40 if available
    if os.path.exists(SAMPLES_40_FILE):
        df_s40 = pd.read_csv(SAMPLES_40_FILE)
        cols_to_use = [c for c in ["example_id", "sample_number", "predicted_intent", "my_acceptable_reply_note"] if c in df_s40.columns]
        df_merged = pd.merge(df_needed, df_s40[cols_to_use], on="example_id", how="left")
    else:
        df_merged = df_needed.copy()
        df_merged["sample_number"] = range(1, len(df_merged) + 1)
        df_merged["predicted_intent"] = "Unspecified"
        df_merged["my_acceptable_reply_note"] = ""

    total_items = len(df_merged)
    scored_ids = load_already_scored_ids()

    print("=" * 80)
    print("SPOTIFYCARES HUMAN RE-SCORING TOOL (38 STALE GEMINI REPLIES)")
    print("=" * 80)
    print(f"Total stale items needing scoring: {total_items}")
    print(f"Already scored in this session:     {len(scored_ids)} / {total_items}")
    print("NOTE: The 2 non-stale examples (E063, E130) will be automatically retained")
    print("      from your original human scores.")
    print("-" * 80)
    print("Score each drafted reply on 5 dimensions (1=Poor, 3=Acceptable, 5=Excellent):")
    print("  1. Groundedness (policy adherence to Spotify support workflows)")
    print("  2. Factual Correctness (accurate technical specs, limits, requirements)")
    print("  3. Tone & Empathy (friendly, empathetic, authentic Twitter voice)")
    print("  4. Actionability (clear next step, DM escalation, or resolution path)")
    print("  5. Conciseness (crisp, focused, appropriate length for social support)")
    print("=" * 80)

    for idx, row in df_merged.iterrows():
        ex_id = str(row["example_id"])
        if ex_id in scored_ids:
            continue

        progress = len(scored_ids) + 1
        s_num = row.get("sample_number", idx + 1)
        pred_intent = row.get("predicted_intent", "Unspecified")
        cust_msg = row["customer_message"]
        new_reply = row["new_drafted_reply"] if "new_drafted_reply" in row else row["drafted_reply"]
        guidance = row.get("my_acceptable_reply_note", "")

        print(f"\n[{progress} / {total_items}] Example ID: {ex_id} (Sample #{s_num})")
        print("-" * 80)
        print("CUSTOMER MESSAGE:")
        print(f"  \"{cust_msg}\"")
        print(f"Predicted Intent: {pred_intent}")
        print("-" * 80)
        print("NEW GEMINI DRAFTED REPLY TO SCORE (BLIND):")
        print(f"  \"{new_reply}\"")
        if not pd.isna(guidance) and str(guidance).strip():
            print(f"Reference Guidance Note: {guidance}")
        print("-" * 80)

        g = get_valid_score("1. Groundedness")
        if g == "quit":
            print(f"\n[Saved] Progress saved ({len(scored_ids)}/{total_items} complete). Exiting...")
            return

        f = get_valid_score("2. Factual Correctness")
        if f == "quit":
            print(f"\n[Saved] Progress saved ({len(scored_ids)}/{total_items} complete). Exiting...")
            return

        t = get_valid_score("3. Tone & Empathy")
        if t == "quit":
            print(f"\n[Saved] Progress saved ({len(scored_ids)}/{total_items} complete). Exiting...")
            return

        a = get_valid_score("4. Actionability")
        if a == "quit":
            print(f"\n[Saved] Progress saved ({len(scored_ids)}/{total_items} complete). Exiting...")
            return

        c = get_valid_score("5. Conciseness")
        if c == "quit":
            print(f"\n[Saved] Progress saved ({len(scored_ids)}/{total_items} complete). Exiting...")
            return

        notes = input("  Optional human notes / remarks (Enter to skip): ").strip()
        overall = round((g + f + t + a + c) / 5.0, 2)

        rec = {
            "sample_number": s_num,
            "example_id": ex_id,
            "customer_message": cust_msg,
            "predicted_intent": pred_intent,
            "drafted_reply": new_reply,
            "groundedness": g,
            "factual_correctness": f,
            "tone_empathy": t,
            "actionability": a,
            "conciseness": c,
            "overall_average": overall,
            "human_notes": notes
        }

        save_score_record(rec)
        scored_ids.add(ex_id)
        print(f"-> Saved {ex_id}! (Overall: {overall}/5.0 | Total: {len(scored_ids)}/{total_items})")

    print("\n" + "=" * 80)
    print("ALL 38 STALE EXAMPLES HAVE BEEN SUCCESSFULLY RE-SCORED!")
    print(f"Scores saved to: {OUTPUT_SCORES_38_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
