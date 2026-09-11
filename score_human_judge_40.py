import os
import sys
import csv
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

CANDIDATE_FILE = os.path.join("data", "human_judge_samples_40.csv")
HUMAN_SCORES_FILE = os.path.join("data", "human_judge_scores_40.csv")

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

def load_scored_ids():
    if not os.path.exists(HUMAN_SCORES_FILE):
        return set()
    try:
        df = pd.read_csv(HUMAN_SCORES_FILE)
        return set(df["example_id"].astype(str).tolist())
    except Exception:
        return set()

def save_score_record(rec):
    file_exists = os.path.exists(HUMAN_SCORES_FILE)
    with open(HUMAN_SCORES_FILE, "a", encoding="utf-8", newline="") as f:
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
        print("    [!] Please enter a valid score between 1 and 5 (or 'q' to quit).")

def main():
    if not os.path.exists(CANDIDATE_FILE):
        print(f"Error: Candidate file not found at {CANDIDATE_FILE}")
        return

    df_candidates = pd.read_csv(CANDIDATE_FILE)
    total_samples = len(df_candidates)
    scored_ids = load_scored_ids()
    
    print("=" * 80)
    print("SPOTIFYCARES HUMAN-JUDGE EVALUATION TOOL (40 RANDOM SAMPLES)")
    print("=" * 80)
    print(f"Total samples to score: {total_samples}")
    print(f"Already scored: {len(scored_ids)} / {total_samples}")
    print("Score each drafted reply on 5 dimensions (1=Poor, 3=Acceptable, 5=Excellent):")
    print("  1. Groundedness (adheres to Spotify support workflows)")
    print("  2. Factual / Policy Correctness (accurate limits, requirements, capabilities)")
    print("  3. Tone & Empathy (friendly, polite, supportive, authentic)")
    print("  4. Actionability (clear next steps, links, or solutions)")
    print("  5. Conciseness (crisp, Twitter-appropriate length)")
    print("=" * 80)

    for idx, row in df_candidates.iterrows():
        ex_id = str(row["example_id"])
        if ex_id in scored_ids:
            continue
            
        progress = len(scored_ids) + 1
        print(f"\n[{progress} / {total_samples}] Example ID: {ex_id} (Sample #{row['sample_number']})")
        print("-" * 80)
        print("CUSTOMER MESSAGE:")
        print(f"  \"{row['customer_message']}\"")
        print(f"Predicted Intent: {row['predicted_intent']}")
        print("-" * 80)
        print("DRAFTED REPLY TO SCORE:")
        print(f"  \"{row['drafted_reply']}\"")
        if not pd.isna(row.get("my_acceptable_reply_note")) and str(row["my_acceptable_reply_note"]).strip():
            print(f"Reference Guidance Note: {row['my_acceptable_reply_note']}")
        print("-" * 80)
        
        g = get_valid_score("1. Groundedness")
        if g == "quit":
            print("\nProgress saved. Exiting...")
            return
            
        f = get_valid_score("2. Factual Correctness")
        if f == "quit":
            print("\nProgress saved. Exiting...")
            return
            
        t = get_valid_score("3. Tone & Empathy")
        if t == "quit":
            print("\nProgress saved. Exiting...")
            return
            
        a = get_valid_score("4. Actionability")
        if a == "quit":
            print("\nProgress saved. Exiting...")
            return
            
        c = get_valid_score("5. Conciseness")
        if c == "quit":
            print("\nProgress saved. Exiting...")
            return
            
        notes = input("  Optional human notes / remarks (Enter to skip): ").strip()
        overall = round((g + f + t + a + c) / 5.0, 2)
        
        record = {
            "sample_number": row["sample_number"],
            "example_id": ex_id,
            "customer_message": row["customer_message"],
            "predicted_intent": row["predicted_intent"],
            "drafted_reply": row["drafted_reply"],
            "groundedness": g,
            "factual_correctness": f,
            "tone_empathy": t,
            "actionability": a,
            "conciseness": c,
            "overall_average": overall,
            "human_notes": notes
        }
        
        save_score_record(record)
        scored_ids.add(ex_id)
        print(f"-> Saved {ex_id} successfully! (Overall: {overall}/5.0 | Total Completed: {len(scored_ids)}/{total_samples})")

    print("\n" + "=" * 80)
    print("CONGRATULATIONS! ALL 40 SAMPLES HAVE BEEN SCORED PERSONALLY!")
    print(f"Scores saved to: {HUMAN_SCORES_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
