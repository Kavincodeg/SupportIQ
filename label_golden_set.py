import csv
import os
import sys
import pandas as pd

# Set up utf-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

CANDIDATES_FILE = os.path.join("data", "golden_set_candidates_180.csv")
LABELED_FILE = os.path.join("data", "golden_set_labeled.csv")

LOCKED_INTENTS = [
    "Subscription & Billing Issues",
    "Playback & Technical Errors",
    "Feature Requests & Device Support",
    "Content Availability & Licensing",
    "Playlist & Library Management",
    "Account Access & Security",
    "Other / Unclear"
]

LABELED_COLUMNS = [
    "example_id",
    "thread_id",
    "customer_message",
    "my_intent_label",
    "escalate_or_auto",
    "acceptable_reply_note",
    "disagree_with_reply_note",
    "spotify_reply",
    "provisional_intent_guess",
    "full_thread_reference"
]

def load_already_labeled_ids():
    if not os.path.exists(LABELED_FILE):
        return set()
    try:
        df_labeled = pd.read_csv(LABELED_FILE)
        return set(df_labeled["example_id"].astype(str).tolist())
    except Exception:
        return set()

def save_single_labeled_record(record_dict):
    file_exists = os.path.exists(LABELED_FILE)
    with open(LABELED_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABELED_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record_dict)

def clear_screen():
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "=" * 80)

def main():
    if not os.path.exists(CANDIDATES_FILE):
        print(f"Error: Candidate file not found at {CANDIDATES_FILE}")
        sys.exit(1)

    df_candidates = pd.read_csv(CANDIDATES_FILE)
    total_examples = len(df_candidates)
    
    already_labeled = load_already_labeled_ids()
    print("=" * 80)
    print("SPOTIFYCARES GOLDEN EVALUATION SET MANUAL LABELING TOOL")
    print("=" * 80)
    print(f"Loaded {total_examples} candidate examples from {CANDIDATES_FILE}.")
    print(f"Already labeled: {len(already_labeled)} / {total_examples}")
    print("Commands at any prompt:")
    print("  'q' or 'quit' to save and exit immediately.")
    print("  'c' to view full thread context.")
    print("=" * 80)

    for idx, row in df_candidates.iterrows():
        example_id = str(row["example_id"])
        if example_id in already_labeled:
            continue

        labeled_count = len(already_labeled)
        clear_screen()
        print(f"Progress: [{labeled_count + 1} / {total_examples}] (Completed: {labeled_count}/{total_examples})")
        print(f"Example ID: {example_id} | Thread ID: {row['thread_id']}")
        print("-" * 80)
        print("CUSTOMER MESSAGE:")
        print(str(row["customer_message"]).strip() if not pd.isna(row["customer_message"]) else "[Empty / Missing Message]")
        print("-" * 80)

        # STEP 1: Intent Label Selection (Blind to provisional guess and spotify reply)
        intent_chosen = None
        while True:
            print("\nSelect Intent Category (enter 1-7, 'c' for full thread, 'q' to quit):")
            for i, cat in enumerate(LOCKED_INTENTS, 1):
                print(f"  [{i}] {cat}")
            
            user_choice = input("\nYour intent choice (1-7): ").strip()
            if user_choice.lower() in ["q", "quit"]:
                print("\nSession saved. Exiting...")
                return
            if user_choice.lower() == "c":
                print("\n--- FULL THREAD CONTEXT ---")
                if pd.isna(row["full_thread_reference"]) or str(row["full_thread_reference"]).strip() == "":
                    print("[No additional context available]")
                else:
                    print(str(row["full_thread_reference"]).strip())
                print("---------------------------\n")
                continue

            if user_choice in ["1", "2", "3", "4", "5", "6", "7"]:
                intent_chosen = LOCKED_INTENTS[int(user_choice) - 1]
                print(f"-> Selected Intent: {intent_chosen}")
                break
            else:
                print("Invalid input. Please enter a number between 1 and 7, 'c', or 'q'.")

        # STEP 2: Show historical Spotify reply and provisional guess now that intent is locked
        print("\n" + "." * 80)
        print("HISTORICAL CONTEXT (For your reference only):")
        print(f"Provisional sampling guess : {str(row['provisional_intent_guess']).strip()}")
        print(f"Historical Spotify reply   :\n{str(row['spotify_reply']).strip() if not pd.isna(row['spotify_reply']) else '[No reply recorded]'}")
        print("." * 80)

        # STEP 3: Escalate or Auto Decision
        routing_chosen = None
        while True:
            print("\nShould this issue be escalated to human support or automated?")
            print("  [1] auto     (can be resolved via canned macro / automated guidance / FAQ link)")
            print("  [2] escalate (requires human agent intervention / account lookup / security action)")
            
            route_choice = input("Your choice (1=auto, 2=escalate, 'q' to quit): ").strip()
            if route_choice.lower() in ["q", "quit"]:
                print("\nSession saved. Exiting...")
                return
            if route_choice in ["1", "auto", "a"]:
                routing_chosen = "auto"
                break
            elif route_choice in ["2", "escalate", "e"]:
                routing_chosen = "escalate"
                break
            else:
                print("Invalid input. Please enter '1' for auto or '2' for escalate.")

        # STEP 4: What should an acceptable reply contain? (Optional)
        acceptable_note = ""
        while True:
            print("\nNote: What key information/action SHOULD an acceptable reply contain?")
            print("(Press Enter to skip, or type note):")
            user_input_note = input("> ").strip()
            if user_input_note.lower() in ["q", "quit"]:
                confirm = input("Quit session? (y/n): ").strip().lower()
                if confirm == "y":
                    print("\nSession saved. Exiting...")
                    return
                else:
                    continue  # re-prompt for the note cleanly without storing 'q'
            acceptable_note = user_input_note
            break

        # STEP 5: Disagreement with historical reply (Optional)
        disagree_note = ""
        while True:
            print("\nNote: Any disagreement with how historical Spotify reply handled this issue?")
            print("(Press Enter to skip, or type note):")
            user_input_disagree = input("> ").strip()
            if user_input_disagree.lower() in ["q", "quit"]:
                confirm = input("Quit session? (y/n): ").strip().lower()
                if confirm == "y":
                    print("\nSession saved. Exiting...")
                    return
                else:
                    continue  # re-prompt for the note cleanly without storing 'q'
            disagree_note = user_input_disagree
            break

        # Save record incrementally
        record = {
            "example_id": example_id,
            "thread_id": row["thread_id"],
            "customer_message": row["customer_message"],
            "my_intent_label": intent_chosen,
            "escalate_or_auto": routing_chosen,
            "acceptable_reply_note": acceptable_note,
            "disagree_with_reply_note": disagree_note,
            "spotify_reply": row["spotify_reply"],
            "provisional_intent_guess": row["provisional_intent_guess"],
            "full_thread_reference": row["full_thread_reference"]
        }
        save_single_labeled_record(record)
        already_labeled.add(example_id)
        print(f"\n[Saved {example_id} successfully! Total labeled: {len(already_labeled)} / {total_examples}]")

    print("\n" + "=" * 80)
    print("CONGRATULATIONS! ALL 180 CANDIDATE EXAMPLES HAVE BEEN LABELED!")
    print(f"Final labeled file saved to: {LABELED_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
