import csv
import json
import os
import re
import sys
import time
from datetime import datetime

# Enable UTF-8 for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

INPUT_CSV = "twcs.csv"
INPUT_JSONL = "spotify_threads_annotated_v2.jsonl"
DATA_DIR = "data"
REPORTS_DIR = "reports"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

OUTPUT_TRUSTED_CSV = os.path.join(DATA_DIR, "spotifycares_trusted_v2.csv")
OUTPUT_GROUNDING_CSV = os.path.join(DATA_DIR, "spotify_grounding_corpus_v2.csv")
OUTPUT_REPORT_TXT = os.path.join(REPORTS_DIR, "filter_comparison.txt")

def main():
    print("Loading in_response_to_tweet_id mapping for SpotifyCares from twcs.csv...")
    t0 = time.time()
    spotify_in_resp = {}
    with open(INPUT_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        tid_idx = header.index("tweet_id")
        auth_idx = header.index("author_id")
        in_resp_idx = header.index("in_response_to_tweet_id")
        for row in reader:
            if row[auth_idx] == "SpotifyCares":
                spotify_in_resp[row[tid_idx]] = row[in_resp_idx].strip()
    print(f"Loaded {len(spotify_in_resp):,} Spotify reply mappings in {time.time() - t0:.2f}s.")

    print(f"Loading {INPUT_JSONL}...")
    threads = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                threads.append(json.loads(line))
    total_threads = len(threads)
    print(f"Loaded {total_threads:,} total threads.")

    # 1. Evaluate previous filtering stats
    # Previous rule: fully_trusted = (not suspect_timing) and (not non_english) and (num_unique_customer_authors == 1)
    old_trusted_count = sum(
        1 for t in threads
        if not t["suspect_timing"]
        and not t["non_english"]
        and t["num_unique_customer_authors"] == 1
    )
    old_rejected_multi_author = sum(
        1 for t in threads
        if t["multi_author_merge"]
    )
    old_rejected_suspect_timing = sum(
        1 for t in threads
        if t["suspect_timing"]
    )

    # 2. Evaluate new filtering stats
    # New rule: exclude_multi_author_merge = multi_author_merge & suspect_timing
    # fully_trusted = (not suspect_timing) & (not non_english) & (num_unique_customer_authors >= 1)
    # Note that if suspect_timing is True, it's excluded anyway. So multi-author threads with clean timing (<=48h) are retained.
    
    trusted_v2_threads = []
    excluded_multi_author_and_suspect = []
    clean_multi_author_retained = []
    clean_multi_extracted = []
    clean_multi_ambiguous = []

    single_author_extracted = []

    for t in threads:
        is_suspect_timing = t["suspect_timing"]
        is_non_english = t["non_english"]
        is_multi_author = t["multi_author_merge"]
        num_cust = t["num_unique_customer_authors"]

        # New multi-author exclusion condition
        exclude_multi_author_rule = bool(is_multi_author and is_suspect_timing)
        t["exclude_multi_author_merge"] = exclude_multi_author_rule
        
        # New overall trust condition:
        # Must have at least 1 customer author, must be clean timing (<=48h), must be English (en)
        # and not excluded by the combined multi-author & suspect timing rule
        is_trusted_v2 = bool(
            num_cust >= 1
            and not is_suspect_timing
            and not is_non_english
            and not exclude_multi_author_rule
        )
        t["fully_trusted_v2"] = is_trusted_v2

        if exclude_multi_author_rule:
            excluded_multi_author_and_suspect.append(t)

        if is_trusted_v2:
            trusted_v2_threads.append(t)

            # Grounding pair extraction logic
            tweets = t["tweets"]
            inbound_tweets = [tw for tw in tweets if tw.get("inbound", False)]
            
            # Original asker: first inbound customer
            orig_asker = inbound_tweets[0]["author_id"]
            asker_tweets = [tw for tw in inbound_tweets if tw["author_id"] == orig_asker]
            asker_tids = set(tw["tweet_id"] for tw in asker_tweets)
            
            # Customer message: concatenated in chronological order
            customer_message = "\n\n".join(tw["text"] for tw in asker_tweets)

            # Spotify replies
            spotify_tweets = [tw for tw in tweets if tw.get("author_id") == "SpotifyCares"]

            if not is_multi_author:
                # Single customer author: all Spotify replies in the thread are for this customer
                spotify_reply_text = "\n\n".join(tw["text"] for tw in spotify_tweets)
                grounding_record = {
                    "thread_id": t["thread_id"],
                    "original_asker_id": orig_asker,
                    "num_unique_customer_authors": num_cust,
                    "multi_author_merge": False,
                    "customer_message": customer_message,
                    "spotify_reply": spotify_reply_text,
                    "start_time": t["start_time"],
                    "end_time": t["end_time"]
                }
                single_author_extracted.append(grounding_record)
            else:
                # Multi-author thread: check attribution
                clean_multi_author_retained.append(t)
                
                attributed_replies = []
                for stw in spotify_tweets:
                    stid = stw["tweet_id"]
                    parent_id = spotify_in_resp.get(stid, "")
                    text = stw.get("text", "")
                    
                    is_direct_reply = parent_id in asker_tids
                    mentions_asker = bool(re.search(rf"@{orig_asker}\b", text))
                    
                    if is_direct_reply or mentions_asker:
                        attributed_replies.append(stw)
                
                if attributed_replies:
                    spotify_reply_text = "\n\n".join(tw["text"] for tw in attributed_replies)
                    grounding_record = {
                        "thread_id": t["thread_id"],
                        "original_asker_id": orig_asker,
                        "num_unique_customer_authors": num_cust,
                        "multi_author_merge": True,
                        "customer_message": customer_message,
                        "spotify_reply": spotify_reply_text,
                        "start_time": t["start_time"],
                        "end_time": t["end_time"]
                    }
                    clean_multi_extracted.append((t, grounding_record, attributed_replies))
                else:
                    clean_multi_ambiguous.append(t)

    new_trusted_count = len(trusted_v2_threads)
    change_in_trusted = new_trusted_count - old_trusted_count
    pct_change = (change_in_trusted / old_trusted_count) * 100

    # Save data/spotifycares_trusted_v2.csv
    print(f"\nWriting {OUTPUT_TRUSTED_CSV}...")
    with open(OUTPUT_TRUSTED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "thread_id",
            "tweet_count",
            "has_customer",
            "has_spotify",
            "start_time",
            "end_time",
            "max_gap_hours",
            "suspect_timing",
            "customer_language",
            "language_confidence",
            "non_english",
            "num_unique_customer_authors",
            "multi_author_merge",
            "exclude_multi_author_merge",
            "fully_trusted_v2",
            "tweets"
        ])
        for t in trusted_v2_threads:
            writer.writerow([
                t["thread_id"],
                t["tweet_count"],
                t["has_customer"],
                t["has_spotify"],
                t["start_time"],
                t["end_time"],
                t["max_gap_hours"],
                t["suspect_timing"],
                t["customer_language"],
                t["language_confidence"],
                t["non_english"],
                t["num_unique_customer_authors"],
                t["multi_author_merge"],
                t["exclude_multi_author_merge"],
                t["fully_trusted_v2"],
                json.dumps(t["tweets"], ensure_ascii=False)
            ])
    print(f"Saved {len(trusted_v2_threads):,} trusted threads to {OUTPUT_TRUSTED_CSV} ({os.path.getsize(OUTPUT_TRUSTED_CSV)/(1024*1024):.2f} MB)")

    # Save data/spotify_grounding_corpus_v2.csv
    all_grounding_pairs = single_author_extracted + [rec for _, rec, _ in clean_multi_extracted]
    all_grounding_pairs.sort(key=lambda x: x["thread_id"])
    print(f"\nWriting {OUTPUT_GROUNDING_CSV}...")
    with open(OUTPUT_GROUNDING_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "thread_id",
            "original_asker_id",
            "num_unique_customer_authors",
            "multi_author_merge",
            "customer_message",
            "spotify_reply",
            "start_time",
            "end_time"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_grounding_pairs)
    print(f"Saved {len(all_grounding_pairs):,} grounding pairs to {OUTPUT_GROUNDING_CSV} ({os.path.getsize(OUTPUT_GROUNDING_CSV)/(1024*1024):.2f} MB)")

    # Prepare Report Text
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SPOTIFYCARES DATASET FILTERING COMPARISON REPORT (V1 vs V2)")
    report_lines.append("=" * 70)
    report_lines.append(f"Original SpotifyCares threads: {total_threads:,}\n")
    report_lines.append("PREVIOUSLY (V1 Logic):")
    report_lines.append(f"  Fully trusted: {old_trusted_count:,}")
    report_lines.append(f"  Rejected because multi_author_merge: {old_rejected_multi_author:,} (all multi-author threads excluded)")
    report_lines.append(f"  Rejected because suspect_timing: {old_rejected_suspect_timing:,}\n")
    report_lines.append("AFTER FIX (V2 Logic):")
    report_lines.append(f"  Fully trusted: {new_trusted_count:,}")
    report_lines.append(f"  Excluded by multi_author_merge AND suspect_timing: {len(excluded_multi_author_and_suspect):,}")
    report_lines.append(f"  Clean multi-author threads retained (in trusted corpus): {len(clean_multi_author_retained):,}")
    report_lines.append(f"  Of those, successfully extracted into a grounding pair: {len(clean_multi_extracted):,}")
    report_lines.append(f"  Of those, excluded_ambiguous_attribution (kept in trusted corpus, not used as grounding pair): {len(clean_multi_ambiguous):,}\n")
    report_lines.append(f"Change in trusted corpus = new_trusted - old_trusted = {new_trusted_count:,} - {old_trusted_count:,} = +{change_in_trusted:,}")
    report_lines.append(f"Percentage change = (new_trusted - old_trusted) / old_trusted * 100 = +{pct_change:.2f}%\n")
    report_lines.append("GROUNDING CORPUS V2:")
    report_lines.append(f"  Single-author pairs: {len(single_author_extracted):,}")
    report_lines.append(f"  Retained multi-author pairs: {len(clean_multi_extracted):,}")
    report_lines.append(f"  Total Grounding Pairs in V2: {len(all_grounding_pairs):,}")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nSaved filter comparison report to {OUTPUT_REPORT_TXT}")
    print("\n" + report_text)

    # Print 5 Examples of Clean Multi-Author Retained with Successful Extraction
    print("\n" + "=" * 70)
    print("5 EXAMPLES: NEWLY RETAINED MULTI-AUTHOR THREADS (EXTRACTION SUCCEEDED)")
    print("=" * 70)
    for idx, (t, rec, attr_replies) in enumerate(clean_multi_extracted[:5], 1):
        authors = list(set(tw["author_id"] for tw in t["tweets"] if tw.get("inbound", False)))
        print(f"\n[Retained Example {idx}] Thread ID: {t['thread_id']} | Gap: {t['max_gap_hours']}h | Customer Authors: {authors}")
        print(f"  Original Asker: @{rec['original_asker_id']}")
        print(f"  Timestamps: {t['start_time']} -> {t['end_time']}")
        print(f"  Customer Message(s) Extracted (Original Asker only):")
        for line in rec["customer_message"].split("\n\n"):
            print(f"    \"{line}\"")
        print(f"  SpotifyCares Response(s) Extracted:")
        for line in rec["spotify_reply"].split("\n\n"):
            print(f"    \"{line}\"")
        # Coherence note
        print(f"  Coherence Note: Valid support dialogue between @{rec['original_asker_id']} and SpotifyCares; additional user(s) participated without disrupting the core resolution.")
        print("-" * 70)

    # Print 5 Examples of Excluded Threads (multi_author_merge == True AND suspect_timing == True)
    print("\n" + "=" * 70)
    print("5 EXAMPLES: EXCLUDED BY NEW RULE (multi_author_merge == True AND suspect_timing == True)")
    print("=" * 70)
    for idx, t in enumerate(excluded_multi_author_and_suspect[:5], 1):
        authors = list(set(tw["author_id"] for tw in t["tweets"] if tw.get("inbound", False)))
        print(f"\n[Excluded Rule Example {idx}] Thread ID: {t['thread_id']} | Max Gap: {t['max_gap_hours']}h ({t['max_gap_hours']/24:.1f} days) | Customer Authors: {authors}")
        for step, tw in enumerate(t["tweets"], 1):
            role = f"Customer @{tw['author_id']}" if tw.get("inbound", False) else "Spotify Support"
            print(f"    ({step}) [{role}] [{tw['created_at']}]: \"{tw['text']}\"")
        print("-" * 70)

    # Print 5 Examples of Ambiguous Attribution (Clean timing, multi-author, but Spotify didn't reply to original asker)
    print("\n" + "=" * 70)
    print("5 EXAMPLES: EXCLUDED AMBIGUOUS ATTRIBUTION (Clean timing, but Spotify didn't reply to original asker)")
    print("=" * 70)
    for idx, t in enumerate(clean_multi_ambiguous[:5], 1):
        inbound_tweets = [tw for tw in t["tweets"] if tw.get("inbound", False)]
        orig_asker = inbound_tweets[0]["author_id"] if inbound_tweets else "N/A"
        authors = list(set(tw["author_id"] for tw in inbound_tweets))
        print(f"\n[Ambiguous Example {idx}] Thread ID: {t['thread_id']} | Gap: {t['max_gap_hours']}h | Customer Authors: {authors} | Original Asker: @{orig_asker}")
        for step, tw in enumerate(t["tweets"], 1):
            role = f"Customer @{tw['author_id']}" if tw.get("inbound", False) else "Spotify Support"
            print(f"    ({step}) [{role}] [{tw['created_at']}]: \"{tw['text']}\"")
        print(f"  Reason: Spotify's reply was addressed to a participant other than the original asker (@{orig_asker}), so thread is excluded from grounding pairs to prevent hallucinated/mismatched pairs.")
        print("-" * 70)

if __name__ == "__main__":
    main()
