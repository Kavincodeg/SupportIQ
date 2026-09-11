# SpotifyCares Dataset Filtering and Grounding Pipeline (V2)

## Overview
This repository contains the reconstructed, quality-checked, and filtered dataset of customer support conversations between Twitter users and **@SpotifyCares**, derived from the Kaggle Customer Support dataset (`twcs.csv`).

---

## 1. Updated Filtering & Trust Rules (V2)

### The Problem in V1
In V1, all threads where `multi_author_merge == True` were unconditionally excluded from the trusted corpus. This proved too aggressive:
- **Legitimate Multi-Party Threads:** Multiple users participate in an active conversation (e.g. asking about the same bug, or discussing a resolution) with normal timing (`max_gap_hours <= 48.0`).
- **Reconstruction Errors:** Accidental merges between completely unrelated conversations, typically characterized by large time gaps (`max_gap_hours > 48.0`).

### The V2 Trust Rule
In V2, a thread is **NOT** rejected simply for having multiple customer authors. Instead, rejection for the multi-author issue occurs **only when BOTH conditions are met**:

$$\text{exclude\_multi\_author\_merge} = (\text{multi\_author\_merge} == \text{True}) \land (\text{suspect\_timing} == \text{True})$$

A thread belongs to the **Fully Trusted Corpus (V2)** if:
1. It has at least one customer author (`num_unique_customer_authors >= 1`).
2. Its timing is clean (`suspect_timing == False`, i.e., $\max(\text{gap}) \le 48\text{ hours}$).
3. Its customer text is in English (`non_english == False`, i.e., detected language is `en`).
4. It is **not** excluded by the combined rule: $\neg (\text{multi\_author\_merge} \land \text{suspect\_timing})$.

---

## 2. Grounding Pair Extraction Rules

For every thread in the trusted corpus, we extract a canonical `customer_message` $\rightarrow$ `spotify_reply` pair for grounding and fine-tuning.

### Definition of "Original Asker"
The **original asker** is strictly defined as:
> **The customer `author_id` of the first inbound tweet in the thread, ordered chronologically by `created_at`; if timestamps within the thread are tied or noisy, falling back to the first inbound tweet in reply-chain order.**

### Customer Message Construction
- Only the original asker's own tweet(s) are included.
- If the original asker posted multiple turns in the thread, they are concatenated in chronological order into a single `customer_message` field.
- Other users' tweets are excluded to ensure clean, single-intent user prompts.

### SpotifyCares Response & Confident Attribution
SpotifyCares replies are included in `spotify_reply` **only** if they can be confidently attributed to the original asker:
1. The reply is a **direct reply-chain response** to one of the original asker's own tweets (`in_response_to_tweet_id` matches an original asker's tweet ID); **OR**
2. The reply **explicitly addresses or mentions** the original asker (e.g. starts with or includes `@<original_asker_id>`).

### Ambiguous-Attribution Exclusion Rule
> **If SpotifyCares' reply in a multi-author thread cannot be confidently attributed to the original asker (e.g., Spotify only responded to a third-party commenter who chimed in, or the brand tweeted at an artist/handle rather than the original requester), DO NOT GUESS.**
>
> Such threads are flagged as `excluded_ambiguous_attribution`.
> - **They remain in the overall trusted conversation corpus** (`spotifycares_trusted_v2.csv`) because they are valid, coherent discussions.
> - **They are excluded from the grounding-pair corpus** (`spotify_grounding_corpus_v2.csv`) to prevent mismatched, noisy, or hallucinated prompt-reply training pairs.

---

## 3. Workspace Structure & Retained Files

> [!NOTE]
> **Workspace Clean-Up:**
> The repository has been streamlined to contain only production pipeline scripts, evaluation harnesses, prompt definitions, verified datasets, and finalized reports.
> The raw 516MB Kaggle source (`twcs.csv`) and early one-off reconstruction scripts have been pruned.

The active workspace contains:

| File / Folder | Description | Purpose |
| :--- | :--- | :--- |
| [`reports/FINAL_REPORT.md`](file:///c:/Users/Kavin/Downloads/Hiver/reports/FINAL_REPORT.md) | Comprehensive 6-section final project report | Submission deliverable |
| [`reports/DECISION_LOG.md`](file:///c:/Users/Kavin/Downloads/Hiver/reports/DECISION_LOG.md) | Standalone 15-point engineering decision log | Submission deliverable |
| [`reports/golden_set_evaluation.md`](file:///c:/Users/Kavin/Downloads/Hiver/reports/golden_set_evaluation.md) | Detailed baseline vs. pipeline benchmark report | Evaluation documentation |
| [`reports/intent_taxonomy.md`](file:///c:/Users/Kavin/Downloads/Hiver/reports/intent_taxonomy.md) | Intent taxonomy derivation & silhouette audit report | Methodology documentation |
| [`reports/filter_comparison.txt`](file:///c:/Users/Kavin/Downloads/Hiver/reports/filter_comparison.txt) | Data cleaning V1 vs V2 recovery metrics | Data audit record |
| [`intent_taxonomy.json`](file:///c:/Users/Kavin/Downloads/Hiver/intent_taxonomy.json) | 7-class taxonomy schema with rubrics & exemplars | Intent definition contract |
| [`prompts/`](file:///c:/Users/Kavin/Downloads/Hiver/prompts) | Production prompts (`classifier`, `reply_drafting`, `judge`) | System prompt assets |
| [`data/`](file:///c:/Users/Kavin/Downloads/Hiver/data) | Ground truth datasets, grounding corpus, and evaluation scores | Core data assets (11 files) |
| [`run_full_support_pipeline.py`](file:///c:/Users/Kavin/Downloads/Hiver/run_full_support_pipeline.py) | End-to-end pipeline (classification, retrieval, drafting, judge) | Primary production pipeline |
| [`evaluate_golden_set.py`](file:///c:/Users/Kavin/Downloads/Hiver/evaluate_golden_set.py) | Lexical and embedding baseline benchmark runner | Baseline benchmarking |
| [`compute_agreement.py`](file:///c:/Users/Kavin/Downloads/Hiver/compute_agreement.py) | Human vs. LLM judge agreement & correlation metrics calculator | Validation harness |
| [`score_human_judge_40.py`](file:///c:/Users/Kavin/Downloads/Hiver/score_human_judge_40.py) | Interactive CLI scoring interface for human validation | Human audit tool |
| [`label_golden_set.py`](file:///c:/Users/Kavin/Downloads/Hiver/label_golden_set.py) | Interactive CLI labeling tool for the 180 golden set | Ground truth labeling tool |
| [`run_filter_v2.py`](file:///c:/Users/Kavin/Downloads/Hiver/run_filter_v2.py) | V2 filtering and grounding corpus extraction script | Corpus generation script |
| [`README.md`](file:///c:/Users/Kavin/Downloads/Hiver/README.md) | Repository documentation, methodology, and setup guide | Project documentation |

---

## 4. Pipeline Execution & Reproducibility

### Running the End-to-End Support Pipeline
To run the full support agent pipeline (classification, 3-shot grounded drafting, escalation policy, and automated LLM judging):
```bash
python run_full_support_pipeline.py
```

### Running the Classifier Benchmarks
To benchmark the lexical and dense embedding baselines against the hand-labeled golden set ($N=180$):
```bash
python evaluate_golden_set.py
```

### Computing Human-LLM Judge Agreement
To compute the statistical correlation (Pearson $r$, Spearman $\rho$, MAE, RMSE) between the human judge ($N=40$) and LLM judge:
```bash
python compute_agreement.py
```

