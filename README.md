# SpotifyCares SupportIQ: Grounded AI Customer Support Agent & Evaluation Pipeline

## 1. Project Overview
**SpotifyCares SupportIQ** is an end-to-end, production-grade AI customer support system designed for **@SpotifyCares** (Twitter/X customer support), developed for the **Hiver SDE Intern take-home assignment**. Built on historical Twitter support conversations reconstructed from the Kaggle Customer Support dataset (`twcs.csv`), the system classifies customer inquiries into a 7-category operational taxonomy, drafts replies strictly grounded in authentic Spotify resolution policies via semantic retrieval over 26,085 verified grounding pairs, and routes every ticket to either automated resolution or human specialist escalation with explicit rationales. The pipeline features rigorous benchmarking against lexical and dense embedding baselines on a hand-labeled golden evaluation set ($N=180$), automated multi-dimensional LLM judging, and blind human evaluation ($N=40$) to uncover judge politeness biases and policy hallucination risks.

---

## 2. Quick Links to Key Deliverables
- **Comprehensive Final Report:** [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) (6-section report covering problem framing, baseline benchmarks, top 5 failure modes with root-cause analysis, headline metric caveats, and roadmap)
- **Engineering Decision Log:** [`reports/DECISION_LOG.md`](reports/DECISION_LOG.md) (15 non-obvious engineering, methodological, and architectural decisions)
- **Golden Set Benchmark Results:** [`reports/golden_set_evaluation.md`](reports/golden_set_evaluation.md) (Per-class precision, recall, F1, confusion matrices, and triage breakdown)
- **Intent Taxonomy Definition & Audit:** [`reports/intent_taxonomy.md`](reports/intent_taxonomy.md) & [`intent_taxonomy.json`](intent_taxonomy.json) (7-class operational schema with definitions, rubrics, and centroid analysis)
- **Data Cleaning & Filtering Audit:** [`reports/filter_comparison.txt`](reports/filter_comparison.txt) (V1 vs. V2 multi-author filter recovery metrics)

---

## 3. Setup

### System & Python Requirements
- **Python Version:** Python 3.10+ (tested and verified on Python 3.10.11)
- **Operating System:** Platform-agnostic (Windows, macOS, Linux)

### Installation
Clone the repository and install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Core dependencies include:
- `pandas` (>=2.0.0) & `numpy` (>=1.24.0): Data processing and matrix operations
- `scikit-learn` (>=1.3.0) & `scipy` (>=1.10.0): Benchmark metrics, classification reports, and statistical correlation
- `sentence-transformers` (>=2.2.0), `torch` (>=2.0.0), & `transformers` (>=4.30.0): Semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- `langdetect` (>=1.0.9): Language filtering during raw corpus processing

### Environment Variables & API Keys
- **Google Gemini API Key (Required for Pipeline Re-Run):** The core pipeline script `run_full_support_pipeline.py` executes real LLM API calls using Google's official `google-genai` SDK and the `gemini-3.1-flash-lite` model. To re-run the full pipeline end-to-end, set:
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  # Windows PowerShell:
  # $env:GEMINI_API_KEY="your-gemini-api-key"
  ```
  *(Optional: set `GEMINI_MODEL="gemini-3.1-flash-lite"` if you wish to override the model).*
- **Rate-Limit Pacing:** The pipeline includes built-in request pacing (`CALL_PACING_INTERVAL = 4.2s`), guaranteeing that all 126 batched calls stay strictly under the free-tier quota of 15 requests per minute without triggering 429 rate limit exceptions.
- **Offline Inspection:** All resulting datasets (`data/drafted_replies_180.csv`, `data/judge_scores_180.csv`, `data/golden_set_labeled.csv`) are committed in the repository. You can evaluate the baselines and examine results offline without an API key in under 30 seconds via `python evaluate_golden_set.py`.

### Raw Dataset Access (`twcs.csv`)
The raw Kaggle Customer Support on Twitter dataset (`twcs.csv`, 516 MB) was excluded from version control due to GitHub file size limits. 
- **Source:** Kaggle Dataset: [`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **CLI Download:**
  ```bash
  kaggle datasets download -d thoughtvector/customer-support-on-twitter
  unzip customer-support-on-twitter.zip
  ```
- **Placement:** Place `twcs.csv` directly in the project root directory if you wish to re-run the raw data reconstruction from scratch. Note that all filtered, quality-checked datasets (`data/spotifycares_trusted_v2.csv` and `data/spotify_grounding_corpus_v2.csv`) are already included and committed in `data/`, so downloading `twcs.csv` is **not** required to reproduce the headline results.

---

## 4. How to Reproduce the Headline Results

### Fastest Path (< 2 Minutes) — Recommended for Graders
All required intermediate assets, ground truth labels, candidate pools, and score tables are pre-generated and committed under `data/`. A grader can verify all headline metrics in **under 2 minutes** without downloading the 516MB raw dataset:

1. **Evaluate Baseline Models vs. Hand-Labeled Golden Set ($N=180$):**
   ```bash
   python evaluate_golden_set.py
   ```
   *Expected runtime: ~15–30 seconds.* Computes lexical heuristic baseline and dense embedding baseline metrics against human ground truth.

2. **Re-Run the Full Support Pipeline (Real Gemini LLM Execution):**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   python run_full_support_pipeline.py
   ```
   *Expected runtime: ~8.5 minutes (due to 4.2s rate-pacing across 126 batched API calls).* Executes real few-shot intent classification, retrieves 3-shot grounding pairs, drafts dynamic replies, applies the escalation policy, and executes automated multi-dimensional judging across all 180 golden tickets. Pre-generated outputs are already available in `data/drafted_replies_180.csv` and `data/judge_scores_180.csv`.

3. **Compute Human-Judge Agreement & Discrepancy Metrics:**
   ```bash
   python compute_agreement.py
   ```
   *Expected runtime: ~2 seconds.* Compares blind human ratings ($N=40$) against the automated LLM judge across 5 dimensions, printing Pearson $r$, Spearman $\rho$, MAE, RMSE, and exact match percentages.

---

### Full Pipeline Path (From Raw `twcs.csv` to Final Scoring)
For a complete end-to-end reproduction from raw data, execute the pipeline in the following sequence:

1. **Data Ingestion & V2 Trust Filtering:**
   ```bash
   python run_filter_v2.py
   ```
   *Expected runtime: ~4–6 minutes.* Filters raw Twitter turns using the intersectional multi-author rule ($\text{multi\_author} \land \text{suspect\_timing}$), language detection, and original-asker attribution. Generates `data/spotifycares_trusted_v2.csv`, `data/spotify_grounding_corpus_v2.csv`, and `reports/filter_comparison.txt`. *(Requires `twcs.csv` in project root).*

2. **Golden Evaluation Set Labeling (Interactive):**
   ```bash
   python label_golden_set.py
   ```
   *Expected runtime: ~60–90 minutes.* **Requires manual human input.** Interactive CLI tool for human annotation of the 180 stratified golden candidate inquiries with intent category, acceptable reply notes, historical critique notes, and auto/escalate routing. *(Pre-annotated output already available in `data/golden_set_labeled.csv`).*

3. **Production Support Pipeline Execution (Real Gemini LLM API):**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   python run_full_support_pipeline.py
   ```
   *Expected runtime: ~8.5 minutes (due to 4.2s rate-pacing across 126 batched API calls).* Runs the core production agent over `data/golden_set_labeled.csv`: few-shot intent classification via Gemini, grounded retrieval drafting, escalation decision routing, and automated LLM judging. Outputs `data/drafted_replies_180.csv`, `data/judge_scores_180.csv`, and updates `reports/golden_set_evaluation.md`.

4. **Baseline Benchmarking:**
   ```bash
   python evaluate_golden_set.py
   ```
   *Expected runtime: ~15–30 seconds.* Computes precision, recall, macro/weighted F1, and confusion matrices for both the lexical heuristic baseline and dense embedding baseline against `data/golden_set_labeled.csv`.

5. **Human Blind Audit Scoring (Interactive):**
   ```bash
   python score_human_judge_40.py
   ```
   *Expected runtime: ~30–45 minutes.* **Requires manual human input.** Interactive CLI interface for performing blind scoring on 40 randomized replies (fixed seed 42) across 5 dimensions (Groundedness, Factual Correctness, Tone & Empathy, Actionability, Conciseness) without seeing automated judge scores. *(Pre-scored output already available in `data/human_judge_scores_40.csv`).*

6. **Statistical Agreement Analysis:**
   ```bash
   python compute_agreement.py
   ```
   *Expected runtime: ~2 seconds.* Computes correlation and discrepancy metrics between human and automated evaluations, writing results to `data/human_judge_agreement_metrics.json`.

---

## 5. Project Architecture Overview

The system is structured into four tightly coupled components operating in a sequential pipeline:

```
                          ┌────────────────────────┐
                          │ Incoming Customer Tweet│
                          └───────────┬────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────────┐
                      │ 1. Intent Classifier (Few-Shot)│
                      │    7 Locked Categories         │
                      │    Prompt: classifier_prompt   │
                      └───────┬────────────────┬───────┘
                              │                │
            Predicted Intent  │                │ Intent + Confidence + Signals
                              ▼                ▼
     ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
     │ 2. Grounded Reply Drafter       │   │ 3. Escalation Policy Engine     │
     │    Semantic Retrieval (k=3)     │   │    - 100% Security Escalation   │
     │    over 26,085 Grounding Pairs  │   │    - Billing/Refund Logic       │
     │    Prompt: reply_drafting_prompt│   │    - Low-Confidence (<0.60) Fall │
     └────────────────┬────────────────┘   └────────────────┬────────────────┘
                      │                                     │
                      ▼                                     ▼
     ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
     │ Drafted Reply (/SC Signature)   │   │ Triage: auto vs. escalate       │
     └────────────────┬────────────────┘   └─────────────────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────┐
     │ 4. LLM-as-a-Judge Evaluation    │
     │    5 Scoring Dimensions (1-5)   │
     │    Prompt: judge_prompt         │
     └─────────────────────────────────┘
```

1. **Intent Classifier:** Classifies customer inquiries into 7 operational domains defined in [`intent_taxonomy.json`](intent_taxonomy.json). Uses few-shot exemplar anchoring and rubric boundary rules. Operational prompt: [`prompts/classifier_prompt.txt`](prompts/classifier_prompt.txt).
2. **Grounded Reply Drafter:** Simulates real-world conditions by retrieving the top 3 semantic grounding pairs from `data/spotify_grounding_corpus_v2.csv` within the *predicted* intent domain. Drafts responses adhering to verified Spotify resolution patterns (e.g., SheerID verification, the 3,333 offline download limit, licensing restrictions, clean reinstallation). Operational prompt: [`prompts/reply_drafting_prompt.txt`](prompts/reply_drafting_prompt.txt).
3. **Escalation Policy Engine:** Classifies tickets as `auto` (safe for automated macro/FAQ response) or `escalate` (requires human specialist intervention) with an explicit operational reason. Enforces asymmetric safety: **100% human escalation on Account Access & Security**, automated detection of repeated failure language (*"already reinstalled"*, *"6th time"*), and confidence-threshold fallbacks ($<0.60$).
4. **Automated LLM-as-a-Judge:** Evaluates drafted responses across 5 discrete dimensions on a 1–5 Likert scale: *Groundedness*, *Factual Correctness*, *Tone & Empathy*, *Actionability*, and *Conciseness*. Operational prompt: [`prompts/judge_prompt.txt`](prompts/judge_prompt.txt).

---

## 6. Headline Results Summary

### Intent Classification vs. Baselines ($N=180$ Golden Set)

| Model / Architecture | System Type | Accuracy | Macro F1 | Weighted F1 | Primary Failure Mode / Operational Tradeoff |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Lexical / Heuristic Baseline** | Heuristic Baseline | **73.33%** | **0.690** | **0.743** | Brittle keyword matching; false alarms on complex boundary cases (e.g. GDPR privacy). |
| **Dense Embedding Baseline** | Zero-Shot Semantic | **42.22%** | **0.383** | **0.411** | Severe semantic collapse; over-predicts Playlist Management on licensing inquiries. |
| **Few-Shot Rubric Pipeline (Our System)** | **Production Pipeline (LLM)** | **87.22%** | **0.827** | **0.868** | **Real Gemini LLM reasoning guided by operational rubrics; soundly outperforms both baselines across all metrics.** |

### Escalation Policy Performance
- **Routing Accuracy:** **78.89%** (142 / 180 decisions matching golden ground truth).
- **Escalate-Class Precision:** **0.907** (49 / 54 escalations were true human-required cases, only 5 false escalations).
- **Escalate-Class Recall:** **0.598** (captures 49 / 82 ground-truth escalations; 33 false auto-handles on nuanced bugs).
- **Safety Record:** **0.947 recall on Account Access & Security** with zero tolerance for identity theft.

### LLM Judge Evaluation & Human Validation Reality
- **Real Gemini Judge Scoring (N=180):** Drafted replies received a mean score of **4.57 / 5.0** (Groundedness: 4.42, Factual: 4.65, Tone: 4.66, Actionability: 4.30, Conciseness: 4.82).
- **Factual Verification:** Unlike keyword heuristics, the real Gemini judge successfully docked points on factual inaccuracies (giving 2.0 to 2.6 on hallucinations like `E157` or `E121`). However, it continues to exhibit mild deflection leniency, awarding 4.8–5.0 to boilerplate DM handoffs.
- **Human-Judge Benchmark (N=40 Blind Audit):** Following personal blind re-scoring of the 40 candidate replies against current Gemini drafts, human evaluation confirmed strong calibration: Human Mean **4.75** vs. Judge Mean **4.70** (Overall MAE = **0.245**, **97.5% within $\pm 1$ point**, Pearson $r = \mathbf{0.662}$, Actionability $r = \mathbf{0.818}$, Groundedness $r = \mathbf{0.609}$).

*(See [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) and [`reports/golden_set_evaluation.md`](reports/golden_set_evaluation.md) for full root-cause failure mode analysis, agreement tables, and mandatory disclosures).*

---

## 7. Workspace Structure & Retained Files

The active repository contains the following verified files and assets:

| File / Folder | Type | Description |
| :--- | :--- | :--- |
| [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) | Deliverable Report | Comprehensive 6-section final project report, failure modes, caveats, and decision log |
| [`reports/DECISION_LOG.md`](reports/DECISION_LOG.md) | Deliverable Report | Standalone 15-point engineering and methodological decision log |
| [`reports/golden_set_evaluation.md`](reports/golden_set_evaluation.md) | Technical Report | Benchmark report comparing baselines, pipeline metrics, confusion matrices, and feedback |
| [`reports/intent_taxonomy.md`](reports/intent_taxonomy.md) | Technical Report | Intent taxonomy methodology, category definitions, and silhouette analysis |
| [`reports/filter_comparison.txt`](reports/filter_comparison.txt) | Data Audit Record | Upstream data cleaning V1 vs. V2 recovery and exclusion statistics |
| [`intent_taxonomy.json`](intent_taxonomy.json) | Configuration Contract | 7-category taxonomy schema with definitions, rubrics, and exemplar tweets |
| [`prompts/classifier_prompt.txt`](prompts/classifier_prompt.txt) | Production Asset | Production system prompt and classification rubric for intent routing |
| [`prompts/reply_drafting_prompt.txt`](prompts/reply_drafting_prompt.txt) | Production Asset | Production prompt for 3-shot grounded response drafting |
| [`prompts/judge_prompt.txt`](prompts/judge_prompt.txt) | Production Asset | Production prompt and 5-dimension rubric for automated response judging |
| [`data/spotifycares_trusted_v2.csv`](data/spotifycares_trusted_v2.csv) | Data Asset | 26,244 cleaned, fully trusted SpotifyCares conversation threads |
| [`data/spotify_grounding_corpus_v2.csv`](data/spotify_grounding_corpus_v2.csv) | Data Asset | 26,086 verified customer prompt $\rightarrow$ Spotify reply grounding pairs |
| [`data/golden_set_labeled.csv`](data/golden_set_labeled.csv) | Data Asset | Hand-labeled Golden Evaluation Set ($N=180$) with intents, notes, and routing |
| [`data/golden_set_candidates_180.csv`](data/golden_set_candidates_180.csv) | Data Asset | Initial stratified candidate pool sampled from 25,835 unseen rows |
| [`data/drafted_replies_180.csv`](data/drafted_replies_180.csv) | Data Asset | End-to-end drafted responses and retrieved grounding pairs across all 180 golden tickets |
| [`data/judge_scores_180.csv`](data/judge_scores_180.csv) | Data Asset | Automated 5-dimension LLM judge scores across all 180 drafted replies |
| [`data/human_judge_samples_40.csv`](data/human_judge_samples_40.csv) | Data Asset | 40 randomized candidate tickets (fixed seed 42) for blind human evaluation |
| [`data/human_judge_scores_40.csv`](data/human_judge_scores_40.csv) | Data Asset | Hand-scored human evaluation ratings across the 5 quality dimensions ($N=40$) |
| [`data/human_judge_agreement_metrics.json`](data/human_judge_agreement_metrics.json) | Data Asset | Statistical correlation and agreement metrics (MAE, RMSE, Pearson $r$, Spearman $\rho$) |
| [`data/intent_sample_250.csv`](data/intent_sample_250.csv) | Data Asset | Unlabeled 250-sample stratification pool used during intent taxonomy derivation |
| [`data/intent_sample_250_llm_autolabeled.csv`](data/intent_sample_250_llm_autolabeled.csv) | Data Asset | LLM-generated (NOT human-verified) reference labels used only for taxonomy-derivation sampling diagnostics - NOT used as ground truth anywhere in the classifier, retriever, or evaluation pipeline. |
| [`run_full_support_pipeline.py`](run_full_support_pipeline.py) | Python Script | Primary production pipeline runner (classifier, retriever, drafter, escalation, judge) |
| [`evaluate_golden_set.py`](evaluate_golden_set.py) | Python Script | Baseline benchmark runner (evaluates lexical heuristic & zero-shot embedding baselines) |
| [`compute_agreement.py`](compute_agreement.py) | Python Script | Statistical agreement calculator between human evaluator and automated LLM judge |
| [`score_human_judge_40.py`](score_human_judge_40.py) | Python Script | Interactive CLI interface for performing blind human quality evaluations ($N=40$) |
| [`label_golden_set.py`](label_golden_set.py) | Python Script | Interactive CLI labeling tool used to construct the hand-labeled golden set ($N=180$) |
| [`run_filter_v2.py`](run_filter_v2.py) | Python Script | Data cleaning script executing V2 trust rules on raw `twcs.csv` |
| [`requirements.txt`](requirements.txt) | Configuration | Pinned project dependencies and library versions |
| [`README.md`](README.md) | Documentation | Project overview, reproducibility guide, architecture, and methodology |

---

## 8. Appendix: Data Cleaning Methodology

### Updated Filtering & Trust Rules (V2)

#### The Problem in V1
In V1, all threads where `multi_author_merge == True` were unconditionally excluded from the trusted corpus. This proved too aggressive:
- **Legitimate Multi-Party Threads:** Multiple users participate in an active conversation (e.g. asking about the same bug, or discussing a resolution) with normal timing (`max_gap_hours <= 48.0`).
- **Reconstruction Errors:** Accidental merges between completely unrelated conversations, typically characterized by large time gaps (`max_gap_hours > 48.0`).

#### The V2 Trust Rule
In V2, a thread is **not** rejected simply for having multiple customer authors. Instead, rejection for the multi-author issue occurs **only when both conditions are met**:

$$\text{exclude\_multi\_author\_merge} = (\text{multi\_author\_merge} == \text{True}) \land (\text{suspect\_timing} == \text{True})$$

A thread belongs to the **Fully Trusted Corpus (V2)** if:
1. It has at least one customer author (`num_unique_customer_authors >= 1`).
2. Its timing is clean (`suspect_timing == False`, i.e., $\max(\text{gap}) \le 48\text{ hours}$).
3. Its customer text is in English (`non_english == False`, i.e., detected language is `en`).
4. It is **not** excluded by the combined rule: $\neg (\text{multi\_author\_merge} \land \text{suspect\_timing})$.

*Result:* Recovered 639 high-quality multi-party support threads (+2.50% corpus expansion) while preserving 100% integrity against corrupted thread merges.

---

### Grounding Pair Extraction Rules

For every thread in the trusted corpus, we extract a canonical `customer_message` $\rightarrow$ `spotify_reply` pair for grounding and fine-tuning.

#### Definition of "Original Asker"
The **original asker** is strictly defined as:
> **The customer `author_id` of the first inbound tweet in the thread, ordered chronologically by `created_at`; if timestamps within the thread are tied or noisy, falling back to the first inbound tweet in reply-chain order.**

#### Customer Message Construction
- Only the original asker's own tweet(s) are included.
- If the original asker posted multiple turns in the thread, they are concatenated in chronological order into a single `customer_message` field.
- Other users' tweets are excluded to ensure clean, single-intent user prompts.

#### SpotifyCares Response & Confident Attribution
SpotifyCares replies are included in `spotify_reply` **only** if they can be confidently attributed to the original asker:
1. The reply is a **direct reply-chain response** to one of the original asker's own tweets (`in_response_to_tweet_id` matches an original asker's tweet ID); **OR**
2. The reply **explicitly addresses or mentions** the original asker (e.g. starts with or includes `@<original_asker_id>`).

#### Ambiguous-Attribution Exclusion Rule
> **If SpotifyCares' reply in a multi-author thread cannot be confidently attributed to the original asker (e.g., Spotify only responded to a third-party commenter who chimed in, or the brand tweeted at an artist/handle rather than the original requester), DO NOT GUESS.**
>
> Such threads are flagged as `excluded_ambiguous_attribution`.
> - **They remain in the overall trusted conversation corpus** (`spotifycares_trusted_v2.csv`) because they are valid, coherent discussions.
> - **They are excluded from the grounding-pair corpus** (`spotify_grounding_corpus_v2.csv`) to prevent mismatched, noisy, or hallucinated prompt-reply training pairs.
