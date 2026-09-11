# SpotifyCares Project Decision Log

Below is the standalone record of the 15 key non-obvious engineering, methodological, and architectural decisions made throughout the lifecycle of this project:

1. **Selection of @SpotifyCares Brand Corpus:**
   Selected SpotifyCares from `twcs.csv` because Spotify operates a high-volume, pure digital subscription product with distinct technical, billing, licensing, and security boundaries, avoiding physical logistics complexities (e.g., lost packages or baggage tracking common in retail/airline datasets).

2. **Transition from V1 to V2 Multi-Author Trust Filter:**
   V1 unconditionally discarded all 1,073 multi-author threads, losing legitimate community discussions. V2 replaced this with an intersectional rule ($\text{multi\_author} \land \text{suspect\_timing}$), recovering 639 trusted threads (+2.50% corpus expansion).

3. **48-Hour Threshold for Suspect Timing:**
   Established 48.0 hours as the maximum inter-tweet gap cutoff; Twitter customer service interactions exceeding 48 hours almost universally represent unrelated conversational restarts or thread reconstruction errors rather than coherent single-issue resolutions.

4. **Strict Original-Asker Attribution Rule:**
   Defined the original asker strictly as the author of the first inbound tweet in the thread. To prevent training hallucinations, all customer text in `customer_message` is restricted exclusively to the original asker, discarding third-party interjections.

5. **Exclusion of Ambiguous Attribution Threads (158 Pairs):**
   Retained 158 multi-author threads in the trusted conversation corpus for reference, but explicitly excluded them from `spotify_grounding_corpus_v2.csv` because Spotify agents addressed third-party commenters rather than the primary requester.

6. **Exclusion of Non-English Tweets (~4% of Corpus):**
   Filtered out non-English customer tweets using automated language detection to preserve clean semantic alignment with English taxonomy rubrics and prevent translation-induced noise in few-shot retrieval.

7. **Intent Taxonomy Fixed at 7 Locked Categories:**
   Standardized on 7 categories after qualitative analysis revealed that clustering algorithms with $k > 7$ created fragmented, overlapping sub-intents (e.g., splitting Student vs. Family billing) that degraded classifier stability.

8. **Folding Free-Tier Ad Complaints into Subscription & Billing:**
   Explicitly mapped substantive ad complaints (frequency, volume jumps, disturbing horror ads) to Subscription & Billing because advertising is the foundational monetization boundary separating Free from Premium tiers.

9. **Treating 10,000-Song Limit as Playlist & Library Management:**
   Classified complaints regarding the 10,000 saved song cap under Playlist & Library Management rather than Feature Requests, as it represents an immediate functional restriction on active library curation.

10. **Independent Sampling for Golden Evaluation Set ($N=180$):**
    Sampled the 180 golden candidates from a fresh pool of 25,835 unseen rows (excluding the 250 examples used in taxonomy derivation), using a fixed seed of 123 for the initial 1,500-row screening/stratification pool, and a fixed seed of 789 for the final randomized shuffle of the 180 selected candidates, to eliminate data leakage and enable exact reproducibility.

11. **15-Candidate Minimum Floor per Stratum in Golden Set:**
    Enforced a minimum sampling floor of 15 candidates per intent category, ensuring low-frequency categories (such as Account Access and Playlist Management) had statistically significant support for evaluation.

12. **Simulating Real-World Deployment in Grounded Reply Retrieval:**
    In Part B, retrieval of grounding examples was driven strictly by the classifier's **predicted intent** rather than the ground-truth label, accurately simulating cascading errors in live production.

13. **Deterministic 100% Escalation for Account Access & Security:**
    Enforced mandatory human escalation on all Account Access & Security predictions regardless of classifier confidence, reflecting zero organizational tolerance for automated handling of identity theft.

14. **Prioritizing Groundedness as the Primary Human Evaluation Dimension:**
    Positioned groundedness (adherence to Spotify policy patterns) as the lead evaluation metric, recognizing that surface fluency is trivial for modern LLMs while policy fidelity is the true failure point.

15. **Blind Human Scoring Protocol ($N=40$, Seed 42):**
    Extracted 40 randomized drafted replies with fixed seed 42 and required personal human evaluation blind to automated LLM judge scores, ensuring an uncorrupted ground truth for judge calibration.
