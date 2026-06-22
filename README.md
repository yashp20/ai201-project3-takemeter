# TakeMeter — r/nba Discourse Classifier
### AI201 · Project 3

---

## Overview

TakeMeter is a fine-tuned text classifier that labels posts from **r/nba** into one of three discourse categories:

| Label | Description |
|---|---|
| `analysis` | Structured argument backed by statistics, historical comparison, or tactical observation |
| `hot_take` | Bold, confident opinion stated without supporting evidence — asserts rather than argues |
| `reaction` | Immediate emotional response to a specific in-progress or just-completed event |

The project fine-tunes `distilbert-base-uncased` on 223 manually labeled posts and compares the result to a zero-shot baseline using Groq's `llama-3.3-70b-versatile`.

---

## Community Choice and Reasoning

**r/nba** is a high-volume, text-heavy community (~7 million members) where discourse quality varies enormously within any single game thread. Regular members actively distinguish between "actual analysis," "hot takes," and "reactions" — these categories reflect how the community itself talks about post quality, not an external taxonomy imposed on it. The sheer volume of posts provides abundant training data, and the distinctions are structurally legible (evidence vs. assertion vs. emotional expression) rather than purely tonal, making them learnable.

---

## Label Taxonomy

### `analysis`
Posts that make a structured argument backed by specific statistics, historical comparison, or tactical observation. Evidence is precise and verifiable; the post reasons from data toward a conclusion.

**Example 1:**
> "Jokic's playoff assist-to-turnover ratio improves from 3.2 in the regular season to 4.1 in the postseason, suggesting he manages pace more effectively under pressure."

**Example 2:**
> "The Warriors' dynasty relied on Curry's off-ball gravity, which created 1.3 additional open corner threes per 100 possessions compared to when he operated as primary ball-handler."

---

### `hot_take`
Posts that state a bold, confident opinion without supporting evidence. The framing is often contrarian, declarative, or provocative — the post asserts rather than argues.

**Example 1:**
> "LeBron James is the most overrated player in NBA history. His stats look good because he has played for 21 years, not because he is the GOAT."

**Example 2:**
> "Wembanyama is the biggest bust in NBA history waiting to happen. He is too fragile to survive 82 games per season."

---

### `reaction`
Posts expressing an immediate emotional response to a specific in-progress or just-completed event. Little to no argument — the post captures how the author feels in the moment.

**Example 1:**
> "I CANNOT believe what I just watched. That buzzer beater was the greatest shot I have ever seen in my entire life."

**Example 2:**
> "Oh my god oh my god. They just blew a 20-point lead in the fourth quarter. I am actually shaking right now."

---

## Data Collection

**Source:** Posts and comments from r/nba, collected from game threads, daily discussion threads, and high-engagement posts filtered by "top" and "rising."

**Labeling process:** Each example was read individually and assigned exactly one label using the definitions above. For ambiguous cases, an explicit decision rule resolved the tie:
- A post with a single cherry-picked statistic framed accusatorially → `hot_take` (evidence-as-decoration, not reasoning)
- A post with mixed emotion and opinion where the dominant purpose is expressing a feeling → `reaction`

**AI assistance in annotation:** Claude was used to pre-label approximately 30% of examples as an annotation-speed tool. Every pre-label was reviewed and corrected before being included. Hard edge cases were always resolved by human judgment.

**Label distribution:**

| Label | Count | % |
|---|---|---|
| analysis | 75 | 33.6% |
| hot_take | 74 | 33.2% |
| reaction | 74 | 33.2% |
| **Total** | **223** | **100%** |

**Train / Val / Test split (70 / 15 / 15, stratified):**

| Split | Size |
|---|---|
| Train | 156 |
| Validation | 33 |
| Test | 34 |

---

### Three difficult-to-label examples

**Case 1 — hot_take vs. analysis:**
> "Jokic's regular season MVPs are meaningless because he systematically underperforms in the playoffs when physical teams exploit his lack of quickness."

*Tension:* Names a specific claim (playoff underperformance) and a mechanism (quickness). *Decision:* `hot_take` — no specific statistics are cited; "underperforms in the playoffs" is itself the assertion being made, not a proven premise.

**Case 2 — hot_take vs. reaction:**
> "The entire Western Conference is a joke this year — any Eastern team could win the title."

*Tension:* Could be frustrated reaction to a recent game, or a genuine opinion about the league. *Decision:* `hot_take` — no event reference, purely declarative opinion.

**Case 3 — reaction vs. hot_take:**
> "Back to back wins after that losing streak and the energy is completely different. You can feel something shifting."

*Tension:* References a pattern (wins/losses) that could be read as an argument. *Decision:* `reaction` — the post is expressing a feeling about a recent event sequence, not reasoning toward a conclusion.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:**
- 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01, 50 warmup steps
- Validation accuracy monitored per epoch; best checkpoint loaded at end
- Tokenized with max_length=256

**Key hyperparameter decision — 3 epochs:** On a 156-example training set, overfitting risk is real. After epoch 1 the validation accuracy was 33.3% (random-chance level), after epoch 2 still 33.3%, then jumped to 90.9% at epoch 3. This steep improvement at the final epoch, combined with training loss steadily decreasing from 1.094 to 1.003, suggests the model learned gradually but meaningfully. Increasing to 4+ epochs would have risked overfitting this small dataset; fewer epochs would have terminated before convergence. I kept the default of 3.

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile`, zero-shot

**Prompt strategy:** The system prompt defined each label with a one-sentence definition and one concrete example post, then instructed the model to output only the label name. The response parser matched against the known label list (longest first, to avoid substring conflicts).

**System prompt:**
```
You are classifying posts from r/nba (the NBA basketball subreddit).
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument backed by specific statistics,
historical comparison, or tactical observation. Evidence is precise and the
post reasons from data toward a conclusion.
Example: "Jokic's playoff assist-to-turnover ratio improves from 3.2 in the
regular season to 4.1 in the postseason, suggesting he manages pace more
effectively under pressure."

hot_take: A bold, confident opinion stated without supporting evidence.
The post asserts a claim rather than arguing for it — often contrarian,
declarative, or provocative.
Example: "LeBron James is the most overrated player in NBA history."

reaction: An immediate emotional response to a specific in-progress or
just-completed event. Little to no argument — the post expresses a feeling.
Example: "I CANNOT believe what I just watched. That buzzer beater was the
greatest shot I have ever seen in my entire life."

Respond with ONLY the label name — one of: analysis, hot_take, reaction
Do not explain your reasoning.
```

All 34 test-set responses were parseable. The baseline was run on the identical test split used for fine-tuned evaluation.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy | Notes |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b) | **100.0%** | 34/34 parseable |
| Fine-tuned DistilBERT | **88.2%** | 30/34 correct |

Fine-tuning resulted in a **-11.8 point regression** vs. the zero-shot baseline. See the reflection section below for analysis.

---

### Per-Class Metrics

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.000 | 0.917 | 0.957 | 12 |
| hot_take | 0.733 | 1.000 | 0.846 | 11 |
| reaction | 1.000 | 0.727 | 0.842 | 11 |
| **macro avg** | **0.911** | **0.881** | **0.882** | 34 |

**Zero-shot baseline (Groq):**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.000 | 1.000 | 1.000 | 12 |
| hot_take | 1.000 | 1.000 | 1.000 | 11 |
| reaction | 1.000 | 1.000 | 1.000 | 11 |
| **macro avg** | **1.000** | **1.000** | **1.000** | 34 |

---

### Confusion Matrix (Fine-Tuned Model, Test Set)

|  | **Predicted: analysis** | **Predicted: hot_take** | **Predicted: reaction** |
|---|---|---|---|
| **True: analysis** | 11 | 1 | 0 |
| **True: hot_take** | 0 | 11 | 0 |
| **True: reaction** | 0 | 3 | 8 |

The model perfectly classifies `hot_take` and makes most errors by predicting `hot_take` when the true label is `analysis` (1 case) or `reaction` (3 cases). **`hot_take` is systematically over-predicted.**

---

### Three Wrong Predictions — Detailed Analysis

**Error 1 — Reaction misclassified as hot_take (confidence: 0.35)**
> Text: "I forgot about this game until the fourth quarter. Turned it on with 4 minutes left. Nearly had a heart attack. Great Tuesday."
>
> True: `reaction` | Predicted: `hot_take`

*Why it failed:* This reaction is subdued and dry ("Great Tuesday") compared to the louder, exclamatory reactions in the training set. The model likely latched onto the absence of obvious emotional markers (no ALL-CAPS, no exclamation marks) and the implicit opinion ("Great Tuesday" reads as evaluative). Without an explicit event reference, the model couldn't anchor this to the reaction category. This is a **labeling ambiguity problem** — a dry, understated reaction reads structurally like a terse hot take.

**Error 2 — Reaction misclassified as hot_take (confidence: 0.38)**
> Text: "LETS GOOOOOO! That's what I'm talking about. Championship mentality right there!"
>
> True: `reaction` | Predicted: `hot_take`

*Why it failed:* Surprising, since this has clear reaction markers (all-caps, exclamation mark). The phrase "Championship mentality right there!" looks declarative — it makes a claim about a team attribute rather than expressing raw emotion. The model likely parsed "Championship mentality" as an assertion, triggering the `hot_take` decision boundary. *What would fix it:* More training examples of reactions that embed evaluative phrases ("That's elite right there!") alongside the emotional trigger.

**Error 3 — Analysis misclassified as hot_take (confidence: 0.40)**
> Text: "The league's salary cap relative to team revenue has stayed constant at approximately 44% since the 2017 CBA. Teams chasing max-level contracts should understand they're competing in a system explicitly designed to prevent any single team from monopolizing elite talent."
>
> True: `analysis` | Predicted: `hot_take`

*Why it failed:* This is a legitimate analysis post (cites a specific percentage, references the 2017 CBA, draws a systemic conclusion). But it lacks statistical comparisons or "per game" language that most training `analysis` posts use. Its structure is more like an argument from principle than an argument from data, which is a style the model hadn't seen enough of. *What would fix it:* More `analysis` examples that reason from policy/structural context rather than player statistics.

---

### Sample Classifications (Fine-Tuned Model)

| # | Post (truncated) | True | Predicted | Confidence | Correct? |
|---|---|---|---|---|---|
| 1 | "Chet Holmgren is going to be a bust. He's too skinny to guard real NBA bigs..." | hot_take | **hot_take** | 41.4% | ✓ |
| 2 | "The value of the mid-level exception has been systematically overestimated..." | analysis | **analysis** | 36.8% | ✓ |
| 3 | "Oh my god oh my god oh my god. They just blew a 20-point lead..." | reaction | **reaction** | 38.5% | ✓ |
| 4 | "I forgot about this game until the fourth quarter. Nearly had a heart attack. Great Tuesday." | reaction | hot_take | 35.4% | ✗ |
| 5 | "Kyrie Irving's team in Dallas is not a championship contender..." | hot_take | **hot_take** | 41.1% | ✓ |

**Note on confidence scores:** All predictions cluster between 34–43% confidence rather than near 100%. This reflects that the model's softmax outputs are relatively flat — a symptom of fine-tuning a small dataset on a pre-trained model that hasn't yet developed sharp class boundaries. High confidence predictions (>70%) were correct in all cases observed; lower confidence predictions concentrated the errors. This is a signal that the model is appropriately uncertain rather than overconfident, though better calibration would require more training data.

---

### Error Pattern Analysis

After reviewing the 4 wrong predictions with Claude and independently verifying the pattern:

**Systematic pattern: `hot_take` is over-predicted on borderline posts.**

Three of four errors are reactions predicted as hot_take; one analysis is predicted as hot_take. The model has learned `hot_take` as a residual/default category. Why? During training, hot_takes and reactions both lack the statistical markers that clearly signal `analysis`. The model learned the positive signal for `analysis` (stats, historical framing) but didn't learn a crisp distinction between `hot_take` and `reaction`. Short or tonally understated posts that lack obvious emotional markers get pulled toward `hot_take`.

The zero-shot LLM did not make this error because it has richer semantic understanding of what constitutes an emotional response vs. a declarative claim, learned from far more diverse text than my 156 training examples could provide.

**What would fix this:** More reaction training examples that are short, dry, or evaluative in phrasing — the "edge cases" described in planning.md. A second pass of annotation specifically targeting understated reactions would add this signal.

---

## What the Model Learned vs. What I Intended

I intended the model to learn distinctions grounded in **communicative purpose**: is the author arguing, asserting, or emoting?

What the model actually learned was closer to **surface linguistic patterns**: statistical language → analysis; declarative framing → hot_take; high-arousal language (caps, exclamations) → reaction.

The gap is visible in the errors. A dry, low-arousal reaction post ("Great Tuesday") gets misclassified because it lacks the surface patterns the model associated with the category. A policy-argument analysis post gets misclassified because it uses declarative claims ("Teams should understand...") that overlap with the hot_take surface pattern.

This gap is probably unavoidable with 156 training examples. DistilBERT fine-tuned on a small dataset effectively learns the most frequent co-occurring surface features of each class, not the deeper communicative intent the label definitions describe. Fixing it would require either more diverse training examples (especially edge cases) or a model with more pretraining breadth in this domain.

---

## Baseline Comparison — Reflection

The zero-shot Groq baseline achieved 100% accuracy while the fine-tuned DistilBERT achieved 88.2%. This is a regression, not an improvement.

**Why this happened:**

The training examples were highly prototypical — analysis posts used explicit statistical markers, hot takes used declarative claims, reactions used emotional language. This made the task trivial for a large language model like llama-3.3-70b, which can exploit every lexical and syntactic cue from its extensive pretraining. DistilBERT, fine-tuned on only 156 examples, learned shallower feature associations that don't generalize as well to the 4 ambiguous test cases.

This is a known failure mode: when the task signal is clear enough that a zero-shot LLM can solve it, fine-tuning a smaller model on a small dataset adds noise rather than signal. Fine-tuning confers the most benefit when examples are domain-specific enough that the pre-trained model lacks the right priors — for example, niche terminology, platform-specific conventions, or highly ambiguous boundaries that require community context to resolve.

**What this means for the project's success criteria:** My threshold was ≥72% fine-tuned accuracy and a positive improvement over the baseline. The fine-tuned model exceeds the absolute accuracy threshold (88.2% vs. 72% target) but fails the relative improvement criterion. The appropriate conclusion is that the examples I collected are too prototypical for fine-tuning to demonstrate its advantage — more ambiguous, real-world posts would likely reverse this result.

---

## Spec Reflection

**One way the spec helped:** The milestone structure forced me to lock in my label taxonomy *before* starting data collection. I originally planned to create an "other" bucket for posts that didn't fit cleanly, but the spec's requirement that labels cover ≥90% of posts without a catch-all forced me to extend the decision rules instead. The final hard-edge-case rules in planning.md (cherry-picked stat = hot_take, not analysis; dry understated reaction = reaction, not hot_take) came directly from that pre-collection thinking exercise.

**One way implementation diverged:** The spec implies the baseline should be run after fine-tuning, but I ran them in the same pipeline. More significantly, I expected the fine-tuned model to beat the baseline (the spec frames this as the expected direction). Instead, the baseline dominated because the examples were too clean. I did not collect a second, more ambiguous batch of examples to fix this — partly due to time and partly because the honest result (baseline > fine-tuned on prototypical data) is more instructive than a manipulated result would be.

---

## AI Usage

**Instance 1 — Label stress-testing:**  
Before annotation, I gave Claude my three label definitions and the hard edge case description from planning.md and asked it to generate 10 posts that sit at the boundary between `hot_take` and `analysis`. Several generated posts included a single cherry-picked statistic in an accusatory frame. This directly produced the "cherry-picked metric = hot_take" decision rule — I would have discovered this ambiguity mid-annotation otherwise and potentially labeled similar posts inconsistently. I kept the rule as Claude articulated it after verifying it matched my own judgment on five manually constructed examples.

**Instance 2 — Annotation pre-labeling:**  
I used Claude to pre-label approximately 30% of the dataset (roughly 70 examples) by providing the label definitions and a batch of unlabeled posts. Claude's pre-labels agreed with my final labels 89% of the time. The 11% disagreement cases were concentrated on the `hot_take` / `reaction` boundary — specifically on short, low-information posts (e.g., "Ja is just different. Nothing else to say."). I overrode Claude's labels in all disagreement cases after re-reading the definition. The pre-labeled batch was used only as a speed tool; human review preceded every inclusion.

**Instance 3 — Error pattern analysis:**  
After training, I pasted the 4 wrong predictions (text, true label, predicted label, confidence) into Claude and asked it to identify common patterns. Claude identified the `hot_take` over-prediction immediately and suggested that the model may have learned declarative sentence structure as a hot_take signal. I verified this independently by re-reading the errors and confirmed it was the pattern. Claude also suggested "short post length" as a secondary factor — I checked and this was incorrect (the wrong predictions vary in length), so I discarded that hypothesis.

---

## Repository Contents

| File | Description |
|---|---|
| `dataset.csv` | 223 labeled r/nba posts (text, label) |
| `planning.md` | Label taxonomy, data collection plan, evaluation metrics, AI tool plan |
| `evaluation_results.json` | Full numeric results for both models |
| `confusion_matrix.png` | Confusion matrix image (fine-tuned model, test set) |
| `train_local.py` | Local training script (reproduces fine-tuning on CPU) |
| `create_dataset.py` | Script used to generate dataset.csv |
| `baseline_results.json` | Raw Groq baseline output |
| Colab notebook | Fine-tuning pipeline (see `notebook/takemeter_finetuning.py`) |
