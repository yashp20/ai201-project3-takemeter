# TakeMeter — Planning Document
### AI201 · Project 3 · r/nba Discourse Classifier

---

## 1. Community

**Chosen community:** r/nba (Reddit's NBA basketball community, ~7 million members)

**Why this community:**  
r/nba is a high-volume text community where discourse quality varies enormously and visibly. Within a single game thread you'll find a rigorously-argued tactical breakdown sitting next to "LeBron is washed change my mind" sitting next to "I CANNOT BELIEVE THAT SHOT." Those three post types are subjectively distinct to anyone who spends time in the subreddit — regular members actively complain about "bad takes flooding the thread" and praise "actual analysis" — yet they're hard to define precisely enough for a model to learn. The community generates hundreds of posts per game day, providing abundant labeled data, and the distinctions map neatly onto a classification task because post types differ structurally (evidence vs. assertion vs. emotional expression), not just tonally.

---

## 2. Label Taxonomy

### `analysis` (label 0)
**Definition:** The post makes a structured argument backed by specific statistics, historical comparison, or tactical observation. Evidence is precise and verifiable; the post reasons from data toward a conclusion rather than simply stating one.

**Example 1 (clear):**  
> "Jokic's playoff assist-to-turnover ratio improves from 3.2 in the regular season to 4.1 in the postseason, suggesting he deliberately tightens his decision-making under pressure."

**Example 2 (clear):**  
> "The Warriors' dynasty relied on a key spacing principle: Curry's off-ball gravity created 1.3 additional open corner threes per 100 possessions compared to when he operated as primary ball-handler."

---

### `hot_take` (label 1)
**Definition:** A bold, confident opinion stated without supporting evidence. The post asserts a claim rather than arguing for it — the framing is often contrarian, declarative, or provocative.

**Example 1 (clear):**  
> "LeBron James is the most overrated player in NBA history. His stats look good because he's played for 21 years, not because he's the GOAT."

**Example 2 (clear):**  
> "Wembanyama is the biggest bust in NBA history waiting to happen. He's too fragile to survive 82 games."

---

### `reaction` (label 2)
**Definition:** An immediate emotional response to a specific in-progress or just-completed event (a game, a play, a trade). Little to no argument — the post expresses a feeling in the moment and often uses excited or distressed language.

**Example 1 (clear):**  
> "I CANNOT believe what I just watched. That buzzer beater was the greatest shot I have ever seen in my entire life."

**Example 2 (clear):**  
> "Oh my god oh my god oh my god. They just blew a 20-point lead in the fourth quarter. I'm actually shaking right now."

---

## 3. Hard Edge Cases

### Primary ambiguity: hot_take vs. analysis

The boundary that required the most deliberate decision rules was between `hot_take` and `analysis`. Some posts include a statistic but frame it in an accusatory or cherry-picked way that mimics analysis without genuinely reasoning from evidence.

**Ambiguous example:**
> "LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."

This post cites a specific statistic. But the statistic is selected for rhetorical effect rather than as part of a broader argument: one cherry-picked metric doesn't constitute a case. The framing ("overrated") and the lack of additional evidence or reasoning mark it as assertion-with-decoration.

**Decision rule:** If removing the opinion framing would leave a self-contained argument that still supports the conclusion, label it `analysis`. If the "evidence" is a single selected metric or is vague enough that removing the opinion framing leaves nothing, label it `hot_take`.

### Secondary ambiguity: reaction vs. hot_take

Some posts express frustration in a way that includes an opinion about a player or team:

> "I'm so angry at LeBron for not taking the last shot. He's a coward."

This mixes emotional reaction with an evaluative claim. **Decision rule:** If the dominant purpose of the post is to express how the author feels about an event (anger, joy, disbelief), label it `reaction`, even if a claim is embedded. Reserve `hot_take` for posts whose primary purpose is the opinion itself, not the feeling.

### Documented difficult annotation cases

**Case 1:**  
> "Jokic's regular season MVPs are meaningless because he systematically underperforms in the playoffs when physical teams exploit his lack of quickness."

*Tension:* Makes a specific claim (playoff underperformance) but no statistics. The "exploit his lack of quickness" framing sounds tactical. **Decision:** `hot_take` — no specific evidence; "underperforms in the playoffs" is itself the asserted claim, not a proven premise.

**Case 2:**  
> "The entire Western Conference is a joke this year — any Eastern team could win the title."

*Tension:* Could be a frustrated reaction (to a bad game) or a genuine take. No event reference. **Decision:** `hot_take` — no specific event trigger, declarative opinion frame.

**Case 3:**  
> "Back to back wins after that losing streak and the energy is completely different. You can feel something shifting."

*Tension:* References a pattern (back-to-back wins, losing streak) but doesn't use it as evidence for an analytical claim. **Decision:** `reaction` — it's reflecting a feeling about a recent event sequence, not arguing toward a conclusion.

---

## 4. Data Collection Plan

**Source:** Posts and comments from r/nba, collected manually from:
- Game threads (reactions predominantly)
- Daily discussion threads (mix of all three)
- Hot posts filtered by "top" and "rising" (analysis and hot_takes predominantly)

**Target counts:** 75 `analysis`, 74 `hot_take`, 74 `reaction` = 223 total  
(Intentionally balanced to avoid class-imbalance problems)

**Imbalance contingency:** If any class falls below 60 examples after collection, specifically search for posts in that category. For `analysis`, search "statistically" or "historically" in the subreddit. For `reaction`, focus on game thread top comments.

**Format:** Single CSV file (`dataset.csv`) with columns `text` and `label`. The training notebook handles the 70/15/15 split automatically.

---

## 5. Evaluation Metrics

**Why accuracy alone is insufficient:**  
This is a 3-class task with balanced classes (roughly 33% each). A majority-class baseline achieves ~33% accuracy. Accuracy doesn't reveal which labels the model confuses or whether it's biased toward one class. With balanced data, macro-averaged F1 is the primary metric — it gives equal weight to each class and penalizes poor performance on any single label.

**Chosen metrics:**
- **Overall accuracy:** Directional comparison to baseline; intuitive summary
- **Per-class precision, recall, F1 (macro average):** Identifies which boundaries the model learned vs. which it confused
- **Confusion matrix:** Shows directionality of errors (e.g., does the model mistake `hot_take` for `analysis` or the reverse?)
- **Confidence calibration (stretch):** Whether high-confidence predictions are actually more accurate

**Success criteria:**  
The classifier is "good enough for deployment in a community tool" if:
- Fine-tuned accuracy ≥ 0.72 on the test set (>2× better than random chance on a 3-class problem)
- No single class F1 below 0.60 (the model can't simply ignore a label)
- Fine-tuned model beats zero-shot baseline by ≥ 0.10 accuracy

A macro-F1 below 0.60 or failure to beat the baseline would indicate the labels are too inconsistent for DistilBERT to learn reliably on 200 examples.

---

## 6. Definition of Success

A classifier is useful to a real community tool if it:
1. Correctly labels ≥72% of held-out posts (vs. ~33% baseline for this 3-class problem)
2. Does not catastrophically fail on any single class (all per-class F1 ≥ 0.60)
3. Meaningfully outperforms a zero-shot LLM baseline (≥10 point accuracy gain), demonstrating that fine-tuning captured community-specific patterns the LLM couldn't infer from its prompt alone

The classifier is **not** expected to handle ironic or highly contextual posts (e.g., a post that looks like a hot take but is quoting someone else sarcastically). Those edge cases require more data and potentially additional labels.

---

## 7. AI Tool Plan

### 7a. Label stress-testing
Before finalizing the taxonomy, I gave Claude the three label definitions and asked it to generate 10 posts that sit at the boundary between `hot_take` and `analysis`. Several of the generated examples were genuinely ambiguous — they included a single statistic framed in an accusatory way. This forced me to write the "cherry-picked metric" decision rule in Section 3 before annotation began, rather than discovering the ambiguity mid-collection.

### 7b. Annotation assistance
For roughly 30% of examples, I used Claude to pre-label a batch by providing the definitions and asking for one label per post. I then reviewed every pre-label before accepting it. Pre-labeling without review was explicitly avoided. For the hard edge cases (Section 3), I always overrode the pre-label with my own judgment. Pre-labeled examples are not specially flagged in the CSV because every entry was human-reviewed before inclusion.

### 7c. Failure analysis
After training, I will paste the list of wrong predictions (text, true label, predicted label) into Claude and ask it to identify common patterns — e.g., whether short posts cluster in errors, whether one label pair is systematically confused, whether sarcasm or hedged language correlates with failure. I will then verify each identified pattern manually by re-reading the relevant examples. The analysis section of the README will document both the AI-suggested patterns and my own verification or correction of them.

---

*Last updated: 2026-06-22*
