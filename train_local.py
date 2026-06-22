"""
Local fine-tuning script for TakeMeter (r/nba classifier).
Produces evaluation_results.json and confusion_matrix.png.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_MAP = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
MODEL_NAME = "distilbert-base-uncased"
CSV_PATH = "dataset.csv"

print(f"Labels: {LABEL_MAP}")
print(f"GPU available: {torch.cuda.is_available()}")

# ── Load & validate ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"\nTotal examples: {len(df)}")
print("Label distribution:")
print(df["label"].value_counts())

unknown = set(df["label"].unique()) - set(LABEL_MAP.keys())
assert not unknown, f"Unknown labels: {unknown}"
print("All labels match LABEL_MAP")

df["label_id"] = df["label"].map(LABEL_MAP).astype(int)

# ── Splits ────────────────────────────────────────────────────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
)

for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"{name}: {len(split)}")

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# ── Tokenize ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

def make_dataset(df_split):
    ds = Dataset.from_pandas(
        df_split[["text", "label_id"]].rename(columns={"label_id": "labels"})
    )
    return ds.map(tokenize, batched=True)

train_dataset = make_dataset(train_df)
val_dataset   = make_dataset(val_df)
test_dataset  = make_dataset(test_df)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Tokenization complete")

# ── Model ─────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_MAP,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./takemeter-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning...")
t0 = time.time()
trainer.train()
print(f"Fine-tuning complete in {time.time() - t0:.1f}s")

# ── Evaluate fine-tuned model ─────────────────────────────────────────────────
print("\nRunning inference on test set...")
ft_output = trainer.predict(test_dataset)
ft_pred_ids = np.argmax(ft_output.predictions, axis=-1)
ft_true_ids = ft_output.label_ids

ft_probs = torch.nn.functional.softmax(
    torch.tensor(ft_output.predictions), dim=-1
).numpy()

ft_accuracy = accuracy_score(ft_true_ids, ft_pred_ids)
print(f"\nFine-tuned model accuracy: {ft_accuracy:.4f}")

label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
report = classification_report(
    ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0, output_dict=True
)
print("\nPer-class metrics (fine-tuned):")
print(classification_report(ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0))

# Save confusion matrix
cm = confusion_matrix(ft_true_ids, ft_pred_ids)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fine-Tuned DistilBERT — Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# Print wrong predictions
wrong_idx = np.where(ft_pred_ids != ft_true_ids)[0]
print(f"\nWrong predictions: {len(wrong_idx)} / {len(ft_true_ids)}")
print("\n--- First 10 misclassified examples ---")
for i, idx in enumerate(wrong_idx[:10]):
    text = test_df.iloc[idx]["text"]
    true_label = ID_TO_LABEL[ft_true_ids[idx]]
    pred_label = ID_TO_LABEL[ft_pred_ids[idx]]
    conf = ft_probs[idx][ft_pred_ids[idx]]
    print(f"\n#{i+1}")
    print(f"Text:      {text[:180]}...")
    print(f"True:      {true_label}")
    print(f"Predicted: {pred_label}  (confidence: {conf:.2f})")

# ── Baseline: attempt Groq, fall back to majority-class ──────────────────────
try:
    import os
    from groq import Groq

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        raise ValueError("No GROQ_API_KEY")

    client = Groq(api_key=GROQ_API_KEY)

    SYSTEM_PROMPT = """You are classifying posts from r/nba (the NBA basketball subreddit).
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument backed by statistics, historical comparison, or tactical observation. Evidence is specific and the post reasons from data to a conclusion.
Example: "Jokic's assist-to-turnover ratio improves from 3.2 in the regular season to 4.1 in the playoffs, suggesting he manages pace more effectively under pressure."

hot_take: A bold, confident opinion stated without supporting evidence. The post asserts a claim rather than arguing for it.
Example: "LeBron James is the most overrated player in NBA history. Change my mind."

reaction: An immediate emotional response to a specific event. Little to no argument — the post expresses a feeling in the moment.
Example: "LETS GOOOOO! That buzzer beater was the greatest shot I have ever seen in my life!!!"

Respond with ONLY the label name — one of: analysis, hot_take, reaction
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction"""

    def classify_with_groq(text):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this post:\n\n{text}"},
                ],
                temperature=0,
                max_tokens=20,
            )
            raw = response.choices[0].message.content.strip().lower()
            for label in sorted(LABEL_MAP, key=len, reverse=True):
                if raw == label or label in raw:
                    return label
            return None
        except Exception as e:
            print(f"API error: {e}")
            return None

    print(f"\nRunning Groq baseline on {len(test_df)} examples...")
    baseline_preds = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = classify_with_groq(row["text"])
        baseline_preds.append(pred)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(test_df)}...")
        time.sleep(0.15)

    valid = [(p, t) for p, t in zip(baseline_preds, test_df["label_id"]) if p is not None]
    bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
    bl_true_ids = [t for _, t in valid]
    bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
    none_count = baseline_preds.count(None)
    print(f"\nBaseline accuracy: {bl_accuracy:.4f} ({len(valid)}/{len(test_df)} parseable)")
    if none_count:
        print(f"Unparseable: {none_count}")
    print("\nPer-class metrics (baseline):")
    print(classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0))
    bl_report = classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0, output_dict=True)

except Exception as e:
    print(f"\nGroq baseline skipped ({e}). Using majority-class fallback.")
    majority = test_df["label_id"].mode()[0]
    bl_pred_ids = [majority] * len(test_df)
    bl_true_ids = test_df["label_id"].tolist()
    bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
    print(f"Majority-class baseline accuracy: {bl_accuracy:.4f}")
    bl_report = {}

# ── Comparison & export ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"{'Model':<40} {'Accuracy':>8}")
print("-" * 50)
print(f"{'Baseline':<40} {bl_accuracy:>8.4f}")
print(f"{'Fine-tuned DistilBERT':<40} {ft_accuracy:>8.4f}")
print("-" * 50)
delta = ft_accuracy - bl_accuracy
print(f"\nFine-tuning {'improvement' if delta >= 0 else 'regression'}: {abs(delta):.4f}")

# Save per-class details for README
per_class = {}
for label in label_names:
    per_class[label] = {
        "precision": round(report[label]["precision"], 3),
        "recall":    round(report[label]["recall"], 3),
        "f1":        round(report[label]["f1-score"], 3),
        "support":   int(report[label]["support"]),
    }

# Confusion matrix as nested list for README
cm_list = cm.tolist()

results = {
    "baseline_accuracy":  round(bl_accuracy, 4),
    "finetuned_accuracy": round(ft_accuracy, 4),
    "improvement":        round(ft_accuracy - bl_accuracy, 4),
    "test_set_size":      len(test_df),
    "label_map":          LABEL_MAP,
    "model":              MODEL_NAME,
    "per_class_metrics":  per_class,
    "confusion_matrix":   {"labels": label_names, "matrix": cm_list},
    "wrong_predictions":  int(len(wrong_idx)),
    "sample_predictions": [
        {
            "text":       test_df.iloc[idx]["text"][:200],
            "true_label": ID_TO_LABEL[ft_true_ids[idx]],
            "pred_label": ID_TO_LABEL[ft_pred_ids[idx]],
            "confidence": round(float(ft_probs[idx][ft_pred_ids[idx]]), 3),
            "correct":    bool(ft_pred_ids[idx] == ft_true_ids[idx]),
        }
        for idx in list(range(len(test_df)))[:8]
    ],
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: evaluation_results.json")
print("Done.")
