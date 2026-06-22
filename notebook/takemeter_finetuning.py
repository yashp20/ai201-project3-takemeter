# -*- coding: utf-8 -*-
"""TakeMeter — Fine-Tuning Notebook (r/nba Discourse Classifier)
AI201 · Project 3

Filled-in version of the starter notebook.
Upload this to Google Colab, set runtime to T4 GPU, and run all cells.
"""

# ── Section 0: Setup ──────────────────────────────────────────────────────────
# Install any dependencies not pre-installed on Colab
# !pip install -q groq python-dotenv
# print("✅ Dependencies ready")

import pandas as pd
import numpy as np
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

print("✅ Imports complete")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

"""---
## Section 1: Load Your Dataset
"""

# ── Label map (r/nba discourse classifier) ────────────────────────────────────
LABEL_MAP = {
    "analysis": 0,   # Structured argument backed by stats/history/tactics
    "hot_take": 1,   # Bold opinion stated without supporting evidence
    "reaction": 2,   # Immediate emotional response to a specific event
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
print(f"Labels: {LABEL_MAP}")
print(f"Number of labels: {NUM_LABELS}")

# In Colab: upload dataset.csv from this repo
# from google.colab import files
# print("Select your labeled dataset CSV file...")
# uploaded = files.upload()
# CSV_PATH = list(uploaded.keys())[0]

# For local/testing: set path directly
CSV_PATH = "dataset.csv"  # change to uploaded path in Colab
print(f"Using: {CSV_PATH}")

# Load and validate
df = pd.read_csv(CSV_PATH)
print(f"Columns: {df.columns.tolist()}")
print(f"Total examples: {len(df)}")
print("\nLabel distribution:")
print(df["label"].value_counts())

unknown = set(df["label"].unique()) - set(LABEL_MAP.keys())
if unknown:
    print(f"\n⚠️  Labels not in LABEL_MAP: {unknown}")
else:
    print("\n✅ All labels match LABEL_MAP")

df["label_id"] = df["label"].map(LABEL_MAP)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

"""---
## Section 2: Prepare Data for Training
"""

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("\nTrain label distribution:")
print(train_df["label"].value_counts())

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

MODEL_NAME = "distilbert-base-uncased"
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
print("✅ Tokenization complete")

"""---
## Section 3: Fine-Tune Your Model
"""

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_MAP,
)
print(f"✅ Model loaded: {MODEL_NAME}")

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

print("Starting fine-tuning... (5–15 minutes on T4 GPU)")
trainer.train()
print("\n✅ Fine-tuning complete")

"""---
## Section 4: Evaluate Fine-Tuned Model on Test Set
"""

print("Running inference on test set...")
ft_output = trainer.predict(test_dataset)
ft_pred_ids = np.argmax(ft_output.predictions, axis=-1)
ft_true_ids = ft_output.label_ids

ft_probs = torch.nn.functional.softmax(
    torch.tensor(ft_output.predictions), dim=-1
).numpy()

ft_accuracy = accuracy_score(ft_true_ids, ft_pred_ids)
print(f"\n🎯 Fine-tuned model accuracy: {ft_accuracy:.3f}")

label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
print("\nPer-class metrics (fine-tuned model):")
print(classification_report(ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0))

cm = confusion_matrix(ft_true_ids, ft_pred_ids)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✅ Saved: confusion_matrix.png")

wrong_idx = np.where(ft_pred_ids != ft_true_ids)[0]
print(f"\nWrong predictions: {len(wrong_idx)} / {len(ft_true_ids)}\n")

for i, idx in enumerate(wrong_idx[:15]):
    text = test_df.iloc[idx]["text"]
    true_label = ID_TO_LABEL[ft_true_ids[idx]]
    pred_label = ID_TO_LABEL[ft_pred_ids[idx]]
    confidence = ft_probs[idx][ft_pred_ids[idx]]
    print(f"--- #{i+1} ---")
    print(f"Text:      {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"True:      {true_label}")
    print(f"Predicted: {pred_label}  (confidence: {confidence:.2f})")
    print()

"""---
## Section 5: Baseline Classifier (Groq)
"""

from groq import Groq

# ── Groq API key ──────────────────────────────────────────────────────────────
# Option A — Colab Secrets (recommended):
# from google.colab import userdata
# GROQ_API_KEY = userdata.get("GROQ_API_KEY")
#
# Option B — paste directly (do not commit to GitHub):
# GROQ_API_KEY = "your_groq_api_key_here"

GROQ_API_KEY = "your_groq_api_key_here"  # ← replace or use Option A

assert GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here", (
    "Add your GROQ_API_KEY via Colab Secrets or paste it above."
)

client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq client initialized")

# ── Classification prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are classifying posts from r/nba (the NBA basketball subreddit).
Assign each post to exactly one of the following categories.

analysis: The post makes a structured argument backed by specific statistics, historical comparison, or tactical observation. Evidence is precise and the post reasons from data toward a conclusion.
Example: "Jokic's playoff assist-to-turnover ratio improves from 3.2 in the regular season to 4.1 in the postseason, suggesting he manages pace more effectively under pressure."

hot_take: A bold, confident opinion stated without supporting evidence. The post asserts a claim rather than arguing for it — often contrarian, declarative, or provocative.
Example: "LeBron James is the most overrated player in NBA history. His stats look good because he has played for 21 years, not because he is the GOAT."

reaction: An immediate emotional response to a specific in-progress or just-completed event. Little to no argument — the post expresses a feeling in the moment.
Example: "I CANNOT believe what I just watched. That buzzer beater was the greatest shot I have ever seen in my entire life."

Respond with ONLY the label name — one of: analysis, hot_take, reaction
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction"""

print("Prompt length:", len(SYSTEM_PROMPT), "characters")

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

print(f"Running baseline on {len(test_df)} examples...")
print("(0.15s delay between requests)\n")

baseline_preds = []
for i, (_, row) in enumerate(test_df.iterrows()):
    pred = classify_with_groq(row["text"])
    baseline_preds.append(pred)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(test_df)} complete...")
    time.sleep(0.15)

none_count = baseline_preds.count(None)
if none_count > 0:
    print(f"\n⚠️  {none_count} responses could not be parsed.")

valid = [(p, t) for p, t in zip(baseline_preds, test_df["label_id"]) if p is not None]
bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
bl_true_ids = [t for _, t in valid]
bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
print(f"🎯 Baseline accuracy: {bl_accuracy:.3f}  "
      f"(evaluated on {len(valid)}/{len(test_df)} parseable responses)")
print()
print("Per-class metrics (baseline):")
print(classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0))

"""---
## Section 6: Compare Results and Export
"""

print("=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"{'Model':<35} {'Accuracy':>8}")
print("-" * 45)
print(f"{'Zero-shot baseline (Groq)':<35} {bl_accuracy:>8.3f}")
print(f"{'Fine-tuned DistilBERT':<35} {ft_accuracy:>8.3f}")
print("-" * 45)
delta = ft_accuracy - bl_accuracy
direction = "improvement" if delta >= 0 else "regression"
print(f"\nFine-tuning {direction}: {abs(delta):.3f}")

results = {
    "baseline_accuracy": round(bl_accuracy, 4),
    "finetuned_accuracy": round(ft_accuracy, 4),
    "improvement": round(ft_accuracy - bl_accuracy, 4),
    "test_set_size": len(test_df),
    "label_map": LABEL_MAP,
    "model": MODEL_NAME,
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Files ready to download:")
print("   evaluation_results.json")
print("   confusion_matrix.png")
print("\nDownload via: Files panel (📁) → right-click → Download")
