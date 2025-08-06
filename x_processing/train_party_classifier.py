import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

CACHE_FILE = "cached_training_data.parquet"

def detect_candidate(text):
    text = str(text).lower()
    candidates = {
        'democrat': ['biden', 'harris', '#bidenharris2024', '#kamalaharris2024', '@joebiden', '@kamalaharris', 'democrats'],
        'republican': ['trump', 'jdvance', 'maga', 'republican', 'trump2024', '@realdonaldtrump']
    }
    for label, keywords in candidates.items():
        if any(k in text for k in keywords):
            return label
    return None

def load_data():
    if os.path.exists(CACHE_FILE):
        return pd.read_parquet(CACHE_FILE)

    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    all_data = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".csv.gz"):
                try:
                    path = os.path.join(root, f)
                    df = pd.read_csv(path, compression="gzip", usecols=["rawContent", "lang"])
                    df = df[df["lang"] == "en"]
                    df["party"] = df["rawContent"].apply(detect_candidate)
                    df = df[df["party"].isin(["democrat", "republican"])]
                    all_data.append(df)
                except Exception as e:
                    print(f"Error in {f}: {e}")

    full_df = pd.concat(all_data) if all_data else pd.DataFrame()
    if not full_df.empty:
        full_df.to_parquet(CACHE_FILE, index=False)
    return full_df

def train():
    df = load_data()
    if df.empty:
        print("No training data found.")
        return

    dataset = Dataset.from_pandas(df[["rawContent", "party"]])
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["rawContent"], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("party", "labels")
    dataset = dataset.class_encode_column("labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="party_classifier_output",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_total_limit=1,
        save_strategy="epoch",
        logging_strategy="epoch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    # Save model and tokenizer manually
    torch.save(model.state_dict(), "party_classifier_model.pt")
    tokenizer.save_pretrained("party_classifier_tokenizer")
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    train()
