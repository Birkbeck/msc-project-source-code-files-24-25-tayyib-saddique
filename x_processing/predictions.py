import os
import pandas as pd
import joblib
import time

MODEL_DIR = "x_processing/models"
UNLABELLED_PARQUET = "x_processing/train_unlabelled.parquet"
OUTPUT_PARQUET = "x_processing/predictions.parquet"

CHUNK_SIZE = 500_000 

def load_model(classifier, party=None, directory=MODEL_DIR):
    """Load best saved model for party or sentiment classification"""
    candidates = []
    for f in os.listdir(directory):
        if not f.endswith(".joblib"):
            continue
        f_lower = f.lower()
        if classifier == "party" and "party_classifier" in f_lower:
            candidates.append(f)
        elif classifier == "sentiment" and party and party.lower() in f_lower and "sentiment_classifier" in f_lower:
            candidates.append(f)
    if not candidates:
        raise FileNotFoundError(f"No model found for classifier='{classifier}' party='{party}'")
    model_path = os.path.join(directory, candidates[0])
    print(f"Loading {classifier} model from: {model_path}")
    return joblib.load(model_path)


def chunked_predict(model, texts, chunk_size=CHUNK_SIZE, label=""):
    """Predict in chunks and log progress"""
    n = len(texts)
    chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    results = []

    start = time.time()
    for idx, (s, e) in enumerate(chunks, 1):
        y_chunk = model.predict(texts[s:e])
        results.extend(y_chunk)

        elapsed = time.time() - start
        avg = elapsed / idx
        remaining = (len(chunks) - idx) * avg
        print(f"[{label}] Chunk {idx}/{len(chunks)} "
              f"→ {e:,}/{n:,} rows done "
              f"(elapsed {elapsed:.1f}s, ~{avg:.1f}s/chunk, ETA {remaining/60:.1f} min)")

    return results

def main():
    start_total = time.time()

    # Load models
    party_model = load_model("party")
    dem_sentiment_model = load_model("sentiment", party="democrat")
    rep_sentiment_model = load_model("sentiment", party="republican")

    # Load data
    start = time.time()
    print(f"Loading unlabelled data from {UNLABELLED_PARQUET}...")
    df = pd.read_parquet(UNLABELLED_PARQUET)
    print(f"Loaded {len(df):,} rows in {time.time() - start:.2f} seconds")

    # Party Prediction
    print("\nPredicting party labels")
    start = time.time()
    df['party'] = chunked_predict(party_model, df['clean_text'].tolist(), label="Party")
    print(f"Party prediction completed in {time.time() - start:.2f} seconds")

    # Sentiment Prediction per Party
    print("\nPredicting sentiment per party")
    for party, model in [("democrat", dem_sentiment_model), ("republican", rep_sentiment_model)]:
        mask = df['party'] == party
        n_party = mask.sum()
        if n_party == 0:
            print(f"No rows for {party}, skipping sentiment prediction.")
            continue

        print(f"\n{party.capitalize()} sentiment: {n_party:,} rows")
        start = time.time()
        df.loc[mask, 'sentiment'] = chunked_predict(model, df.loc[mask, 'clean_text'].tolist(), label=party)
        print(f"{party.capitalize()} sentiment prediction done in {time.time() - start:.2f} seconds")

    print("\nSaving predictions...")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Predictions saved to {OUTPUT_PARQUET}")

    # Calculate positive sentiment ratio per party
    sentiment_summary = {}
    for party in ["democrat", "republican"]:
        mask = df['party'] == party
        if mask.sum() > 0:
            total = mask.sum()
            positive = (df.loc[mask, 'sentiment'] == "positive").sum()
            ratio = positive / total
            sentiment_summary[party] = (positive, total, ratio)

    summary_file_path = "x_processing/sentiment_summary.txt"
    with open(summary_file, "w") as f:
        for party, (pos, total, ratio) in sentiment_summary.items():
            f.write(f"{party.capitalize()}:\n")
            f.write(f"  Positive sentiment: {pos}/{total} ({ratio:.2%})\n\n")
    print(f"Sentiment summary saved to {summary_file}")

    print(f"\nTotal workflow completed in {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    main()
