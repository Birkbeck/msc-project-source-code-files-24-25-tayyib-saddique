import os
import pandas as pd
import joblib
import time

MODEL_DIR = "x_processing/models"
UNLABELLED_PARQUET = "x_processing/train_unlabelled.parquet"
OUTPUT_PARQUET = "x_processing/predictions_with_timestamps.parquet"

# Date filtering configuration
CAMPAIGN_START_DATE = pd.Timestamp("2024-05-01")
CHUNK_SIZE = 500_000 

def convert_to_timestamp(df):
    """Create timestamp column from date/epoch"""
    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    
    if 'epoch' in df.columns:
        missing_mask = df['timestamp'].isna() & df['epoch'].notna()
        df.loc[missing_mask, 'timestamp'] = pd.to_datetime(df.loc[missing_mask, 'epoch'], unit='s')
    
    return df


def filter_campaign_period(df):
    mask = (df['timestamp'] >= CAMPAIGN_START_DATE)
    filtered_df = df[mask].copy()    
    return filtered_df


def load_model(classifier, party=None):
    """Load trained model"""
    if classifier == "party":
        filename = "LightGBM_party_classifier.joblib"
    elif classifier == "sentiment":
        if not party:
            raise ValueError("Party must be specified for sentiment classifier")
        filename = f"LinearSVC_{party.lower()}_sentiment_classifier.joblib"
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loaded: {filename}")
    return joblib.load(model_path)


def chunked_predict(model, texts, chunk_size=CHUNK_SIZE, label=""):
    """Predict in chunks with progress tracking"""
    n = len(texts)
    results = []
    start = time.time()

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        results.extend(model.predict(texts[i:end]))
        
        chunk_num = (i // chunk_size) + 1
        total_chunks = (n + chunk_size - 1) // chunk_size
        elapsed = time.time() - start
        
        print(f"[{label}] {chunk_num}/{total_chunks} chunks ({end:,}/{n:,} rows, {elapsed:.1f}s elapsed)")

    return results


def print_summary(df):
    """Print sentiment summary and save to file"""
    summary_file = "x_processing/sentiment_summary.txt"
    summary_lines = []

    for party in ["democrat", "republican"]:
        mask = df['party'] == party
        total = mask.sum()
        positive = (df.loc[mask, 'sentiment'] == "positive").sum()
        negative = total - positive
        pos_ratio = positive / total if total > 0 else 0
        neg_ratio = negative / total if total > 0 else 0

        # Print to console
        print(f"\n{party.capitalize()}:")
        print(f"  Total: {total:,}")
        print(f"  Positive: {positive:,} ({pos_ratio:.2%})")
        print(f"  Negative: {negative:,} ({neg_ratio:.2%})")

        # Prepare lines to write to file
        summary_lines.append(f"{party.capitalize()}:\n")
        summary_lines.append(f"  Positive sentiment: {positive}/{total} ({pos_ratio:.2%})\n")
        summary_lines.append(f"  Negative sentiment: {negative}/{total} ({neg_ratio:.2%})\n\n")

    # Save to file
    with open(summary_file, "w") as f:
        f.writelines(summary_lines)

    print(f"\nSentiment summary saved to {summary_file}")


def main():
    start_total = time.time()
    
    # Check if predictions already exist
    if os.path.exists(OUTPUT_PARQUET):
        print(f"\nLoading existing predictions from {OUTPUT_PARQUET}")
        df = pd.read_parquet(OUTPUT_PARQUET)
        print(f"Loaded {len(df):,} tweets")
        print_summary(df)
        print(f"\nTotal workflow completed in {time.time() - start_total:.2f} seconds")
    else:
        # Load models
        print("\nLoading models")
        party_model = load_model("party")
        dem_model = load_model("sentiment", party="democrat")
        rep_model = load_model("sentiment", party="republican")

        # Load and filter data
        print(f"\nLoading data from {UNLABELLED_PARQUET}")
        df = pd.read_parquet(UNLABELLED_PARQUET)
        print(f"  Loaded {len(df):,} tweets")

        df = filter_campaign_period(df)

        # Predict party
        print("\nClassifying party affiliation")
        df['party'] = chunked_predict(party_model, df['clean_text'].tolist(), label="Party")

        print(f"\nParty distribution:")
        for party, count in df['party'].value_counts().items():
            print(f"{party.capitalize()}: {count:,} ({count/len(df):.1%})")

        # Predict sentiment by party
        print("\nClassifying sentiment")
        for party, model in [("democrat", dem_model), ("republican", rep_model)]:
            mask = df['party'] == party
            if mask.sum() > 0:
                df.loc[mask, 'sentiment'] = chunked_predict(
                    model, 
                    df.loc[mask, 'clean_text'].tolist(), 
                    label=party.capitalize()
                )

        # Save results
        print(f"\nSaving to {OUTPUT_PARQUET}")
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print("Saved")

        # Print summary
        print_summary(df)
    
    print(f"\nTotal workflow completed in {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    main()