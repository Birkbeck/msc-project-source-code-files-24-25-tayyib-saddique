import os
import pandas as pd
import torch
from transformers import pipeline

# Global models (single GPU)
party_classifier = None
sentiment_classifier = None

# Candidate labels
labels = ["democrat", "republican", "unknown"]

def init_models(device_id=0):
    """Initialize models once on the GPU"""
    global party_classifier, sentiment_classifier
    party_classifier = pipeline(
        "zero-shot-classification",
        model="mlburnham/Political_DEBATE_base_v1.0",
        device=device_id
    )
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device_id
    )
    print(f"Models loaded on GPU {device_id}")

def classify_party(texts, batch_size=128):
    """Batch zero-shot classification"""
    results = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = party_classifier(batch, candidate_labels=labels, truncation=True)
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            results.extend([r['labels'][0].lower() for r in batch_results])
    return results

def classify_sentiment(texts, batch_size=256):
    """Batch sentiment classification"""
    results = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results.extend(sentiment_classifier(batch))
    return results

def process_file(file_path):
    """Process a single file, fully batched"""
    try:
        df = pd.read_csv(file_path, compression="gzip", usecols=[
            "id", "rawContent", "lang", "date", "replyCount", 
            "retweetCount", "likeCount", "quoteCount", "hashtags", "viewCount"
        ])
        df = df[df["lang"] == "en"]
        if df.empty:
            return None

        # Fully batched party classification
        df['party'] = classify_party(df['rawContent'].tolist())

        # Batch sentiment classification per party
        df['sentiment'] = classify_sentiment(df['rawContent'].tolist())

        # Save output
        output_dir = os.path.join(os.getcwd(), "x_processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace(".csv.gz", "_processed.parquet"))
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
