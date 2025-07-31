import os
import glob
import emoji
import pandas as pd
import swifter
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- Setup ---
tokeniser = TweetTokenizer()

# Use tweet-optimized, efficient sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Batch-capable pipeline
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    batch_size=32,
    device=-1  # CPU
)

input_dir = os.path.join(os.getcwd(), "x-24-us-election")
output_dir = os.path.join(input_dir, "processed")
os.makedirs(output_dir, exist_ok=True)

print("Input dir:", input_dir)

# --- Candidate Keywords ---
candidates = {
    'democrat': ['biden', 'harris', '#bidenharris2024', '#kamalaharris2024', '@joebiden', '@kamalaharris', 'democrats'],
    'republican': ['trump', 'jdvance', 'vance', 'maga', 'republican', 'trump2024', '@realdonaldtrump']
}
flat_keywords = '|'.join(candidates['democrat'] + candidates['republican'])

# --- Functions ---
def detect_candidate(text):
    text = text.lower()
    for cand, terms in candidates.items():
        if any(term in text for term in terms):
            return cand
    return None

def clean_text(text):
    text = emoji.replace_emoji(text.lower(), replace='')
    tokens = tokeniser.tokenize(text)
    return ' '.join(tokens)

def get_sentiment_scores(texts):
    try:
        results = sentiment_classifier(texts)
        scores = []
        for r in results:
            label = r['label'].lower()
            score = r['score']
            scores.append(score if label == 'positive' else -score)
        return scores
    except Exception as e:
        print("Batch sentiment failed:", e)
        return [0.0] * len(texts)

def process_file(file_path):
    try:
        df = pd.read_csv(
            file_path,
            compression='gzip',
            usecols=["id", "text", "lang", "date", "replyCount", "retweetCount", "likeCount", "quoteCount", "hashtags", "viewCount"]
        )
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

    df = df[df['lang'] == 'en']

    if df.empty:
        return None

    # Filter early
    df = df[df['text'].str.contains(flat_keywords, case=False, na=False)]
    df['candidate'] = df['text'].swifter.apply(detect_candidate)
    df = df[df['candidate'].notna()]

    if df.empty:
        return None

    df['clean'] = df['text'].swifter.apply(clean_text)

    # Batched sentiment scoring
    batch_size = 16
    all_clean = df['clean'].tolist()
    all_scores = []
    _ = sentiment_classifier("Model warmup")  # Warm up model

    for i in range(0, len(all_clean), batch_size):
        batch = all_clean[i:i+batch_size]
        print(f"Scoring batch {i//batch_size + 1}/{len(all_clean)//batch_size + 1}")
        try:
            scores = get_sentiment_scores(batch)
            all_scores.extend(scores)
        except Exception as e:
            print(f"Failed scoring batch {i}: {e}")
            all_scores.extend([0.0] * len(batch))    
    df['sentiment'] = all_scores

    return df[['id', 'date', 'candidate', 'text', 'clean', 'sentiment']]

# --- Main Processing Loop ---
def find_files(input_dir):
    pattern = os.path.join(input_dir, "**", "*.csv.gz")
    return list(glob.iglob(pattern, recursive=True))

files = find_files(input_dir)
print(f"Found {len(files)} files.")

limit = 2  # adjust or remove to process all
processed_count = 0

for file_path in tqdm(files, desc="Processing files"):
    if limit and processed_count >= limit:
        print(f"Stopping after {processed_count} files (limit hit).")
        break

    df = process_file(file_path)
    if df is None or df.empty:
        continue

    output_file = os.path.join(
        output_dir,
        os.path.basename(file_path).replace(".csv.gz", "_processed.parquet")
    )

    try:
        df.to_parquet(output_file, index=False)
        processed_count += 1
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Failed to save {output_file}: {e}")

print(f"{processed_count} file(s) processed.")
