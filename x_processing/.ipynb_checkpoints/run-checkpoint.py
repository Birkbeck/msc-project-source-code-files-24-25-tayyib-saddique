import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vader = SentimentIntensityAnalyzer()

CANDIDATE_KEYWORDS = {
    'democrat': ['#bidenharris2024', '#kamalaharris2024', '@joebiden', '@kamalaharris', 'democrats'],
    'republican': ['#maga', 'republican', '#trump2024', '@realdonaldtrump']
}
STRONG_SENTIMENT_THRESHOLD = 0.8

# Preprocessing & Weak labeling
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+|@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s#]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

def detect_candidate(text):
    text = text.lower()
    for candidate, keywords in CANDIDATE_KEYWORDS.items():
        if any(k in text for k in keywords):
            return candidate
    return None

def label_sentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score >= STRONG_SENTIMENT_THRESHOLD:
        return 'positive'
    elif score <= -STRONG_SENTIMENT_THRESHOLD:
        return 'negative'
    else:
        return None

# File Processing
def load_preprocess_weak_label(file_path):
    try:
        df = pd.read_csv(
            file_path,
            compression="gzip",
            usecols=["id", "rawContent", "lang"],
            dtype={"id": str, "rawContent": str, "lang": str}
        )
        df = df[df["lang"] == "en"]
        if df.empty:
            return None, None

        df = df.rename(columns={"rawContent": "text"})
        df['clean_text'] = df['text'].apply(preprocess)
        df['party'] = df['text'].apply(detect_candidate)
        df['sentiment'] = df['text'].apply(label_sentiment)

        labeled = df.dropna(subset=['party', 'sentiment'])
        unlabeled = df[df['party'].isna() | df['sentiment'].isna()]

        if labeled.empty and unlabeled.empty:
            return None, None

        return labeled[['clean_text', 'party', 'sentiment']], unlabeled[['id', 'clean_text', 'text']]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def find_all_files_recursively(directory, extension=".csv.gz"):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def main():
    INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "x-24-us-election"))

    all_files = find_all_files_recursively(INPUT_DIR)
    print(f"Found {len(all_files)} files")

    labeled_dfs = []
    unlabeled_dfs = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(load_preprocess_weak_label, file): file for file in all_files}

        for i, future in enumerate(as_completed(futures)):
            file = futures[future]
            try:
                labeled, unlabeled = future.result()
                if labeled is not None:
                    labeled_dfs.append(labeled)
                if unlabeled is not None and not unlabeled.empty:
                    unlabeled_dfs.append(unlabeled)
            except Exception as e:
                print(f"Error in processing {file}: {e}")
            if i % 50 == 0:
                print(f"Processed {i}/{len(all_files)} files")

    if not labeled_dfs:
        print("No labeled data found.")
        return

    # Save labeled dataset
    labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    print(f"Total labeled samples: {len(labeled_df)}")
    labeled_df.to_parquet("train_labeled.parquet", index=False)

    # Save unlabeled dataset
    if unlabeled_dfs:
        unlabeled_df = pd.concat(unlabeled_dfs, ignore_index=True)
        print(f"Total unlabeled samples: {len(unlabeled_df)}")
        unlabeled_df.to_parquet("train_unlabeled.parquet", index=False)
    else:
        unlabeled_df = None
        print("No unlabeled data found.")

    # Train Classifier
    labeled_df['label'] = labeled_df['party'] + "_" + labeled_df['sentiment']

    X = labeled_df['clean_text']
    y = labeled_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=300, n_jobs=-1))
    ])

    print("Training classifier...")
    pipeline.fit(X_train, y_train)

    print("Evaluating classifier...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_path = "msc-project-source-code-files-24-25-tayyib-saddique/party_sentiment_classifier.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
