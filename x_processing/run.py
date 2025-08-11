import os
import re
import time
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
from sklearn.metrics import classification_report, confusion_matrix
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
        # Keep raw sentiment score as well as label
        df['sentiment_score'] = df['text'].apply(lambda t: vader.polarity_scores(t)['compound'])
        df['sentiment'] = df['sentiment_score'].apply(
            lambda score: 'positive' if score >= STRONG_SENTIMENT_THRESHOLD else
                          'negative' if score <= -STRONG_SENTIMENT_THRESHOLD else None)

        labeled = df.dropna(subset=['party', 'sentiment'])
        unlabeled = df[df['party'].isna() | df['sentiment'].isna()]

        if labeled.empty and unlabeled.empty:
            return None, None

        return labeled[['clean_text', 'party', 'sentiment', 'sentiment_score']], unlabeled[['id', 'clean_text', 'text']]
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

    start_time = time.time()
    all_files = find_all_files_recursively(INPUT_DIR)
    print(f"Found {len(all_files)} files")

    labeled_dfs = []
    unlabeled_dfs = []

    t1 = time.time()
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
    t2 = time.time()
    print(f"Preprocessing and weak labeling took {t2 - t1:.2f} seconds")

    if not labeled_dfs:
        print("No labeled data found.")
        return

    # Save labeled dataset with sentiment_score
    labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    print(f"Total labeled samples: {len(labeled_df)}")
    labeled_df.to_parquet("x_processing/train_labelled.parquet", index=False)

    # Save unlabeled dataset
    if unlabeled_dfs:
        unlabeled_df = pd.concat(unlabeled_dfs, ignore_index=True)
        print(f"Total unlabeled samples: {len(unlabeled_df)}")
        unlabeled_df.to_parquet("x_processing/train_unlabelled.parquet", index=False)
    else:
        unlabeled_df = None
        print("No unlabeled data found.")

    # Train Party Classifier
    print("\nTraining party classifier...")
    t3 = time.time()
    X_party = labeled_df['clean_text']
    y_party = labeled_df['party']

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_party, y_party, test_size=0.2, random_state=42)

    party_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=300, n_jobs=-1))
    ])

    party_pipeline.fit(X_train_p, y_train_p)
    y_pred_p = party_pipeline.predict(X_test_p)
    print("Party classifier report:")
    print(classification_report(y_test_p, y_pred_p))
    
    print("Party confusion matrix:")
    print(confusion_matrix(y_test_p, y_pred_p))
    
    t4 = time.time()
    print(f"Party classifier training and evaluation took {t4 - t3:.2f} seconds")

    # Train Sentiment Classifiers separately for each party
    sentiment_pipelines = {}
    for party in ['democrat', 'republican']:
        print(f"\nTraining sentiment classifier for {party}...")
        t_start = time.time()
        party_data = labeled_df[labeled_df['party'] == party]

        X_sent = party_data['clean_text']
        y_sent = party_data['sentiment']

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42)

        sentiment_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=300, n_jobs=-1))
        ])

        sentiment_pipeline.fit(X_train_s, y_train_s)
        y_pred_s = sentiment_pipeline.predict(X_test_s)
        print(f"{party.capitalize()} sentiment classifier report:")
        print(classification_report(y_test_s, y_pred_s))
        
        print(f"{party.capitalize()} sentiment confusion matrix:")
        print(confusion_matrix(y_test_s, y_pred_s))
        
        sentiment_pipelines[party] = sentiment_pipeline
        t_end = time.time()
        print(f"{party.capitalize()} sentiment classifier training and evaluation took {t_end - t_start:.2f} seconds")

    # Save models
    model_dir = "x_processing/models"
    os.makedirs(model_dir, exist_ok=True)

    party_model_path = os.path.join(model_dir, "party_classifier.joblib")
    joblib.dump(party_pipeline, party_model_path)
    print(f"Party classifier saved to {party_model_path}")

    for party, model in sentiment_pipelines.items():
        path = os.path.join(model_dir, f"sentiment_classifier_{party}.joblib")
        joblib.dump(model, path)
        print(f"{party.capitalize()} sentiment classifier saved to {path}")

    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
