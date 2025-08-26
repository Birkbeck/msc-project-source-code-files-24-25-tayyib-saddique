import os
import re
import time
import numpy as np
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import joblib
import torch

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vader = SentimentIntensityAnalyzer()

# Constants
CANDIDATE_KEYWORDS = {
    'democrat': ['#bidenharris2024', '#kamalaharris2024', '@joebiden', '@kamalaharris', 'democrats'],
    'republican': ['#maga', 'republican', '#trump2024', '@realdonaldtrump']
}
STRONG_SENTIMENT_THRESHOLD = 0.8
MODEL_DIR = "x_processing/models/experiments"
labelled_parquet = "x_processing/train_labelled.parquet"
unlabelled_parquet = "x_processing/train_unlabelled.parquet"

# ---------------- Preprocessing & Weak labeling ----------------
def preprocess(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+|https\S+|@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s#']", " ", text)
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
        df['sentiment_score'] = df['text'].apply(lambda t: vader.polarity_scores(t)['compound'])
        df['sentiment'] = df['sentiment_score'].apply(
            lambda score: 'positive' if score >= STRONG_SENTIMENT_THRESHOLD else 'negative'
            if score <= -STRONG_SENTIMENT_THRESHOLD else None
        )

        labelled = df.dropna(subset=['party', 'sentiment'])
        unlabelled = df[df['party'].isna() | df['sentiment'].isna()]

        if labelled.empty and unlabelled.empty:
            return None, None

        return (
            labelled[['clean_text', 'party', 'sentiment', 'sentiment_score']],
            unlabelled[['id', 'clean_text', 'text']]
        )
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

# ---------------- Evaluation ----------------
def evaluate_model(pipeline, X_test, y_test, model_name, output_path=None):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = (
        f"\n{model_name}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Classification Report:\n{classification_report(y_test, y_pred)}\n"
        f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n"
    )
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return acc

# ---------------- Transformers ----------------
def vader_sentiment_features(X):
    """ Returns a NumPy array with 4 columns: neg, neu, pos, compound scores. """
    features = []
    for t in X:
        scores = vader.polarity_scores(t)
        features.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
    return np.array(features)

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=256, device='cuda:0'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values.ravel().tolist()
        else:
            X = list(X)

        with torch.no_grad():
            embeddings = self.model.encode(
                X,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        return embeddings

# ---------------- Model Candidates ----------------
def get_model_candidates(for_embeddings=False):
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, solver="saga", n_jobs=-1, random_state=42
        ),
        "LinearSVC": LinearSVC(
            max_iter=3000, random_state=42
        ),
        "SGDClassifier": SGDClassifier(
            max_iter=1000, tol=1e-3, loss='hinge', penalty='l2', n_jobs=-1, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_features='sqrt', max_depth=12, n_jobs=-1, random_state=42
        ),
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=12, num_leaves=31, learning_rate=0.05,
            n_jobs=-1, random_state=42, force_col_wise=True
        )
    }

    if for_embeddings:
        candidates.pop("LinearSVC", None)
        candidates.pop("MultinomialNB", None)
    return candidates

#  Pipeline Builder 
def build_pipeline(clf, feature_mode="tfidf", max_features=20000, batch_size=256, pca_components=256):
    if feature_mode == "tfidf":
        features = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    elif feature_mode == "embeddings":
        features = Pipeline([
            ('embed', EmbeddingTransformer(batch_size=batch_size)),
            ('scale', StandardScaler(with_mean=False)),
            ('pca', PCA(n_components=pca_components))
        ])
    else:
        raise ValueError("Invalid feature_mode. Choose from ['tfidf', 'embeddings'].")

    pipeline = Pipeline([
        ('features', features),
        ('clf', clf)
    ])
    return pipeline

#  Training 
def train_and_save_top_models(X, y, model_candidates, task_name, feature_mode='tfidf',
                              tfidf_max_features=20000, test_size=0.2, output_file=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"X_train: {len(X_train)}, y_train: {len(y_train)}")
    print(f"X_test: {len(X_test)}, y_test: {len(y_test)}")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Distribution report for {task_name}\n\n"
                    f"X_train: {len(X_train)}, y_train: {len(y_train)}\n"
                    f"X_test: {len(X_test)}, y_test: {len(y_test)}\n\n")

    def check_distribution(name, labels):
        counts = Counter(labels)
        total = sum(counts.values())
        dist = {cls: f"{count} ({count / total:.2%})" for cls, count in counts.items()}
        report_line = f"{task_name} - {name} distribution: {dist}"
        print(report_line)
        if output_file:
            with open(output_file, 'a') as f:
                f.write(report_line + "\n")

    check_distribution("Train", y_train)
    check_distribution("Test", y_test)
    check_distribution("Full", y)

    results = []
    print(f"\nTraining models for {task_name} using feature mode: {feature_mode}")

    for name, clf in model_candidates.items():
        print(f"\n{name} for {task_name} ({feature_mode})")
        start_time = time.time()
        pipeline = build_pipeline(clf, feature_mode=feature_mode, max_features=tfidf_max_features)
        pipeline.fit(X_train, y_train)

        report_path = os.path.join(MODEL_DIR, f"{task_name}_{name}_{feature_mode}_report.txt")
        acc = evaluate_model(pipeline, X_test, y_test, f"{task_name} ({name}, {feature_mode})", output_path=report_path)
        print(f"Training + evaluation time: {time.time() - start_time:.2f} seconds")

        results.append((name, pipeline, acc))

    results.sort(key=lambda x: x[2], reverse=True)

    for rank, (name, model, acc) in enumerate(results[:2], start=1):
        path = os.path.join(MODEL_DIR, f"top{rank}_{name}_{task_name}_{feature_mode}_classifier.joblib")
        joblib.dump(model, path)
        print(f"Saved top{rank} model: {path} (Accuracy: {acc:.4f})")

    return results

#  Main 
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    total_start = time.time()

    # Load or preprocess data
    start = time.time()
    if os.path.exists(labelled_parquet):
        print(f"Loading labelled data from {labelled_parquet}")
        labelled_df = pd.read_parquet(labelled_parquet)
        print(f"Total rows in labelled data is {len(labelled_df)}")
    else:
        print("Preprocessing raw data...")
        INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "x-24-us-election"))
        all_files = find_all_files_recursively(INPUT_DIR)
        print(f"Found {len(all_files)} files")

        labelled_dfs, unlabelled_dfs = [], []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(load_preprocess_weak_label, file): file for file in all_files}
            for i, future in enumerate(as_completed(futures)):
                file = futures[future]
                try:
                    labelled, unlabelled = future.result()
                    if labelled is not None:
                        labelled_dfs.append(labelled)
                    if unlabelled is not None and not unlabelled.empty:
                        unlabelled_dfs.append(unlabelled)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

                if i % 50 == 0:
                    print(f"Processed {i}/{len(all_files)} files")

        if not labelled_dfs:
            print("No labelled data found, exiting.")
            return

        labelled_df = pd.concat(labelled_dfs, ignore_index=True)
        labelled_df.to_parquet(labelled_parquet, index=False)
        print(f"Labelled data saved to {labelled_parquet}")
        print(f"Total rows in labelled data is {len(labelled_df)}")

        if unlabelled_dfs:
            unlabelled_df = pd.concat(unlabelled_dfs, ignore_index=True)
            unlabelled_df.to_parquet(unlabelled_parquet, index=False)
            print(f"Unlabelled data saved to {unlabelled_parquet}")
            print(f"Total rows in unlabelled data is {len(unlabelled_df)}")

        print(f"Data loading / preprocessing took {time.time() - start:.2f} seconds")

    # Prepare datasets
    labelled_df.dropna(subset=['party', 'sentiment'], inplace=True)
    labelled_df.reset_index(drop=True, inplace=True)

    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']

    for feature_mode in ['tfidf', 'embeddings']:
        print(f"\nTraining party classifiers with feature mode: {feature_mode}")
        model_candidates = get_model_candidates(for_embeddings=(feature_mode == "embeddings"))
        train_and_save_top_models(
            X_party, y_party, model_candidates,
            task_name="party", feature_mode=feature_mode,
            output_file=os.path.join(MODEL_DIR, f"party_distribution.txt")
        )

    # Sentiment Classification per Party
    for party in ['democrat', 'republican']:
        party_data = labelled_df[labelled_df['party'] == party]
        X_sent = party_data['clean_text']
        y_sent = party_data['sentiment']

        for feature_mode in ['tfidf', 'embeddings']:
            print(f"\nTraining sentiment classifiers for {party} with feature mode: {feature_mode}")
            model_candidates = get_model_candidates(for_embeddings=(feature_mode == "embeddings"))
            train_and_save_top_models(
                X_sent, y_sent, model_candidates,
                task_name=f"{party}_sentiment", feature_mode=feature_mode,
                output_file=os.path.join(MODEL_DIR, f"{party}_sentiment_distribution.txt")
            )

    print(f"\nTotal pipeline execution took {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
