import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

MODEL_DIR = "x_processing/models"
DATA_DIR = "x_processing"
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "x-24-us-election"))

LABELLED_PARQUET = f"{DATA_DIR}/train_labelled.parquet"
UNLABELLED_PARQUET = f"{DATA_DIR}/train_unlabelled.parquet"

CAMPAIGN_START_DATE = pd.Timestamp("2024-05-01")
STRONG_SENTIMENT_THRESHOLD = 0.8

CANDIDATE_KEYWORDS = {
    "democrat": ["#bidenharris2024", "#kamalaharris2024", "@joebiden", "@kamalaharris", "democrats"],
    "republican": ["#maga", "#trump2024", "@realdonaldtrump", "republican"]
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
vader = SentimentIntensityAnalyzer()

def preprocess(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s#']", " ", text)

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(tokens)

def detect_party(text):
    text = text.lower()
    for party, keys in CANDIDATE_KEYWORDS.items():
        if any(k in text for k in keys):
            return party
    return None

def label_sentiment(text):
    score = vader.polarity_scores(text)["compound"]
    if score >= STRONG_SENTIMENT_THRESHOLD:
        return "positive"
    if score <= -STRONG_SENTIMENT_THRESHOLD:
        return "negative"
    return None

def load_and_label_file(path):
    try:
        df = pd.read_csv(
            path,
            compression="gzip",
            usecols=["id", "rawContent", "lang", "date", "epoch"]
        )

        df = df[df["lang"] == "en"]
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
        df.loc[df["timestamp"].isna(), "timestamp"] = pd.to_datetime(
            df["epoch"], unit="s", errors="coerce"
        )

        df = df[df["timestamp"] >= CAMPAIGN_START_DATE]
        if df.empty:
            return None, None

        df = df.rename(columns={"rawContent": "text"})
        df["clean_text"] = df["text"].apply(preprocess)
        df["party"] = df["text"].apply(detect_party)
        df["sentiment"] = df["text"].apply(label_sentiment)

        labelled = df.dropna(subset=["party", "sentiment"])
        unlabelled = df[df["party"].isna() | df["sentiment"].isna()]

        return labelled, unlabelled

    except Exception as e:
        return None, None

def find_all_files_recursively(directory, extension=".csv.gz"):
    """
    Recursively find all files with given extension.
    
    Args:
        directory (str): Root directory to search
        extension (str): File extension to match
        
    Returns:
        list: List of file paths
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def train_and_save(X, y, model, name, task, max_features, n_splits=5):
    """
    Train a model, perform cross-validation, save model and reports.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1,2))),
        ("clf", model)
    ])

    #  Cross-Validation 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=skf,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        n_jobs=-1, return_train_score=False
    )

    cv_summary = (
        f"{name} [{task}] Cross-Validation Results ({n_splits}-fold):\n"
        f"Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}\n"
        f"Precision: {cv_results['test_precision_macro'].mean():.4f} ± {cv_results['test_precision_macro'].std():.4f}\n"
        f"Recall:    {cv_results['test_recall_macro'].mean():.4f} ± {cv_results['test_recall_macro'].std():.4f}\n"
        f"F1 Score:  {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}\n"
    )
    print("\n" + cv_summary)

    # Save CV report
    cv_path = f"{MODEL_DIR}/{name}_{task}_cv_results.txt"
    with open(cv_path, "w") as f:
        f.write(cv_summary)
    print(f"Saved cross-validation report:{cv_path}")

    #  Train on full data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    #  Evaluate on test set 
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    eval_path = f"{MODEL_DIR}/{name}_{task}_evaluation.txt"
    with open(eval_path, "w") as f:
        f.write(f"Model: {name} [{task}]\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(cm)
    print(f"Saved evaluation report:{eval_path}")

    #  Save model 
    model_path = f"{MODEL_DIR}/{name}_{task}_classifier.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model:{model_path}")

    return pipeline, cv_results


def main():
    start = time.time()

    if os.path.exists(LABELLED_PARQUET):
        labelled = pd.read_parquet(LABELLED_PARQUET)
        unlabelled = pd.read_parquet(UNLABELLED_PARQUET)
    else:
        files = find_all_files_recursively(INPUT_DIR)
        labelled_parts, unlabelled_parts = [], []

        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(load_and_label_file, f) for f in files]
            for fut in as_completed(futures):
                l, u = fut.result()
                if l is not None and not l.empty:
                    labelled_parts.append(l)
                if u is not None and not u.empty:
                    unlabelled_parts.append(u)

        labelled = pd.concat(labelled_parts, ignore_index=True)
        unlabelled = pd.concat(unlabelled_parts, ignore_index=True)

        labelled.to_parquet(LABELLED_PARQUET, index=False)
        unlabelled.to_parquet(UNLABELLED_PARQUET, index=False)


    # Model Training
    train_and_save(
        labelled["clean_text"],
        labelled["party"],
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=30,
            num_leaves=63,
            random_state=42,
            force_row_wise=True
        ),
        "LightGBM",
        "party",
        20000
    )

    for party in ["democrat", "republican"]:
        subset = labelled[labelled["party"] == party]

        train_and_save(
            subset["clean_text"],
            subset["sentiment"],
            LogisticRegression(max_iter=300, n_jobs=-1),
            "LogisticRegression",
            f"{party}_sentiment",
            10000
        )

        train_and_save(
            subset["clean_text"],
            subset["sentiment"],
            LinearSVC(max_iter=3000),
            "LinearSVC",
            f"{party}_sentiment",
            10000
        )

    print(f"\nTotal runtime: {(time.time()-start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
