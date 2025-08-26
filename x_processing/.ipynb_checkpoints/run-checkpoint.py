import os
import re
import time
import pandas as pd
import emoji
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
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
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

MODEL_DIR = "x_processing/models"
labelled_parquet = "x_processing/train_labelled.parquet"
unlabelled_parquet = "x_processing/train_unlabelled.parquet"

# Preprocessing + weak labeling

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
        df['sentiment'] = df['text'].apply(label_sentiment)

        labelled = df.dropna(subset=['party', 'sentiment'])
        unlabelled = df[df['party'].isna() | df['sentiment'].isna()]

        if labelled.empty and unlabelled.empty:
            return None, None

        return labelled[['clean_text', 'party', 'sentiment']], unlabelled[['id', 'clean_text', 'text']]
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


# Model training helpers
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


def train_and_save_models(X, y, model_candidates, task_name, tfidf_max_features=20000, ngram_range=(1, 2)):
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, clf in model_candidates.items():
        print(f"\nTraining {name} for {task_name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=tfidf_max_features, ngram_range=ngram_range)),
            ('clf', clf)
        ])
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start_time

        report_path = os.path.join(MODEL_DIR, f"{task_name}_{name}___report.txt")
        acc = evaluate_model(pipeline, X_test, y_test, f"{task_name} ({name})", output_path=report_path)

        print(f"Training + evaluation time for {name} ({task_name}): {elapsed:.2f} seconds")
        results.append((name, pipeline, acc))

    # Save models
    for name, model, acc in results:
        path = os.path.join(MODEL_DIR, f"{name}_{task_name}_classifier.joblib")
        joblib.dump(model, path)
        print(f"Saved {name} for {task_name} with accuracy {acc:.4f} -> {path}")

    return results


# Main pipeline
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    total_start = time.time()

    # Load/preprocess data
    start = time.time()
    if os.path.exists(labelled_parquet):
        print(f"Loading labelled data from {labelled_parquet}")
        labelled_df = pd.read_parquet(labelled_parquet)
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

        if unlabelled_dfs:
            unlabelled_df = pd.concat(unlabelled_dfs, ignore_index=True)
            unlabelled_df.to_parquet(unlabelled_parquet, index=False)
            print(f"Unlabelled data saved to {unlabelled_parquet}")
    print(f"Data loading / preprocessing took {time.time() - start:.2f} seconds")

    # Drop rows with missing labels
    labelled_df.dropna(subset=['party', 'sentiment'], inplace=True)
    labelled_df.reset_index(drop=True, inplace=True)

    # Party classification (LightGBM only)
    print("\nTraining party classifier (LightGBM only):")
    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']
    party_models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=30,
            num_leaves=63,
            n_jobs=-1,
            random_state=42,
            force_row_wise=True
        )
    }
    train_and_save_models(X_party, y_party, party_models, task_name="party", tfidf_max_features=20000)

    # Sentiment classification

    # Democrat Sentiment Models
    dem_sentiment_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=300,
            C=13.826,
            n_jobs=-1,
            verbose=1
        ),
        "LinearSVC": LinearSVC(max_iter=3000,
                               C=0.6338,
                               loss='squared_hinge',
                              )
    }

    print("\nTraining sentiment classifiers for Democrats (LogReg + LinearSVC):")
    dem_data = labelled_df[labelled_df['party'] == 'democrat']
    X_dem = dem_data['clean_text']
    y_dem = dem_data['sentiment']

    train_and_save_models(
        X_dem, y_dem, dem_sentiment_models, task_name="democrat_sentiment", tfidf_max_features=10000
    )

    # Republican Sentiment Models
    rep_sentiment_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=300,
            C=13.826,
            n_jobs=-1,
            verbose=1
        ),
        "LinearSVC": LinearSVC(max_iter=3000,
                              C=0.367,
                              loss='squared_hinge')
    }

    print("\nTraining sentiment classifiers for Republicans (LogReg + LinearSVC):")
    rep_data = labelled_df[labelled_df['party'] == 'republican']
    X_rep = rep_data['clean_text']
    y_rep = rep_data['sentiment']

    train_and_save_models(
        X_rep, y_rep, rep_sentiment_models, task_name="republican_sentiment", tfidf_max_features=10000
    )

    print(f"\nTotal runtime: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
