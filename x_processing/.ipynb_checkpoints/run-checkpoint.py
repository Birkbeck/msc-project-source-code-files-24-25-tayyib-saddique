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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
STRONG_SENTIMENT_THRESHOLD = 0.6

MODEL_DIR = "x_processing/models"
labelled_parquet = "x_processing/train_labelled.parquet"
unlabelled_parquet = "x_processing/train_unlabelled.parquet"

# --- Preprocessing & Weak labeling ---
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
            lambda score: 'positive' if score >= STRONG_SENTIMENT_THRESHOLD else
                          'negative' if score <= -STRONG_SENTIMENT_THRESHOLD else None)

        labelled = df.dropna(subset=['party', 'sentiment'])
        unlabelled = df[df['party'].isna() | df['sentiment'].isna()]

        if labelled.empty and unlabelled.empty:
            return None, None

        return labelled[['clean_text', 'party', 'sentiment', 'sentiment_score']], unlabelled[['id', 'clean_text', 'text']]
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

# Model Evaluation Helper
def evaluate_model(pipeline, X_test, y_test, model_name, output_path=None):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = f"\n{model_name}\nAccuracy: {acc:.4f}\nClassification Report:\n{classification_report(y_test, y_pred)}\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n"
    print(report)
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    return acc

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    total_start = time.time()

    # Data Loading / Preprocessing
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

    # Prepare datasets
    labelled_df.dropna(subset=['party', 'sentiment'], inplace=True)
    labelled_df.reset_index(drop=True, inplace=True)

    model_candidates = {
        "LogisticRegression": LogisticRegression(max_iter=300, n_jobs=-1),
        "LinearSVC": LinearSVC(max_iter=3000),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1),
        "MultinomialNB": MultinomialNB()
    }

    # --- Train & evaluate Party classifiers ---
    print("\nTraining party classifiers:")
    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_party, y_party, test_size=0.2, random_state=42)

    best_party_acc = 0
    best_party_model = None
    best_party_name = None

    for name, clf in model_candidates.items():
        print(f"{name} for party classification")
        start_time = time.time()

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ('clf', clf)
        ])
        pipeline.fit(X_train_p, y_train_p)

        report_path = os.path.join(MODEL_DIR, f"{name}_party_report.txt")
        acc = evaluate_model(pipeline, X_test_p, y_test_p, f"Party classifier ({name})", output_path=report_path)

        elapsed = time.time() - start_time
        print(f"Training + evaluation time for {name} (party): {elapsed:.2f} seconds")

        if acc > best_party_acc:
            best_party_acc = acc
            best_party_model = pipeline
            best_party_name = name

    party_model_path = os.path.join(MODEL_DIR, f"{best_party_name}_party_classifier.joblib")
    joblib.dump(best_party_model, party_model_path)
    print(f"\nBest party classifier: {best_party_name} saved to {party_model_path}")

    #Train & evaluate Sentiment classifiers per party
    sentiment_pipelines = {}
    for party in ['democrat', 'republican']:
        print(f"\nTraining sentiment classifiers for {party}:")
        party_data = labelled_df[labelled_df['party'] == party]
        X_sent = party_data['clean_text']
        y_sent = party_data['sentiment']
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42)

        best_sent_acc = 0
        best_sent_model = None
        best_sent_name = None

        for name, clf in model_candidates.items():
            print(f"{name} for {party} sentiment classification")
            start_time = time.time()

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', clf)
            ])
            pipeline.fit(X_train_s, y_train_s)

            report_path = os.path.join(MODEL_DIR, f"{party}_{name}_sentiment_report.txt")
            acc = evaluate_model(pipeline, X_test_s, y_test_s, f"{party.capitalize()} sentiment classifier ({name})", output_path=report_path)

            elapsed = time.time() - start_time
            print(f"Training + evaluation time for {name} ({party} sentiment): {elapsed:.2f} seconds")

            if acc > best_sent_acc:
                best_sent_acc = acc
                best_sent_model = pipeline
                best_sent_name = name

        sentiment_model_path = os.path.join(MODEL_DIR, f"{party}_{best_sent_name}_sentiment_classifier.joblib")
        joblib.dump(best_sent_model, sentiment_model_path)
        print(f"Best {party} sentiment classifier: {best_sent_name} saved to {sentiment_model_path}")
        sentiment_pipelines[party] = best_sent_model

    print(f"\nTotal runtime: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
