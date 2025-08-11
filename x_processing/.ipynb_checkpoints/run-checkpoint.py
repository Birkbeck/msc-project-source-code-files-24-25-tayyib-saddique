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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#  NLTK Setup 
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

# Preprocessing 
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

# File processing
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

# Candidates for supervised ML
MODEL_CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, class_weight='balanced'),
    "LinearSVC": LinearSVC(max_iter=2000, class_weight='balanced'),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
}

def train_best_model(X_train, X_test, y_train, y_test, model_candidates):
    best_model = None
    best_score = 0
    best_name = None

    for name, model in model_candidates.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ('scaler', MaxAbsScaler() if name != "MultinomialNB" else "passthrough"),
            ('clf', model)
        ])

        pipeline.fit(X_train, y_train)
        score = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"{name} Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name

    print(f"Best model: {best_name} (Accuracy: {best_score:.4f})\n")
    return best_model, best_name, best_score

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

    labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    print(f"Total labeled samples: {len(labeled_df)}")
    labeled_df.to_parquet("x_processing/train_labelled.parquet", index=False)

    if unlabeled_dfs:
        unlabeled_df = pd.concat(unlabeled_dfs, ignore_index=True)
        unlabeled_df.to_parquet("x_processing/train_unlabelled.parquet", index=False)
    else:
        unlabeled_df = None

    # Party Classifier
    print("=== Training Party Classifier ===")
    X_party = labeled_df['clean_text']
    y_party = labeled_df['party']

    X_train, X_test, y_train, y_test = train_test_split(X_party, y_party, test_size=0.2, random_state=42)
    best_party_model, _, _ = train_best_model(X_train, X_test, y_train, y_test, MODEL_CANDIDATES)
    joblib.dump(best_party_model, "x_processing/party_classifier.joblib")

    # Sentiment Classifiers for each party 
    for party in ["democrat", "republican"]:
        print(f"=== Training Sentiment Classifier for {party} ===")
        df_party = labeled_df[labeled_df['party'] == party]
        X_sent = df_party['clean_text']
        y_sent = df_party['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42)
        best_sent_model, _, _ = train_best_model(X_train, X_test, y_train, y_test, MODEL_CANDIDATES)
        joblib.dump(best_sent_model, f"x_processing/{party}_sentiment_classifier.joblib")


if __name__ == "__main__":
    main()
