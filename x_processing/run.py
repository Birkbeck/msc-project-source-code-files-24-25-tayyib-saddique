import os
import re
import time
import warnings
from datetime import datetime
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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Directories
MODEL_DIR = "x_processing/models"
OUTPUT_DIR = "x_processing/outputs"
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "x-24-us-election"))

# File paths
LABELLED_PARQUET = "x_processing/train_labelled.parquet"
UNLABELLED_PARQUET = "x_processing/train_unlabelled.parquet"
PREDICTIONS_PARQUET = "x_processing/predictions_with_timestamps.parquet"

# Configuration
CANDIDATE_KEYWORDS = {
    'democrat': ['#bidenharris2024', '#kamalaharris2024', '@joebiden', '@kamalaharris', 'democrats'],
    'republican': ['#maga', 'republican', '#trump2024', '@realdonaldtrump']
}
STRONG_SENTIMENT_THRESHOLD = 0.8
CAMPAIGN_START_DATE = pd.Timestamp("2024-05-01")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vader = SentimentIntensityAnalyzer()


def preprocess(text):
    """
    Preprocess tweet text: lowercase, remove URLs/mentions, lemmatize, remove stopwords.
    
    Args:
        text (str): Raw tweet text
        
    Returns:
        str: Cleaned text
    """
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+|https\S+|@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s#']", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def detect_candidate(text):
    """
    Detect party affiliation based on keywords.
    
    Args:
        text (str): Tweet text
        
    Returns:
        str or None: 'democrat', 'republican', or None
    """
    text = text.lower()
    for candidate, keywords in CANDIDATE_KEYWORDS.items():
        if any(k in text for k in keywords):
            return candidate
    return None


def label_sentiment(text):
    """
    Label sentiment using VADER with conservative threshold.
    
    Args:
        text (str): Tweet text
        
    Returns:
        str or None: 'positive', 'negative', or None
    """
    score = vader.polarity_scores(text)['compound']
    if score >= STRONG_SENTIMENT_THRESHOLD:
        return 'positive'
    elif score <= -STRONG_SENTIMENT_THRESHOLD:
        return 'negative'
    return None


def convert_to_timestamp(df):
    """
    Create 'timestamp' column from date/epoch, coerce errors.
    Uses epoch as fallback for missing dates.
    
    Args:
        df (pd.DataFrame): DataFrame with 'date' and 'epoch' columns
        
    Returns:
        pd.DataFrame: DataFrame with 'timestamp' column
    """
    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    missing_date_mask = df['timestamp'].isna() & df['epoch'].notna()
    df.loc[missing_date_mask, 'timestamp'] = pd.to_datetime(
        df.loc[missing_date_mask, 'epoch'], unit='s'
    )
    return df


def filter_campaign_tweets(df):
    """
    Keep only tweets after campaign start date (May 1, 2024).
    
    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    df = convert_to_timestamp(df)
    return df[df['timestamp'] >= CAMPAIGN_START_DATE]


def load_preprocess_weak_label(file_path):
    """
    Load, preprocess, and weakly label a single CSV file.
    
    Args:
        file_path (str): Path to gzipped CSV file
        
    Returns:
        tuple: (labelled_df, unlabelled_df) or (None, None)
    """
    try:
        df = pd.read_csv(
            file_path,
            compression="gzip",
            usecols=["id", "rawContent", "lang", "date", "epoch"],
            dtype={"id": str, "rawContent": str, "lang": str, "date": str, "epoch": object}
        )
        
        # Filter English tweets only
        df = df[df["lang"] == "en"]
        if df.empty:
            return None, None

        # Filter to campaign period
        df = filter_campaign_tweets(df)
        if df.empty:
            return None, None
        
        # Rename and preprocess
        df = df.rename(columns={"rawContent": "text"})
        df['clean_text'] = df['text'].apply(preprocess)
        
        # Weak labeling
        df['party'] = df['text'].apply(detect_candidate)
        df['sentiment_score'] = df['text'].apply(
            lambda t: vader.polarity_scores(t)['compound']
        )
        df['sentiment'] = df['sentiment_score'].apply(
            lambda score: 'positive' if score >= STRONG_SENTIMENT_THRESHOLD 
            else 'negative' if score <= -STRONG_SENTIMENT_THRESHOLD 
            else None
        )

        # Split labeled and unlabeled
        labelled = df.dropna(subset=['party', 'sentiment'])
        unlabelled = df[df['party'].isna() | df['sentiment'].isna()]

        if labelled.empty and unlabelled.empty:
            return None, None

        return (
            labelled[['clean_text', 'date', 'party', 'sentiment', 'sentiment_score', 'timestamp']], 
            unlabelled[['id', 'clean_text', 'text', 'date', 'timestamp']]
        )
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
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


def check_existing_models(task_name, model_names):
    """
    Check if models for a given task already exist.
    
    Args:
        task_name (str): Task identifier (e.g., 'party', 'democrat_sentiment')
        model_names (list): List of model names to check
        
    Returns:
        dict: {model_name: model_path} for existing models, or empty dict if any missing
    """
    existing_models = {}
    all_exist = True
    
    for name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{name}_{task_name}_classifier.joblib")
        if os.path.exists(model_path):
            existing_models[name] = model_path
        else:
            all_exist = False
            break
    
    return existing_models if all_exist else {}


def load_existing_models(task_name, model_names):
    """
    Load existing trained models.
    
    Args:
        task_name (str): Task identifier
        model_names (list): List of model names to load
        
    Returns:
        dict: {model_name: loaded_pipeline}
    """
    loaded_models = {}
    print(f"\nLoading existing models for {task_name}:")
    
    for name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{name}_{task_name}_classifier.joblib")
        try:
            pipeline = joblib.load(model_path)
            loaded_models[name] = pipeline
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return loaded_models


def evaluate_model(pipeline, X_test, y_test, model_name, output_path=None):
    """
    Evaluate model on test set and optionally save report.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test labels
        model_name (str): Model name for reporting
        output_path (str, optional): Path to save evaluation report
        
    Returns:
        float: Test accuracy
    """
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = (
        f"\n{model_name}\n"
        f"Accuracy: {acc:.4f}\n\n"
        f"Classification Report:\n{classification_report(y_test, y_pred)}\n"
        f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n"
    )
    print(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return acc


def train_and_save_models(X, y, model_candidates, task_name, 
                          tfidf_max_features=20000, ngram_range=(1, 2), 
                          n_splits=5):
    """
    Train models with cross-validation, evaluate on test set, and save.
    
    Args:
        X: Features (text)
        y: Labels
        model_candidates (dict): Dictionary of {name: classifier}
        task_name (str): Task identifier (e.g., 'party', 'democrat_sentiment')
        tfidf_max_features (int): Maximum TF-IDF features
        ngram_range (tuple): N-gram range for TF-IDF
        n_splits (int): Number of CV folds
        
    Returns:
        list: List of result dictionaries
    """
    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    for name, clf in model_candidates.items():        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=tfidf_max_features, ngram_range=ngram_range)),
            ('clf', clf)
        ])

        # Cross-validation
        print("Running cross-validation...")
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=skf,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            n_jobs=-1, return_train_score=False
        )

        cv_summary = (
            f"\n{name} Cross-Validation Results ({task_name}):\n"
            f"Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}\n"
            f"Precision: {cv_results['test_precision_macro'].mean():.4f} ± {cv_results['test_precision_macro'].std():.4f}\n"
            f"Recall:    {cv_results['test_recall_macro'].mean():.4f} ± {cv_results['test_recall_macro'].std():.4f}\n"
            f"F1:        {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}\n"
        )
        print(cv_summary)

        # Train final model on full training set
        print("Training final model...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"Training time: {elapsed:.2f} seconds")
        
        # Evaluate on test set
        eval_output_path = os.path.join(OUTPUT_DIR, f"{name}_{task_name}_evaluation.txt")
        test_acc = evaluate_model(pipeline, X_test, y_test, f"{name} ({task_name})", 
                                  output_path=eval_output_path)

        # Save CV results
        cv_output_path = os.path.join(OUTPUT_DIR, f"{name}_{task_name}_cv_results.txt")
        with open(cv_output_path, "w") as f:
            f.write(cv_summary)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}_{task_name}_classifier.joblib")
        joblib.dump(pipeline, model_path)
        print(f"Model saved: {model_path}")

        results.append({
            "model": name,
            "pipeline": pipeline,
            "cv_results": cv_results,
            "test_accuracy": test_acc,
            "model_path": model_path,
            "cv_report_path": cv_output_path,
            "eval_report_path": eval_output_path
        })
        
    return results


def main():
    """
    Main pipeline: preprocess, train (if needed), predict, and prepare for analysis.
    """
    # Create directories
    for directory in [MODEL_DIR, OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    total_start = time.time()

    
    start = time.time()
    if os.path.exists(LABELLED_PARQUET) and os.path.exists(UNLABELLED_PARQUET):
        print(f"\nLoading existing data:")
        print(f"  Labelled:   {LABELLED_PARQUET}")
        print(f"  Unlabelled: {UNLABELLED_PARQUET}")
        labelled_df = pd.read_parquet(LABELLED_PARQUET)
        unlabelled_df = pd.read_parquet(UNLABELLED_PARQUET)
    else:
        print("\nPreprocessing raw data from CSV files...")
        print(f"Campaign period: {CAMPAIGN_START_DATE} onwards")
        all_files = find_all_files_recursively(INPUT_DIR)
        print(f"Found {len(all_files)} files to process")

        labelled_dfs, unlabelled_dfs = [], []

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(load_preprocess_weak_label, file): file 
                      for file in all_files}

            for i, future in enumerate(as_completed(futures), 1):
                file = futures[future]
                try:
                    labelled, unlabelled = future.result()
                    if labelled is not None and not labelled.empty:
                        labelled_dfs.append(labelled)
                    if unlabelled is not None and not unlabelled.empty:
                        unlabelled_dfs.append(unlabelled)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

                if i % 50 == 0:
                    print(f"Processed {i}/{len(all_files)} files...")

        if not labelled_dfs:
            print("ERROR: No labelled data found. Exiting.")
            return
        
        labelled_df = pd.concat(labelled_dfs, ignore_index=True)
        unlabelled_df = pd.concat(unlabelled_dfs, ignore_index=True) if unlabelled_dfs else pd.DataFrame()

        labelled_df.to_parquet(LABELLED_PARQUET, index=False)
        unlabelled_df.to_parquet(UNLABELLED_PARQUET, index=False)

        print(f"\nSaved:")
        print(f"  Labelled:   {LABELLED_PARQUET}")
        print(f"  Unlabelled: {UNLABELLED_PARQUET}")

    # Clean data
    labelled_df.dropna(subset=['party', 'sentiment', 'clean_text'], inplace=True)
    labelled_df = labelled_df[labelled_df['clean_text'].str.len() > 0]
    labelled_df.reset_index(drop=True, inplace=True)
    
    print(f"\nData loading completed in {time.time() - start:.2f} seconds")
    print(f"\nDataset Summary:")
    print(f"  Labelled tweets:   {len(labelled_df):,}")
    print(f"  Unlabelled tweets: {len(unlabelled_df):,}")
    print(f"\nParty Distribution (Labelled):")
    print(labelled_df['party'].value_counts())
    print(f"\nSentiment Distribution (Labelled):")
    print(labelled_df.groupby(['party', 'sentiment']).size())
    

    # Party Classification Models
    party_model_names = ["LightGBM"]
    existing_party_models = check_existing_models("party", party_model_names)
    
    # if existing_party_models:
    #     print(f"\nFound existing party models")
    #     party_models = load_existing_models("party", party_model_names)
    # else:
    print("\nTraining party classifier (LightGBM)...")
    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']

    party_model_candidates = {
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

    train_and_save_models(
        X_party, y_party, party_model_candidates, 
        task_name="party", tfidf_max_features=20000
    )

    # Democrat Sentiment Models
    sentiment_model_names = ["LogisticRegression", "LinearSVC"]
    existing_dem_models = check_existing_models("democrat_sentiment", sentiment_model_names)
    
    # if existing_dem_models:
    #     print(f"\nFound existing Democrat sentiment models")
    #     dem_models = load_existing_models("democrat_sentiment", sentiment_model_names)
    # else:
    print("\nTraining sentiment classifiers for Democrats...")
    dem_data = labelled_df[labelled_df['party'] == 'democrat']
    X_dem = dem_data['clean_text']
    y_dem = dem_data['sentiment']

    dem_sentiment_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=300,
            C=21.368329072358737,
            n_jobs=-1,
            verbose=1
        ),
        "LinearSVC": LinearSVC(
            max_iter=3000,
            C=0.6338653441536255,
            loss='squared_hinge',
        )
    }

    train_and_save_models(
        X_dem, y_dem, dem_sentiment_models, 
        task_name="democrat_sentiment", tfidf_max_features=10000
    )

    # Republican Sentiment Models
    existing_rep_models = check_existing_models("republican_sentiment", sentiment_model_names)
    
#     if existing_rep_models:
#         print(f"\nFound existing Republican sentiment models")
#         rep_models = load_existing_models("republican_sentiment", sentiment_model_names)
#     else:
    print("\nTraining sentiment classifiers for Republicans...")
    rep_data = labelled_df[labelled_df['party'] == 'republican']
    X_rep = rep_data['clean_text']
    y_rep = rep_data['sentiment']

    rep_sentiment_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=300,
            C=13.826232179369853,
            n_jobs=-1,
            verbose=1
        ),
        "LinearSVC": LinearSVC(
            max_iter=3000,
            C=0.6338653441536255,
            loss='squared_hinge'
        )
    }

    train_and_save_models(
        X_rep, y_rep, rep_sentiment_models, 
        task_name="republican_sentiment", tfidf_max_features=10000
    )
    
    # if not os.path.exists(PREDICTIONS_PARQUET):
    print(f"\nApplying models to {len(unlabelled_df):,} unlabelled tweets...")

    # Load trained models (use LinearSVC for sentiment as it typically performs best)
    party_model = joblib.load(os.path.join(MODEL_DIR, "LightGBM_party_classifier.joblib"))
    dem_sent_model = joblib.load(os.path.join(MODEL_DIR, "LinearSVC_democrat_sentiment_classifier.joblib"))
    rep_sent_model = joblib.load(os.path.join(MODEL_DIR, "LinearSVC_republican_sentiment_classifier.joblib"))

    # Predict party
    print("Classifying party affiliation...")
    unlabelled_df['party'] = party_model.predict(unlabelled_df['clean_text'])

    # Predict sentiment for each party
    print("Classifying sentiment for Democrats...")
    dem_mask = unlabelled_df['party'] == 'democrat'
    unlabelled_df.loc[dem_mask, 'sentiment'] = dem_sent_model.predict(
        unlabelled_df.loc[dem_mask, 'clean_text']
    )

    print("Classifying sentiment for Republicans...")
    rep_mask = unlabelled_df['party'] == 'republican'
    unlabelled_df.loc[rep_mask, 'sentiment'] = rep_sent_model.predict(
        unlabelled_df.loc[rep_mask, 'clean_text']
    )

    # Save predictions
    unlabelled_df.to_parquet(PREDICTIONS_PARQUET, index=False)
    print(f"\nPredictions saved: {PREDICTIONS_PARQUET}")
    # else:
    #     print(f"\nLoading existing predictions: {PREDICTIONS_PARQUET}")
    #     unlabelled_df = pd.read_parquet(PREDICTIONS_PARQUET)
    
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(unlabelled_df):,}")
    print(f"\nParty Distribution (Predicted):")
    print(unlabelled_df['party'].value_counts())
    print(f"\nSentiment by Party (Predicted):")
    print(unlabelled_df.groupby(['party', 'sentiment']).size())
    
    total_time = time.time() - total_start
    print(f"Total runtime: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")

if __name__ == "__main__":
    main()