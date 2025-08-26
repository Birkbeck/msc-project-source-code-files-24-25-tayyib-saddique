import os
import time
import joblib
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# CONFIG
MODEL_DIR = "x_processing/models/experiments"
FINE_TUNED_DIR = os.path.join(MODEL_DIR, "fine_tuned")
ALL_REPORTS_DIR = os.path.join(FINE_TUNED_DIR, "reports")
os.makedirs(ALL_REPORTS_DIR, exist_ok=True)

MODEL_N_ITER = {
    'LogisticRegression': 30,
    'LinearSVC': 30,
    'LightGBM': 20
}

labelled_parquet = "x_processing/train_labelled.parquet"
os.makedirs(FINE_TUNED_DIR, exist_ok=True)


# FUNCTIONS
def split_data(labelled_df, label_col, test_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(
        labelled_df['clean_text'], labelled_df[label_col],
        test_size=test_size, stratify=labelled_df[label_col], random_state=42
    )
    return X_train, X_val, y_train, y_val


def get_model_candidates(party=False):
    if party:
        return ['LightGBM', 'LinearSVC']
    else:
        return ['LinearSVC', 'LogisticRegression']


def hyperparameter_tune_pipeline(
    X_tfidf, y_train, model_type, n_iter=None, cv_folds=2, n_jobs=8
):
    if n_iter is None:
        n_iter = MODEL_N_ITER.get(model_type, 10)

    # Classifier + parameter distributions
    if model_type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=500, random_state=42)
        param_distributions = {
            'C': scipy.stats.loguniform(0.01, 100),
        }

    elif model_type == "LinearSVC":
        clf = LinearSVC(max_iter=3000, random_state=42)
        param_distributions = {
            'C': scipy.stats.loguniform(0.01, 100),
            'loss': ['hinge', 'squared_hinge'],
        }

    elif model_type == "LightGBM":
        clf = lgb.LGBMClassifier(n_jobs=8, force_row_wise=True, random_state=42)
        param_distributions = {
            'num_leaves': [31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 500],
            'max_depth': [-1, 30],
        }

    else:
        raise ValueError("Unsupported model_type")

    # Randomized search
    random_search = RandomizedSearchCV(
        clf,
        param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=n_jobs,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_tfidf, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def train_and_save_top_models(X, y, model_candidates, task_name, output_file):
    X_train_text, X_val_text, y_train, y_val = split_data(pd.DataFrame({'clean_text': X, 'label': y}), 'label')

    # Precompute TF-IDF once
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)

    best_acc = 0
    best_model_info = None

    for model_type in model_candidates:
        print(f"\nTraining {model_type} for task {task_name}")

        # Sample for tuning
        sample_frac = 0.2
        n_sample = int(X_train_tfidf.shape[0] * sample_frac)
        sample_idx = np.random.choice(X_train_tfidf.shape[0], n_sample, replace=False)
        X_sample = X_train_tfidf[sample_idx]
        y_sample = y_train.iloc[sample_idx]

        # Hyperparameter tuning
        best_estimator, best_params = hyperparameter_tune_pipeline(
            X_sample, y_sample, model_type
        )

        # Retrain on full training set
        best_estimator.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred = best_estimator.predict(X_val_tfidf)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        print(f"{model_type} validation accuracy: {acc:.4f}")

        # Save report
        model_report_path = os.path.join(ALL_REPORTS_DIR, f"{task_name}_{model_type}_report.txt")
        with open(model_report_path, 'w') as f:
            f.write(f"Task: {task_name}\nModel: {model_type}\nValidation Accuracy: {acc:.4f}\n\n")
            f.write("Best Parameters:\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\nClassification Report:\n")
            for label, metrics in report.items():
                f.write(f"{label}: {metrics}\n")
        print(f"Saved report for {model_type} at {model_report_path}")

        if acc > best_acc:
            best_acc = acc
            best_model_info = {
                'model': Pipeline([("tfidf", vectorizer), ("clf", best_estimator)]),
                'model_type': model_type,
                'accuracy': acc,
                'report': report,
                'best_params': best_params
            }

    # Save best model
    if best_model_info:
        save_path = f"{FINE_TUNED_DIR}/{best_model_info['model_type']}_{task_name}_best_model.joblib"
        joblib.dump(best_model_info['model'], save_path)
        print(f"\nSaved best model ({best_model_info['model_type']}) with accuracy {best_model_info['accuracy']:.4f} to {save_path}")

        # Save summary
        with open(output_file, 'w') as f:
            f.write(f"{best_model_info['model_type']}: {best_model_info['accuracy']:.4f} -> {save_path}\n")


# MAIN
def main():
    total_start = time.time()
    print(f"Loading labelled data from {labelled_parquet}")
    labelled_df = pd.read_parquet(labelled_parquet)
    print(f"Total rows in labelled data: {len(labelled_df)}")

    # Party Classification
    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']
    print("\nTraining party classifiers")
    train_and_save_top_models(
        X_party, y_party, get_model_candidates(party=True),
        task_name="party",
        output_file=os.path.join(MODEL_DIR, "party_distribution.txt")
    )

    # Sentiment Classification per Party
    for party in ['democrat', 'republican']:
        party_data = labelled_df[labelled_df['party'] == party]
        X_sent = party_data['clean_text']
        y_sent = party_data['sentiment']
        print(f"\nTraining sentiment classifiers for {party}")
        train_and_save_top_models(
            X_sent, y_sent, get_model_candidates(party=False),
            task_name=f"{party}_sentiment",
            output_file=os.path.join(MODEL_DIR, f"{party}_sentiment_distribution.txt")
        )

    print(f"\nTotal pipeline execution took {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
