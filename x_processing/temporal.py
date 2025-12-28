import os
import time
import warnings
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

MODEL_DIR = "x_processing/models"
OUTPUT_DIR = "x_processing/outputs"
FIGURE_DIR = "x_processing/figures"
PREDICTIONS_PARQUET = "x_processing/predictions_with_timestamps.parquet"

BIDEN_START = datetime(2024, 5, 1)
BIDEN_END = datetime(2024, 7, 21)
HARRIS_START = datetime(2024, 7, 22)
ELECTION_DAY = datetime(2024, 11, 5)

def assign_campaign_phase(ts):
    if pd.isna(ts):
        return "Unknown"
    if BIDEN_START <= ts <= BIDEN_END:
        return "Biden Campaign"
    elif HARRIS_START <= ts < ELECTION_DAY:
        return "Harris Campaign"
    elif ts >= ELECTION_DAY:
        return "Post-Election"
    else:
        return "Other"

def compute_sentiment_gap(df, freq='W'):
    df = df.dropna(subset=['party', 'sentiment', 'timestamp']).copy()
    df['week'] = df['timestamp'].dt.to_period(freq)
    
    weekly = df.groupby(['week', 'party']).agg(
        total_tweets=('sentiment', 'count'),
        positive_tweets=('sentiment', lambda x: (x=='positive').sum())
    ).reset_index()
    
    weekly['positive_ratio'] = weekly['positive_tweets'] / weekly['total_tweets']
    
    pivot = weekly.pivot(index='week', columns='party', values='positive_ratio').fillna(0)
    pivot['sentiment_gap'] = pivot.get('republican', 0) - pivot.get('democrat', 0)
    return pivot.reset_index()

def plot_sentiment_gap(weekly_gap, output_path):
    plt.figure(figsize=(12, 5))
    plt.plot(weekly_gap['week'].astype(str), weekly_gap.get('democrat', 0), label='Democrat', color='#0015BC')
    plt.plot(weekly_gap['week'].astype(str), weekly_gap.get('republican', 0), label='Republican', color='#E81B23')
    plt.plot(weekly_gap['week'].astype(str), weekly_gap['sentiment_gap'], label='Rep-Dem Gap', color='green', linestyle='--')
    plt.xticks(rotation=45)
    plt.xlabel('Week')
    plt.ylabel('Positive Sentiment Ratio')
    plt.title('Weekly Positive Sentiment & Gap')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def compute_volume_trends(df, freq='W'):
    df = df.dropna(subset=['party', 'timestamp']).copy()
    df['week'] = df['timestamp'].dt.to_period(freq)
    
    weekly_volume = df.groupby(['week', 'party']).size().reset_index(name='tweet_count')
    pivot = weekly_volume.pivot(index='week', columns='party', values='tweet_count').fillna(0)
    return pivot.reset_index()

def plot_volume_trends(weekly_volume, output_path):
    plt.figure(figsize=(12, 5))
    plt.bar(weekly_volume['week'].astype(str), weekly_volume.get('democrat', 0)/1e6, label='Democrat', alpha=0.8, color='#0015BC')
    plt.bar(weekly_volume['week'].astype(str), weekly_volume.get('republican', 0)/1e6, bottom=weekly_volume.get('democrat', 0)/1e6, label='Republican', alpha=0.8, color='#E81B23')
    plt.xticks(rotation=45)
    plt.ylabel('Tweet Volume (millions)')
    plt.xlabel('Week')
    plt.title('Weekly Tweet Volume by Party')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def generate_predictions(unlabelled_df):
    print(f"\nGenerating predictions for {len(unlabelled_df):,} tweets...")
    
    # Load pre-trained models
    party_model = joblib.load(os.path.join(MODEL_DIR, "LightGBM_party_classifier.joblib"))
    dem_sent_model = joblib.load(os.path.join(MODEL_DIR, "LinearSVC_democrat_sentiment_classifier.joblib"))
    rep_sent_model = joblib.load(os.path.join(MODEL_DIR, "LinearSVC_republican_sentiment_classifier.joblib"))
    
    # Party prediction
    print("Classifying party affiliation...")
    unlabelled_df['party'] = party_model.predict(unlabelled_df['clean_text'])
    
    # Sentiment prediction per party
    print("Classifying sentiment for Democrats...")
    dem_mask = unlabelled_df['party'] == 'democrat'
    unlabelled_df.loc[dem_mask, 'sentiment'] = dem_sent_model.predict(unlabelled_df.loc[dem_mask, 'clean_text'])
    
    print("Classifying sentiment for Republicans...")
    rep_mask = unlabelled_df['party'] == 'republican'
    unlabelled_df.loc[rep_mask, 'sentiment'] = rep_sent_model.predict(unlabelled_df.loc[rep_mask, 'clean_text'])
    
    # Save
    unlabelled_df.to_parquet(PREDICTIONS_PARQUET, index=False)
    print(f"Predictions saved: {PREDICTIONS_PARQUET}")
    return unlabelled_df

def phase_based_sentiment(df):
    df['phase'] = df['timestamp'].apply(assign_campaign_phase)
    
    results = []
    for party in ['democrat', 'republican']:
        for phase in ['Biden Campaign', 'Harris Campaign', 'Post-Election']:
            subset = df[(df['party']==party) & (df['phase']==phase)]
            if subset.empty:
                continue
            total = len(subset)
            pos = (subset['sentiment']=='positive').sum()
            neg = (subset['sentiment']=='negative').sum()
            results.append({
                'party': party,
                'phase': phase,
                'total_tweets': total,
                'positive_ratio': pos / total,
                'negative_ratio': neg / total
            })
    return pd.DataFrame(results)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # Load or generate predictions
    if os.path.exists(PREDICTIONS_PARQUET):
        print(f"\nLoading existing predictions: {PREDICTIONS_PARQUET}")
        df = pd.read_parquet(PREDICTIONS_PARQUET)
    else:
        # Assume unlabelled_df already exists with 'clean_text' and 'timestamp'
        unlabelled_df = pd.read_parquet("x_processing/train_unlabelled.parquet")
        df = generate_predictions(unlabelled_df)
    
    # Phase-based sentiment
    phase_results = phase_based_sentiment(df)
    phase_csv = os.path.join(OUTPUT_DIR, "phase_sentiment.csv")
    phase_results.to_csv(phase_csv, index=False)
    print(f"\nPhase sentiment saved: {phase_csv}")
    
    # Sentiment gap over time
    weekly_gap = compute_sentiment_gap(df)
    gap_fig = os.path.join(FIGURE_DIR, "weekly_sentiment_gap.png")
    plot_sentiment_gap(weekly_gap, gap_fig)
    
    # Volume trends
    weekly_volume = compute_volume_trends(df)
    volume_fig = os.path.join(FIGURE_DIR, "weekly_volume.png")
    plot_volume_trends(weekly_volume, volume_fig)
    
if __name__ == "__main__":
    main()
