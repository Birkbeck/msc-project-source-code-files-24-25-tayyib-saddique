import os
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import joblib

warnings.filterwarnings("ignore")

# --- Configuration & Paths ---
MODEL_DIR = "x_processing/models"
OUTPUT_DIR = "x_processing/outputs"
FIGURE_DIR = "x_processing/figures"
PREDICTIONS_PARQUET = "x_processing/predictions_with_timestamps.parquet"
INPUT_DATA = "x_processing/train_unlabelled.parquet"

# Temporal Scope
ANALYSIS_START = datetime(2024, 5, 1)
ANALYSIS_END = datetime(2024, 11, 30, 23, 59, 59)

# Key Events
EVENTS = {
    "Trump Assassination Attempt": datetime(2024, 7, 13),
    "Biden Withdrawal": datetime(2024, 7, 21),
    "Harris Nomination": datetime(2024, 8, 6)
}

# Campaign Phases
PHASES = {
    "Biden Campaign": (datetime(2024, 5, 1), datetime(2024, 7, 21)),
    "Harris Campaign": (datetime(2024, 7, 22), datetime(2024, 11, 5)),
    "Post-Election": (datetime(2024, 11, 6), datetime(2024, 11, 30))
}

def assign_campaign_phase(ts):
    if pd.isna(ts): return "Unknown"
    for phase, (start, end) in PHASES.items():
        if start <= ts <= end: return phase
    return "Outside Scope"

def generate_discourse_predictions(df):
    """
    Classifies 'discourse_alignment' (Topic) and 'sentiment' (Tone).
    Renamed from 'party' to reflect that we are tracking discourse target.
    """
    print(f"Generating predictions for {len(df):,} tweets...")
    
    party_model = joblib.load(os.path.join(MODEL_DIR, "LightGBM_party_classifier.joblib"))
    dem_model = joblib.load(os.path.join(MODEL_DIR, "LinearSVC_democrat_sentiment_classifier.joblib"))
    rep_model = joblib.load(os.path.join(MODEL_DIR, "LinearSVC_republican_sentiment_classifier.joblib"))

    # Track who the discourse is ABOUT
    df['discourse_alignment'] = party_model.predict(df['clean_text'])
    
    dem_mask = df['discourse_alignment'] == 'democrat'
    rep_mask = df['discourse_alignment'] == 'republican'

    df.loc[dem_mask, 'sentiment'] = dem_model.predict(df.loc[dem_mask, 'clean_text'])
    df.loc[rep_mask, 'sentiment'] = rep_model.predict(df.loc[rep_mask, 'clean_text'])

    df.to_parquet(PREDICTIONS_PARQUET, index=False)
    return df

def analyze_event_impact(df, window_days=7):
    """Statistically evaluates if an event shifted the tone of party-aligned discourse."""
    results = []
    for event, date in EVENTS.items():
        for align in ['democrat', 'republican']:
            pre = df[(df['discourse_alignment'] == align) & 
                     (df['timestamp'] >= date - timedelta(days=window_days)) & (df['timestamp'] < date)]
            post = df[(df['discourse_alignment'] == align) & 
                      (df['timestamp'] >= date) & (df['timestamp'] <= date + timedelta(days=window_days))]

            if len(pre) < 15 or len(post) < 15: continue

            # Chi-Square Test for Sentiment Distribution Shift
            contingency = pd.crosstab(['pre']*len(pre) + ['post']*len(post),
                                      list(pre['sentiment']) + list(post['sentiment']))
            _, p_val, _, _ = stats.chi2_contingency(contingency)

            results.append({
                'event': event, 'alignment': align, 
                'shift': (post['sentiment'] == 'positive').mean() - (pre['sentiment'] == 'positive').mean(),
                'significant': p_val < 0.05
            })
    return pd.DataFrame(results)

def plot_rolling_discourse(df):
    """Visualizes the 7-day rolling sentiment of party-aligned discourse."""
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby(['date', 'discourse_alignment']).agg(
        pos_ratio=('sentiment', lambda x: (x == 'positive').mean())
    ).reset_index()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {'democrat': '#1f77b4', 'republican': '#d62728'}

    for align in ['democrat', 'republican']:
        data = daily[daily['discourse_alignment'] == align].sort_values('date')
        rolling = data['pos_ratio'].rolling(7).mean()
        ax.plot(data['date'], rolling, label=f"{align.capitalize()}-Aligned Discourse", color=colors[align], lw=2.5)

    for event, dt in EVENTS.items():
        ax.axvline(x=dt.date(), color='gray', ls='--', alpha=0.6)
        ax.text(dt.date(), ax.get_ylim()[1]*0.95, f" {event}", rotation=90, size=9)

    ax.set_title("2024 Election: Sentiment of Party-Aligned Discourse", fontsize=15)
    ax.set_ylabel("Positive Sentiment Ratio (7-Day MA)")
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "discourse_sentiment.png"), dpi=300)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = pd.read_parquet(PREDICTIONS_PARQUET) if os.path.exists(PREDICTIONS_PARQUET) else \
         generate_discourse_predictions(pd.read_parquet(INPUT_DATA))

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'] >= ANALYSIS_START) & (df['timestamp'] <= ANALYSIS_END)]
    
    # Run Analyses
    event_impact = analyze_event_impact(df)
    event_impact.to_csv(os.path.join(OUTPUT_DIR, "event_discourse_impact.csv"), index=False)
    
    plot_rolling_discourse(df)
    print("Pipeline Updated: Terminology now reflects 'Aligned Discourse'.")

if __name__ == "__main__":
    main()