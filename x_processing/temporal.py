import os
import warnings
from datetime import datetime
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

PATHS = {
    "models": "x_processing/models",
    "output": "x_processing/outputs",
    "input": "x_processing/train_unlabelled.parquet",
    "predictions": "x_processing/predictions_with_timestamps.parquet"
}

# Key campaign events
EVENTS = {
    "Biden Exit": datetime(2024, 7, 21),
    "Harris Entry": datetime(2024, 7, 22),
    "Trump Assassination": datetime(2024, 10, 15),  # Example date
    "Election Day": datetime(2024, 11, 5)
}

def assign_campaign_phase(ts):
    """Segment tweets into structural campaign periods based on dates."""
    if pd.isna(ts): 
        return "Unknown"
    
    if datetime(2024, 5, 1) <= ts <= EVENTS["Biden Exit"]:
        return "Biden Campaign"
    elif EVENTS["Harris Entry"] <= ts < EVENTS["Election Day"]:
        return "Harris Campaign"
    elif ts >= EVENTS["Election Day"]:
        return "Post-Election"
    return "Other"

def assign_event(ts):
    """Flag tweets corresponding to key campaign events."""
    if pd.isna(ts):
        return "No Event"
    for event_name, event_date in EVENTS.items():
        # Mark tweets on the exact day of the event
        if ts.date() == event_date.date():
            return event_name
    return "No Event"

def generate_predictions(df):
    """Apply pre-trained classifiers to unlabelled data."""
    print("Loading models and classifying tweets...")
    p_model = joblib.load(os.path.join(PATHS["models"], "LightGBM_party_classifier.joblib"))
    d_model = joblib.load(os.path.join(PATHS["models"], "LinearSVC_democrat_sentiment_classifier.joblib"))
    r_model = joblib.load(os.path.join(PATHS["models"], "LinearSVC_republican_sentiment_classifier.joblib"))
    
    # Predict Party
    df['party'] = p_model.predict(df['clean_text'])
    
    # Predict Sentiment based on Party
    for party, model in [('democrat', d_model), ('republican', r_model)]:
        mask = df['party'] == party
        if mask.any():
            df.loc[mask, 'sentiment'] = model.predict(df.loc[mask, 'clean_text'])
    
    df.to_parquet(PATHS["predictions"], index=False)
    return df

def calculate_swings(df):
    """Calculate sentiment swings across campaign phases."""
    core_df = df[df['phase'].isin(["Biden Campaign", "Harris Campaign"])]
    core_df = core_df.dropna(subset=['sentiment', 'party'])
    
    # % Positive sentiment by phase and party
    metrics = core_df.groupby(['phase', 'party'])['sentiment'].apply(
        lambda x: (x == 'positive').mean()
    ).unstack()
    
    # Calculate swing from Biden -> Harris
    if 'Harris Campaign' in metrics.index and 'Biden Campaign' in metrics.index:
        swing = metrics.loc['Harris Campaign'] - metrics.loc['Biden Campaign']
        swing.name = 'Sentiment Swing'
        return pd.concat([metrics, swing.to_frame().T])
    return metrics

def event_sentiment_summary(df):
    """Summarize sentiment for key campaign events."""
    event_summary = df.groupby(['event', 'party'])['sentiment'].apply(
        lambda x: (x == 'positive').mean()
    ).unstack()
    return event_summary

def main():
    os.makedirs(PATHS["output"], exist_ok=True)

    # Load or Generate Predictions
    if os.path.exists(PATHS["predictions"]):
        print(f"Loading existing predictions from {PATHS['predictions']}")
        df = pd.read_parquet(PATHS["predictions"])
    else:
        raw_df = pd.read_parquet(PATHS["input"])
        df = generate_predictions(raw_df)

    # Assign campaign phases and events
    df['phase'] = df['timestamp'].apply(assign_campaign_phase)
    df['event'] = df['timestamp'].apply(assign_event)

    # Phase Swing Analysis
    swing_results = calculate_swings(df)
    swing_results.to_csv(os.path.join(PATHS["output"], "campaign_swing_analysis.csv"))

    # Event-Driven Sentiment Summary
    event_summary = event_sentiment_summary(df)
    event_summary.to_csv(os.path.join(PATHS["output"], "event_sentiment_summary.csv"))

    print("Phase Swing Analysis:")
    print(swing_results)
    print("\nEvent-Driven Sentiment Summary:")
    print(event_summary)
    print(f"\nReports saved to: {PATHS['output']}")

if __name__ == "__main__":
    main()
