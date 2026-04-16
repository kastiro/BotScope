"""
BotScope — Model Evaluation Script
====================================
Reproduces the evaluation results reported in the thesis.

Loads all labelled user profiles directly from the Cresci et al. dataset
CSVs (data_source/), applies the same 80/20 stratified split
(random_state=42) used during training, runs the trained Decision Tree
Classifier on the held-out test set, and prints:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion matrix (TN, FP, FN, TP)
  - Feature importances

Usage:
    pip install -r requirements.txt
    python scripts/evaluate_model.py

Run from the project root directory (where model/ and data_source/ live).
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "model", "bot_detector.pkl")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data_source")

BOT_FOLDERS = [
    "social_spambots_1.csv",
    "social_spambots_2.csv",
    "social_spambots_3.csv",
    "traditional_spambots_1.csv",
    "traditional_spambots_2.csv",
    "traditional_spambots_3.csv",
    "traditional_spambots_4.csv",
]
HUMAN_FOLDER = "genuine_accounts.csv"

FEATURE_NAMES = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "reputation",
    "post_to_follower_ratio",
]


def load_users_csv(folder_path, is_bot):
    """Load users.csv from a dataset folder and compute the 5 features."""
    csv_path = os.path.join(folder_path, "users.csv")
    df = pd.read_csv(csv_path, usecols=["statuses_count", "followers_count", "friends_count"])

    df["statuses_count"]   = df["statuses_count"].fillna(0).astype(float)
    df["followers_count"]  = df["followers_count"].fillna(0).astype(float)
    df["friends_count"]    = df["friends_count"].fillna(0).astype(float)

    df["reputation"]            = df["followers_count"] / (df["followers_count"] + df["friends_count"] + 1)
    df["post_to_follower_ratio"] = df["statuses_count"]  / (df["followers_count"] + 1)
    df["is_bot"]                = int(is_bot)

    return df[FEATURE_NAMES + ["is_bot"]]


def load_dataset():
    bot_frames   = []
    human_frames = []

    for folder in BOT_FOLDERS:
        path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(path):
            print(f"WARNING: missing folder {path} — skipping")
            continue
        bot_frames.append(load_users_csv(path, is_bot=True))

    human_path = os.path.join(DATA_DIR, HUMAN_FOLDER)
    if os.path.isdir(human_path):
        human_frames.append(load_users_csv(human_path, is_bot=False))
    else:
        print(f"WARNING: missing folder {human_path}")

    return pd.concat(bot_frames + human_frames, ignore_index=True)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading dataset from CSV files...")
    df = load_dataset()

    print(f"Total labelled profiles: {len(df)}")
    print(f"  Bots:   {df['is_bot'].sum()}")
    print(f"  Humans: {(df['is_bot'] == 0).sum()}")

    X = df[FEATURE_NAMES]
    y = df["is_bot"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTest set size: {len(X_test)} records")
    print(f"  Bots in test:   {y_test.sum()}")
    print(f"  Humans in test: {(y_test == 0).sum()}")

    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)

    print("Running predictions on test set...")
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")

    print("\nCONFUSION MATRIX")
    print("-" * 30)
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives  (TP): {tp}")

    print("\nFEATURE IMPORTANCES")
    print("-" * 30)
    importances = model.feature_importances_
    for name, imp in sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name:<25}: {imp * 100:.1f}%")

    print("=" * 50)


if __name__ == "__main__":
    main()
