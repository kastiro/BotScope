"""
BotScope — Model Evaluation Script
====================================
Reproduces the evaluation results reported in the thesis.

Loads all labelled user profiles from the database, applies the same
80/20 stratified split (random_state=42) used during training, runs the
trained Decision Tree Classifier on the held-out test set, and prints:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion matrix (TN, FP, FN, TP)
  - Feature importances

Usage:
    python scripts/evaluate_model.py

Run from the project root directory (where spyfind.db and model/ live).
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "spyfind.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "bot_detector.pkl")

os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

from app.database import SessionLocal
from app.models import User, BotDetection


def load_dataset(db):
    """Load all labelled users and compute the 5 features."""
    users = (
        db.query(User)
        .join(BotDetection, User.id == BotDetection.user_id)
        .all()
    )

    records = []
    for user in users:
        statuses  = user.posts_count or 0
        followers = user.followers_count or 0
        friends   = user.following_count or 0
        reputation          = followers / (followers + friends + 1)
        post_to_follower    = statuses  / (followers + 1)
        is_bot = user.bot_status.is_bot if user.bot_status else False

        records.append({
            "statuses_count":        statuses,
            "followers_count":       followers,
            "friends_count":         friends,
            "reputation":            reputation,
            "post_to_follower_ratio": post_to_follower,
            "is_bot":                int(is_bot),
        })

    return pd.DataFrame(records)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    print("Loading dataset from database...")
    db = SessionLocal()
    try:
        df = load_dataset(db)
    finally:
        db.close()

    print(f"Total labelled profiles: {len(df)}")
    print(f"  Bots:   {df['is_bot'].sum()}")
    print(f"  Humans: {(df['is_bot'] == 0).sum()}")

    feature_names = [
        "statuses_count",
        "followers_count",
        "friends_count",
        "reputation",
        "post_to_follower_ratio",
    ]

    X = df[feature_names]
    y = df["is_bot"]

    # Same split used during training
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
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name:<25}: {imp * 100:.1f}%")

    print("=" * 50)


if __name__ == "__main__":
    main()
