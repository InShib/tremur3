
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Import the same feature extraction function and dataset loader
from app import load_data

print("Loading dataset...")
X, y, groups, windows = load_data()

print(f"Total samples: {len(y)}")
print(f"Total segments: {len(set(groups))}")

# Use the same segment-based split as the main application
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.22,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

# Use the same model settings as app.py
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
)

print("\nTraining evaluation model...")
model.fit(X[train_idx], y[train_idx])

print("Testing model...")
pred = model.predict(X[test_idx])

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy_score(y[test_idx], pred) * 100:.2f}%")
print(f"Macro F1-score: {f1_score(y[test_idx], pred, average='macro', zero_division=0) * 100:.2f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y[test_idx],
    pred,
    labels=[0, 1, 2, 3],
    zero_division=0,
    digits=3
))

print("\n========== CONFUSION MATRIX ==========")
print("Rows = actual labels; columns = predicted labels")
print(confusion_matrix(y[test_idx], pred, labels=[0, 1, 2, 3]))

print("\nEvaluation complete. Your dashboard files were not changed.")
