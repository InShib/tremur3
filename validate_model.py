
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from app import load_data

print("Loading dataset...")
X, y, groups, windows = load_data()

print("Samples:", len(y))
print("Segments:", len(set(groups)))

# Keep all windows from a segment together.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracies = []
macro_f1s = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
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

    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])

    accuracy = accuracy_score(y[test_idx], predictions)
    macro_f1 = f1_score(
        y[test_idx],
        predictions,
        average="macro",
        zero_division=0
    )

    accuracies.append(accuracy)
    macro_f1s.append(macro_f1)

    print(
        f"Fold {fold}: "
        f"Accuracy={accuracy * 100:.2f}% | "
        f"Macro F1={macro_f1 * 100:.2f}%"
    )

print("\n===== 5-FOLD RESULTS =====")
print(f"Mean accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"Accuracy std:  {np.std(accuracies) * 100:.2f} percentage points")
print(f"Mean macro F1: {np.mean(macro_f1s) * 100:.2f}%")
print(f"F1 std:        {np.std(macro_f1s) * 100:.2f} percentage points")
