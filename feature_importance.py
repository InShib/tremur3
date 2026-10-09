
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

from app import load_data

# Feature names must match the order in extract_features()
axis_features = [
    "Mean", "Std deviation", "Minimum", "Maximum",
    "RMS", "Mean absolute change", "Interquartile range",
    "Dominant frequency", "Spectral centroid",
    "3-12 Hz power ratio", "AC RMS"
]

feature_names = []
for axis in ["X", "Y", "Z"]:
    feature_names.extend([f"{axis}_{name}" for name in axis_features])

feature_names.extend([
    "Magnitude mean",
    "Magnitude standard deviation",
    "Magnitude RMS variation",
    "Magnitude mean absolute change",
    "Magnitude dominant frequency",
    "Magnitude spectral centroid",
    "Magnitude 3-12 Hz power ratio"
])

print("Loading dataset...")
X, y, groups, windows = load_data()

print("Feature count:", X.shape[1])
print("Sample count:", len(y))

# Keep segments separate between training and testing
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.22,
    random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups))

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

print("\nTraining model...")
model.fit(X[train_idx], y[train_idx])

# Built-in Random Forest feature importance
rf = model.named_steps["randomforestclassifier"]
importance = rf.feature_importances_

print("\n===== TOP 15 FEATURES =====")
ranked = np.argsort(importance)[::-1]

for rank, i in enumerate(ranked[:15], start=1):
    print(f"{rank:2}. {feature_names[i]:40} {importance[i]:.4f}")

# Permutation importance on held-out segments
print("\nCalculating permutation importance...")
result = permutation_importance(
    model,
    X[test_idx],
    y[test_idx],
    scoring="f1_macro",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

print("\n===== TOP 15 HELD-OUT FEATURES =====")
ranked_perm = np.argsort(result.importances_mean)[::-1]

for rank, i in enumerate(ranked_perm[:15], start=1):
    print(
        f"{rank:2}. {feature_names[i]:40} "
        f"{result.importances_mean[i]:+.4f}"
    )

# Save a graph of the top 15 permutation importances
top = ranked_perm[:15][::-1]

plt.figure(figsize=(11, 7))
plt.barh(
    [feature_names[i] for i in top],
    result.importances_mean[top],
    xerr=result.importances_std[top]
)
plt.xlabel("Change in macro F1 when feature is shuffled")
plt.title("TremorSense: Held-out Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=160)
plt.show()

print("\nGraph saved as feature_importance.png")
print("Evaluation complete. The existing app and model were not changed.")
