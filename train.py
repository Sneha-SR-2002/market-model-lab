from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Find our project folder.
project_folder = Path(__file__).resolve().parent

# Create a folder for models and experiment results.
results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

# Read the prepared data.
input_file = project_folder / "data" / "model_dataset.csv"
dataset = pd.read_csv(input_file, index_col="Date", parse_dates=True)
dataset = dataset.sort_index()

# These five clues are the ONLY inputs allowed into the models.
features = [
    "return_1d",
    "return_5d",
    "ma_ratio_20",
    "volatility_10",
    "range_ratio",
]

# Set chronological boundaries: 60% training, 20% validation, 20% test.
train_end = int(len(dataset) * 0.60)
validation_end = int(len(dataset) * 0.80)

# Leave one row unused at each boundary because answers use tomorrow.
train_data = dataset.iloc[:train_end - 1].copy()
validation_data = dataset.iloc[train_end:validation_end - 1].copy()
test_data = dataset.iloc[validation_end:].copy()

# Separate the training clues from their correct answers.
X_train = train_data[features]
y_train = train_data["target"]

# Separate the practice-exam clues from their correct answers.
X_validation = validation_data[features]
y_validation = validation_data["target"]

# Confirm that training contains both possible answers.
if y_train.nunique() != 2:
    raise ValueError("Training data must contain both 0 and 1 answers.")

# Create three competitors.
models = {
    "baseline": DummyClassifier(strategy="most_frequent"),

    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ),

    "boosted_trees": HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=7,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    ),
}

# Prepare a list to collect their scores.
scores = []

# Teach and evaluate each competitor.
for name, model in models.items():

    # Learn using training data only.
    model.fit(X_train, y_train)

    # Answer the validation questions without seeing their answers.
    predictions = model.predict(X_validation)

    # Compare predictions with the correct validation answers.
    accuracy = accuracy_score(y_validation, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        predictions,
    )

    # Add this competitor's scores to our list.
    scores.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    })

# Turn the scores into a table.
score_table = pd.DataFrame(scores)

# Rank models by validation balanced accuracy.
score_table = score_table.sort_values(
    "balanced_accuracy",
    ascending=False,
    kind="stable",
)

# Select the highest-ranked competitor, even if the baseline wins.
chosen_name = score_table.iloc[0]["model"]
chosen_model = models[chosen_name]

# Save the scoreboard and the selected trained model.
score_table.to_csv(results_folder / "validation_scores.csv", index=False)
joblib.dump(chosen_model, results_folder / "selected_model.joblib")

# Find the probability column representing an up day.
up_column = list(chosen_model.classes_).index(1)

# Save validation predictions for our next skip-rule experiment.
validation_output = validation_data.copy()
validation_output["probability_up"] = chosen_model.predict_proba(
    X_validation
)[:, up_column]

validation_output.to_csv(
    results_folder / "validation_predictions.csv"
)

# Record our decisions so later scripts use the same setup.
settings = {
    "features": features,
    "selected_model": chosen_name,
    "train_end": train_end,
    "validation_end": validation_end,
    "training_rows": len(train_data),
    "validation_rows": len(validation_data),
    "test_rows": len(test_data),
}

settings_file = results_folder / "training_settings.json"
settings_file.write_text(
    json.dumps(settings, indent=2),
    encoding="utf-8",
)

# Display the actual results.
print("\nVALIDATION SCORES:")
print(score_table.to_string(index=False))

print("\nSelected model:", chosen_name)
print("Training rows:", len(train_data))
print("Validation rows:", len(validation_data))
print("Reserved test rows:", len(test_data))
print("The final test has NOT been evaluated.")
print("Results saved to:", results_folder)