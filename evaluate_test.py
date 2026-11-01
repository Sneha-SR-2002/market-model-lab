from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Find the project and results folders.
project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "results"

# Read the decisions already made using validation data.
training_settings = json.loads(
    (results_folder / "training_settings.json").read_text(
        encoding="utf-8"
    )
)

rule_settings = json.loads(
    (results_folder / "selected_rule.json").read_text(
        encoding="utf-8"
    )
)

# Load the saved model without training it again.
model = joblib.load(results_folder / "selected_model.joblib")

# Read the prepared data.
dataset = pd.read_csv(
    project_folder / "data" / "model_dataset.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

# Retrieve the same features and split boundaries used earlier.
features = training_settings["features"]
train_end = training_settings["train_end"]
validation_end = training_settings["validation_end"]

# Recover training rows and the reserved final-test rows.
train_data = dataset.iloc[:train_end - 1].copy()
test_data = dataset.iloc[validation_end:].copy()

# Ask the selected model to predict the final-test answers.
X_test = test_data[features]
y_test = test_data["target"]
predictions = model.predict(X_test)

# Also evaluate a simple baseline trained on the original training rows.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(train_data[features], train_data["target"])
baseline_predictions = baseline.predict(X_test)

# Save classification scores for both competitors.
classification_scores = pd.DataFrame([
    {
        "model": "baseline",
        "accuracy": accuracy_score(y_test, baseline_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, baseline_predictions
        ),
    },
    {
        "model": training_settings["selected_model"],
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
    },
])

classification_scores.to_csv(
    results_folder / "test_classification_scores.csv",
    index=False,
)

# Retrieve the selected model's estimated up-day probabilities.
up_column = list(model.classes_).index(1)
probability_up = model.predict_proba(X_test)[:, up_column]

# Use the threshold and costs selected BEFORE seeing this test.
threshold = rule_settings["selected_threshold"]
cost = rule_settings["cost_bps"] / 10000

# None means validation selected cash: do not enter any trades.
if threshold is None:
    selected_trades = np.zeros(len(test_data), dtype=int)
else:
    selected_trades = (probability_up >= threshold).astype(int)

# Compare four predefined actions on exactly the same test dates.
strategies = {
    "selected_rule": selected_trades,
    "model_at_50_percent": (probability_up >= 0.50).astype(int),
    "always_long_intraday": np.ones(len(test_data), dtype=int),
    "cash": np.zeros(len(test_data), dtype=int),
}

# Prepare the daily evidence table.
daily_results = test_data[
    ["execution_date", "target", "next_intraday_return"]
].copy()

daily_results["probability_up"] = probability_up
summary_rows = []

# Calculate each strategy's outcome without changing its rule.
for name, trades in strategies.items():

    # Deduct the complete trading cost only on active days.
    net_return = trades * (
        test_data["next_intraday_return"] - cost
    )

    # Calculate the balance, starting from 1.
    equity = (1 + net_return).cumprod()

    # Include the starting balance when measuring drawdown.
    balances = np.r_[1.0, equity.to_numpy()]
    previous_peaks = np.maximum.accumulate(balances)
    drawdown = balances / previous_peaks - 1

    # Record trading activity and performance.
    summary_rows.append({
        "strategy": name,
        "trades": int(trades.sum()),
        "coverage": float(trades.mean()),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
    })

    # Keep daily results so we can inspect individual outcomes.
    daily_results[name + "_trade"] = trades
    daily_results[name + "_net_return"] = net_return
    daily_results[name + "_equity"] = equity

# Save the final-test evidence.
summary = pd.DataFrame(summary_rows)

summary.to_csv(
    results_folder / "test_strategy_comparison.csv",
    index=False,
)

daily_results.to_csv(
    results_folder / "test_daily_results.csv"
)

# Display the measured results.
print("\nFINAL-TEST CLASSIFICATION SCORES:")
print(classification_scores.to_string(index=False))

print("\nFINAL-TEST STRATEGY COMPARISON:")
print(summary.to_string(index=False))

print("\nFrozen threshold:", threshold)
print("Test rows:", len(test_data))
print("No model or threshold was changed using these test results.")