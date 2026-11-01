from pathlib import Path
import json
import math
import pandas as pd

# Find our project and results folders.
project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "results"

# Read the selected model's validation predictions.
input_file = results_folder / "validation_predictions.csv"
validation = pd.read_csv(
    input_file,
    index_col="Date",
    parse_dates=True,
)

# Assume a total entry-plus-exit cost of 0.10% per trade.
cost_bps = 10
cost_per_trade = cost_bps / 10000

# Try four rules chosen before looking at final-test results.
thresholds = [0.50, 0.55, 0.60, 0.65]

# Require enough trades before trusting a validation result.
minimum_trades = max(20, math.ceil(len(validation) * 0.10))

# Prepare a list for the experiment results.
experiments = []

# Test each rule on validation data only.
for threshold in thresholds:

    # Trade only when the estimated up probability is high enough.
    trade = (
        validation["probability_up"] >= threshold
    ).astype(int)

    # Earn tomorrow's open-to-close return, minus costs, if trading.
    # A skipped day earns zero.
    net_return = trade * (
        validation["next_intraday_return"] - cost_per_trade
    )

    # Start with a balance of 1 and compound the daily results.
    balance = (1 + net_return).cumprod()

    # Count trades and calculate the final return.
    number_of_trades = int(trade.sum())
    total_return = float(balance.iloc[-1] - 1)

    # Record this rule's results.
    experiments.append({
        "threshold": threshold,
        "trades": number_of_trades,
        "coverage": float(trade.mean()),
        "total_return": total_return,
    })

# Turn the experiment list into a table.
comparison = pd.DataFrame(experiments)

# Keep rules with enough trades and positive validation returns.
eligible = comparison[
    (comparison["trades"] >= minimum_trades)
    & (comparison["total_return"] > 0)
]

# If no rule qualifies, choose to stay in cash.
selected_threshold = None

if not eligible.empty:

    # Otherwise choose the qualifying rule with the highest return.
    best_rule = eligible.sort_values(
        "total_return",
        ascending=False,
        kind="stable",
    ).iloc[0]

    selected_threshold = float(best_rule["threshold"])

# Save every rule's result, including unsuccessful ones.
comparison.to_csv(
    results_folder / "validation_rule_comparison.csv",
    index=False,
)

# Record our decision before opening the final test.
settings = {
    "selected_threshold": selected_threshold,
    "cost_bps": cost_bps,
    "minimum_validation_trades": minimum_trades,
    "selection_data": "validation only",
    "selection_rule": (
        "Highest positive validation return among qualifying rules; "
        "otherwise cash."
    ),
}

settings_file = results_folder / "selected_rule.json"
settings_file.write_text(
    json.dumps(settings, indent=2),
    encoding="utf-8",
)

# Display the actual experiment results.
print("\nVALIDATION RULE COMPARISON:")
print(comparison.to_string(index=False))

print("\nMinimum required trades:", minimum_trades)

if selected_threshold is None:
    print("Selected rule: stay in cash; no threshold qualified.")
else:
    print("Selected probability threshold:", selected_threshold)

print("The final test has NOT been evaluated.")
print("Decision saved to:", settings_file)