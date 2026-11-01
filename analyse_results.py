from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Find the project and results folders.
project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "results"

# Read the daily results already calculated.
daily = pd.read_csv(
    results_folder / "test_daily_results.csv",
    index_col="Date",
    parse_dates=True,
)

# Recognise the actual trading dates.
daily["execution_date"] = pd.to_datetime(
    daily["execution_date"]
)

# Give each strategy a readable name for the chart.
strategy_names = {
    "selected_rule": "Selected rule",
    "model_at_50_percent": "Model at 50% threshold",
    "always_long_intraday": "Always long intraday",
    "cash": "Cash",
}

# Create a chart.
fig, ax = plt.subplots(figsize=(11, 6))

# Draw each strategy's saved balance history.
for name, label in strategy_names.items():

    # Use a dashed line for cash to distinguish overlapping lines.
    line_style = "--" if name == "cash" else "-"

    ax.plot(
        daily["execution_date"],
        daily[name + "_equity"],
        label=label,
        linestyle=line_style,
    )

# Explain the chart.
ax.set_title(
    "Final-test strategy comparison\n"
    "Historical index-price simulation after assumed costs"
)
ax.set_xlabel("Trading date")
ax.set_ylabel("Balance: starting value = 1")
ax.legend()
ax.grid(alpha=0.3)

# Keep labels inside the image and save it.
fig.tight_layout()
chart_file = results_folder / "test_strategy_chart.png"
fig.savefig(chart_file, dpi=150)
plt.close(fig)

# Read the original training boundaries.
settings = json.loads(
    (results_folder / "training_settings.json").read_text(
        encoding="utf-8"
    )
)

# Read the prepared feature table.
dataset = pd.read_csv(
    project_folder / "data" / "model_dataset.csv",
    index_col="Date",
    parse_dates=True,
).sort_index()

# Recover only the original training rows.
training = dataset.iloc[:settings["train_end"] - 1]

# Learn the dividing line from TRAINING volatility only.
volatility_cutoff = training["volatility_10"].median()

# Match each test prediction date with its known volatility.
daily["volatility_10"] = dataset["volatility_10"].reindex(
    daily.index
)

# Stop if dates do not match correctly.
if daily["volatility_10"].isna().any():
    raise ValueError("Some test dates do not match the prepared data.")

# Label each prediction day using its available recent history.
daily["regime"] = np.where(
    daily["volatility_10"] > volatility_cutoff,
    "volatile",
    "calm",
)

# Prepare a list for the market-condition report.
report_rows = []

# Examine calm and volatile conditions separately.
for regime, group in daily.groupby("regime"):

    # Examine every predefined strategy in that condition.
    for name, label in strategy_names.items():

        # Select the rows where this strategy actually traded.
        traded = group[group[name + "_trade"] == 1]

        # A winning trade has a positive return AFTER costs.
        if traded.empty:
            win_rate = np.nan
        else:
            win_rate = (
                traded[name + "_net_return"] > 0
            ).mean()

        # Include skipped days when calculating daily averages.
        report_rows.append({
            "regime": regime,
            "strategy": label,
            "days": len(group),
            "trades": len(traded),
            "coverage": group[name + "_trade"].mean(),
            "net_trade_win_rate": win_rate,
            "mean_net_daily_return": (
                group[name + "_net_return"].mean()
            ),
        })

# Save the report and the daily condition labels.
report = pd.DataFrame(report_rows)

report.to_csv(
    results_folder / "market_condition_report.csv",
    index=False,
)

daily.to_csv(
    results_folder / "test_results_with_regimes.csv"
)

# Display the findings.
print("\nCALM VS VOLATILE MARKET REPORT:")
print(report.to_string(index=False))

print("\nTraining volatility cutoff:", volatility_cutoff)
print("Chart saved to:", chart_file)
print("No model or trading rule was changed.")