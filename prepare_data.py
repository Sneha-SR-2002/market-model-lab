from pathlib import Path
import numpy as np
import pandas as pd

# Find this project's folder.
project_folder = Path(__file__).resolve().parent

# Locate the price file we downloaded.
input_file = project_folder / "data" / "nifty50_prices.csv"

# Read the file and recognise Date as a date column.
prices = pd.read_csv(input_file, index_col="Date", parse_dates=True)

# Put the dates in order.
prices = prices.sort_index()

# Choose the price columns we need.
price_columns = ["Open", "High", "Low", "Close"]

# Stop if any dates are missing or repeated.
if prices.index.isna().any() or prices.index.duplicated().any():
    raise ValueError("Missing or repeated dates found. Check the price file.")

# Stop if prices are missing, infinite or zero/negative.
price_values = prices[price_columns].to_numpy(dtype=float)

if not np.isfinite(price_values).all() or (price_values <= 0).any():
    raise ValueError("Invalid prices found. Check the price file.")

# Create a new table with the same dates.
dataset = pd.DataFrame(index=prices.index)

# Give the closing-price column a shorter name.
close = prices["Close"]

# CLUE 1: Today's change compared with the previous trading day.
dataset["return_1d"] = close.pct_change(fill_method=None)

# CLUE 2: Price change over five trading days.
dataset["return_5d"] = close.pct_change(5, fill_method=None)

# CLUE 3: How far today's price is above or below its 20-day average.
dataset["ma_ratio_20"] = close / close.rolling(20).mean() - 1

# CLUE 4: How jumpy daily returns have been over ten trading days.
dataset["volatility_10"] = dataset["return_1d"].rolling(10).std()

# CLUE 5: Today's high-to-low range compared with its closing price.
dataset["range_ratio"] = (prices["High"] - prices["Low"]) / close

# Calculate each day's opening-to-closing price change.
intraday_return = prices["Close"] / prices["Open"] - 1

# Move tomorrow's result beside today's clues.
dataset["next_intraday_return"] = intraday_return.shift(-1)

# Record the trading date that the answer belongs to.
dates = pd.Series(prices.index, index=prices.index)
dataset["execution_date"] = dates.shift(-1)

# Remove rows without enough history or without a known next-day answer.
dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

# Answer: 1 means tomorrow closes above its open; otherwise 0.
dataset["target"] = (dataset["next_intraday_return"] > 0).astype(int)

# Save our prepared worksheet.
output_file = project_folder / "data" / "model_dataset.csv"
dataset.to_csv(output_file)

# Show what we created.
print(dataset.head())
print("Original rows:", len(prices))
print("Prepared rows:", len(dataset))
print("Removed rows:", len(prices) - len(dataset))
print("Saved to:", output_file)