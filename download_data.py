from pathlib import Path
import yfinance as yf

# Create a folder for our market data.
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Download daily NIFTY 50 prices.
prices = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2026-01-01",
    interval="1d",
    auto_adjust=True,
    multi_level_index=False,
    progress=False,
)

# Stop with a message if no data arrives.
if prices is None or prices.empty:
    raise RuntimeError("Download failed. Check your internet and try again.")

# Arrange dates from oldest to newest.
prices = prices.sort_index()
prices.index.name = "Date"

# Save the prices as a spreadsheet-style CSV file.
file_path = data_folder / "nifty50_prices.csv"
prices.to_csv(file_path)

# Display the first five rows and the saved location.
print(prices.head())
print("Total rows:", len(prices))
print("Saved to:", file_path.resolve())