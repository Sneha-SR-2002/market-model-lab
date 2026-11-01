from pathlib import Path
import sqlite3
import pandas as pd

# Find the main project folder.
project = Path(__file__).resolve().parent

# Locate the saved results.
results = project / "results"

# Choose the database filename.
database_file = results / "research.sqlite"

# Match database table names to existing CSV files.
files = {
    "daily_results": results / "test_results_with_regimes.csv",
    "strategy_results": results / "test_strategy_comparison.csv",
    "validation_scores": (
        project / "results_neural" / "validation_comparison.csv"
    ),
}

# Open the database; create it if it does not exist.
with sqlite3.connect(database_file) as connection:

    # Visit each CSV file.
    for table_name, file_path in files.items():

        # Read the CSV into a spreadsheet-like table.
        table = pd.read_csv(file_path)

        # Store it in the database.
        table.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )

    # Ask SQL to rank models by validation balanced accuracy.
    query = """
        SELECT model, accuracy, balanced_accuracy
        FROM validation_scores
        ORDER BY balanced_accuracy DESC;
    """

    # Run the SQL question and collect its answer.
    ranking = pd.read_sql_query(query, connection)

    # Ask SQL to count model trades in each market condition.
    regime_query = """
        SELECT
            regime,
            COUNT(*) AS days,
            SUM(model_at_50_percent_trade) AS trades
        FROM daily_results
        GROUP BY regime;
    """

    regime_summary = pd.read_sql_query(
        regime_query,
        connection,
    )

# Show the answers.
print("\nMODEL RANKING:")
print(ranking.to_string(index=False))

print("\nTRADES BY MARKET CONDITION:")
print(regime_summary.to_string(index=False))

print("\nDatabase saved to:", database_file)