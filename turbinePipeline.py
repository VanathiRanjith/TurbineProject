import pandas as pd
import numpy as np
import glob
import os
import sqlite3
import dask.dataframe as dd
import matplotlib.pyplot as plt

# Define paths
RAW_DATA_PATH = "data/raw/"
BRONZE_PATH = "data/bronze/"
SILVER_PATH = "data/silver/"
GOLD_PATH = "data/gold/"
OUTPUT_DIR = "output/"
DB_PATH = "output/turbine_data.db"

# Ensure necessary directories exist
for path in [BRONZE_PATH, SILVER_PATH, GOLD_PATH, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)


def load_data(data_path=RAW_DATA_PATH):
    """Load turbine data from CSV files efficiently."""
    files = glob.glob(os.path.join(data_path, "data_group_*.csv"))
    if not files:
        print("No CSV files found in", data_path)
        return pd.DataFrame()

    df = dd.read_csv(files).compute()
    df.to_csv(os.path.join(BRONZE_PATH, "raw_combined_data.csv"), index=False)
    return df


def clean_data(df):
    """Clean data by handling missing values and ensuring validity."""
    if "power_output" not in df.columns:
        raise KeyError("Missing 'power_output' column in the dataset")

    df["power_output"] = df["power_output"].interpolate(method="linear")
    df = df.dropna(subset=["turbine_id", "power_output"])
    df.to_csv(os.path.join(SILVER_PATH, "cleaned_data.csv"), index=False)
    return df


def compute_statistics(df):
    """Compute min, max, and mean power output per turbine per day."""
    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    stats = df.groupby(["turbine_id", "date"])["power_output"].agg(["min", "max", "mean"]).reset_index()
    stats.to_csv(os.path.join(GOLD_PATH, "summary_statistics.csv"), index=False)
    return stats


def detect_anomalies(df):
    """Identify anomalies where power output is outside 2 standard deviations per turbine per day."""
    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    anomalies = df.groupby(["turbine_id", "date"]).apply(
        lambda group: group[
            np.abs(group['power_output'] - group['power_output'].mean()) > 2 * group['power_output'].std()]
    ).reset_index(drop=True)

    anomalies.to_csv(os.path.join(GOLD_PATH, "anomalies.csv"), index=False)
    return anomalies


def store_data_to_db(cleaned_data, summary_stats, anomalies):
    """Store processed data in a SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cleaned_data.to_sql("cleaned_data", conn, if_exists="replace", index=False)
    summary_stats.to_sql("summary_statistics", conn, if_exists="replace", index=False)
    anomalies.to_sql("anomalies", conn, if_exists="replace", index=False)
    conn.close()
    print("Data successfully stored in database.")


def visualize_anomalies(df, anomalies):
    """Generate time-series visualization of anomalies."""
    if df.empty:
        print("No data to visualize!")
        return

    plt.figure(figsize=(12, 6))
    for turbine in df["turbine_id"].unique():
        subset = df[df["turbine_id"] == turbine]
        plt.plot(subset["timestamp"], subset["power_output"], label=f"Turbine {turbine}", alpha=0.6)

    if not anomalies.empty:
        plt.scatter(anomalies["timestamp"], anomalies["power_output"], color="red", label="Anomalies")

    plt.title("Turbine Power Output Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power Output (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_detection_time_series.png"))
    plt.close()