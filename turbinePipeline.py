import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
import os
import sqlite3
import matplotlib.pyplot as plt

# Medallion Architecture Layers
BRONZE_PATH = "data/raw/"
SILVER_PATH = "data/processed/"
GOLD_PATH = "data/aggregated/"
DB_PATH = "output/turbine_data.db"
OUTPUT_DIR = "output/"


# Load Data (Bronze Layer)
def load_data():
    """Load turbine data from CSV files into the Bronze layer."""
    files = glob.glob(os.path.join(BRONZE_PATH, "data_group_*.csv"))
    if not files:
        print("No CSV files found!")
        return pd.DataFrame()

    df = dd.read_csv(files).compute()
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp
    return df


# Clean Data (Silver Layer)
def clean_data(df):
    """Clean data by handling missing values and ensuring validity."""
    if "power_output" not in df.columns:
        raise KeyError("Missing 'power_output' column in the dataset")

    df["power_output"] = df["power_output"].interpolate(method="linear")
    df = df.dropna(subset=["turbine_id", "power_output"])
    return df


# Compute Statistics (Gold Layer)
def compute_statistics(df):
    """Compute min, max, and mean power output per turbine per day."""
    df['date'] = df['timestamp'].dt.date  # Extract date
    return df.groupby(["turbine_id", "date"])['power_output'].agg(["min", "max", "mean"]).reset_index()


# Detect Anomalies (Gold Layer)
def detect_anomalies(df):
    """Identify anomalies using 2 standard deviations from the mean."""
    df['date'] = df['timestamp'].dt.date
    anomalies = df.groupby(["turbine_id", "date"]).apply(
        lambda group: group[
            np.abs(group['power_output'] - group['power_output'].mean()) > 2 * group['power_output'].std()
            ]
    ).reset_index(drop=True)
    return anomalies


# Store Data in Database
def store_data_to_db(cleaned_data, summary_stats, anomalies):
    """Store processed data in a SQLite database."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cleaned_data.to_sql("cleaned_data", conn, if_exists="replace", index=False)
    summary_stats.to_sql("summary_statistics", conn, if_exists="replace", index=False)
    anomalies.to_sql("anomalies", conn, if_exists="replace", index=False)
    conn.close()
    print("Data successfully stored in database.")


# Visualizations
def visualize_statistics(summary_stats):
    """Generate bar plot for turbine summary statistics."""
    plt.figure(figsize=(12, 6))
    summary_stats.plot(x="turbine_id", y=["min", "max", "mean"], kind="bar", legend=True)
    plt.title("Summary Statistics per Turbine")
    plt.xlabel("Turbine ID")
    plt.ylabel("Power Output (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_statistics.png"))
    plt.close()


def visualize_anomalies(df, anomalies):
    """Generate time-series visualization of anomalies."""
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
