import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Define directories
DATA_PATH = r"C:\Users\India\PycharmProjects\Turbine_project\Data"
OUTPUT_DIR = "output/"


def load_data():
    """Load turbine data from CSV files."""
    files = glob.glob(os.path.join(DATA_PATH, "data_group_*.csv"))
    if not files:
        print("No CSV files found!")
        return pd.DataFrame()

    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def clean_data(df):
    """Clean data by imputing missing values instead of dropping."""
    if "power_output" not in df.columns:
        raise KeyError("Missing 'power_output' column in the dataset")

    # Impute missing values using the median power output per turbine
    df["power_output"] = df.groupby("turbine_id")["power_output"].transform(lambda x: x.fillna(x.median()))

    return df


def remove_outliers(df):
    """Remove extreme outliers using Z-score method (threshold = 3 SD)."""
    df["z_score"] = df.groupby("turbine_id")["power_output"].transform(lambda x: zscore(x, nan_policy='omit'))
    return df[(df["z_score"].abs() <= 3)].drop(columns=["z_score"])  # Keep only valid data


def compute_statistics(df):
    """Compute min, max, and mean power output for each turbine."""
    if df.empty:
        return pd.DataFrame()

    return df.groupby("turbine_id")["power_output"].agg(["min", "max", "mean"]).reset_index()


def detect_anomalies(df):
    """Detect anomalies using 2.0 standard deviations."""
    if df.empty:
        return pd.DataFrame()

    stats = df.groupby("turbine_id")["power_output"].agg(["mean", "std"]).reset_index()
    df = df.merge(stats, on="turbine_id")

    df["anomaly"] = (df["power_output"] < df["mean"] - (2.0 * df["std"])) | (
            df["power_output"] > df["mean"] + (2.0 * df["std"]))

    return df[df["anomaly"]]


def store_data(cleaned_data, summary_stats, anomalies):
    """Save processed data and visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cleaned_data.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
    summary_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)
    anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)

    print("Data saved successfully.")


def visualize_anomalies(df, anomalies):
    """Generate and save anomaly visualization."""
    if df.empty:
        print("No data to visualize!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot normal turbine data
    plt.scatter(df["turbine_id"], df["power_output"], label="Normal Data", alpha=0.5, color="blue")

    # Plot detected anomalies
    if not anomalies.empty:
        plt.scatter(anomalies["turbine_id"], anomalies["power_output"], color="red", label="Anomalies", marker="x")

    plt.title("Anomaly Detection in Turbine Power Output")
    plt.xlabel("Turbine ID")
    plt.ylabel("Power Output (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_detection.png"))
    plt.close()

    print("Anomaly visualization saved.")

