import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where CSV files are stored
DATA_PATH = "Data/"
OUTPUT_DIR = "output/"


def load_data():
    """ Load turbine data from multiple CSV files. """
    files = glob.glob(os.path.join(DATA_PATH, "data_group_*.csv"))
    data_frames = []

    for file in files:
        df = pd.read_csv(file)
        df['file_source'] = os.path.basename(file)  # Track source file
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


def clean_data(df):
    """ Clean data by handling missing values and removing outliers. """
    df.dropna(inplace=True)

    # Remove outliers using Z-score
    df = df[(np.abs(df['power_output'] - df['power_output'].mean()) / df['power_output'].std()) < 3]

    return df


def compute_statistics(df):
    """ Calculate min, max, and mean power output for each turbine. """
    return df.groupby("turbine_id")["power_output"].agg(["min", "max", "mean"]).reset_index()


def detect_anomalies(df):
    """ Detect turbines with power output outside 1.5 standard deviations per turbine. """

    # Compute mean and standard deviation per turbine
    stats = df.groupby("turbine_id")["power_output"].agg(["mean", "std"]).reset_index()
    df = df.merge(stats, on="turbine_id", how="left")  # Merge stats back

    # Debugging: Print computed mean & std dev for each turbine
    print("\nComputed Mean & Std Dev Per Turbine:")
    print(stats)

    # Replace NaN standard deviations (happens when only one data point exists per turbine)
    df["std"].fillna(0, inplace=True)

    # **Lower the threshold to 1.5 standard deviations instead of 2.0**
    df["anomaly"] = (df["std"] > 0) & (
            (df["power_output"] < (df["mean"] - 1.5 * df["std"])) |
            (df["power_output"] > (df["mean"] + 1.5 * df["std"]))
    )

    # Debugging: Print rows that are marked as anomalies
    anomalies = df[df["anomaly"] == True]
    print("\nAnomalies Detected:")
    print(anomalies)

    return anomalies  # Return only anomalies

def store_data(clean_df, stats_df, anomalies_df):
    """ Store processed data as CSV files in the output directory. """

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the cleaned data
    clean_df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)

    # Save summary statistics
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)

    # Save anomalies detected
    anomalies_df.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)

    print(f"Data saved in the '{OUTPUT_DIR}' directory.")


def visualize_data(clean_df, anomalies_df, stats_df):
    """ Generate and save visualizations for power output trends, anomalies, and summary statistics. """

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1️⃣ Power Output Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(clean_df["power_output"], bins=30, color="skyblue", edgecolor="black")
    plt.axvline(clean_df["power_output"].mean(), color='red', linestyle='dashed', linewidth=2,
                label="Mean Power Output")
    plt.xlabel("Power Output (MW)")
    plt.ylabel("Frequency")
    plt.title("Power Output Distribution")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "power_output_distribution.png"))  # Save instead of show
    plt.close()  # Close the figure to free memory

    # 2️⃣ Scatter plot of anomalies
    plt.figure(figsize=(12, 6))
    plt.scatter(clean_df["turbine_id"], clean_df["power_output"], label="Normal Data", alpha=0.6)
    plt.scatter(anomalies_df["turbine_id"], anomalies_df["power_output"], color='red', label="Anomalies", marker="x")
    plt.xlabel("Turbine ID")
    plt.ylabel("Power Output (MW)")
    plt.title("Anomaly Detection in Power Output")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_detection.png"))  # Save instead of show
    plt.close()

    # 3️⃣ Summary statistics: Min, Max, Mean power per turbine
    plt.figure(figsize=(12, 6))
    turbines = stats_df["turbine_id"]
    plt.plot(turbines, stats_df["min"], marker='o', linestyle='dashed', label="Min Power Output")
    plt.plot(turbines, stats_df["max"], marker='o', linestyle='dashed', label="Max Power Output")
    plt.plot(turbines, stats_df["mean"], marker='o', linestyle='solid', label="Mean Power Output", color='black')
    plt.xlabel("Turbine ID")
    plt.ylabel("Power Output (MW)")
    plt.title("Turbine Power Output Summary")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "turbine_power_summary.png"))  # Save instead of show
    plt.close()

    print(f"Visualization images saved in '{OUTPUT_DIR}' directory.")

