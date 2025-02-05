from turbinePipeline import load_data, clean_data, compute_statistics, detect_anomalies, remove_outliers, store_data, visualize_anomalies, OUTPUT_DIR

def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Removing outliers...")
    df = remove_outliers(df)

    print("Computing statistics...")
    stats = compute_statistics(df)

    print("Detecting anomalies...")
    anomalies = detect_anomalies(df)

    print("Storing results...")
    store_data(df, stats, anomalies)

    print("Generating anomaly visualization...")
    visualize_anomalies(df, anomalies)

    print("Pipeline execution complete.")


if __name__ == "__main__":
    main()