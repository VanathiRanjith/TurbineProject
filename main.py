from turbinePipeline import load_data, clean_data, compute_statistics, detect_anomalies, store_data_to_db, visualize_anomalies, OUTPUT_DIR

def run_pipeline():
    """Orchestrates the full data pipeline."""
    print("Loading data...")
    df = load_data()

    if df.empty:
        print("No data loaded. Exiting pipeline.")
        return

    print("Cleaning data...")
    cleaned_df = clean_data(df)

    print("Computing statistics...")
    stats_df = compute_statistics(cleaned_df)

    print("Detecting anomalies...")
    anomalies_df = detect_anomalies(cleaned_df)

    print("Storing data in database...")
    store_data_to_db(cleaned_df, stats_df, anomalies_df)

    print("Generating visualizations...")
    visualize_anomalies(cleaned_df, anomalies_df)

    print("Pipeline execution completed.")


if __name__ == "__main__":
    run_pipeline()