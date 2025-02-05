from turbinePipeline import load_data, clean_data, compute_statistics, detect_anomalies, store_data, visualize_data

def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    clean_df = clean_data(df)

    print("Computing statistics...")
    stats_df = compute_statistics(clean_df)

    print("Detecting anomalies...")
    anomalies_df = detect_anomalies(clean_df)

    print("Storing data...")
    store_data(clean_df, stats_df, anomalies_df)

    print("Generating visualizations...")
    visualize_data(clean_df, anomalies_df, stats_df)

    print("Pipeline execution completed!")

if __name__ == "__main__":
    main()
