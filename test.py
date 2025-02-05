import pytest
import pandas as pd
import os
from turbinePipeline import load_data, clean_data, compute_statistics, detect_anomalies, store_data

# Define test output directory
TEST_OUTPUT_DIR = "test_output/"

@pytest.fixture(scope="module")
def sample_data():
    """
    Fixture to create a sample DataFrame for testing.
    """
    data = {
        "turbine_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "power_output": [10, 12, 11, 100, 9, 20, 22, 21, 200, 19]  # 100 and 200 should be detected as anomalies
    }
    return pd.DataFrame(data)

def test_load_data():
    """
    Test that load_data() reads CSV files correctly.
    """
    df = load_data()
    assert df is not None, "Data should not be None."
    assert not df.empty, "Loaded data should not be empty."
    assert "turbine_id" in df.columns, "Column 'turbine_id' should be in the DataFrame."

def test_clean_data(sample_data):
    """
    Test that clean_data() handles missing values and removes outliers.
    """
    cleaned_df = clean_data(sample_data)
    assert cleaned_df is not None, "Cleaned data should not be None."
    assert cleaned_df.isnull().sum().sum() == 0, "There should be no missing values after cleaning."

def test_compute_statistics(sample_data):
    """
    Test that compute_statistics() correctly calculates min, max, and mean power output.
    """
    stats_df = compute_statistics(sample_data)
    assert stats_df is not None, "Summary statistics should not be None."
    assert "min" in stats_df.columns, "Min power output should be included."
    assert "max" in stats_df.columns, "Max power output should be included."
    assert "mean" in stats_df.columns, "Mean power output should be included."

def test_detect_anomalies(sample_data):
    """
    Test that detect_anomalies() correctly identifies anomalies.
    """
    # Create a dataset where anomalies should exist
    test_df = pd.DataFrame({
        'turbine_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'power_output': [10, 12, 11, 100, 9, 20, 22, 21, 200, 19]  # 100 and 200 should be anomalies
    })

    print("\nInput Data for Anomaly Detection:")
    print(test_df)

    anomalies_df = detect_anomalies(test_df)

    print("\nDetected anomalies:\n", anomalies_df)  # Debugging output

    # Ensure anomalies are detected
    assert anomalies_df is not None, "Anomalies DataFrame should not be None."
    assert not anomalies_df.empty, "Anomalies should be detected."
    assert 100 in anomalies_df["power_output"].values, "Value 100 should be an anomaly."
    assert 200 in anomalies_df["power_output"].values, "Value 200 should be an anomaly."

def test_store_data(sample_data):
    """
    Test that store_data() correctly saves CSV files.
    """
    # Create test data
    cleaned_df = clean_data(sample_data)
    stats_df = compute_statistics(cleaned_df)
    anomalies_df = detect_anomalies(cleaned_df)

    # Save files
    store_data(cleaned_df, stats_df, anomalies_df)

    # Verify file existence
    assert os.path.exists("output/cleaned_data.csv"), "Cleaned data file should be saved."
    assert os.path.exists("output/summary_statistics.csv"), "Summary statistics file should be saved."
    assert os.path.exists("output/anomalies.csv"), "Anomalies file should be saved."

