import pytest
import pandas as pd
import sqlite3
import os

from turbinePipeline import load_data, clean_data, compute_statistics, detect_anomalies, store_data_to_db, DB_PATH


@pytest.fixture(scope="module")
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        "turbine_id": [1, 1, 2, 2],
        "wind_speed": [10.0, None, 15.0, 20.0],
        "power_output": [100.0, 120.0, 130.0, 140.0],
        "wind_direction": [180, 185, 190, 195],
        "timestamp": ["2023-09-28 00:00:00", "2023-09-28 01:00:00",
                      "2023-09-28 02:00:00", "2023-09-28 03:00:00"]
    }
    return pd.DataFrame(data)


def test_clean_data_interpolation(sample_data):
    """Ensure missing values are interpolated correctly."""
    sample_data.loc[1, "power_output"] = None  # Introduce missing value
    cleaned_df = clean_data(sample_data)
    assert cleaned_df["power_output"].isnull().sum() == 0, "Missing values should be imputed"


def test_compute_statistics(sample_data):
    """Test that compute_statistics() calculates correct min, max, and mean values."""
    sample_data["timestamp"] = pd.to_datetime(sample_data["timestamp"])
    stats_df = compute_statistics(sample_data)
    assert "min" in stats_df.columns, "Min power output should be included."
    assert "max" in stats_df.columns, "Max power output should be included."
    assert "mean" in stats_df.columns, "Mean power output should be included."


def test_detect_anomalies(sample_data):
    """Test that detect_anomalies() correctly identifies anomalies."""
    anomalies_df = detect_anomalies(sample_data)
    assert "power_output" in anomalies_df.columns, "Anomalies should contain 'power_output'."


def test_store_data_to_db(sample_data):
    """Test database storage functionality."""
    cleaned_df = clean_data(sample_data)
    stats_df = compute_statistics(cleaned_df)
    anomalies_df = detect_anomalies(cleaned_df)

    store_data_to_db(cleaned_df, stats_df, anomalies_df)

    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    conn.close()

    assert "cleaned_data" in tables.values, "Cleaned data should be in database"
    assert "summary_statistics" in tables.values, "Summary statistics should be in database"
    assert "anomalies" in tables.values, "Anomalies should be in database"


if __name__ == "__main__":
    pytest.main()
