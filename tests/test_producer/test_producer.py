import pytest
import numpy as np
from src.producer import generate_time_stamp_range, assign_df_random_time_stamp
import pandas as pd

def test_generate_time_stamp_range():
    current_timestamp = pd.Timestamp("2025-08-21 12:00:00")
    periods = 4

    result = generate_time_stamp_range(current_timestamp, periods)


    assert isinstance(result, np.ndarray), "Output should be a numpy array"
    assert result.dtype == "datetime64[ns]", "Array should have dtype datetime64[ns]"
    assert len(result) == periods, f"Array should have {periods} elements"
    assert result[0] == current_timestamp - pd.Timedelta(minutes=15), "First element should be 15 mins before current timestamp"
    assert result[-1] == current_timestamp, "Last element should be the current timestamp"

    # Check that elements are evenly spaced
    deltas = np.diff(result)
    assert all(deltas == deltas[0]), "Timestamps should be evenly spaced"

def test_assign_df_random_time_stamp_length():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    timestamps = np.array(pd.date_range("2025-08-21 12:00:00", periods=10, freq="T"))

    result = assign_df_random_time_stamp(df, timestamps)

    assert isinstance(result, list), "Result should be a list"
    assert len(result) == len(df), "Result length should match number of rows in df"
    assert all(isinstance(ts, int) for ts in result), "All elements should be np.datetime64"

def test_assign_df_random_time_stamp_values_in_range():
    df = pd.DataFrame({"id": range(20)})
    timestamps = np.array(pd.date_range("2025-08-21 12:00:00", periods=15, freq="T"))

    
    result = assign_df_random_time_stamp(df, timestamps)

    # Every timestamp in result must exist in timestamps array
    dt = pd.to_datetime(result)
    assert all(ts in timestamps for ts in dt), "All timestamps should be drawn from input timestamps"

def test_assign_df_random_time_stamp_randomness():
    
    df = pd.DataFrame({"id": range(50)})
    timestamps = np.array(pd.date_range("2025-08-21 12:00:00", periods=5, freq="T"))
    np.random.seed(42)
    
    result1 = assign_df_random_time_stamp(df, timestamps)
    result2 = assign_df_random_time_stamp(df, timestamps)


    assert result1 != result2, "Two calls should produce different assignments most of the time"

def test_assign_df_random_time_stamp_empty_df():
    
    df = pd.DataFrame(columns=["id"])
    timestamps = np.array(pd.date_range("2025-08-21 12:00:00", periods=5, freq="T"))

    
    result = assign_df_random_time_stamp(df, timestamps)


    assert result == [], "Empty DataFrame should return an empty list"