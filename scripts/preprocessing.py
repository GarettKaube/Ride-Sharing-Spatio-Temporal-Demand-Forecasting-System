import pandas as pd
import numpy as np
import sys
import duckdb
from src.utils import fetch_df_from_query
from meteostat import Point, Daily, Hourly
from datetime import datetime
from pathlib import Path

N_TIME_STEPS_BACK = 15

path = Path(__file__).resolve().parent

sys.path.insert(0, str(path))


def fetch_data_from_duckdb(conn):
    query = """
     SELECT
        pickup_community_area,
        time_bucket(INTERVAL '30 minutes', trip_start_timestamp) as interval,
        count(trip_id) as num_rides,
        AVG(fare) as average_fare,
        AVG(trip_total) as average_trip_total,
        MAX(trip_start_day_of_week) as trip_start_day_of_week,
        MAX(trip_start_hour) as trip_start_hour
    FROM RIDES_EXTRA_VARS
    GROUP BY pickup_community_area, interval
    """

    return fetch_df_from_query(query, conn)

def fetch_weather_data():
    start = datetime(2020, 1, 1)
    end = datetime(2023, 2, 1)
    location = Point(41.8667, -87.6, 181)
    try:
        weather_hourly = Hourly(location, start, end)
        weather = weather_hourly.fetch()
        return weather
    except Exception as e:
        print("Failed to get weather data:", e)
        print(type(e))
        raise


def load_weather_data():
    weather = None
    try:
        weather = pd.read_csv("./weatherdata/weather.csv")
    except FileNotFoundError:
        weather = fetch_weather_data()
        weather.to_csv("./weatherdata/weather.csv")

    if weather:
        weather = weather.dropna(axis=1, thresh=int(0.8 * weather.shape[0]))
        weather['time'] = pd.to_datetime(weather['time'])

        # Lag hourly data to avoid data leakage when merging together
        # with 30 minute interval data
        cols_to_lag = weather.columns.drop("time")

        for col in cols_to_lag:
            weather[col] = weather[col].shift(-1)
        return weather


def cyclical_transform(x, period):
    return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)


def transform(df: pd.DataFrame, weather:pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["interval", "pickup_community_area"]) \
        .reset_index(drop=True)

    df['target'] = df['num_rides'] \
        .shift(-1) \
        .iloc[:-1] # Remove the row with a nan target from shifting

    # Na pickup communities are pickups outside of Chicago by definition
    df = df[~df["pickup_community_area"].isna()]

    df['pickup_community_area'] = df['pickup_community_area'].astype(int)

    # How="left" since weather data is on hour interval
    df = df.dropna() \
        .merge(weather, left_on="interval", right_on="time", how="left") \
        .ffill() \
        .dropna()

    df['trip_start_hour_sin'] = df['trip_start_hour'].apply(lambda x: cyclical_transform(x, 24)[0])
    df['trip_start_hour_cos'] = df['trip_start_hour'].apply(lambda x: cyclical_transform(x, 24)[1])

    df['trip_start_day_of_week_sin'] = df['trip_start_day_of_week'].apply(lambda x: cyclical_transform(x, 7)[0])
    df['trip_start_day_of_week_cos'] = df['trip_start_day_of_week'].apply(lambda x: cyclical_transform(x, 7)[1])

    df = df.drop(['trip_start_hour', 'trip_start_day_of_week'], axis=1)

    return df


def main():
    with duckdb.connect(":memory:") as conn:
        print('yo')
        df = fetch_data_from_duckdb(conn)

        weather = load_weather_data()

        preprocessed_data = transform(df, weather)

        processed_data_path = "./data/processed_data.csv"
        preprocessed_data.to_csv(processed_data_path)


if __name__ == "__main__":
    main()