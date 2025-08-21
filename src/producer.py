"""
Script for simulating incoming ride requests to send to the ML system, starting with Kinesis, then Lambda for
processing and then forecasting.
"""
import logging
import time
import numpy as np
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from src.logging_config import setup_logging
import datetime
import json
from pandas import IndexSlice as idx
from src.utils import get_param
from typing import Tuple

import os

load_dotenv()

date_time = datetime.datetime.now()
date_time = date_time.strftime("%m-%d-%Y_%H-%M-%S")
setup_logging(log_file_path=f'./logs/producer/log_{date_time}.log')

logger = logging.getLogger("producer")

# Increase this to increase the rate of rides
STREAM_SPEED_FACTOR = int(get_param("STREAM_SPEED_FACTOR"))

STREAM_NAME = os.environ["DATA_STREAM_NAME"]

# Makes simulation simulate every SIMULATION_STRIDEth event
SIMULATION_STRIDE = 100

SIMULATION_START_INDEX = 150000

AWS_REGION = os.environ.get("REGION")

kinesis_client = boto3.client("kinesis", region_name=AWS_REGION)


def generate_time_stamp_range(current_timestamp: pd.Timestamp, periods:int) -> np.ndarray:
    """
    Given a pickup time stamp from the rides data, a numpy array of random dates is
    generated ranging from 15 minutes prior of current_timestamp to current_timestamp
    :param current_timestamp: pd.Timestamp to generate dates between [current_timestamp - 15mins, current_timestamp]
    :param periods:
    :return: np.ndarray np.datetime64
    """
    start = current_timestamp - pd.Timedelta(minutes=15)
    end = current_timestamp
    return pd.date_range(start=start, end=end, periods=periods).to_numpy()


def assign_df_random_time_stamp(df:pd.DataFrame, timestamps:np.ndarray) -> list:
    """
    :param df: Pandas DataFrame to assign timestamps
    :param timestamps: np.array of timestamps
    :return: list of randomized timestamps
    """
    # Use uniform dist to select random indices for timestamps for pickup times
    random_u = np.random.randint(low=0, high=timestamps.shape[0], size=df.shape[0])
    if random_u.shape[0] != df.shape[0]:
        raise ValueError(f"random_u.shape[0] does not equal df_at_timestamp.shape[0]")

    return  timestamps[random_u].tolist()


def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly shuffles Pandas DataFrame
    :param df: pd.DataFrame
    :return: shuffled pd.DataFrame
    """
    return df.sample(frac=1)


def generate_random_timestamps(df: pd.DataFrame, periods=1000) -> pd.DataFrame:
    """
    Addresses that the pickup times are rounded to nearest 15 minute by
    constructing a random timestamp for each observation that is within the range of the
    observations pickup time stamp and the past 15 minutes of the pickup.
    :param df: pandas dataframe with datetime index
    :param periods number of possible time stamps to be selected when random sampling
    :return: pd.DataFrame
    """
    unique_timestamps = df.index.unique()

    all_timestamps = []
    for time_stamp in unique_timestamps:
        df_at_timestamp = df.loc[[time_stamp]]

        # Shuffle the data at the time stamp since the data is sorted by (pickup time, pickup community)
        # So that we do not simulate all of one community before the next
        df.loc[[time_stamp]] = shuffle_df(df_at_timestamp)

        # Generate time stamps between 15 minutes ago and present
        time_stamps = generate_time_stamp_range(time_stamp, periods)

        # Sample time stamps
        time_stamps = assign_df_random_time_stamp(df_at_timestamp, time_stamps)

        all_timestamps.extend(time_stamps)

    df['random_time_stamp'] = all_timestamps
    return  df


def get_time_deltas_seconds(time_stamps: pd.DataFrame) -> list:
    """
    Differences the input time_stamps and converts the differences to seconds
    :param time_stamps:
    :return: list of deltas in seconds
    """
    time_deltas = [np.nan]
    for i in range(1, time_stamps.shape[0]):
        delta = (time_stamps[i] - time_stamps[i-1]).total_seconds()
        time_deltas.append(delta)

    return time_deltas


def shift_aggregates(df:pd.DataFrame, length: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    To implement rolling aggregates, this function removes to most historic timestamp for each community
    if there is more than length timestamps present.
    :param df: expects df with multiindex (community, time_stamps)
    :param length: int of maximum number of timestamps that each community can have
    :return: Tuple[pd.DataFrame, pd.Timestamp] shifted dataframe and the timestamp that was removed
    """
    removed_ts = None

    if len(df.index.get_level_values(1).unique()) > length:
        next_timestamp = df.index.get_level_values(1).unique()[1]
        removed_ts = df.index.get_level_values(1).unique()[0]
        df = df.loc[idx[:, next_timestamp:], :]
    return  df, removed_ts


def aggregate_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Groups by df by pickup_community_area and 30 minute time intervals and calculates statistics for each group.
    The function ensures each possible 30 minute time interval timestamp exists for each community.
    :param df: df to aggregate
    :return: aggregated df
    """
    aggregate =  df[[
            "pickup_community_area",
            "random_time_stamp",
            "trip_id", "fare",
            "trip_total",
            "trip_start_day_of_week",
            "trip_start_hour",
        ]]\
        .groupby(["pickup_community_area", pd.Grouper(key="random_time_stamp", freq="30min")])\
        .agg({
            "trip_id": 'count',
            "trip_total": 'mean',
            "trip_start_day_of_week": 'max',
            "trip_start_hour": 'max'
        })
    communities = aggregate.index.get_level_values(0).unique()
    time_stamps = aggregate.index.get_level_values(1).unique()

    full_index = pd.MultiIndex.from_product(
        [communities, time_stamps], names=["pickup_community_area", "random_time_stamp"]
    )

    aggregate = aggregate.reindex(full_index)
    return aggregate


def update_current_time(current_time: pd.Timestamp, item: pd.Series) -> pd.Timestamp:
    """
    compares current time and item['trip_start_timestamp'] to update current_time
    :param current_time: pd.Timestamp to be updated
    :param item: pd.Series containing trip_start_timestamp datetime column
    :return:
    """
    if current_time is None:
        current_time = item['trip_start_timestamp']
    elif current_time - item['trip_start_timestamp'] != datetime.timedelta(0):
        current_time = item['trip_start_timestamp']
    return current_time


def simulate_data(df: pd.DataFrame, stream: str, stream_speed_factor: int | float) -> None:
    """
    :param df: rides data to loop over to simulate
    :param stream: AWS Kinesis stream name
    :param stream_speed_factor: speeds up or slows down the simulation
    :return: None
    """
    # Buffer for observed data
    data = pd.DataFrame()

    current_time = None
    for index, item in df.iterrows():
        logger.info(f"Sending {item}")

        wait = item['time_delta'] / stream_speed_factor

        logger.info("Sending record to Kinesis")
        send_to_kinesis(item, stream=stream)

        data = pd.concat([data, item.to_frame().T])
        try:
            data = data.to_frame()
        except AttributeError:
            pass

        current_time = update_current_time(current_time, item)

        aggregate = aggregate_data(data)


        # shift the aggregate forward one timestamp if we have
        # more than length unique time stamps
        length = 3
        aggregate, removed_ts = shift_aggregates(aggregate, length)

        # Shift observed data
        if removed_ts:
            data = data[data['random_time_stamp'] >= (removed_ts + datetime.timedelta(minutes=30))]

        logger.info(aggregate)

        if len(aggregate.index.get_level_values(1).unique()) > length:
            logger.info(f"{aggregate}")
            raise ValueError("len(aggregate.index.get_level_values(0)) > 16")

        trip_counts = aggregate["trip_id"]
        logger.info(f"aggregated: {trip_counts}")
        logger.info(f"waiting {wait} seconds")
        time.sleep(wait)


def send_to_kinesis(x:pd.Series, stream:str):
    data = x.to_json()
    data = json.dumps(data)

    partition = str(int(x['pickup_community_area']))

    try:
        response = kinesis_client.put_record(
            StreamName=stream,
            Data=data,
            PartitionKey=partition
        )
    except ClientError:
        logger.exception("Failed to put data into stream. partition key: %s", partition)
        raise
    else:
        return response


def main():
    sim_data = pd.read_csv("./data/simulation_data_communities.csv") \
        .set_index("trip_start_timestamp") \
        .sort_index(ascending=True).iloc[SIMULATION_START_INDEX::SIMULATION_STRIDE]

    sim_data.index = pd.to_datetime(sim_data.index)

    sim_data = generate_random_timestamps(sim_data)
    sim_data['random_time_stamp'] = pd.to_datetime(sim_data['random_time_stamp'])
    sim_data = sim_data.sort_values(by='random_time_stamp')

    # Create variable that contains the time difference between current pickup time and last rides
    # pickup time
    sim_data['time_delta'] = get_time_deltas_seconds(sim_data['random_time_stamp'])
    sim_data['time_delta'] = sim_data['time_delta'].fillna(0.0)

    sim_data = sim_data.reset_index()

    simulate_data(sim_data, stream=STREAM_NAME, stream_speed_factor=STREAM_SPEED_FACTOR)


if __name__ == "__main__":
    main()