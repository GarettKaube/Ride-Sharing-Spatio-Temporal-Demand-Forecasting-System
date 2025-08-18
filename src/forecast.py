import json
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import io
from pandas import IndexSlice as idx
import requests
import os

INFERENCE_SERVER_IP = os.environ.get("INFERENCE_SERVER_IP")
INFERENCE_SERVER_PORT = os.environ.get("INFERENCE_SERVER_PORT")

SERVER_URL = f"http://{INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}" + "/invocations/"

COMMUNITIES_TO_FORECAST = [8, 33, 32]

REGION = "us-west-1"

ssm = boto3.client("ssm", region_name=REGION)
response = ssm.get_parameter(Name="N_LAGS")
N_LAGS = int(response['Parameter']['Value'])

ENV = os.environ.get("ENVIRONMENT", "DEV")

s3 = boto3.client("s3", region_name=REGION)
DEST_BUCKET = os.environ.get("DEST_BUCKET")


def get_buffered_data():
    client = boto3.resource("dynamodb", region_name="us-west-1")
    table = client.Table('RideKinesisStreamBuffer')

    response = table.scan()
    items = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    return items


def format_data(items):
    data = pd.json_normalize(items)

    # Remove 'data' prefixes from columns
    data.columns = data.columns.str.replace('data.', '')

    # Convert data types
    data['pickup_community_area'] = data['pickup_community_area'].astype(float)
    data['trip_total'] = data['trip_total'].astype(float)
    data['fare'] = data['fare'].astype(float)
    data['trip_start_day_of_week'] = data['trip_start_day_of_week'].astype(float)
    data['trip_start_hour'] = data['trip_start_hour'].astype(float)
    data['random_time_stamp'] = data['random_time_stamp'].astype(float)
    data['ttl'] = data['ttl'].astype(float)

    # Fix timestamps
    data['random_time_stamp'] = pd.to_datetime(data['random_time_stamp'], utc=True, unit='ms')
    data['ttl'] = pd.to_datetime(data['ttl'], utc=True, unit="s")

    return data


def aggregate_data(df: pd.DataFrame):
    aggregate = df[[
        "pickup_community_area",
        "random_time_stamp",
        "trip_id", "fare",
        "trip_total",
        "trip_start_day_of_week",
        "trip_start_hour",
    ]] \
        .groupby(["pickup_community_area", pd.Grouper(key="random_time_stamp", freq="30min")]) \
        .agg({
        "trip_id": 'count',
        "fare": 'mean',
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


def load_weather_data(bucket='mybucket1654'):
    key = 'weather_features.csv'

    response = s3.get_object(Bucket=bucket, Key=key)

    body = response['Body'].read()
    weather_features = pd.read_csv(io.BytesIO(body))
    weather_features['time'] = pd.to_datetime(weather_features['time'], utc=True)

    return weather_features


def join_weather_data(df:pd.DataFrame, weather:pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df['interval'] = df['random_time_stamp'].dt.floor('30min')
    df = df.merge(weather, left_on='interval', right_on='time', how='left')\
        .ffill()\
        .drop(["time", "random_time_stamp"], axis=1)
    return df


def encode_periodic_features(df:pd.DataFrame):
    def cyclical_transform(x, period):
        return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)

    df['trip_start_hour_sin'] = df['trip_start_hour'].apply(lambda x: cyclical_transform(x, 24)[0])
    df['trip_start_hour_cos'] = df['trip_start_hour'].apply(lambda x: cyclical_transform(x, 24)[1])

    df['trip_start_day_of_week_sin'] = df['trip_start_day_of_week'].apply(lambda x: cyclical_transform(x, 7)[0])
    df['trip_start_day_of_week_cos'] = df['trip_start_day_of_week'].apply(lambda x: cyclical_transform(x, 7)[1])

    df = df.drop(['trip_start_hour', 'trip_start_day_of_week'], axis=1)

    return df


def fill_absent_communities(X):
    """
    For a time stamp t we observe a matrix (n_nodes, n_features). If a community had no counts at time t, then
    n_nodes will be less than expected so we just insert the node into the observation with default values.
    Note: this process could probably be sped up by using pd.MultiIndex.from_product(
        [communities, time_stamps], names=["pickup_community_area", "random_time_stamp"]
    ) instead then a ffill after
    :param X:
    :return:
    """
    weather_features = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres',
                        'coco']

    community_areas = set(range(1, 77 + 1))
    missing_com = community_areas - set(X['pickup_community_area'])

    day_of_week_sin = X["trip_start_day_of_week_sin"].iat[0]
    day_of_week_cos = X["trip_start_day_of_week_cos"].iat[0]

    hour_sin = X["trip_start_hour_sin"].iat[0]
    hour_cos = X["trip_start_hour_cos"].iat[0]

    def fill_val(val=None):
        if val:
            return [val for com in missing_com]

        return [0 for com in missing_com]

    columns = X.columns.drop(
        ["pickup_community_area",
         "trip_start_day_of_week_sin",
         "trip_start_day_of_week_cos",
         "trip_start_hour_sin",
         "trip_start_hour_cos"] + weather_features
    )

    data = {
        "pickup_community_area": [com for com in missing_com],
        "trip_start_day_of_week_sin": fill_val(day_of_week_sin),
        "trip_start_day_of_week_cos": fill_val(day_of_week_cos),
        "trip_start_hour_sin": fill_val(hour_sin),
        "trip_start_hour_cos": fill_val(hour_cos)
    }

    # Fill in the weather
    for weather_fet in weather_features:
        data.update({weather_fet: fill_val(X[weather_fet].iloc[-1])})

    # Fill in other unknown columns with 0
    more_data = {
        col: fill_val() for col in columns
    }

    data.update(more_data)

    new_df = pd.DataFrame(data)

    X = pd.concat([X, new_df], axis=0, ignore_index=True) \
        .sort_values(by="pickup_community_area")
    return X


def make_features_array(df:pd.DataFrame) -> list:
    groups = ['interval']
    graph_ts = df.groupby(groups)[df.columns.drop(groups)]
    features = []
    for g, val in graph_ts:
        val = fill_absent_communities(val)

        feat = val.drop(['pickup_community_area'], axis=1) \
            .to_numpy()
        features.append(feat)

    return features


def lag_features_np(features: np.ndarray, n_time_steps_back: int) -> np.ndarray:
    """Reduced memory consumption of this algorithm by 100% by efficiently using numpy"""
    features = np.flip(features, axis=0)
    # Empty array shaped (n_time_stamps_after_lag, n_time_steps_back + 1, n_nodes, n_features)
    added_lagged_features = np.empty(
        (
            len(features) - n_time_steps_back,
            1 + n_time_steps_back,
            features.shape[1],
            features.shape[2]
        )
    )

    for i, X in enumerate(features[:len(features) - n_time_steps_back]):
        i_old = i

        lagged = np.empty(
            (
                n_time_steps_back + 1,
                features.shape[1],
                features.shape[2]
            )
        )

        i += n_time_steps_back + 1

        lagged[0, :, :] = X

        # Loop over n_time_steps previous feature matrices up to current time i
        for j, X_lagged in enumerate(features[i - n_time_steps_back: i]):
            lagged[j + 1, :, :] = X_lagged

        added_lagged_features[i_old, :, :, :] = lagged

    added_lagged_features = np.flip(added_lagged_features, axis=0)
    added_lagged_features = np.flip(added_lagged_features, axis=1)

    return added_lagged_features


def handle_not_enough_observations(features: np.ndarray[float], n_lags: int) -> np.ndarray[float]:
    """
    Appends an array full of zeros before the observed features to account for not
    enough past observations to construct n_lags of the observations
    :param features: np.ndarray[float] shape: (time_stamps, n_nodes, n_features)
    :param n_lags:
    :return: np.ndarray[float]
    """
    difference = features.shape[0] - (n_lags + 1)
    if difference < 0:
        n_nodes = features.shape[1]
        n_features = features.shape[2]
        zeros = np.zeros(shape = (abs(difference), n_nodes, n_features))
        features = np.concatenate([zeros, features], axis=0)
    return features


def shift_aggregates(df:pd.DataFrame, length:int) -> pd.DataFrame:
    """
    Ensures each community has at most length number of timestamps by removing the most historical
    timestamp for each community
    expects df with multiindex (community, time_stamps)
    :param df:
    :param length:
    :return:
    """
    if len(df.index.get_level_values(1).unique()) > length:
        next_timestamp = df.index.get_level_values(1).unique()[1]
        df = df.loc[idx[:, next_timestamp:], :]
    return  df


def get_forecast(X:list | np.ndarray, date, communities:list | None, 
                 server_url:str):
    
    if isinstance(X, np.ndarray):
        X = X.tolist()

    body = {
        "date": date,
        "X": X,
    }

    if communities:
        body["communities"] = communities

    response = requests.post(url=server_url, json=body)

    if response.status_code == 200:
        return response.json()


def save_forecast_to_s3(forecast: list[list], date, bucket:str, env:str) -> None:
    date_str = str(date)\
        .replace(":", "_")\
        .replace(" ", "_")\
        .replace(".", "_")

    for community_num, prediction in forecast:
        data = {
            "date": str(date),
            "community_num": community_num,
            "forecast": prediction[0]
        }

        json_data = json.dumps(data).format("UTF-8")
        try:
            response = s3.put_object(
                Body=json_data,
                Bucket=bucket,
                key=f"{env}/community_{community_num}_forecasts/{community_num}_{date_str}.json"
            )
        except ClientError as e:
            print(f"failed to put forecast to s3: {community_num}, {date_str}", e)
            raise


def lambda_handler(event, context):

    items = get_buffered_data()

    data = format_data(items)
    agg = aggregate_data(data)\
        .fillna(0.00)

    agg = shift_aggregates(agg, N_LAGS + 1)

    # Get and join weather data
    weather = load_weather_data()
    data_weather = join_weather_data(agg, weather)

    # Manage periodicity of day of week and hour of day
    data_weather = encode_periodic_features(data_weather)

    # Prepare features into a matrix shaped (n_time_stamps, n_nodes, n_features)
    features = make_features_array(data_weather)
    features = np.array(features, dtype=float)

    # Ensure there are enough observations to create N_LAGS of the features
    features = handle_not_enough_observations(features, N_LAGS)

    # Output shaped (n_time_stamps, N_LAGS + 1, N_NODES, N_FEATURES)
    lagged_features = lag_features_np(features, N_LAGS)

    date = str(data_weather.index[-1])
    forecast = get_forecast(
        X=lagged_features,
        date=date,
        communities=COMMUNITIES_TO_FORECAST,
        server_url=SERVER_URL
    )

    save_forecast_to_s3(forecast, date, DEST_BUCKET, ENV)

    return {
        'statusCode': 200,
        'body': json.dumps(forecast)
    }

