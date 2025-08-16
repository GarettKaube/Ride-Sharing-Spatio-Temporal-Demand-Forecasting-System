"""
Simulates incoming ride requests to send to the ML system, starting with Kinesis, then Lambda for
processing and then FastAPI for forecasting.
"""
from src.utils import get_param
from src.producer import generate_random_timestamps, get_time_deltas_seconds, simulate_data
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# Increase this to increase the rate of rides
STREAM_SPEED_FACTOR = int(get_param("STREAM_SPEED_FACTOR"))

STREAM_NAME = os.environ["DATA_STREAM_NAME"]

# Makes simulation simulate every SIMULATION_STRIDEth event
SIMULATION_STRIDE = 100

SIMULATION_START_INDEX = 150000


def main():
    sim_data = pd.read_csv("data/simulation_data_communities.csv") \
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
