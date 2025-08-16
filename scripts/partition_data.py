"""
Script which splits up the massive rides.csv in taxidata folder into smaller parquet files
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

chunks = pd.read_csv(BASE_DIR / "taxidata/rides.csv", chunksize=100000)

data_path = "data"
saved_file_name = "rides_chunk"

for i, chunk in enumerate(chunks):
    file_name = saved_file_name + f"_{i}"
    chunk.to_parquet(data_path +"/" + file_name + ".parquet")
