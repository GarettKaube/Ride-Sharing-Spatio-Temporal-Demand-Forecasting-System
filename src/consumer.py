"""
Unused python file just for testing the Kinesis stream
"""

import logging
import time
import numpy as np
import boto3
import pandas as pd
from botocore.exceptions import ClientError
import time
from logging_config import setup_logging
import datetime
import json
import os

date_time = datetime.datetime.now()
date_time = date_time.strftime("%m-%d-%Y_%H-%M-%S")
setup_logging(log_file_path=f'./logs/consumer_log_{date_time}.log')

logger = logging.getLogger("consumer")

STREAM_SCALE_FACTOR = os.environ.get("STREAM_SCALE_FACTOR")
STREAM_NAME = os.environ["DATA_STREAM_NAME"]

class StreamProcessor:
    def __init__(self, kinesis_client, stream_name):
        self.kinesis_client = kinesis_client
        self.details = None
        self.name = stream_name
        self.stream_exists_waiter = kinesis_client.get_waiter("stream_exists")


    def get_records(self, max_records):
        try:
            response = self.kinesis_client.get_shard_iterator(
                StreamName=self.name,
                ShardId=self.details["Shards"][0]["ShardId"],
                ShardIteratorType="LATEST"
            )
            shard_iter = response["ShardIterator"]
            record_count = 0
            while record_count < max_records:
                response = self.kinesis_client.get_records(
                    ShardIterator=shard_iter, Limit=10
                )
                shard_iter = response["NextShardIterator"]
                records = response["Records"]
                if len(records) > 0:
                    logger.info("Got %s records.", len(records))
                record_count += len(records)
                yield records

        except ClientError:
            logger.exception("")
            raise

    def process_records(self, max_records):
        start_timestamp = None
        for items in self.get_records(max_records):
            if items:
                data = [
                    json.loads(json.loads(item['Data'])) for item in items
                ]

                if start_timestamp is None:
                    trip_start_timestamps = [
                        datetime.datetime.fromtimestamp(item['trip_start_timestamp'] / 1000) for item in data
                    ]
                    start_timestamp = min(trip_start_timestamps)

                print(start_timestamp)

    def describe(self, name):
        """
        Gets metadata about a stream.

        :param name: The name of the stream.
        :return: Metadata about the stream.
        """
        try:
            response = self.kinesis_client.describe_stream(StreamName=name)
            self.name = name
            self.details = response["StreamDescription"]
            logger.info("Got stream %s.", name)
        except ClientError:
            logger.exception("Couldn't get %s.", name)
            raise
        else:
            return self.details


def main():
    kinesis_client = boto3.client("kinesis", region_name="us-west-1")

    stream = StreamProcessor(kinesis_client, stream_name=STREAM_NAME)
    stream.describe(name=STREAM_NAME)
    stream.process_records(10)


if __name__ == "__main__":
    main()
