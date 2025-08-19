import json
import base64
import boto3
from botocore.exceptions import ClientError
from decimal import Decimal
import time
import os

REGION = "us-west-1"

ssm = boto3.client("ssm", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)

response = ssm.get_parameter(Name="STREAM_SPEED_FACTOR")
STREAM_SPEED_FACTOR = int(response['Parameter']['Value'])

response = ssm.get_parameter(Name="N_LAGS")
N_LAGS = int(response['Parameter']['Value'])
# Number of time steps total including current and lagged values
N_TIME_STEPS = N_LAGS + 1

ENV = os.environ.get("ENVIRONMENT", "DEV")

BUFFER_TTL = (3000 * N_TIME_STEPS) / STREAM_SPEED_FACTOR

BUFFER_TABLE_NAME = f"RideKinesisStreamBuffer{ENV}"
PROCESSING_TABLE_NAME = f"RideKinesisProcessor{ENV}"


def try_add_item_to_db(trip_id: str, arrival: float, table_name:str) -> bool:
    """ Adds the event to a dynamodb to ensure we do not process the record
    again.
    """
    in_db = False
    client = boto3.resource("dynamodb", region_name=REGION)
    table = client.Table(table_name)

    item = {
        'id': str(trip_id),
        'ttl': Decimal(str(arrival + 3000)) 
    }

    try:
        table.put_item(
            Item=item,
            ConditionExpression='attribute_not_exists(id)'
        )
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            print("Item already in db")
            in_db = True
        else:
            raise
    return in_db


def add_data_to_buffer(data, arrival_time, table_name:str):
    print("Adding data to buffer")
    print(f"ttl: {arrival_time + BUFFER_TTL}")

    client = boto3.resource("dynamodb", region_name=REGION)
    table = client.Table(table_name)

    data = data.copy()

    try:
        id = data.pop('trip_id')
        table.put_item(
            TableName=table_name,
            Item={
                "trip_id": str(id),
                "random_time_stamp": Decimal(str(data.pop('random_time_stamp'))),
                "ttl": Decimal(str(arrival_time + BUFFER_TTL)),
                "data": {
                    "trip_start_timestamp": Decimal(str(data['trip_start_timestamp'])),
                    "trip_end_timestamp": str(data['trip_end_timestamp']),
                    "trip_seconds": str(int(data['trip_seconds'])),
                    "trip_miles": Decimal(str(data['trip_miles'])),
                    "pickup_census_tract": str(data['pickup_census_tract']),
                    "dropoff_census_tract": str(data['dropoff_census_tract']),
                    "pickup_community_area": str(int(data['pickup_community_area'])),
                    "dropoff_community_area": str(data['dropoff_community_area']),
                    "fare": Decimal(str(data['fare'])),
                    "tip": Decimal(str(data['tip'])),
                    "additional_charges": Decimal(str(data['additional_charges'])),
                    "trip_total": Decimal(str(data['trip_total'])),
                    "shared_trip_authorized": str(data['shared_trip_authorized']),
                    "trips_pooled": str(int(data['trips_pooled'])),
                    "pickup_centroid_latitude": Decimal(str(data['pickup_centroid_latitude'])),
                    "pickup_centroid_longitude": Decimal(str(data['pickup_centroid_longitude'])),
                    "pickup_centroid_location": str(data['pickup_centroid_location']),
                    "dropoff_centroid_latitude": str(data['dropoff_centroid_latitude']),
                    "dropoff_centroid_longitude": str(data['dropoff_centroid_longitude']),
                    "dropoff_centroid_location": str(data['dropoff_centroid_location']),
                    "trip_start_year": str(int(data['trip_start_year'])),
                    "trip_start_month": str(int(data['trip_start_month'])),
                    "trip_start_day": str(int(data['trip_start_day'])),
                    "trip_start_day_of_week": str(int(data['trip_start_day_of_week'])),
                    "trip_start_hour": str(int(data['trip_start_hour'])),
                    "row_number_for_community": str(int(data['row_number_for_community'])),
                    "time_delta": Decimal(str(data['time_delta']))
                }
            }
        )
    except ClientError:
        print(f"Failed to put item id: {id}")

    except KeyError:
        print("data is missing key 'trip_id'.")
        raise


def invoke_get_rides_forecast() -> bool:
    # The getRidesForecast does not need any actual input since the function 
    # will just read from the buffer
    dummy_payload = {
        "key1": "helloworld"
    }

    lambda_client.invoke(
        FunctionName="getRidesForecast",
        InvocationType="Event",
        payload=json.dumps(dummy_payload)
    )

    return True


def lambda_handler(event, context):
    n_events = len(event['Records'])
    print(f"received {n_events} records")

    for event in event['Records']:
        record = event['kinesis']
        
        payload = base64.b64decode(record['data']).decode('utf-8')
        data = json.loads(json.loads(payload))

        # Time when lambda actually started processing the event
        processing_time = time.time()
        
        trip_id = data['trip_id']
        print(f"received trip_id: {trip_id}")
 
        # Enable idempotency
        if not try_add_item_to_db(data['trip_id'], processing_time, PROCESSING_TABLE_NAME):
            # Add data to retention buffer
            add_data_to_buffer(data, processing_time, BUFFER_TABLE_NAME)


    invoke_get_rides_forecast()
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully processed {n_events}')
    }
