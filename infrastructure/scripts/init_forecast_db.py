"""
Script for creating a dynamodb table called RideKinesisProcessor to track records that were already processed
"""
import boto3
from botocore.exceptions import ClientError
import os

ENV = os.environ.get("ENVIRONMENT", "DEV")
AWS_REGION = os.environ.get("REGION")

class RideForecastTable:
    def __init__(self, client, name):
        self.dynamodb_client = client
        self.table = None
        self.table_name = name

    def create_table(self):
        try:
            self.table = self.dynamodb_client.create_table(
                TableName=self.table_name,
                KeySchema = [
                    {"AttributeName": "community", "KeyType": "HASH"},
                    {"AttributeName": "date", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "community", "AttributeType": "N"},
                    {"AttributeName": "date", "AttributeType": "N"},
                ],
                BillingMode="PAY_PER_REQUEST"
            )

            client = boto3.resource("dynamodb", region_name="us-west-1")
            table = client.Table(self.table_name)
            table.wait_until_exists()
        except ClientError as err:
            print("Failed to create table. code: %s, %s",
                err.response["Error"]["Code"], err.response["Error"]["Message"]
            )
            raise
        else:
            return self.table

    def enable_ttl(self):
        self.dynamodb_client.update_time_to_live(
            TableName=self.table_name,
            TimeToLiveSpecification={
                'Enabled': True,
                'AttributeName': 'ttl'
            }
        )


def main():
    client = boto3.client("dynamodb", region_name=AWS_REGION)

    table = RideForecastTable(client, name=f"RideForecast{ENV}")
    table.create_table()
    table.enable_ttl()

if __name__ == "__main__":
    main()