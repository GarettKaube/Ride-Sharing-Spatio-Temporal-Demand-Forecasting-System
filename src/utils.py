import duckdb
import boto3


def fetch_df_from_query(query, conn:duckdb.DuckDBPyConnection):
    return conn.execute(query).fetch_df()

def get_param(name:str):
    ssm = boto3.client("ssm", region_name="us-west-1")
    response = ssm.get_parameter(Name=name)
    return response['Parameter']['Value']


