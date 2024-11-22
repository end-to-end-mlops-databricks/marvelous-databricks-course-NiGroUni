# Databricks notebook source
# MAGIC %md
# MAGIC # Send request to the endpoint from normal and skewed distribution

# COMMAND ----------

# MAGIC %pip install /Volumes/mlops_test/power_consumptions/packages/housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
import os
import datetime
import itertools
from pyspark.sql import SparkSession

from power_consumption.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# Load configuration
try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name


inference_data_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal_nico") \
                        .withColumn(config.id_col, col(config.id_col).cast("string")) \
                        .toPandas()
inference_data_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed_nico") \
                        .withColumn(config.id_col, col(config.id_col).cast("string")) \
                        .toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico") \
                        .withColumn(config.id_col, col(config.id_col).cast("string")) \
                        .toPandas()


# COMMAND ----------

try:
    # notebook way
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
except AttributeError:
    # local/databricks connect way
    token = os.environ.get("DATABRICKS_TOKEN")
host = spark.conf.get("spark.databricks.workspaceUrl")


# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

# Required columns for inference
required_columns = config.num_features 
required_columns.append(config.id_col)

# Sample records from inference datasets
sampled_normal_records = inference_data_normal[required_columns].to_dict(orient="records")
sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")

# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/power-consumption-model-serving/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response

# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="power-consumption-model-serving",
        dataframe_records=[dataframe_record]
    )
    return response


# COMMAND ----------

# Loop over test records and send requests for 20 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=2)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# Loop over normal records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=2)
for index, record in enumerate(itertools.cycle(sampled_normal_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for normal data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

# Loop over skewed records and send requests for 30 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=3)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)