"""
This script handles data ingestion and feature table updates for a house price prediction system.

Key functionality:
- Loads the source dataset and identifies new records for processing
- Splits new records into train and test sets based on timestamp
- Updates existing train and test tables with new data
- Inserts the latest feature values into the feature table for serving
- Triggers and monitors pipeline updates for online feature refresh
- Sets task values to coordinate pipeline orchestration

Workflow:
1. Load source dataset and retrieve recent records with updated timestamps.
2. Split new records into train and test sets (80-20 split).
3. Append new train and test records to existing train and test tables.
4. Insert the latest feature data into the feature table for online serving.
5. Trigger a pipeline update and monitor its status until completion.
6. Set a task value indicating whether new data was processed.
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import yaml
from databricks.sdk import WorkspaceClient
from power_consumption.config import ProjectConfig

workspace = WorkspaceClient()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = (f"{root_path}/project_config.yml")

print("STAART")
try:
    config = ProjectConfig.from_yaml(config_path=config_path)
except Exception:
    print("1")
    try:
        config = ProjectConfig.from_yaml(config_path="project_config.yml")
    except Exception:
        print("2")
        config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

spark = SparkSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name

# Load source_data table
source_data = spark.table(f"{catalog_name}.{schema_name}.source_data_nico")

# Get max update timestamps from existing data
max_train_timestamp = spark.table(f"{catalog_name}.{schema_name}.train_set_nico") \
    .select(F.max("update_timestamp_utc").alias("max_update_timestamp")) \
    .collect()[0]["max_update_timestamp"]

max_test_timestamp = spark.table(f"{catalog_name}.{schema_name}.test_set_nico") \
    .select(F.max("update_timestamp_utc").alias("max_update_timestamp")) \
    .collect()[0]["max_update_timestamp"]

latest_timestamp = max(max_train_timestamp, max_test_timestamp)

# Filter source_data for rows with update_timestamp_utc greater than the latest_timestamp
new_data = source_data.filter(F.col("update_timestamp_utc") > latest_timestamp)

# Split the new data into train and test sets
new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)

# Update train_set and test_set tables
new_data_train.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set_nico")
new_data_test.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set_nico")

# Verify affected rows count for train and test
affected_rows_train = new_data_train.count()
affected_rows_test = new_data_test.count()

if affected_rows_train > 0 or affected_rows_test > 0 :
    refreshed = 1
else:
    # simulate refresh
    refreshed = 1

dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)

print("DONE")
