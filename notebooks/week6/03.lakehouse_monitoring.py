# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_test/power_consumptions/packages/housing_price-0.0.1-py3-none-any.whl


# COMMAND ----------

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
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

workspace = WorkspaceClient()

monitoring_table = f"{catalog_name}.{schema_name}.model_monitoring_nico"

workspace.quality_monitors.create(
    table_name=monitoring_table,
    assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
    output_schema_name=f"{catalog_name}.{schema_name}",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
        prediction_col="prediction",
        timestamp_col="timestamp",
        granularities=["5 minutes"],
        model_id_col="model_name",
        label_col="power_consumption",
    ),
)

spark.sql(f"ALTER TABLE {monitoring_table} "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# COMMAND ----------

## How to delete a monitor
# workspace.quality_monitors.delete(
#     table_name="mlops_test.power_consumptions.model_monitoring"
# )

# COMMAND ----------

