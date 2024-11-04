# Databricks notebook source
from pyspark.sql import SparkSession

from power_consumption.data_processor import DataProcessor
from power_consumption.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# path stuff is annoying, in vs code it seems like everything gets executed from repo main level
try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Preprocess data
data_processor = DataProcessor(config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
print("DONE")
