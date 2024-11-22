# Databricks notebook source
# MAGIC %md
# MAGIC # Generate synthetic datasets for inference

# COMMAND ----------

# MAGIC %pip install /Volumes/mlops_test/power_consumptions/packages/housing_price-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading tables

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
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

# Ensure config.id_col column is cast to string in Spark before converting to Pandas
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_nico") \
                 .withColumn(config.id_col, col(config.id_col).cast("string")) \
                 .toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico") \
                 .withColumn(config.id_col, col(config.id_col).cast("string")) \
                 .toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate 2 synthetic datasets, similar distribution to the existing data and skewed

# COMMAND ----------

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype
import numpy as np
import pandas as pd

def create_synthetic_data(df, drift=False, num_rows=100):
    synthetic_data = pd.DataFrame()
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != config.id_col:
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, num_rows)
        
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))
            
        elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(df[column].dtype, pd.StringDtype):
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(
                    # np.random.randint(min_date.value, max_date.value, num_rows)
                    np.random.randint(0, 100, num_rows)
                )
            else:
                synthetic_data[column] = [min_date] * num_rows
        
        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)
    
    synthetic_data[config.id_col] = df[config.id_col]

    if drift:
        # Skew the top features to introduce drift
        top_features = ["Temperature", "Humidity", "Wind_Speed"]  # Select top 2 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

    return synthetic_data

# Generate and visualize fake data

combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(id for id in combined_set[config.id_col])

synthetic_data_normal = create_synthetic_data(train_set,  drift=False, num_rows=200)
synthetic_data_skewed = create_synthetic_data(train_set, drift=True, num_rows=200)

print(synthetic_data_normal.dtypes)
print(synthetic_data_normal.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add timestamp

# COMMAND ----------

synthetic_normal_df = spark.createDataFrame(synthetic_data_normal)
synthetic_normal_df_with_ts = synthetic_normal_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

synthetic_normal_df_with_ts.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.inference_set_normal_nico"
)

synthetic_skewed_df = spark.createDataFrame(synthetic_data_skewed)
synthetic_skewed_df_with_ts = synthetic_skewed_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

synthetic_skewed_df_with_ts.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.inference_set_skewed_nico"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to feature table

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.power_features
    SELECT DateTime, Temperature, Humidity, Wind_Speed
    FROM {catalog_name}.{schema_name}.inference_set_normal_nico
""")

#write into feature table; update online table
spark.sql(f"""
    INSERT INTO {catalog_name}.{schema_name}.power_features
    SELECT DateTime, Temperature, Humidity, Wind_Speed
    FROM {catalog_name}.{schema_name}.inference_set_skewed_nico
""")
  
# update_response = workspace.pipelines.start_update(
#     pipeline_id=pipeline_id, full_refresh=False)
# while True:
#     update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, 
#                             update_id=update_response.update_id)
#     state = update_info.update.state.value
#     if state == 'COMPLETED':
#         break
#     elif state in ['FAILED', 'CANCELED']:
#         raise SystemError("Online table failed to update.")
#     elif state == 'WAITING_FOR_RESOURCES':
#         print("Pipeline is waiting for resources.")
#     else:
#         print(f"Pipeline is in {state} state.")
#     time.sleep(30)
