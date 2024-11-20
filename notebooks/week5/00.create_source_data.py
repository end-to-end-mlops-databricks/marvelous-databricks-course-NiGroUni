import pandas as pd
import numpy as np
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from power_consumption.config import ProjectConfig
from pyspark.sql import SparkSession
from pyspark.errors.exceptions.connect import AnalysisException

# Load configuration
try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
spark = SparkSession.builder.getOrCreate()

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_nico").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(id for id in combined_set[config.id_col])

# Define function to create synthetic data without random state
def create_synthetic_data(df, num_rows=100):
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

    return synthetic_data

# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)

# spark.sql(f"DROP TABLE {catalog_name}.{schema_name}.source_data_nico")

# fails when table has not been created yet
try:
    existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data_nico").schema
    synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)
except AnalysisException:
    synthetic_spark_df = spark.createDataFrame(synthetic_df)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.source_data_nico"
)

print("DONE")
