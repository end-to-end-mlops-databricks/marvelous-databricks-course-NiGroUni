# Databricks notebook source
# MAGIC %pip install /Volumes/main/default/file_exchange/nico/power_consumption-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import functions as F
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from power_consumption.config import ProjectConfig


# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")


try:
    config = ProjectConfig.from_yaml(config_path="project_config.yml")
except Exception:
    config = ProjectConfig.from_yaml(config_path="../../project_config.yml")


# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name
mlflow_experiment_name = config.mlflow_experiment_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.power_features"
function_name = f"{catalog_name}.{schema_name}.round_temperature"


# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_nico")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico")


# COMMAND ----------
# Create or replace the power_features table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.power_features
(DateTime STRING NOT NULL,
 Temperature DOUBLE,
 Humidity DOUBLE,
 Wind_Speed DOUBLE);
""")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.power_features "
          "ADD CONSTRAINT power_pk PRIMARY KEY(DateTime);")

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.power_features "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# Insert data into the feature table from both train and test sets
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.power_features "
          f"SELECT DateTime, Temperature, Humidity, Wind_Speed FROM {catalog_name}.{schema_name}.train_set_nico")
spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.power_features "
          f"SELECT DateTime, Temperature, Humidity, Wind_Speed FROM {catalog_name}.{schema_name}.test_set_nico")

# COMMAND ----------
# Define a function to calculate the power's age using the current year and YearBuilt
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(temperature DOUBLE)
RETURNS INT
LANGUAGE PYTHON AS
$$
return round(temperature)
$$
""")
# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_nico").drop("Humidity", "Wind_Speed")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico").toPandas()

# COMMAND ----------

# TODO: This leads to unauthorized error as Maria said

# Feature engineering setup

training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["Humidity", "Wind_Speed"],
            lookup_key="DateTime",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="temperature_rounded",
            input_bindings={"temperature": "Temperature"},
        ),
    ],
    # exclude_columns=["bla"]
)
# COMMAND ----------

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate temperature_rounded for training and test set
test_set["temperature_rounded"] = test_set["Temperature"].round()

# Split features and target
X_train = training_df[num_features + ["temperature_rounded"]]
y_train = training_df[target]
X_test = test_set[num_features + ["temperature_rounded"]]
y_test = test_set[target]
# Setup model pipeline
pipeline = Pipeline(
    steps=[("regressor", LGBMRegressor(**parameters))]
)

# Set and start MLflow experiment
mlflow.set_experiment(experiment_name=mlflow_experiment_name)
git_sha = "bla"

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=f"{catalog_name}.{schema_name}.power_consumptions_model_fe")
    