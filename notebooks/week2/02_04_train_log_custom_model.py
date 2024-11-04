# Databricks notebook source
import json
from lightgbm import LGBMRegressor
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from power_consumption.config import ProjectConfig
from power_consumption.utils import adjust_predictions

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# COMMAND ----------

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

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set_nico")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico").toPandas()

X_train = train_set[num_features]
y_train = train_set[target]

X_test = test_set[num_features]
y_test = test_set[target]

# COMMAND ----------

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[
    ('regressor', LGBMRegressor(**parameters))
])


# COMMAND ----------
mlflow.set_experiment(experiment_name=mlflow_experiment_name)
git_sha = "bla"

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}",
          "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(
    train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set",
    version="0")
    mlflow.log_input(dataset, context="training")
    
    # mlflow.sklearn.log_model(
    #     sk_model=pipeline,
    #     artifact_path="lightgbm-pipeline-model",
    #     signature=signature
    # )

# COMMAND ----------

class PowerConsumptionModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(
                predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")



# COMMAND ----------

wrapped_model = PowerConsumptionModelWrapper(pipeline) # we pass the loaded model to the wrapper

# COMMAND ----------


with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    
    run_id = run.info.run_id
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-power-consumption-model",
        signature=infer_signature(model_input=[], model_output=[])
    )

# COMMAND ----------
loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/pyfunc-power-consumption-model')
loaded_model.unwrap_python_model()

# COMMAND ----------
model_name = f"{catalog_name}.{schema_name}.power_consumption_model_pyfunc"

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/pyfunc-power-consumption-model',
    name=model_name,
    tags={"git_sha": f"{git_sha}"})
# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")  
 
model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)
# COMMAND ----------
model

# COMMAND ----------

print("DONE")

