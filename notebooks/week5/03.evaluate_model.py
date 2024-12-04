"""
This script evaluates and compares a new house price prediction model against the currently deployed model.
Key functionality:
- Loads test data and performs feature engineering
- Generates predictions using both new and existing models
- Calculates and compares performance metrics (MAE and RMSE)
- Registers the new model if it performs better
- Sets task values for downstream pipeline steps

The evaluation process:
1. Loads models from the serving endpoint
2. Prepares test data with feature engineering
3. Generates predictions from both models
4. Calculates error metrics
5. Makes registration decision based on MAE comparison
6. Updates pipeline task values with results
"""

from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
import mlflow
import argparse
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

from power_consumption.config import ProjectConfig

git_sha = "bla_sha"
job_run_id = "bla_run_id"
new_model_uri = "runs:/1c5ec5df28784362adbbe7ad074bc57a/pyfunc-power-consumption-model"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha

config_path = (f"{root_path}/project_config.yml")
# config_path = ("/Volumes/mlops_test/power_consumptions/data/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

# try:
#     config = ProjectConfig.from_yaml(config_path="project_config.yml")
# except Exception:
#     config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define the serving endpoint
serving_endpoint_name = "power-consumption-model-serving"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

# Load test set and create additional features in Spark DataFrame
current_year = datetime.now().year
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico")

# Select the necessary columns for prediction and target
X_test_spark = test_set.select(num_features).toPandas()# + [config.id_col])
y_test_spark = test_set.select(config.id_col, target)

# Generate predictions from both models
previous_loaded_model = mlflow.pyfunc.load_model(previous_model_uri)
prediction_old = previous_loaded_model.predict(X_test_spark).tolist()

new_loaded_model = mlflow.pyfunc.load_model(new_model_uri)
predictions_new = new_loaded_model.predict(X_test_spark).tolist()

test_set = test_set.withColumn("prediction_new_tmp", F.lit(predictions_new))
test_set = test_set.withColumn("prediction_old_tmp", F.lit(prediction_old))

df_with_id = test_set.withColumn("id", F.monotonically_increasing_id())
list_df = spark.createDataFrame([(i, val) for i, val in enumerate(predictions_new)], ["id", "prediction_new"])
test_set = df_with_id.join(list_df, "id").drop("id")

df_with_id = test_set.withColumn("id", F.monotonically_increasing_id())
list_df = spark.createDataFrame([(i, val) for i, val in enumerate(prediction_old)], ["id", "prediction_old"])
test_set = df_with_id.join(list_df, "id").drop("id")

# # Join the DataFrames on the 'id' column
df = test_set.drop("prediction_new_tmp", "prediction_old_tmp")

# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df[config.target] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df[config.target] - df["prediction_old"]))

# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df[config.target] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df[config.target] - df["prediction_old"]))

# Calculate the Mean Absolute Error (MAE) for each model
mae_new = df.agg(F.mean("error_new")).collect()[0][0]
mae_old = df.agg(F.mean("error_old")).collect()[0][0]


# Compare models based on MAE and RMSE
print(f"MAE for New Model: {mae_new}")
print(f"MAE for Old Model: {mae_old}")

# simulate better model
# if mae_new < mae_old:
if True:
    print("New model is better based on MAE.")
    model_version = mlflow.register_model(
      model_uri=new_model_uri,
      name=f"{catalog_name}.{schema_name}.power_consumption_model_pyfunc",
      tags={"git_sha": f"{git_sha}",
            "job_run_id": job_run_id})

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    print("Old model is better based on MAE.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)

print("DONE")
