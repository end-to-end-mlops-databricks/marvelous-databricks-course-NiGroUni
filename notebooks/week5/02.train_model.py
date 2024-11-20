"""
This script trains a LightGBM model for house price prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom calculated house age feature.
"""

from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.utils.environment import _mlflow_conda_env
import argparse
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
from power_consumption.model import PowerConsumptionModelWrapper

git_sha = "bla_sha"
job_run_id = "bla_run_id"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
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
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id


config_path = (f"{root_path}/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)

# try:
#     config = ProjectConfig.from_yaml(config_path="project_config.yml")
# except Exception:
#     config = ProjectConfig.from_yaml(config_path="../../project_config.yml")


# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.power_features"
# function_name = f"{catalog_name}.{schema_name}.calculate_house_age"

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set_nico").toPandas()#.drop("OverallQual", "GrLivArea", "GarageCars")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_nico").toPandas()

# Load feature-engineered DataFrame
training_df = train_set

# Split features and target
X_train = training_df[num_features]
y_train = training_df[target]
X_test = test_set[num_features]
y_test = test_set[target]

# Setup preprocessing and model pipeline
pipeline = Pipeline(
    steps=[("regressor", LGBMRegressor(**parameters))]
)

mlflow.set_experiment(experiment_name=config.mlflow_experiment_name)

with mlflow.start_run(tags={"branch": "week5",
                            "git_sha": f"{git_sha}",
                            "job_run_id": job_run_id}) as run:
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


    wrapped_model = PowerConsumptionModelWrapper(pipeline) # we pass the loaded model to the wrapper
    example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
    example_prediction = wrapped_model.predict(context=None, model_input=example_input)

    signature = infer_signature(model_input=X_train, model_output=example_prediction)

    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/power_consumption-0.0.1-py3-none-any.whl",
                             ],
        additional_conda_channels=None,
    )

    run_id = run.info.run_id
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-power-consumption-model",
        code_paths = [f"{root_path}/notebooks/power_consumption-0.0.1-py3-none-any.whl"],
        signature=signature,
        conda_env=conda_env
    )


model_uri=f'runs:/{run_id}/pyfunc-power-consumption-model'
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)

print(model_uri)
print("DONE")
