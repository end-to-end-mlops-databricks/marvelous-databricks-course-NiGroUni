from pyspark.sql.functions import col
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from databricks.sdk import WorkspaceClient
import argparse

from power_consumption.config import ProjectConfig

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
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

try:
    config = ProjectConfig.from_yaml(config_path=config_path)
except Exception:
    try:
        config = ProjectConfig.from_yaml(config_path="project_config.yml")
    except Exception:
        config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

inf_table = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.`power-consumption-model-serving_payload`")

request_schema = StructType([
    StructField("dataframe_records", ArrayType(StructType([
        StructField("Temperature", DoubleType(), True),
        StructField("Humidity", DoubleType(), True),
        StructField("Wind_speed", DoubleType(), True),
        StructField(config.id_col, StringType(), True)
    ])), True)  
])

response_schema = StructType([
    StructField("predictions", ArrayType(DoubleType()), True),
    StructField("databricks_output", StructType([
        StructField("trace", StringType(), True),
        StructField("databricks_request_id", StringType(), True)
    ]), True)
])


inf_table_parsed = inf_table.withColumn("parsed_request", 
                                        F.from_json(F.col("request"),
                                                    request_schema))

inf_table_parsed = inf_table_parsed.withColumn("parsed_response",
                                               F.from_json(F.col("response"),
                                                           response_schema))

df_exploded = inf_table_parsed.withColumn("record",
                                          F.explode(F.col("parsed_request.dataframe_records")))



df_final = df_exploded.select(
    F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
    "timestamp_ms",
    "databricks_request_id",
    "execution_time_ms",
    F.col(f"record.{config.id_col}").alias(config.id_col),
    F.col("record.Temperature").alias("Temperature"),
    F.col("record.Humidity").alias("Humidity"),
    F.col("record.Wind_speed").alias("Wind_speed"),
    F.col("parsed_response.predictions")[0].alias("prediction"),
    F.lit("power_consumption_model_pyfunc").alias("model_name")
)


test_set = spark.table(f"{catalog_name}.{schema_name}.train_set_nico")
inference_set_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal_nico")
inference_set_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed_nico")

inference_set = inference_set_normal.union(inference_set_skewed)


df_final_with_status = df_final \
    .join(test_set.select(config.id_col, config.target), on=config.id_col, how="left") \
    .withColumnRenamed(config.target, "power_consumption_test") \
    .join(inference_set.select(config.id_col, config.target), on=config.id_col, how="left") \
    .withColumnRenamed(config.target, "power_consumption_inference") \
    .select(
        "*",  
        F.coalesce(F.col("power_consumption_test"), F.col("power_consumption_inference")).alias("power_consumption")
    ) \
    .drop("power_consumption_test", "power_consumption_inference") \
    .withColumn("power_consumption", F.col("power_consumption").cast("double")) \
    .withColumn("prediction", F.col("prediction").cast("double")) \
    .dropna(subset=["power_consumption", "prediction"])

power_features = spark.table(f"{catalog_name}.{schema_name}.power_features")

df_final_with_features = df_final_with_status.drop("Wind_speed") \
    .join(power_features.select(config.id_col, "Wind_speed"), on=config.id_col, how="left")


df_final_with_features.write.format("delta").mode("append")\
    .saveAsTable(f"{catalog_name}.{schema_name}.model_monitoring_nico")


workspace.quality_monitors.run_refresh(
    table_name=f"{catalog_name}.{schema_name}.model_monitoring_nico"
)
