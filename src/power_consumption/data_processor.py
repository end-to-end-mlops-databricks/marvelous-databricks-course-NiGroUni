from pyspark.sql import SparkSession
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

spark = SparkSession.builder.appName("DataProcessor").getOrCreate()


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None
        self.df = self.load_data()
        # self.df.show()
        print(self.df.head())

    def load_data(self):
        return spark.sql(f"SELECT * FROM {self.config.catalog_name}.{self.config.schema_name}.power_consumption_nico").toPandas()

    def preprocess(self):
        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Fill missing values with mean or default values
        self.df.fillna(0, inplace=True)

        # Extract target and relevant features
        target = self.config.target
        id_col = self.config.id_col
        relevant_columns = num_features + [target] + [id_col]
        self.df = self.df[relevant_columns]

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.df, test_size=test_size, random_state=random_state)

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set = spark.createDataFrame(train_set)   
        
        test_set = spark.createDataFrame(test_set)

        train_set.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set_nico")
        
        test_set.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set_nico")

        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set_nico "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set_nico "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
