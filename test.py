from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# Initialisiere die SparkSession
spark = SparkSession.builder \
    .appName("Simple DataFrame") \
    .getOrCreate()

print("Hello World")

# Definiere das Schema des DataFrames
schema = StructType([
    StructField("Spalte1", StringType(), True),
    StructField("Spalte2", StringType(), True)
])

# Erstelle die Daten für den DataFrame
data = [
    ("Zeile1_Spalte1", "Zeile1_Spalte2"),
    ("Zeile2_Spalte1", "Zeile2_Spalte2"),
    ("Zeile3_Spalte1", "Zeile3_Spalte2"),
    ("Zeile4_Spalte1", "Zeile4_Spalte2"),
    ("Zeile5_Spalte1", "Zeile5_Spalte2")
]

# Erstelle den DataFrame
df = spark.createDataFrame(data, schema)

# Zeige den DataFrame an
df.show()

print("HELLO WORLD 2")