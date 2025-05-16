from pyspark.sql import SparkSession
from pyspark.sql.functions import col

#--------------
#   QUERY 1
#--------------

#The five types of plays are classified by their riskiness.
#It helps to identify the tactics of the AI

spark = SparkSession.builder \
    .appName("TGVD_GenericQuery") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

path_training = "CardsParquetData/trained_blackjack.parquet"
path_match = "CardsParquetData/played_blackjack.parquet"

df = spark.read.parquet(path_training)
df_ = df.rdd.zipWithIndex().toDF(["data", "index"])

print(df.count())

#We can try to see how the AI agent evolves along iterations
#For that we will divide the dataset in parts (as we did 100000 matches and
#we only took a match for each 50 matches we will divide it into 5 parts)

CHUNK_SIZE = 440
print(df_.show(truncate =  False))
df_.select("data").printSchema()

from pyspark.sql.functions import floor

df_clean = df_.select(
    col("index"),
    #col("data.Timestamp").alias("Timestamp"),
    col("data.Shown_cards").alias("Hand_-1"),
    col("data.`Hand 0`").alias("Hand_0"),
    col("data.`Hand 1`").alias("Hand_1"),
    col("data.`Hand 2`").alias("Hand_2"),
    col("data.`Hand 3`").alias("Hand_3"),
    col("data.`Hand 4`").alias("Hand_4"),
    col("data.`Hand 5`").alias("Hand_5"),
    col("data.`Hand 6`").alias("Hand_6"),
    col("data.`Hand 7`").alias("Hand_7")
)

df_clean.show(5)

df_chunks = df_clean.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))

df_chunks.show(5)

# Contar jugadas no nulas
from pyspark.sql.functions import expr, when

hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5", "Hand_6", "Hand_7"]
df_moves = df_chunks.withColumn("n_moves",
    1 + sum([
        when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), 1).otherwise(0)
        for i in range(1, 8)
    ])
)

df_moves.show(5)

df_risk = df_moves.withColumn(
    "Risk Level",
    when(col("n_moves") <= 1, "Safe")
    .when(col("n_moves") == 2, "Tactical")
    .when(col("n_moves") == 3, "Aggressive")
    .when(col("n_moves") == 4, "Risky")
    .otherwise("Are you crazy?")
)

df_risk.show(5)

#--------------
#   QUERY 2
#--------------

#The types of strategies, (safe, tactical, risky…) and it’s proportion
#It helps to identify which strategy is the common one.

df_fin = df_risk.select(col("Chunk Number"), col("n_moves"), col("Risk Level")).groupBy("Chunk Number", "Risk Level").count().orderBy("Chunk Number", "count")
print(df_fin.show(30))

df_query2 = df_fin.withColumn("Proportion", round(col("count")*100/CHUNK_SIZE, 2))
df_query2.show(30)