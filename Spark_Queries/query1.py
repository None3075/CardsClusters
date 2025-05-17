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

CHUNK_SIZE = 500
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
from pyspark.sql.functions import expr, when, coalesce

hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5", "Hand_6", "Hand_7"]
df_moves = df_chunks.withColumn("n_moves",
    1 + sum([
        when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), 1).otherwise(0)
        for i in range(1, len(hand_cols))
    ])
)

df_moves.show(5)

rev_hand_cols = ["Hand_7", "Hand_6", "Hand_5", "Hand_4", "Hand_3", "Hand_2", "Hand_1", "Hand_0"]
df_result_train = df_moves.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
df_result_train.show()

df_result_train = df_result_train.withColumn(
    "Result",
    when(col("Final_Hand").isNull(), "Unknown")
    .when(col("Final_Hand")[0] > 21, "Lose")
    .when(col("Final_Hand")[1] > 21, "Win")
    .when(col("Final_Hand")[0] > col("Final_Hand")[1], "Win")
    .when(col("Final_Hand")[0] < col("Final_Hand")[1], "Lose")
    .otherwise("Draw")
)
df_result_train.show()

probabilities = [0.25, 0.5, 0.75]

quartiles = df_result_train.select("n_moves").approxQuantile("n_moves", probabilities, 0.01)

df_risk = df_moves.withColumn(
    "Risk Level",
    when(col("n_moves") <= quartiles[0], "Safe")
    .when((col("n_moves") > quartiles[0]) & (col("n_moves") <= quartiles[1]), "Tactical")
    .when((col("n_moves") > quartiles[1]) & (col("n_moves") <= quartiles[2]), "Risky")
    .otherwise("Suicidal")
)

df_risk.show()

df_risk = df_risk.groupBy("Chunk Number", "Risk Level").count().orderBy("Chunk Number", "count")
df_risk.show(truncate = False)