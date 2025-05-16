from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sys

#--------------
#   QUERY 3
#--------------

#The chances of winning given the first hand
#It helps to maximize the profit and minimize the loss by evaluating the initial hand

spark = SparkSession.builder \
    .appName("TGVD_GenericQuery") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

path_training = "CardsParquetData/trained_blackjack.parquet"
path_match = "CardsParquetData/played_blackjack.parquet"

df_train = spark.read.parquet(path_training)
df_play = spark.read.parquet(path_match)

df_play.show()
df_train.show()

from pyspark.sql.functions import floor
from pyspark.sql.functions import monotonically_increasing_id

df_train = df_train.withColumn("index", (monotonically_increasing_id() + 1))
df_play = df_play.withColumn("index", (monotonically_increasing_id() + 1))

CHUNK_SIZE = 440
df_chunks_train = df_train.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))
df_chunks_play = df_play.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))

df_chunks_play.show()

#Funtion to detect if we have a win, draw or a lose
from pyspark.sql.functions import coalesce, when

rev_hand_cols = ["Hand 7", "Hand 6", "Hand 5", "Hand 4", "Hand 3", "Hand 2", "Hand 1", "Hand 0"]
df_result_train = df_chunks_train.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
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

rev_hand_cols = ["Hand 5", "Hand 4", "Hand 3", "Hand 2", "Hand 1", "Hand 0"]
df_result_play = df_chunks_play.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
df_result_play.show()

df_result_play = df_result_play.withColumn(
    "Result",
    when(col("Final_Hand").isNull(), "Unknown")
    .when(col("Final_Hand")[0] > 21, "Lose")
    .when(col("Final_Hand")[1] > 21, "Win")
    .when(col("Final_Hand")[0] > col("Final_Hand")[1], "Win")
    .when(col("Final_Hand")[0] < col("Final_Hand")[1], "Lose")
    .otherwise("Draw")
)
df_result_play.show()

df_clean_play = df_result_play.withColumn("1st Hand Card", col("Shown_cards")[0]).select("Chunk Number", "Final_Hand", "Result", "1st Hand Card")
df_clean_play.show()
df_clean_train = df_result_play.withColumn("1st Hand Card", col("Shown_cards")[0]).select("Chunk Number", "Final_Hand", "Result", "1st Hand Card")
df_clean_train.show()

df_clean_play = df_clean_play.groupBy("1st Hand Card", "Result", "Chunk Number").count().orderBy(["Chunk Number", "count", "1st Hand Card"], ascending = False)
df_clean_play.show()

df_clean_train = df_clean_train.groupBy("1st Hand Card", "Result", "Chunk Number").count().orderBy(["Chunk Number", "count", "1st Hand Card"], ascending = False)
df_clean_train.show()

from pyspark.sql.functions import sum, count

total_df = df_clean_play.groupBy("1st Hand Card", "Chunk Number").agg(count("*").alias("TotalGames"))
wins_df = df_clean_play.filter(col("Result") == "Win").groupBy("1st Hand Card", "Chunk Number").agg(count("*").alias("Wins"))

df_winrate = total_df.join(wins_df, on=["1st Hand Card", "Chunk Number"], how="left").fillna(0, subset=["Wins"]).withColumn("Winning Rate Proportion", col("Wins") / col("TotalGames"))
df_winrate.orderBy("Winning Rate Proportion", ascending = False).show()