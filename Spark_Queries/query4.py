from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import monotonically_increasing_id

#The distance from 21 at the end of each match.
#It helps to know the winning chance of the current hand

spark = SparkSession.builder \
    .appName("TGVD_GenericQuery") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

path_training = "CardsParquetData/trained_blackjack.parquet"
path_match = "CardsParquetData/played_blackjack.parquet"

df_train = spark.read.parquet(path_training)
df_play = spark.read.parquet(path_match)
df_train = df_train.withColumn("index", (monotonically_increasing_id() + 1))
df_play = df_play.withColumn("index", (monotonically_increasing_id() + 1))

df_train.show(5)
df_play.show(5)

from pyspark.sql.functions import floor, coalesce, when

#We do the chunk part just for the train (to analize the behaviour change through the iterations of the training)
CHUNK_SIZE = 500
df_chunks_train = df_train.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))

rev_hand_cols = ["Hand 7", "Hand 6", "Hand 5", "Hand 4", "Hand 3", "Hand 2", "Hand 1", "Hand 0"]
df_result_train = df_chunks_train.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
df_result_train = df_result_train.withColumn(
    "Result",
    when(col("Final_Hand").isNull(), "Unknown")
    .when(col("Final_Hand")[0] > 21, "Lose")
    .when(col("Final_Hand")[1] > 21, "Win")
    .when(col("Final_Hand")[0] > col("Final_Hand")[1], "Win")
    .when(col("Final_Hand")[0] < col("Final_Hand")[1], "Lose")
    .otherwise("Draw")
).select("Result", "Final_Hand", "Chunk Number")
df_result_train.show(10)

rev_hand_cols = ["Hand 5", "Hand 4", "Hand 3", "Hand 2", "Hand 1", "Hand 0"]
df_result_play = df_play.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
df_result_play = df_result_play.withColumn(
    "Result",
    when(col("Final_Hand").isNull(), "Unknown")
    .when(col("Final_Hand")[0] > 21, "Lose")
    .when(col("Final_Hand")[1] > 21, "Win")
    .when(col("Final_Hand")[0] > col("Final_Hand")[1], "Win")
    .when(col("Final_Hand")[0] < col("Final_Hand")[1], "Lose")
    .otherwise("Draw")
).select("Result", "Final_Hand")
df_result_play.show(10)

#Tal vez podriamos hacer un histograma de la distancia hasta 21 por result
from pyspark.sql.functions import sum

df_train = df_result_train.withColumn("Agent_hand", col("Final_Hand")[0]).withColumn("Distance_from_21", col("Final_Hand")[0] - 21).filter(col("Distance_from_21") <= 0).groupBy("Result", "Agent_hand", "Chunk Number", "Distance_from_21").count()
df_train.show(10)

df_play = df_result_play.withColumn("Agent_hand", col("Final_Hand")[0]).withColumn("Distance_from_21", col("Final_Hand")[0] - 21).filter(col("Distance_from_21") <= 0).groupBy("Result", "Agent_hand", "Distance_from_21").count()
df_play.show(10)

total_df = df_train.groupBy("Agent_hand", "Chunk Number", "Distance_from_21").agg(sum("count").alias("TotalGames"))
wins_df = df_train.filter(col("Result") == "Win").groupBy("Agent_hand", "Chunk Number", "Distance_from_21").agg(sum("count").alias("Wins"))
df_winrate_train = total_df.join(wins_df, on=["Agent_hand", "Chunk Number", "Distance_from_21"], how="left").fillna(0, subset=["Wins"]).withColumn("Winning Rate Proportion", col("Wins") / col("TotalGames")).orderBy("Winning Rate Proportion", ascending = False)
df_winrate_train.show(10)

total_df = df_play.groupBy("Agent_hand", "Distance_from_21").agg(sum("count").alias("TotalGames"))
wins_df = df_play.filter(col("Result") == "Win").groupBy("Agent_hand", "Distance_from_21").agg(sum("count").alias("Wins"))
df_winrate_play = total_df.join(wins_df, on=["Agent_hand", "Distance_from_21"], how="left").fillna(0, subset=["Wins"]).withColumn("Winning Rate Proportion", col("Wins") / col("TotalGames")).orderBy("Winning Rate Proportion", ascending = False)
df_winrate_play.show(1000)