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

print(df_play.count())
print(df_train.count())

from pyspark.sql.functions import floor
from pyspark.sql.functions import monotonically_increasing_id

df_train = df_train.withColumn("index", (monotonically_increasing_id() + 1))
df_play = df_play.withColumn("index", (monotonically_increasing_id() + 1))

#We do the chunk part just for the train (to analize the behaviour change through the iterations of the training)
CHUNK_SIZE = 500
df_chunks_train = df_train.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))

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
df_result_play = df_play.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
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

df_clean_play = df_result_play.withColumn("1st Hand Card", col("Shown_cards")[0]).select("Final_Hand", "Result", "1st Hand Card")
df_clean_play.show()
df_clean_train = df_result_train.withColumn("1st Hand Card", col("Shown_cards")[0]).select("Chunk Number", "Final_Hand", "Result", "1st Hand Card")
df_clean_train.show()

df_clean_play = df_clean_play.groupBy("1st Hand Card", "Result").count().orderBy(["count", "1st Hand Card"], ascending = False)
df_clean_play.show()

df_clean_train = df_clean_train.groupBy("1st Hand Card", "Result", "Chunk Number").count().orderBy(["Chunk Number", "count", "1st Hand Card"], ascending = False)
df_clean_train.show()

from pyspark.sql.functions import sum, count

total_df = df_clean_train.groupBy("1st Hand Card", "Chunk Number").agg(sum("count").alias("TotalGames"))
wins_df = df_clean_train.filter(col("Result") == "Win").groupBy("1st Hand Card", "Chunk Number").agg(sum("count").alias("Wins"))
df_winrate_train = total_df.join(wins_df, on=["1st Hand Card", "Chunk Number"], how="left").fillna(0, subset=["Wins"]).withColumn("Winning Rate Proportion", col("Wins") / col("TotalGames")).orderBy("Winning Rate Proportion", ascending = False)
df_winrate_train.show(10)

total_df = df_clean_play.groupBy("1st Hand Card").agg(sum("count").alias("TotalGames"))
wins_df = df_clean_play.filter(col("Result") == "Win").groupBy("1st Hand Card").agg(sum("count").alias("Wins"))
df_winrate_play = total_df.join(wins_df, on=["1st Hand Card"], how="left").fillna(0, subset=["Wins"]).withColumn("Winning Rate Proportion", col("Wins") / col("TotalGames")).orderBy("Winning Rate Proportion", ascending = False)
df_winrate_play.show(1000)

#We prepare the data in order to do a collect_set aggregation.
#With this function we will be able to see how many unique move values (n_moves) there are per first hand, indicating variability in the strategy.
from functools import reduce
from pyspark.sql import functions as F

df_moves_train = df_result_train.select(
    col("index"),
    col("Shown_cards").alias("Hand_-1"),
    col("Hand 0").alias("Hand_0"),
    col("Hand 1").alias("Hand_1"),
    col("Hand 2").alias("Hand_2"),
    col("Hand 3").alias("Hand_3"),
    col("Hand 4").alias("Hand_4"),
    col("Hand 5").alias("Hand_5"),
    col("Hand 6").alias("Hand_6"),
    col("Hand 7").alias("Hand_7"),
    col("Chunk Number")
)
hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5", "Hand_6", "Hand_7"]
df_moves_train = df_moves_train.withColumn("n_moves",
    1 + reduce(
        lambda a, b: a + b,
        [
            when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), F.lit(1)).otherwise(F.lit(0))
            for i in range(1, len(hand_cols))
        ]
    )
)

df_moves_play = df_result_play.select(
    col("index"),
    col("Shown_cards").alias("Hand_-1"),
    col("Hand 0").alias("Hand_0"),
    col("Hand 1").alias("Hand_1"),
    col("Hand 2").alias("Hand_2"),
    col("Hand 3").alias("Hand_3"),
    col("Hand 4").alias("Hand_4"),
    col("Hand 5").alias("Hand_5")
)
hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5"]
df_moves_play = df_moves_play.withColumn("n_moves",
    1 + reduce(
        lambda a, b: a + b,
        [
            when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), F.lit(1)).otherwise(F.lit(0))
            for i in range(1, len(hand_cols))
        ]
    )
)

df_moves_train.show()
df_moves_play.show()

#We prepare the data in order to do a collect_set aggregation.
#With this function we will be able to see how many unique move values (n_moves) there are per first hand, indicating variability in the strategy.
from pyspark.sql.functions import count, collect_set, round

df_fin_train = df_moves_train.select("Chunk Number", "Hand_-1", "n_moves") \
    .groupBy("Chunk Number", "Hand_-1") \
    .agg(
        count("*").alias("count"),
        round((count("*") * 100 / CHUNK_SIZE), 2).alias("Proportion"),
        collect_set("n_moves").alias("Unique_Moves")
    ) \
    .orderBy("Chunk Number", "count", ascending = False)

df_fin_train.show(truncate=False)

df_fin_play = df_moves_play.select("Hand_-1", "n_moves") \
    .groupBy("Hand_-1") \
    .agg(
        count("*").alias("count"),
        round((count("*") * 100 / CHUNK_SIZE), 2).alias("Proportion"),
        collect_set("n_moves").alias("Unique_Moves")
    ) \
    .orderBy("count", ascending = False)

df_fin_play.show(truncate = False)