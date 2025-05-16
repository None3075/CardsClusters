from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import monotonically_increasing_id

#The chances of taking a card based on the card of the opponent and the current hand.
#It help to understand how the model is working

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

from pyspark.sql.functions import floor

CHUNK_SIZE = 500
df_clean = df_train.select(
    col("index"),
    col("Shown_cards").alias("Hand_-1"),
    col("Hand 0").alias("Hand_0"),
    col("Hand 1").alias("Hand_1"),
    col("Hand 2").alias("Hand_2"),
    col("Hand 3").alias("Hand_3"),
    col("Hand 4").alias("Hand_4"),
    col("Hand 5").alias("Hand_5"),
    col("Hand 6").alias("Hand_6"),
    col("Hand 7").alias("Hand_7")
)
df_chunks = df_clean.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))

from pyspark.sql.functions import expr, when, coalesce

hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5", "Hand_6", "Hand_7"]
df_moves_train = df_chunks.withColumn("n_hits",
    1 + sum([
        when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), 1).otherwise(0)
        for i in range(1, len(hand_cols))
    ])
)

df_moves_train.show(5)

df_clean = df_play.select(
    col("index"),
    col("Shown_cards").alias("Hand_-1"),
    col("Hand 0").alias("Hand_0"),
    col("Hand 1").alias("Hand_1"),
    col("Hand 2").alias("Hand_2"),
    col("Hand 3").alias("Hand_3"),
    col("Hand 4").alias("Hand_4"),
    col("Hand 5").alias("Hand_5")
)
df_chunks = df_clean.withColumn("Chunk Number", floor(col("index")/CHUNK_SIZE))

from pyspark.sql.functions import expr, when, coalesce

hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5"]
df_moves_play = df_chunks.withColumn("n_hits",
    1 + sum([
        when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), 1).otherwise(0)
        for i in range(1, len(hand_cols))
    ])
)

df_moves_play.show(5)

from pyspark.sql.functions import variance

'''
Variance:
    - It helps you know if for a combination of (my card, opponent's card) the number of hits is constant (low variance, consistent decisions) or changes a lot (high variance, doubtful decisions).

Percentile:
    - You can see if the number of hits is generally low, medium, or high, without depending on the average (which is a restricted function).

Collect Set:
    - It gives you the different number of hits the model has made for a combination (my card, opponent's card). If there are many different options, it's a sign of inconsistency or exploration.
'''

df_moves_train = df_moves_train.withColumn("Agent_1st_Card", col("Hand_-1")[0])
df_moves_train = df_moves_train.withColumn("Opponent_1st_Card", col("Hand_-1")[1])

df_hits_stats = df_moves_train.groupBy("Agent_1st_Card", "Opponent_1st_Card", "Chunk Number") \
    .agg(
        variance("n_hits").alias("Hits_Variance"),
        expr("percentile(n_hits, array(0.25, 0.5, 0.75)) as Hits_Quartiles"),
        expr("collect_set(n_hits) as Unique_Hits")
    )

df_hits_stats.show()

df_moves_play = df_moves_play.withColumn("Agent_1st_Card", col("Hand_-1")[0])
df_moves_play = df_moves_play.withColumn("Opponent_1st_Card", col("Hand_-1")[1])

df_stats_play = df_moves_play.groupBy("Agent_1st_Card", "Opponent_1st_Card") \
    .agg(
        variance("n_hits").alias("Hits_Variance"),
        expr("percentile(n_hits, array(0.25, 0.5, 0.75)) as Hits_Quartiles"),
        expr("collect_set(n_hits) as Unique_Hits")
    )

df_stats_play.show()