from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import monotonically_increasing_id

#The five types of plays are classified by their riskiness.
#It helps to identify the tactics of the AI

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

count_play = df_play.count()

#Recycled part from query 1
#-----------------------------------------------------
#Using the dataset where the model has been trained
#-----------------------------------------------------
CHUNK_SIZE = 500
from pyspark.sql.functions import floor
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
df_moves = df_chunks.withColumn("n_moves",
    1 + sum([
        when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), 1).otherwise(0)
        for i in range(1, len(hand_cols))
    ])
)
rev_hand_cols = ["Hand_7", "Hand_6", "Hand_5", "Hand_4", "Hand_3", "Hand_2", "Hand_1", "Hand_0"]
df_result_train = df_moves.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
df_result_train = df_result_train.withColumn(
    "Result",
    when(col("Final_Hand").isNull(), "Unknown")
    .when(col("Final_Hand")[0] > 21, "Lose")
    .when(col("Final_Hand")[1] > 21, "Win")
    .when(col("Final_Hand")[0] > col("Final_Hand")[1], "Win")
    .when(col("Final_Hand")[0] < col("Final_Hand")[1], "Lose")
    .otherwise("Draw")
)
probabilities = [0.25, 0.5, 0.75]
quartiles = df_result_train.select("n_moves").approxQuantile("n_moves", probabilities, 0.01)
df_risk_train = df_moves.withColumn(
    "Risk Level",
    when(col("n_moves") <= quartiles[0], "Safe")
    .when((col("n_moves") > quartiles[0]) & (col("n_moves") <= quartiles[1]), "Tactical")
    .when((col("n_moves") > quartiles[1]) & (col("n_moves") <= quartiles[2]), "Risky")
    .otherwise("Why would you risk yourself so much?!")
)

#-----------------------------------------------------
#Using the dataset of the model playing against itself
#-----------------------------------------------------
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
from pyspark.sql.functions import expr, when, coalesce
hand_cols = ["Hand_0", "Hand_1", "Hand_2", "Hand_3", "Hand_4", "Hand_5"]
df_moves = df_clean.withColumn("n_moves",
    1 + sum([
        when((col(f"Hand_{i-1}")[0].isNotNull()) & (col(f"Hand_{i-1}")[0] != col(f"Hand_{i-2}")[0]), 1).otherwise(0)
        for i in range(1, len(hand_cols))
    ])
)
rev_hand_cols = ["Hand_5", "Hand_4", "Hand_3", "Hand_2", "Hand_1", "Hand_0"]
df_result_train = df_moves.withColumn("Final_Hand", coalesce(*[col(c) for c in rev_hand_cols]))
df_result_train = df_result_train.withColumn(
    "Result",
    when(col("Final_Hand").isNull(), "Unknown")
    .when(col("Final_Hand")[0] > 21, "Lose")
    .when(col("Final_Hand")[1] > 21, "Win")
    .when(col("Final_Hand")[0] > col("Final_Hand")[1], "Win")
    .when(col("Final_Hand")[0] < col("Final_Hand")[1], "Lose")
    .otherwise("Draw")
)
probabilities = [0.25, 0.5, 0.75]
quartiles = df_result_train.select("n_moves").approxQuantile("n_moves", probabilities, 0.01)
df_risk_play = df_moves.withColumn(
    "Risk Level",
    when(col("n_moves") <= quartiles[0], "Safe")
    .when((col("n_moves") > quartiles[0]) & (col("n_moves") <= quartiles[1]), "Tactical")
    .when((col("n_moves") > quartiles[1]) & (col("n_moves") <= quartiles[2]), "Risky")
    .otherwise("Why would you risk yourself so much?!")
)

#--------------
#   QUERY 2
#--------------

#The types of strategies, (safe, tactical, risky…) and it’s proportion
#It helps to identify which strategy is the common one.
from pyspark.sql.functions import round

df_fin = df_risk_train.select(col("Chunk Number"), col("n_moves"), col("Risk Level")).groupBy("Chunk Number", "Risk Level").count().orderBy("Chunk Number", "count")
df_query2 = df_fin.withColumn("Proportion", round(col("count")*100/CHUNK_SIZE, 2))
df_query2.show()

df_fin = df_risk_play.select(col("n_moves"), col("Risk Level")).groupBy("Risk Level").count().orderBy("count")
df_query2 = df_fin.withColumn("Proportion", round(col("count")*100/count_play, 2))
df_query2.show(truncate = False)