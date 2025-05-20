from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import expr, current_timestamp

spark = SparkSession.builder \
    .appName("KafkaNewsEventsConsumer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Kafka connection parameters
kafka_bootstrap_servers = "172.31.25.179:9094"  # TODO CHANGE IF CHANGE FROM LOCAL TO SERVER
kafka_topic = "news_events"

schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("article_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("category", StringType(), True),
    StructField("location", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("session_id", StringType(), True)
])

streaming_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .option("startingOffsets", "earliest") \
    .load()

parsed_streaming_df = streaming_df \
    .selectExpr("CAST(value AS STRING)") \
    .select(
        from_json(col("value"), schema).alias("data")
    ) \
    .select("data.*")

df_with_timestamp = parsed_streaming_df \
    .withColumn("event_time", to_timestamp("timestamp"))

active_sessions_by_country = df_with_timestamp \
    .filter(df_with_timestamp.event_time >= expr("current_timestamp() - interval 5 minutes")) \
    .withWatermark("event_time", "1 minute") \
    .groupBy(
        window("event_time", "5 minutes", "5 minutes"),
        "location"
    ) \
    .count() \
    .select("window", "location", "count") \
    .orderBy(col("window.end").desc(),col("count").desc())

query = active_sessions_by_country \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .option("numRows", 10) \
    .start()

query.awaitTermination()