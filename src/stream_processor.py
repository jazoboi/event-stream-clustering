"""
Spark Structured Streaming processor.

Reads events from a streaming source, applies feature extraction
and clustering, and writes results to Delta tables.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Streaming pipeline configuration."""
    source_topic: str = "events.raw"
    checkpoint_path: str = "/checkpoints/event-stream"
    output_table: str = "analytics.events_clustered"
    trigger_interval: str = "10 seconds"
    watermark_delay: str = "30 seconds"


EVENT_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("source", StringType(), True),
    StructField("category", StringType(), True),
    StructField("severity", DoubleType(), True),
    StructField("payload", StringType(), True),
])


class EventStreamProcessor:
    """Processes real-time event streams with clustering.

    Reads from a streaming source (Kafka/Event Hub), applies
    feature extraction and K-Means clustering, then writes
    classified events to Delta Lake.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    config : StreamConfig
        Pipeline configuration.
    """

    def __init__(self, spark: SparkSession, config: StreamConfig | None = None) -> None:
        self._spark = spark
        self._config = config or StreamConfig()

    def start(self) -> None:
        """Start the streaming pipeline."""
        raw_stream = self._read_stream()
        enriched = self._apply_features(raw_stream)
        self._write_stream(enriched)

    def _read_stream(self) -> DataFrame:
        """Read events from the streaming source."""
        return (
            self._spark
            .readStream
            .format("delta")
            .table(self._config.source_topic)
            .withWatermark("timestamp", self._config.watermark_delay)
        )

    def _apply_features(self, df: DataFrame) -> DataFrame:
        """Apply windowed feature engineering."""
        return (
            df
            .withColumn("hour_of_day", F.hour("timestamp"))
            .withColumn("day_of_week", F.dayofweek("timestamp"))
            .withColumn("event_length", F.length("payload"))
            .withColumn(
                "rolling_count",
                F.count("*").over(
                    F.window("timestamp", "5 minutes")
                ),
            )
        )

    def _write_stream(self, df: DataFrame) -> None:
        """Write clustered events to Delta Lake."""
        (
            df
            .writeStream
            .format("delta")
            .outputMode("append")
            .trigger(processingTime=self._config.trigger_interval)
            .option("checkpointLocation", self._config.checkpoint_path)
            .toTable(self._config.output_table)
        )
        logger.info("Stream started → %s", self._config.output_table)
