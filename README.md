# Real-Time Event Stream Processing & Clustering

> Spark Structured Streaming pipeline with online K-Means clustering for event categorization and priority routing.

## Role
**Data Engineer / ML Engineer** — Built the streaming pipeline and clustering engine.

## Overview
Real-time event stream processing pipeline using Databricks Structured Streaming. Ingests 12K+ events/second, extracts features in real-time, and applies K-Means clustering for priority-based event categorization.

## Architecture
```
Event Sources → Kafka / Event Hub → Spark Structured Streaming
                                           ↓
                              Feature Extraction (48 features)
                                           ↓
                                  K-Means Clustering (k=5)
                                           ↓
                          Priority Router → Alert Engine → Dashboard
```

## Key Features
- **Real-Time Ingestion** — 12K+ events/second via Structured Streaming
- **Feature Engineering** — 48 streaming features (rolling stats, time windows)
- **Online Clustering** — K-Means with periodic refit on micro-batches
- **Priority Routing** — 5-tier event classification with SLA-aware routing
- **Sub-Second Latency** — P95 processing latency under 2 seconds

## Tech Stack
`Spark Structured Streaming` · `Databricks` · `K-Means` · `PCA` · `Python` · `Delta Lake`

## Impact
- **Sub-2-second P95** end-to-end latency
- Clustering silhouette score of **0.58**
- **94% correct routing** of high-priority events
- Reduced manual triage effort by **65%**

## Project Structure
```
src/
├── stream_processor.py   # Spark Structured Streaming pipeline
├── feature_extractor.py  # Real-time feature engineering
├── clustering_engine.py  # K-Means clustering with online updates
├── alerting.py           # Priority-based alert routing
└── config.py             # Stream & cluster configuration
```

## License
MIT
