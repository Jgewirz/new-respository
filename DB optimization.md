or a low-latency odds → normalize → detect edge → (paper/live) execute → monitor pipeline, here’s how those tools fit.

Core picks (strong fit)

Redpanda or Kafka (pick one)
Use as your event bus for odds updates, Kalshi ticks/orderbook deltas, opportunities, executions.

Redpanda: easiest “Kafka-compatible” option to run + fast + fewer moving parts.

Kafka: if you already have Kafka ops maturity or need the broader ecosystem.

Polars (over Pandas for the hot path)
Use for fast, columnar transforms: devig, implied probs, joins, feature calc. Polars is typically faster + lower memory than Pandas for this kind of stream batching.

Grafana + Telegraf
Real-time observability: latency, message rates, dropped updates, arb hit-rate, PnL, exposure, error budgets.
Telegraf ships metrics to your time-series backend; Grafana visualizes + alerts.

Apache Flink (only if you truly need streaming joins/state at scale)
Great for stateful stream processing (sliding windows, exactly-once-ish semantics, keyed state).
But it’s heavy—don’t start here unless you’re already drowning in event volume or need complex streaming correlations.

Nice-to-have (fit depends)

Pandas
Keep for notebooks, offline research, backtests, quick debugging. Avoid for your real-time core if latency matters.

Superset
Good for internal analytics on Postgres/ClickHouse: strategy performance, cohorts, “which markets are most mispriced,” etc.
Use when you want a self-serve SQL BI layer.

PowerBI
Similar purpose to Superset but more “enterprise/reporting.” Use if your team lives in Microsoft and wants polished dashboards.

Not a match for the core (but could be adjacent)

MindsDB
Useful if you want “ML in SQL” (forecasting, anomaly detection) without building ML infra.
For arb, it’s usually not the bottleneck—nice later for anomaly/risk models, not day-one.

Recommended stack for your project (lean + fast)

Redpanda + Polars + Postgres + Grafana/Telegraf
(Add Flink later only if you need serious stateful streaming.)

If you tell me your expected throughput (updates/sec) and whether you’re doing orderbook deltas or just ticker/last, I’ll map this into an exact architecture (topics, consumers, storage, and dashboards).