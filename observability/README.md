# Observability Stack

This folder runs the local observability services for the project:

- Kafka for application metrics events
- Kafka UI for inspecting topics and messages
- Prometheus for scraping Triton metrics
- Grafana for dashboards

## Services

- Kafka broker: `localhost:9092`
- Kafka UI: `http://localhost:8080`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
  - user: `admin`
  - password: `admin`

## Start

```powershell
cd observability
docker compose up -d
```

## Stop

```powershell
cd observability
docker compose down
```

## Kafka Topic

The compose stack creates this topic automatically:

- `dnn_partition.metrics`

## Client Publishing

Install Kafka support in your Python environment if it is not already present:

```powershell
pip install kafka-python
```

Run the client with Kafka only:

```powershell
python -m dnn_partition.client.main `
  --metrics-sink kafka `
  --kafka-bootstrap-servers 127.0.0.1:9092 `
  --kafka-topic dnn_partition.metrics `
  --mode full_local `
  --model resnet18 `
  --video assets/videos/input.mp4
```

Run the client with both CSV and Kafka during migration:

```powershell
python -m dnn_partition.client.main `
  --metrics-sink both `
  --metrics-csv artifacts/logs/metrics.csv `
  --kafka-bootstrap-servers 127.0.0.1:9092 `
  --kafka-topic dnn_partition.metrics `
  --mode full_local `
  --model resnet18 `
  --video assets/videos/input.mp4
```

## Grafana Setup

Prometheus is provisioned automatically as the default data source.

For Kafka event visualization:

1. Open Grafana.
2. Go to `Connections -> Data sources`.
3. Add the Kafka data source plugin.
4. Point it to `localhost:9092`.
5. Choose the topic `dnn_partition.metrics`.

Kafka UI at `http://localhost:8080` is the fastest way to confirm messages are arriving before building dashboards.

## Triton Metrics

Prometheus is already configured to scrape Triton at `host.docker.internal:8002`.

If Triton is running on another host, update `observability/prometheus/prometheus.yml` and change the target under the `triton` job.
