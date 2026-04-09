# Runbook

This file contains the detailed operational commands for `dnn_partition`.

## Environment

Use one consistent Python environment for each machine.

Windows Conda base example:

```powershell
conda activate base
python -c "import torch; print(torch.__version__)"
```

Windows `.venv` example:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__)"
```

Linux / Jetson:

```bash
python3 -c "import torch; print(torch.__version__)"
```

## Assets

- Checkpoints live under `dnn_partition/assets/models`
- Sample video lives at `dnn_partition/assets/videos/input.mp4`

## Triton Naming

- Full models: `{model_name}_full`
- Tail models: `{model_name}_tail_after_{partition_point}`

Examples:

- `resnet18_full`
- `resnet18_tail_after_stem`
- `resnet18_tail_after_layer2_0`

## Environment Variables

Client defaults can be overridden with:

- `DNN_PARTITION_TRITON_URL`
- `DNN_PARTITION_VIDEO_PATH`
- `DNN_PARTITION_METRICS_CSV`
- `DNN_PARTITION_METRICS_SINK`
- `DNN_PARTITION_KAFKA_BOOTSTRAP_SERVERS`
- `DNN_PARTITION_KAFKA_TOPIC`
- `DNN_PARTITION_KAFKA_CLIENT_ID`
- `DNN_PARTITION_KAFKA_QUEUE_SIZE`
- `DNN_PARTITION_MODE`
- `DNN_PARTITION_MODEL`
- `DNN_PARTITION_PARTITION_POINT`
- `DNN_PARTITION_MAX_REQUESTS`

PowerShell example:

```powershell
$env:DNN_PARTITION_TRITON_URL = "172.22.231.61:8001"
$env:DNN_PARTITION_MODE = "full_server"
$env:DNN_PARTITION_MODEL = "resnet18"
$env:DNN_PARTITION_METRICS_SINK = "kafka"
$env:DNN_PARTITION_KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
python -m dnn_partition.client.main
```

Linux example:

```bash
export DNN_PARTITION_TRITON_URL=172.22.231.61:8001
export DNN_PARTITION_MODE=full_server
export DNN_PARTITION_MODEL=resnet18
export DNN_PARTITION_METRICS_SINK=kafka
export DNN_PARTITION_KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
python3 -m dnn_partition.client.main
```

TOML config file example:

```powershell
Copy-Item client\client_config.toml.example client\client_config.toml
python -m dnn_partition.client.main --config client\client_config.toml
```

## Local Observability Stack

Run this once on the current machine to start Kafka, Kafka UI, Prometheus, and Grafana:

```powershell
cd observability
docker compose up -d
```

Services:

- Kafka: `localhost:9092`
- Kafka UI: `http://localhost:8080`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Grafana credentials:

- user: `admin`
- password: `admin`

The compose stack creates the Kafka topic `dnn_partition.metrics`.

## Server Setup

Run server-side module commands from the parent directory of `dnn_partition`.

### Export Triton repo

One model:

```bash
python3 -m dnn_partition.server.repository_builder --repo-dir ~/dnn_partition/server/dynamic_model_repo --model resnet18 --device cpu
```

All models:

```bash
python3 -m dnn_partition.server.repository_builder --repo-dir ~/dnn_partition/server/dynamic_model_repo --model all --device cpu
```

### Remove old generated repo

```bash
rm -rf ~/dnn_partition/server/dynamic_model_repo
```

### Start Triton

```bash
cd ~/dnn_partition/server
docker compose down
docker compose up -d
```

### Health checks

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/resnet18_full/ready
curl http://localhost:8000/v2/models/resnet18_tail_after_layer2_0/ready
curl http://localhost:8000/v2/models/resnet18_tail_after_stem/ready
```

### Synthetic server load

GPU:

```bash
python3 -m dnn_partition.server.gpu_load --device cuda --workers 1 --matrix-size 2048 --utilization 0.70
```

CPU:

```bash
python3 -m dnn_partition.server.gpu_load --device cpu --workers 1 --matrix-size 2048 --utilization 0.70
```

Fixed duration:

```bash
python3 -m dnn_partition.server.gpu_load --device cuda --workers 2 --matrix-size 2048 --utilization 0.80 --duration-s 60
```

Stop with `Ctrl+C`.

## Client Setup

### Main modes

- `full_local`
- `full_server`
- `split`

### Full local

```bash
python3 -m dnn_partition.client.main --mode full_local --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

### Full server

```bash
python3 -m dnn_partition.client.main --mode full_server --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

### Split

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

### Limit requests

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --max-requests 20
```

### Force device

GPU:

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --device cuda
```

CPU:

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --device cpu
```

## Dynamic Runtime Control

### Start controllable client

```bash
python3 -m dnn_partition.client.main --mode full_local --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --control-host 0.0.0.0 --control-port 5055 --print-every 5
```

### Send commands

Split:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode split --model resnet18 --partition-point layer2.0
```

Full server:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode full_server --model resnet18
```

Full local:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode full_local --model resnet18
```

## Controller Scripts

### Simple mode sender

From the parent directory:

```bash
python3 dnn_partition/controller/send_mode.py
```

From inside `dnn_partition`:

```bash
python3 controller/send_mode.py
```

### Triton background load

Basic:

```bash
python3 dnn_partition/controller/background_client.py --server-url 172.22.231.61:8001 --model-name resnet18_full --target-rps 10 --workers 2
```

Heavier:

```bash
python3 dnn_partition/controller/background_client.py --server-url 172.22.231.61:8001 --model-name resnet18_full --target-rps 60 --workers 8
```

Heavier model:

```bash
python3 dnn_partition/controller/background_client.py --server-url 172.22.231.61:8001 --model-name resnet50_full --target-rps 20 --workers 4 --input-mode random
```

Multiple processes:

```bash
python3 dnn_partition/controller/background_client.py --server-url 172.22.231.61:8001 --model-name resnet18_full --target-rps 90 --workers 4 --processes 3
```

Stop with `Ctrl+C`.

## Logging

Default client CSV:

- `dnn_partition/artifacts/logs/metrics.csv`

Override:

```bash
python3 -m dnn_partition.client.main --metrics-csv artifacts/logs/my_run.csv
```

Check CSV on Linux:

```bash
ls -l ~/dnn_partition/artifacts/logs/metrics.csv
tail -n 5 ~/dnn_partition/artifacts/logs/metrics.csv
```

Check CSV on Windows:

```powershell
Get-Content dnn_partition\artifacts\logs\metrics.csv -Tail 5
```

Kafka only:

```powershell
python -m dnn_partition.client.main `
  --metrics-sink kafka `
  --kafka-bootstrap-servers 127.0.0.1:9092 `
  --kafka-topic dnn_partition.metrics `
  --mode split `
  --model resnet18 `
  --partition-point layer2.0 `
  --video assets/videos/input.mp4
```

CSV and Kafka together:

```powershell
python -m dnn_partition.client.main `
  --metrics-sink both `
  --metrics-csv artifacts/logs/metrics.csv `
  --kafka-bootstrap-servers 127.0.0.1:9092 `
  --kafka-topic dnn_partition.metrics `
  --mode split `
  --model resnet18 `
  --partition-point layer2.0 `
  --video assets/videos/input.mp4
```

No metrics sink:

```powershell
python -m dnn_partition.client.main --metrics-sink none
```

The Kafka publisher uses a background thread and bounded queue, so metric publishing does not block the inference loop. If the queue fills, the client drops excess metrics and prints a warning instead of slowing inference.

## Triton-side Statistics

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/resnet18_full/stats
curl http://localhost:8000/v2/models/resnet18_tail_after_layer2_0/stats
curl http://localhost:8002/metrics
```

Prometheus in the observability stack scrapes Triton from `host.docker.internal:8002`. If your Triton server runs elsewhere, update `observability/prometheus/prometheus.yml`.

## Split Execution Logic

- `full_server`
  - client preprocesses input
  - Triton runs `resnet18_full`

- `split stem`
  - client runs `stem`
  - Triton runs `resnet18_tail_after_stem`

- `split layer4.0`
  - client runs through `layer4.0`
  - Triton runs `layer4.1` and `head`

- `split layer4.1`
  - client runs through `layer4.1`
  - Triton runs only `head`
