# Dynamic DNN Partitioning

This project packages client-side inference, Triton server export, dynamic runtime switching, and lightweight load-generation utilities into one self-contained `dnn_partition` folder.

Supported models:

- `resnet18`
- `resnet50`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`

## Structure

- `assets/`: bundled checkpoints and sample videos
- `client/`: runtime, scheduling, logging, and live mode switching
- `common/`: shared model loading and partition logic
- `controller/`: lightweight scripts for remote control and background Triton load
- `server/`: Triton export utilities, compose file, and synthetic server load tools

## ResNet Partitioning

Current ResNet partition layout:

- `stem`
- `layer1.0`
- `layer1.1`
- `layer2.0`
- `layer2.1`
- `layer3.0`
- `layer3.1`
- `layer4.0`
- `layer4.1`
- `head`

Valid split points exclude the terminal `head`.

## Quick Start

Server side:

```bash
python3 -m dnn_partition.server.repository_builder --repo-dir ~/dnn_partition/server/dynamic_model_repo --model resnet18 --device cpu
cd ~/dnn_partition/server
docker compose up -d
```

Client side:

```bash
python3 -m dnn_partition.client.main --mode full_server --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

Controllable client:

```bash
python3 -m dnn_partition.client.main --mode full_local --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --control-host 0.0.0.0 --control-port 5055 --print-every 5
```

Send a live mode switch from another machine:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode split --model resnet18 --partition-point layer2.0
```

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

## Logs

The client writes CSV metrics by default to:

- `dnn_partition/artifacts/logs/metrics.csv`

Kafka publishing can be enabled with `--metrics-sink kafka` or `--metrics-sink both`.

Console progress output is printed every `--print-every N` requests.

## Additional Tools

- `controller/background_client.py`: request-based Triton load generator
- `server/gpu_load.py`: synthetic CPU/GPU load generator on the server machine
- `controller/send_mode.py`: simple UDP mode sender

## Detailed Operations

See [RUNBOOK.md](RUNBOOK.md) for:

- full setup steps
- local Kafka + Grafana setup
- Triton export commands
- client run modes
- dynamic control examples
- background load generation
- Triton stats and logging checks
