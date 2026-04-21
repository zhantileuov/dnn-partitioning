# Triton Perf Analyzer

Use this folder to stress the Triton server with NVIDIA `perf_analyzer`.

## 1. Export models and start Triton

Run from the parent directory that contains `dnn_partition`:

```bash
cd /home/edp
python3 -m dnn_partition.server.repository_builder \
  --repo-dir /home/edp/dnn_partition/server/dynamic_model_repo \
  --model resnet18 \
  --device cpu

cd /home/edp/dnn_partition/server
docker compose down
docker compose up -d
```

Check Triton is ready:

```bash
curl http://127.0.0.1:8000/v2/health/ready
curl http://127.0.0.1:8000/v2/models/resnet18_full/ready
```

## 2. Run Perf Analyzer

If `perf_analyzer` is installed locally:

```bash
cd /home/edp/dnn_partition
perf_analyzer \
  -m resnet18_full \
  -i grpc \
  -u 127.0.0.1:8001 \
  --input-data random \
  --concurrency-range 1:16:2
```

If it is not installed locally, use the Triton SDK container:

```bash
docker run --rm --net=host nvcr.io/nvidia/tritonserver:26.01-py3-sdk \
  perf_analyzer \
  -m resnet18_full \
  -i grpc \
  -u 127.0.0.1:8001 \
  --input-data random \
  --concurrency-range 1:16:2
```

## 3. Easier wrapper command

From this repo:

```bash
cd /home/edp/dnn_partition
./perf_analyser/run_perf_analyzer.sh
```

Useful overrides:

```bash
MODEL=resnet50_full ./perf_analyser/run_perf_analyzer.sh
MODEL=resnet18_tail_after_layer2_0 ./perf_analyser/run_perf_analyzer.sh
URL=172.22.231.61:8001 ./perf_analyser/run_perf_analyzer.sh
CONCURRENCY_RANGE=1:32:4 ./perf_analyser/run_perf_analyzer.sh
USE_DOCKER=1 ./perf_analyser/run_perf_analyzer.sh
```

## Common model names

Full models:

```text
resnet18_full
resnet50_full
mobilenet_v2_full
mobilenet_v3_large_full
mobilenet_v3_small_full
```

Split tail examples:

```text
resnet18_tail_after_stem
resnet18_tail_after_layer1_0
resnet18_tail_after_layer2_0
resnet18_tail_after_layer3_0
resnet18_tail_after_layer4_0
```

## Watch server metrics while stressing

In another terminal on the Triton server:

```bash
curl http://127.0.0.1:8002/metrics
```

Or publish the server metrics to Kafka:

```bash
cd /home/edp
python3 -m dnn_partition.server.kafka_metrics_publisher \
  --triton-metrics-url http://127.0.0.1:8002/metrics \
  --kafka-bootstrap-servers 172.22.229.75:9092 \
  --topic dnn_partition.server_metrics \
  --server-id jetson-orin-1 \
  --window-s 1.0
```

