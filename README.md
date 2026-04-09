# Dynamic DNN Partitioning

This project packages client-side inference, Triton server export, dynamic runtime switching, and lightweight load-generation utilities into one self-contained `dnn_partition` folder.

Supported models:

- `resnet18`
- `resnet50`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`

## Folder Layout

- `assets/`
  - `models/`: bundled `.pth` checkpoints used by the partition manager
  - `videos/`: local sample input videos for client testing
- `client/`
  - `main.py`: main client entrypoint
  - `local_executor.py`: local preprocessing and prefix/full execution
  - `scheduler.py`: static and remote-controlled mode selection
  - `runtime.py`: per-frame runtime loop and CSV logging
  - `remote_control.py`: CLI sender for live control commands
- `common/`
  - `partition_manager.py`: model loading, partition definitions, prefix/tail/full builders
  - `catalog.py`: checkpoint lookup
  - `naming.py`: Triton model naming helpers
  - `types.py`: shared dataclasses
- `controller/`
  - `send_mode.py`: simple UDP command sender
  - `background_client.py`: standalone Triton background load generator
- `server/`
  - `repository_builder.py`: export full and tail ONNX models for Triton
  - `build_per_model_repos.py`: create one Triton repo per architecture
  - `docker-compose.yml`: Triton compose file
  - `gpu_load.py`: standalone server-side GPU load generator

## Environment

The project works best when the interpreter and terminal both use the same Python environment.

Example Conda base activation on Windows:

```powershell
conda activate base
python -c "import torch; print(torch.__version__)"
```

Example `.venv` activation on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__)"
```

On Linux / Jetson:

```bash
python3 -c "import torch; print(torch.__version__)"
```

## Self-contained Assets

- Checkpoints live under `dnn_partition/assets/models`
- A sample video is bundled at `dnn_partition/assets/videos/input.mp4`
- Client-relative defaults resolve against the `dnn_partition` root

## ResNet Partition Points

Current ResNet partition list:

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

Valid split points exclude the terminal `head`, because no server-side tail exists after it.

So valid split values for `resnet18` are:

- `stem`
- `layer1.0`
- `layer1.1`
- `layer2.0`
- `layer2.1`
- `layer3.0`
- `layer3.1`
- `layer4.0`
- `layer4.1`

## Triton Naming

- Full models: `{model_name}_full`
- Tail models: `{model_name}_tail_after_{partition_point}`
- Partition names are sanitized by replacing `.` with `_`

Examples:

- `resnet18_full`
- `resnet18_tail_after_stem`
- `resnet18_tail_after_layer2_0`
- `mobilenet_v2_tail_after_features_7`

## Environment Variables

Client defaults can be overridden with environment variables:

- `DNN_PARTITION_TRITON_URL`
- `DNN_PARTITION_VIDEO_PATH`
- `DNN_PARTITION_METRICS_CSV`
- `DNN_PARTITION_MODE`
- `DNN_PARTITION_MODEL`
- `DNN_PARTITION_PARTITION_POINT`
- `DNN_PARTITION_MAX_REQUESTS`

Example:

```powershell
$env:DNN_PARTITION_TRITON_URL = "172.22.231.61:8001"
$env:DNN_PARTITION_MODE = "full_server"
$env:DNN_PARTITION_MODEL = "resnet18"
python -m dnn_partition.client.main
```

Linux:

```bash
export DNN_PARTITION_TRITON_URL=172.22.231.61:8001
export DNN_PARTITION_MODE=full_server
export DNN_PARTITION_MODEL=resnet18
python3 -m dnn_partition.client.main
```

## Server Setup

Run commands from the parent directory of `dnn_partition`, not from inside the package folder.

### 1. Export Triton models

Export one model:

```bash
python3 -m dnn_partition.server.repository_builder --repo-dir ~/dnn_partition/server/dynamic_model_repo --model resnet18 --device cpu
```

Export all models:

```bash
python3 -m dnn_partition.server.repository_builder --repo-dir ~/dnn_partition/server/dynamic_model_repo --model all --device cpu
```

### 2. Remove an old generated Triton repo

```bash
rm -rf ~/dnn_partition/server/dynamic_model_repo
```

### 3. Start Triton with Docker Compose

The compose file mounts `./dynamic_model_repo` into the container as `/models`.

```bash
cd ~/dnn_partition/server
docker compose down
docker compose up -d
```

### 4. Check Triton health

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/resnet18_full/ready
curl http://localhost:8000/v2/models/resnet18_tail_after_layer2_0/ready
curl http://localhost:8000/v2/models/resnet18_tail_after_stem/ready
```

### 5. Generate extra server GPU load

```bash
python3 -m dnn_partition.server.gpu_load --device cuda --workers 1 --matrix-size 2048 --utilization 0.70
```

Run for a fixed duration:

```bash
python3 -m dnn_partition.server.gpu_load --device cuda --workers 2 --matrix-size 2048 --utilization 0.80 --duration-s 60
```

## Client Setup

### Main client modes

- `full_local`: client runs the full model locally
- `full_server`: client preprocesses input and Triton runs the full model
- `split`: client runs a prefix and Triton runs the tail

### Start client in full local mode

```bash
python3 -m dnn_partition.client.main --mode full_local --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

### Start client in full server mode

```bash
python3 -m dnn_partition.client.main --mode full_server --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

### Start client in split mode

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --print-every 5
```

### Limit the number of processed frames

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --max-requests 20
```

### Force device

Force GPU:

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --device cuda
```

Force CPU:

```bash
python3 -m dnn_partition.client.main --mode split --model resnet18 --partition-point layer2.0 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --device cpu
```

## Dynamic Runtime Control

The client can start in any mode and then accept remote mode-switch commands from another machine without restarting.

### Start a controllable client

```bash
python3 -m dnn_partition.client.main --mode full_local --model resnet18 --video assets/videos/input.mp4 --triton-url 172.22.231.61:8001 --control-host 0.0.0.0 --control-port 5055 --print-every 5
```

Notes:

- `--control-host 0.0.0.0` means listen on all interfaces on the client machine
- the sender must target the client machine IP, not `0.0.0.0`

### Send control commands from another machine

Switch to split:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode split --model resnet18 --partition-point layer2.0
```

Switch to full server:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode full_server --model resnet18
```

Switch to full local:

```bash
python3 -m dnn_partition.client.remote_control --host 192.168.1.50 --port 5055 --mode full_local --model resnet18
```

Behavior:

- listener runs on a background thread
- current mode changes on the next processed frame
- invalid commands are ignored

## Controller Scripts

### Simple UDP sender

Edit `controller/send_mode.py` and run:

```bash
python3 dnn_partition/controller/send_mode.py
```

or from inside `dnn_partition`:

```bash
python3 controller/send_mode.py
```

### Background Triton load generator

This script creates request-based server load using Triton inference requests instead of generic stress tools.

Basic example:

```bash
python3 dnn_partition/controller/background_client.py --server-url 172.22.231.61:8001 --model-name resnet18_full --target-rps 10 --workers 2
```

Heavier load:

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

Stop it with `Ctrl+C`.

## Logging

The client prints periodic summaries to the console and also writes one CSV row per processed request.

Default CSV path:

- `dnn_partition/artifacts/logs/metrics.csv`

Override it with:

```bash
python3 -m dnn_partition.client.main --metrics-csv artifacts/logs/my_run.csv
```

### What the console log means

Example:

```text
[client] processed=3120 frame_id=3119 mode=split model=resnet18 partition=layer4.1 client=38.44ms transfer=5.58ms server=n/a e2e=45.21ms
```

Fields:

- `client`: local preprocessing plus local model work
- `transfer`: Triton request round-trip time
- `server`: currently `n/a` because explicit Triton-side timing is not yet injected into the CSV
- `e2e`: end-to-end request latency

### Check the CSV

Linux:

```bash
ls -l ~/dnn_partition/artifacts/logs/metrics.csv
tail -n 5 ~/dnn_partition/artifacts/logs/metrics.csv
```

Windows:

```powershell
Get-Content dnn_partition\artifacts\logs\metrics.csv -Tail 5
```

## Triton-side statistics

Check Triton health:

```bash
curl http://localhost:8000/v2/health/ready
```

Check model statistics:

```bash
curl http://localhost:8000/v2/models/resnet18_full/stats
curl http://localhost:8000/v2/models/resnet18_tail_after_layer2_0/stats
```

Check Prometheus metrics:

```bash
curl http://localhost:8002/metrics
```

## How split execution works

Examples:

- `full_server`
  - client preprocesses the frame
  - client sends input tensor to Triton
  - Triton runs `resnet18_full`

- `split stem`
  - client runs `stem`
  - client sends the output activation
  - Triton runs `resnet18_tail_after_stem`

- `split layer4.0`
  - client runs through `layer4.0`
  - Triton runs `layer4.1` and `head`

- `split layer4.1`
  - client runs through `layer4.1`
  - Triton runs only `head`

## Notes

- Use earlier split points like `stem`, `layer1.0`, or `layer2.0` when the client device is weak
- Very late split points like `layer4.1` often reduce transfer time but increase client compute too much
- `full_server` can be faster than `split` if the client becomes the bottleneck
