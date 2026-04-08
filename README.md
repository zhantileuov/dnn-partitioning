# Dynamic DNN Partitioning

This package generalizes the older ResNet18-only Triton prototype into a shared experiment framework for:

- `resnet18`
- `resnet50`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`

It uses the exact valid partition boundaries already established for those models:

- ResNet: stem modules plus complete residual blocks
- MobileNet: `features.*`, pooling, flatten, and classifier modules

## Layout

- `assets/`
  - `models/`: bundled `.pth` checkpoints used by `PartitionManager`
  - `videos/`: local sample input videos for client testing
- `common/`
  - `partition_manager.py`: one source of truth for model loading, valid partition points, full/prefix/tail builders, and Triton naming
  - `naming.py`: full/tail Triton model naming helpers
  - `types.py`: shared request/plan dataclasses
- `server/`
  - `repository_builder.py`: exports one full ONNX model and one tail ONNX model per valid partition point
- `client/`
  - `video_source.py`: looping video reader
  - `local_executor.py`: preprocessing + local full/prefix execution
  - `triton_client.py`: Triton gRPC wrapper
  - `scheduler.py`: in-process schedulers
  - `runtime_selector.py`: resolves `(mode, model_name, partition_point)` into executable plans
  - `metrics.py`: CSV logging
  - `runtime.py`: continuous inference loop over video frames

## Triton naming

- Full models: `{model_name}_full`
- Tail models: `{model_name}_tail_after_{partition_point}`
- Partition names are sanitized by replacing `.` with `_`

Examples:

- `resnet18_full`
- `resnet18_tail_after_layer2_0`
- `resnet50_tail_after_layer3_2`
- `mobilenet_v2_tail_after_features_7`
- `mobilenet_v3_small_tail_after_features_5`

## Notes

- The repository builder exports experiment-first ONNX models for Triton.
- `full_server` in the current runtime sends the preprocessed tensor to the Triton full model. If you later need strict raw-frame upload to Triton, add a Python backend or ensemble preprocessing stage on the server side.
- `server_processing_time` is kept in the metrics schema but is `None` unless you add Triton tracing or server-side instrumentation.

## Export example

```powershell
python -m dnn_partition.server.repository_builder --repo-dir dnn_partition\server\dynamic_model_repo --model all --device cpu
```

## Runtime integration sketch

```python
from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.client.local_executor import LocalExecutor
from dnn_partition.client.metrics import CsvMetricsLogger
from dnn_partition.client.runtime import DynamicPartitionRuntime
from dnn_partition.client.runtime_selector import ClientRuntimeSelector
from dnn_partition.client.scheduler import StaticScheduler
from dnn_partition.client.triton_client import TritonRequestClient
from dnn_partition.client.video_source import LoopingVideoFrameSource

pm = PartitionManager()
selector = ClientRuntimeSelector(pm, StaticScheduler(mode="split", model_name="resnet18", partition_point="layer2.0"))
runtime = DynamicPartitionRuntime(
    video_source=LoopingVideoFrameSource("assets/videos/input.mp4"),
    selector=selector,
    local_executor=LocalExecutor(pm),
    metrics_logger=CsvMetricsLogger("artifacts/logs/metrics.csv"),
    triton_client=TritonRequestClient("localhost:8001"),
)
runtime.run(max_requests=100)
```

## Self-contained assets

- Checkpoints now live under `dnn_partition/assets/models`.
- A sample recording is bundled as `dnn_partition/assets/videos/input.mp4`.
- The default client config resolves paths relative to the `dnn_partition` folder, so you can copy just this folder and keep the basic runtime assets together.
