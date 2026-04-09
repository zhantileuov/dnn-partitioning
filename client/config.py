import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ClientConfig:
    triton_url: str = os.environ.get("DNN_PARTITION_TRITON_URL", "127.0.0.1:8001")
    video_path: str = os.environ.get("DNN_PARTITION_VIDEO_PATH", "assets/videos/input.mp4")
    metrics_csv: str = os.environ.get("DNN_PARTITION_METRICS_CSV", "artifacts/logs/metrics.csv")
    metrics_sink: str = os.environ.get("DNN_PARTITION_METRICS_SINK", "csv")
    kafka_bootstrap_servers: str = os.environ.get("DNN_PARTITION_KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    kafka_topic: str = os.environ.get("DNN_PARTITION_KAFKA_TOPIC", "dnn_partition.metrics")
    kafka_client_id: str = os.environ.get("DNN_PARTITION_KAFKA_CLIENT_ID", "dnn-partition-client")
    kafka_queue_size: int = int(os.environ.get("DNN_PARTITION_KAFKA_QUEUE_SIZE", "1000"))
    mode: str = os.environ.get("DNN_PARTITION_MODE", "full_local")
    model_name: str = os.environ.get("DNN_PARTITION_MODEL", "resnet18")
    partition_point: Optional[str] = os.environ.get("DNN_PARTITION_PARTITION_POINT") or None
    max_requests: Optional[int] = (
        int(os.environ["DNN_PARTITION_MAX_REQUESTS"])
        if os.environ.get("DNN_PARTITION_MAX_REQUESTS")
        else None
    )


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]
