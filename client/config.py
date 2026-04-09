import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
    device: Optional[str] = os.environ.get("DNN_PARTITION_DEVICE") or None
    print_every: int = int(os.environ.get("DNN_PARTITION_PRINT_EVERY", "5"))
    control_host: Optional[str] = os.environ.get("DNN_PARTITION_CONTROL_HOST") or None
    control_port: Optional[int] = (
        int(os.environ["DNN_PARTITION_CONTROL_PORT"])
        if os.environ.get("DNN_PARTITION_CONTROL_PORT")
        else None
    )
    max_requests: Optional[int] = (
        int(os.environ["DNN_PARTITION_MAX_REQUESTS"])
        if os.environ.get("DNN_PARTITION_MAX_REQUESTS")
        else None
    )


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_client_config_path() -> Path:
    return default_project_root() / "client" / "client_config.toml"


def load_client_config(path: Path) -> ClientConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    client = raw.get("client", {})
    kafka = raw.get("kafka", {})
    control = raw.get("control", {})

    return ClientConfig(
        triton_url=_pick_str(client.get("triton_url"), ClientConfig.triton_url),
        video_path=_pick_str(client.get("video_path"), ClientConfig.video_path),
        metrics_csv=_pick_str(client.get("metrics_csv"), ClientConfig.metrics_csv),
        metrics_sink=_pick_str(client.get("metrics_sink"), ClientConfig.metrics_sink),
        kafka_bootstrap_servers=_pick_str(
            kafka.get("bootstrap_servers"),
            ClientConfig.kafka_bootstrap_servers,
        ),
        kafka_topic=_pick_str(kafka.get("topic"), ClientConfig.kafka_topic),
        kafka_client_id=_pick_str(kafka.get("client_id"), ClientConfig.kafka_client_id),
        kafka_queue_size=_pick_int(kafka.get("queue_size"), ClientConfig.kafka_queue_size),
        mode=_pick_str(client.get("mode"), ClientConfig.mode),
        model_name=_pick_str(client.get("model"), ClientConfig.model_name),
        partition_point=_pick_optional_str(client.get("partition_point"), ClientConfig.partition_point),
        device=_pick_optional_str(client.get("device"), ClientConfig.device),
        print_every=_pick_int(client.get("print_every"), ClientConfig.print_every),
        control_host=_pick_optional_str(control.get("host"), ClientConfig.control_host),
        control_port=_pick_optional_int(control.get("port"), ClientConfig.control_port),
        max_requests=_pick_optional_int(client.get("max_requests"), ClientConfig.max_requests),
    )


def _pick_str(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value.strip() else fallback


def _pick_int(value: Any, fallback: int) -> int:
    return int(value) if isinstance(value, int) else fallback


def _pick_optional_str(value: Any, fallback: Optional[str]) -> Optional[str]:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value or None
    return fallback


def _pick_optional_int(value: Any, fallback: Optional[int]) -> Optional[int]:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    return fallback
