import argparse
from pathlib import Path

from dnn_partition.common.partition_manager import PartitionManager

from .config import ClientConfig, default_client_config_path, default_project_root, load_client_config
from .jetson_telemetry import build_jetson_telemetry_sampler
from .local_executor import LocalExecutor
from .metrics import build_metrics_logger
from .prewarm import build_prewarm_plans, run_prewarm
from .runtime import DynamicPartitionRuntime
from .runtime_selector import ClientRuntimeSelector
from .scheduler import RemoteControlledScheduler, SchedulerDecision, StaticScheduler
from .triton_client import TritonRequestClient
from .video_source import LoopingVideoFrameSource


def parse_args() -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    config_path = Path(bootstrap_args.config) if bootstrap_args.config else default_client_config_path()
    defaults = load_client_config(config_path) if config_path.exists() else ClientConfig()

    parser = argparse.ArgumentParser(description="Run the dynamic DNN partitioning client runtime.")
    parser.add_argument(
        "--config",
        default=str(config_path) if config_path.exists() else None,
        help="Path to a TOML config file.",
    )
    parser.add_argument("--video", default=defaults.video_path, help="Path to the input video.")
    parser.add_argument("--metrics-csv", default=defaults.metrics_csv, help="Path to the output metrics CSV.")
    parser.add_argument(
        "--metrics-sink",
        default=defaults.metrics_sink,
        choices=["csv", "kafka", "both", "none"],
        help="Where to send request metrics.",
    )
    parser.add_argument(
        "--kafka-bootstrap-servers",
        default=defaults.kafka_bootstrap_servers,
        help="Comma-separated Kafka bootstrap servers.",
    )
    parser.add_argument("--kafka-topic", default=defaults.kafka_topic, help="Kafka topic for request metrics.")
    parser.add_argument("--kafka-client-id", default=defaults.kafka_client_id, help="Kafka client id.")
    parser.add_argument(
        "--kafka-queue-size",
        type=int,
        default=defaults.kafka_queue_size,
        help="Maximum number of metrics buffered locally before Kafka send.",
    )
    parser.add_argument("--mode", default=defaults.mode, choices=["full_local", "full_server", "split"])
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--partition-point", default=defaults.partition_point)
    parser.add_argument(
        "--prewarm-mode",
        default=defaults.prewarm_mode,
        choices=["off", "current", "all"],
        help="Warm selected execution paths before the real run without logging warm-up metrics.",
    )
    parser.add_argument("--triton-url", default=defaults.triton_url)
    parser.add_argument("--max-requests", type=int, default=defaults.max_requests)
    parser.add_argument("--device", default=defaults.device, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=defaults.print_every,
        help="Print a short runtime summary every N processed requests. Use 0 to disable.",
    )
    parser.add_argument(
        "--control-host",
        default=defaults.control_host,
        help="Bind address for remote control commands. If set together with --control-port, the client listens for live updates.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=defaults.control_port,
        help="UDP port for remote control commands from another machine.",
    )
    parser.add_argument(
        "--jetson-telemetry-enabled",
        dest="jetson_telemetry_enabled",
        action="store_true",
        default=defaults.jetson_telemetry_enabled,
        help="Enable background Jetson jtop telemetry sampling.",
    )
    parser.add_argument(
        "--disable-jetson-telemetry",
        dest="jetson_telemetry_enabled",
        action="store_false",
        help="Disable background Jetson jtop telemetry sampling.",
    )
    parser.add_argument(
        "--jetson-telemetry-interval-s",
        type=float,
        default=defaults.jetson_telemetry_interval_s,
        help="Background jtop sampling interval in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = default_project_root()
    if args.max_requests is not None and args.max_requests <= 0:
        args.max_requests = None
    if args.control_port is not None and args.control_port <= 0:
        args.control_port = None
    if args.partition_point == "":
        args.partition_point = None
    if args.device == "":
        args.device = None

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.is_absolute():
        metrics_path = root / metrics_path

    metrics_logger = None
    jetson_telemetry = None
    partition_manager = PartitionManager()
    initial_decision = SchedulerDecision(
        mode=args.mode,
        model_name=args.model,
        partition_point=args.partition_point,
    )
    if args.control_host is not None or args.control_port is not None:
        scheduler = RemoteControlledScheduler(
            initial_decision=initial_decision,
            valid_models=partition_manager.list_models(),
            partition_provider=partition_manager.list_partition_points,
            host=args.control_host or "0.0.0.0",
            port=args.control_port or 5055,
        )
        scheduler.start()
    else:
        scheduler = StaticScheduler(
            mode=args.mode,
            model_name=args.model,
            partition_point=args.partition_point,
        )

    try:
        selector = ClientRuntimeSelector(partition_manager, scheduler)
        local_executor = LocalExecutor(partition_manager, device=args.device)
        jetson_telemetry = build_jetson_telemetry_sampler(
            enabled=args.jetson_telemetry_enabled,
            interval_s=args.jetson_telemetry_interval_s,
        )
        triton_client = TritonRequestClient(args.triton_url)
        prewarm_video_source = LoopingVideoFrameSource(video_path)
        try:
            prewarm_video_source.open()
            _, sample_frame = prewarm_video_source.read()
        finally:
            prewarm_video_source.close()
        prewarm_plans = build_prewarm_plans(
            partition_manager=partition_manager,
            mode=args.mode,
            model_name=args.model,
            partition_point=args.partition_point,
            prewarm_mode=args.prewarm_mode,
        )
        run_prewarm(
            local_executor=local_executor,
            triton_client=triton_client,
            plans=prewarm_plans,
            sample_frame=sample_frame,
        )
        metrics_logger = build_metrics_logger(
            sink=args.metrics_sink,
            csv_path=metrics_path,
            kafka_bootstrap_servers=args.kafka_bootstrap_servers,
            kafka_topic=args.kafka_topic,
            kafka_client_id=args.kafka_client_id,
            kafka_queue_size=args.kafka_queue_size,
        )

        runtime = DynamicPartitionRuntime(
            video_source=LoopingVideoFrameSource(video_path),
            selector=selector,
            local_executor=local_executor,
            metrics_logger=metrics_logger,
            jetson_telemetry=jetson_telemetry,
            triton_client=triton_client,
            print_every=args.print_every,
        )
        runtime.run(max_requests=args.max_requests)
    finally:
        if jetson_telemetry is not None:
            jetson_telemetry.close()
        if metrics_logger is not None:
            metrics_logger.close()
        if isinstance(scheduler, RemoteControlledScheduler):
            scheduler.stop()


if __name__ == "__main__":
    main()
