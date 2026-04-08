import argparse
from pathlib import Path

from dnn_partition.common.partition_manager import PartitionManager

from .config import ClientConfig, default_project_root
from .local_executor import LocalExecutor
from .metrics import CsvMetricsLogger
from .runtime import DynamicPartitionRuntime
from .runtime_selector import ClientRuntimeSelector
from .scheduler import RemoteControlledScheduler, SchedulerDecision, StaticScheduler
from .triton_client import TritonRequestClient
from .video_source import LoopingVideoFrameSource


def parse_args() -> argparse.Namespace:
    defaults = ClientConfig()
    parser = argparse.ArgumentParser(description="Run the dynamic DNN partitioning client runtime.")
    parser.add_argument("--video", default=defaults.video_path, help="Path to the input video.")
    parser.add_argument("--metrics-csv", default=defaults.metrics_csv, help="Path to the output metrics CSV.")
    parser.add_argument("--mode", default=defaults.mode, choices=["full_local", "full_server", "split"])
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--partition-point", default=defaults.partition_point)
    parser.add_argument("--triton-url", default=defaults.triton_url)
    parser.add_argument("--max-requests", type=int, default=defaults.max_requests)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print a short runtime summary every N processed requests. Use 0 to disable.",
    )
    parser.add_argument(
        "--control-host",
        default=None,
        help="Bind address for remote control commands. If set together with --control-port, the client listens for live updates.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help="UDP port for remote control commands from another machine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = default_project_root()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = root / video_path

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.is_absolute():
        metrics_path = root / metrics_path

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
        metrics_logger = CsvMetricsLogger(metrics_path)
        triton_client = TritonRequestClient(args.triton_url)

        runtime = DynamicPartitionRuntime(
            video_source=LoopingVideoFrameSource(video_path),
            selector=selector,
            local_executor=local_executor,
            metrics_logger=metrics_logger,
            triton_client=triton_client,
            print_every=args.print_every,
        )
        runtime.run(max_requests=args.max_requests)
    finally:
        if isinstance(scheduler, RemoteControlledScheduler):
            scheduler.stop()


if __name__ == "__main__":
    main()
