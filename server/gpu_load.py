import argparse
import signal
import threading
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controllable background GPU load with PyTorch.")
    parser.add_argument("--device", default="cuda", help="Torch device to use, e.g. cuda or cuda:0.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads.")
    parser.add_argument("--matrix-size", type=int, default=2048, help="Square matrix size for matmul workload.")
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.7,
        help="Approximate duty cycle between 0.0 and 1.0. Higher means more sustained load.",
    )
    parser.add_argument("--log-every", type=float, default=5.0, help="Print progress every N seconds.")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional total runtime in seconds.")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32", help="Tensor dtype for the load.")
    return parser.parse_args()


class LoadStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._ops = 0

    def add(self, count: int = 1) -> None:
        with self._lock:
            self._ops += count

    def snapshot(self) -> int:
        with self._lock:
            return self._ops


def worker_loop(
    worker_id: int,
    device: torch.device,
    matrix_size: int,
    utilization: float,
    dtype: torch.dtype,
    stop_event: threading.Event,
    stats: LoadStats,
) -> None:
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)

    while not stop_event.is_set():
        cycle_start = time.perf_counter()
        _ = torch.mm(a, b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        busy_s = time.perf_counter() - cycle_start
        stats.add()

        if utilization < 1.0:
            target_cycle_s = busy_s / max(utilization, 1e-3)
            sleep_s = max(0.0, target_cycle_s - busy_s)
            if sleep_s > 0:
                stop_event.wait(sleep_s)


def stats_loop(
    stats: LoadStats,
    stop_event: threading.Event,
    log_every: float,
    matrix_size: int,
    workers: int,
    utilization: float,
) -> None:
    last_ops = stats.snapshot()
    last_time = time.perf_counter()
    while not stop_event.wait(log_every):
        now = time.perf_counter()
        ops = stats.snapshot()
        dt = max(1e-9, now - last_time)
        delta = ops - last_ops
        print(
            "[gpu-load] "
            f"workers={workers} matrix_size={matrix_size} utilization={utilization:.2f} "
            f"matmuls_per_s={delta / dt:.2f} total_matmuls={ops}"
        )
        last_ops = ops
        last_time = now


def main() -> int:
    args = parse_args()
    if not (0.0 < args.utilization <= 1.0):
        raise SystemExit("--utilization must be in the range (0.0, 1.0].")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.matrix_size <= 0:
        raise SystemExit("--matrix-size must be > 0.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    stop_event = threading.Event()
    stats = LoadStats()

    def handle_signal(signum, frame):  # pragma: no cover
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(
        "[gpu-load] starting "
        f"device={device} workers={args.workers} matrix_size={args.matrix_size} "
        f"utilization={args.utilization:.2f} dtype={args.dtype}"
    )

    workers = [
        threading.Thread(
            target=worker_loop,
            args=(index, device, args.matrix_size, args.utilization, dtype, stop_event, stats),
            name=f"gpu-load-worker-{index}",
            daemon=True,
        )
        for index in range(args.workers)
    ]
    logger = threading.Thread(
        target=stats_loop,
        args=(stats, stop_event, args.log_every, args.matrix_size, args.workers, args.utilization),
        name="gpu-load-logger",
        daemon=True,
    )

    logger.start()
    for worker in workers:
        worker.start()

    if args.duration_s is not None:
        stop_event.wait(args.duration_s)
        stop_event.set()

    for worker in workers:
        worker.join()
    logger.join(timeout=args.log_every + 1.0)

    print(f"[gpu-load] stopped total_matmuls={stats.snapshot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
