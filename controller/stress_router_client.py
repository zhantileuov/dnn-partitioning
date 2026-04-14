import argparse
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import tritonclient.grpc as grpcclient
except ImportError:  # pragma: no cover
    grpcclient = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight stress-router Triton load generator.")
    parser.add_argument("--server-url", default="127.0.0.1:8001", help="Triton gRPC endpoint.")
    parser.add_argument("--router-model-name", default="stress_router", help="Wrapper model name.")
    parser.add_argument("--target-model-name", required=True, help="Actual tail model to stress.")
    parser.add_argument("--target-rps", type=float, default=5.0, help="Total target requests per second.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads per process.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Per-request Triton timeout.")
    parser.add_argument("--log-every", type=float, default=5.0, help="Print stats every N seconds.")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional total runtime in seconds.")
    parser.add_argument("--max-requests", type=int, default=None, help="Stop after this many sends per process.")
    parser.add_argument("--processes", type=int, default=1, help="Optional number of OS processes.")
    parser.add_argument("--process-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--spawned-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


@dataclass
class StatsSnapshot:
    sent: int
    completed: int
    failed: int
    in_flight: int
    avg_latency_ms: float


class SharedStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._start = time.perf_counter()
        self._sent = 0
        self._completed = 0
        self._failed = 0
        self._in_flight = 0
        self._latency_sum_s = 0.0

    def mark_sent(self) -> None:
        with self._lock:
            self._sent += 1
            self._in_flight += 1

    def mark_done(self, latency_s: float, failed: bool = False) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if failed:
                self._failed += 1
                return
            self._completed += 1
            self._latency_sum_s += latency_s

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            avg_latency_ms = (self._latency_sum_s / self._completed * 1000.0) if self._completed else 0.0
            return StatsSnapshot(
                sent=self._sent,
                completed=self._completed,
                failed=self._failed,
                in_flight=self._in_flight,
                avg_latency_ms=avg_latency_ms,
            )

    def elapsed_s(self) -> float:
        return max(1e-9, time.perf_counter() - self._start)


class StressRouterWorker(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        server_url: str,
        router_model_name: str,
        target_model_name: str,
        rps: float,
        timeout_s: float,
        stats: SharedStats,
        stop_event: threading.Event,
        max_requests: Optional[int] = None,
    ):
        super().__init__(name=f"stress-router-worker-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self.server_url = server_url
        self.router_model_name = router_model_name
        self.target_model_name = target_model_name
        self.rps = rps
        self.timeout_s = timeout_s
        self.stats = stats
        self.stop_event = stop_event
        self.max_requests = max_requests
        self._sent_local = 0
        self._control_payload = np.array([target_model_name], dtype=object)

    def _callback(self, sent_at: float):
        def complete(result, error):
            latency_s = time.perf_counter() - sent_at
            if error is not None:
                self.stats.mark_done(latency_s, failed=True)
                return
            try:
                status = result.as_numpy("STATUS")
                if status is None or int(status.reshape(-1)[0]) != 0:
                    self.stats.mark_done(latency_s, failed=True)
                    return
            except Exception:
                self.stats.mark_done(latency_s, failed=True)
                return
            self.stats.mark_done(latency_s, failed=False)

        return complete

    def run(self) -> None:
        if grpcclient is None:
            raise RuntimeError("tritonclient.grpc is required to run stress_router_client.py")

        client = grpcclient.InferenceServerClient(url=self.server_url, verbose=False)
        interval_s = (1.0 / self.rps) if self.rps > 0 else 0.0
        next_send = time.perf_counter()

        while not self.stop_event.is_set():
            if self.max_requests is not None and self._sent_local >= self.max_requests:
                break

            now = time.perf_counter()
            if interval_s > 0 and now < next_send:
                time.sleep(min(next_send - now, 0.01))
                continue
            if interval_s > 0:
                next_send += interval_s

            infer_input = grpcclient.InferInput("TARGET_MODEL_NAME", [1], "BYTES")
            infer_input.set_data_from_numpy(self._control_payload)
            output = grpcclient.InferRequestedOutput("STATUS")

            sent_at = time.perf_counter()
            self.stats.mark_sent()
            self._sent_local += 1
            try:
                client.async_infer(
                    model_name=self.router_model_name,
                    inputs=[infer_input],
                    outputs=[output],
                    callback=self._callback(sent_at),
                    client_timeout=self.timeout_s,
                )
            except Exception:
                self.stats.mark_done(time.perf_counter() - sent_at, failed=True)


def stats_loop(stats: SharedStats, stop_event: threading.Event, log_every: float, label: str) -> None:
    last_snapshot = stats.snapshot()
    last_time = time.perf_counter()
    while not stop_event.wait(log_every):
        now = time.perf_counter()
        snapshot = stats.snapshot()
        dt = max(1e-9, now - last_time)
        interval_completed = snapshot.completed - last_snapshot.completed
        interval_failed = snapshot.failed - last_snapshot.failed
        achieved_rps = interval_completed / dt
        overall_rps = snapshot.completed / stats.elapsed_s()
        print(
            f"[{label}] sent={snapshot.sent} completed={snapshot.completed} failed={snapshot.failed} "
            f"in_flight={snapshot.in_flight} achieved_rps={achieved_rps:.2f} overall_rps={overall_rps:.2f} "
            f"avg_latency={snapshot.avg_latency_ms:.2f}ms interval_failed={interval_failed}"
        )
        last_snapshot = snapshot
        last_time = now


def split_total(total: float, count: int) -> list[float]:
    base = total / count
    return [base for _ in range(count)]


def split_integer_total(total: Optional[int], count: int) -> list[Optional[int]]:
    if total is None:
        return [None for _ in range(count)]
    base, extra = divmod(total, count)
    return [base + (1 if index < extra else 0) for index in range(count)]


def run_single_process(args: argparse.Namespace) -> int:
    if grpcclient is None:
        raise SystemExit("tritonclient.grpc is required. Install tritonclient.")
    if args.target_rps <= 0:
        raise SystemExit("--target-rps must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    stats = SharedStats()
    stop_event = threading.Event()
    label = f"stress-router-p{args.process_index}"

    def handle_signal(signum, frame):  # pragma: no cover
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    worker_rps = split_total(args.target_rps, args.workers)
    worker_limits = split_integer_total(args.max_requests, args.workers)
    workers = [
        StressRouterWorker(
            worker_id=index,
            server_url=args.server_url,
            router_model_name=args.router_model_name,
            target_model_name=args.target_model_name,
            rps=worker_rps[index],
            timeout_s=args.timeout_s,
            stats=stats,
            stop_event=stop_event,
            max_requests=worker_limits[index],
        )
        for index in range(args.workers)
    ]

    logger = threading.Thread(
        target=stats_loop,
        args=(stats, stop_event, args.log_every, label),
        name=f"{label}-logger",
        daemon=True,
    )

    print(
        f"[{label}] starting server={args.server_url} router={args.router_model_name} "
        f"target_model={args.target_model_name} target_rps={args.target_rps:.2f} workers={args.workers}"
    )
    logger.start()
    for worker in workers:
        worker.start()

    try:
        if args.duration_s is not None:
            stop_event.wait(args.duration_s)
            stop_event.set()
        for worker in workers:
            worker.join()
    finally:
        stop_event.set()
        logger.join(timeout=args.log_every + 1.0)

    snapshot = stats.snapshot()
    print(
        f"[{label}] finished sent={snapshot.sent} completed={snapshot.completed} failed={snapshot.failed} "
        f"avg_latency={snapshot.avg_latency_ms:.2f}ms overall_rps={snapshot.completed / stats.elapsed_s():.2f}"
    )
    return 0 if snapshot.failed == 0 else 1


def run_multi_process(args: argparse.Namespace) -> int:
    total_processes = args.processes
    per_process_rps = split_total(args.target_rps, total_processes)
    per_process_limits = split_integer_total(args.max_requests, total_processes)
    procs = []
    for index in range(total_processes):
        cmd = [
            sys.executable,
            __file__,
            "--server-url",
            args.server_url,
            "--router-model-name",
            args.router_model_name,
            "--target-model-name",
            args.target_model_name,
            "--target-rps",
            str(per_process_rps[index]),
            "--workers",
            str(args.workers),
            "--timeout-s",
            str(args.timeout_s),
            "--log-every",
            str(args.log_every),
            "--process-index",
            str(index),
            "--spawned-child",
        ]
        if args.duration_s is not None:
            cmd.extend(["--duration-s", str(args.duration_s)])
        if per_process_limits[index] is not None:
            cmd.extend(["--max-requests", str(per_process_limits[index])])
        procs.append(subprocess.Popen(cmd))

    try:
        exit_codes = [proc.wait() for proc in procs]
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        exit_codes = [proc.wait() for proc in procs]
    return 0 if all(code == 0 for code in exit_codes) else 1


def main() -> int:
    args = parse_args()
    if args.processes > 1 and not args.spawned_child:
        return run_multi_process(args)
    return run_single_process(args)


if __name__ == "__main__":
    raise SystemExit(main())
