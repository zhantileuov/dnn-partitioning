import time
import uuid

from dnn_partition.common.types import RequestMetrics

from .jetson_telemetry import NullJetsonTelemetry
from .local_executor import LocalExecutor
from .metrics import MetricsLogger
from .runtime_selector import ClientRuntimeSelector
from .triton_client import TritonRequestClient
from .video_source import LoopingVideoFrameSource


class DynamicPartitionRuntime:
    def __init__(
        self,
        video_source: LoopingVideoFrameSource,
        selector: ClientRuntimeSelector,
        local_executor: LocalExecutor,
        metrics_logger: MetricsLogger,
        jetson_telemetry=None,
        triton_client=None,
        print_every: int = 0,
    ):
        self.video_source = video_source
        self.selector = selector
        self.local_executor = local_executor
        self.metrics_logger = metrics_logger
        self.jetson_telemetry = jetson_telemetry or NullJetsonTelemetry()
        self.triton_client = triton_client
        self.print_every = max(0, int(print_every))

    def _print_progress(self, processed: int, metrics: RequestMetrics) -> None:
        client_ms = metrics.client_processing_time * 1000.0
        transfer_ms = metrics.transfer_time * 1000.0
        server_ms = metrics.server_processing_time * 1000.0 if metrics.server_processing_time is not None else None
        e2e_ms = metrics.e2e_latency * 1000.0
        loop_ms = metrics.loop_latency * 1000.0 if metrics.loop_latency is not None else None
        read_ms = metrics.frame_read_time * 1000.0 if metrics.frame_read_time is not None else None
        overhead_ms = metrics.post_e2e_overhead * 1000.0 if metrics.post_e2e_overhead is not None else None
        print(
            "[client] "
            f"processed={processed} "
            f"frame_id={metrics.frame_id} "
            f"mode={metrics.mode} "
            f"model={metrics.model_name} "
            f"partition={metrics.partition_point or 'full'} "
            f"client={client_ms:.2f}ms "
            f"transfer={transfer_ms:.2f}ms "
            f"server={(f'{server_ms:.2f}ms' if server_ms is not None else 'n/a')} "
            f"e2e={e2e_ms:.2f}ms "
            f"loop={(f'{loop_ms:.2f}ms' if loop_ms is not None else 'n/a')} "
            f"read={(f'{read_ms:.2f}ms' if read_ms is not None else 'n/a')} "
            f"overhead={(f'{overhead_ms:.2f}ms' if overhead_ms is not None else 'n/a')} "
            f"power={(f'{metrics.latest_sampled_power_w:.2f}W' if metrics.latest_sampled_power_w is not None else 'n/a')} "
            f"temp={(f'{metrics.latest_sampled_temp_c:.2f}C' if metrics.latest_sampled_temp_c is not None else 'n/a')} "
            f"cpu_avg={(f'{metrics.latest_avg_cpu_util:.2f}%' if metrics.latest_avg_cpu_util is not None else 'n/a')} "
            f"gpu_avg={(f'{metrics.latest_avg_gpu_util:.2f}%' if metrics.latest_avg_gpu_util is not None else 'n/a')}"
        )

    def run(self, max_requests=None) -> None:
        self.video_source.open()
        processed = 0
        last_metrics = None
        try:
            while max_requests is None or processed < max_requests:
                loop_start = time.perf_counter()
                e2e_start = loop_start
                read_start = time.perf_counter()
                frame_id, frame = self.video_source.read()
                frame_read_time = time.perf_counter() - read_start
                plan = self.selector.next_plan(frame_id, last_metrics=last_metrics)
                request_id = str(uuid.uuid4())

                preprocess_start = time.perf_counter()
                x = self.local_executor.preprocess_frame(frame)
                client_processing_time = time.perf_counter() - preprocess_start

                transfer_time = 0.0
                server_processing_time = None
                bytes_sent = 0
                bytes_received = 0

                if plan.mode == "full_local":
                    _ = self.local_executor.run_full(plan.model_name, x).detach().float().cpu().numpy()
                elif plan.mode == "full_server":
                    if self.triton_client is None:
                        raise RuntimeError("Triton client is required for full_server mode")
                    array = x.detach().float().cpu().numpy()
                    _, transfer_time, server_processing_time, bytes_sent, bytes_received = self.triton_client.infer(
                        plan.triton_model_name,
                        array,
                    )
                elif plan.mode == "split":
                    if self.triton_client is None:
                        raise RuntimeError("Triton client is required for split mode")
                    prefix_start = time.perf_counter()
                    activation = self.local_executor.run_prefix(plan.model_name, plan.partition_point, x)
                    client_processing_time += time.perf_counter() - prefix_start
                    array = activation.detach().float().cpu().numpy()
                    _, transfer_time, server_processing_time, bytes_sent, bytes_received = self.triton_client.infer(
                        plan.triton_model_name,
                        array,
                    )
                else:
                    raise ValueError(f"Unsupported mode: {plan.mode}")

                e2e_latency = time.perf_counter() - e2e_start
                telemetry = self.jetson_telemetry.get_latest()
                loop_latency = time.perf_counter() - loop_start
                metrics = RequestMetrics(
                    request_id=request_id,
                    timestamp=time.time(),
                    frame_id=frame_id,
                    model_name=plan.model_name,
                    partition_point=plan.partition_point,
                    mode=plan.mode,
                    client_processing_time=client_processing_time,
                    transfer_time=transfer_time,
                    server_processing_time=server_processing_time,
                    e2e_latency=e2e_latency,
                    loop_latency=loop_latency,
                    frame_read_time=frame_read_time,
                    post_e2e_overhead=loop_latency - e2e_latency,
                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                    latest_sampled_power_w=telemetry.latest_sampled_power_w,
                    latest_sampled_temp_c=telemetry.latest_sampled_temp_c,
                    latest_power_w=telemetry.latest_power_w,
                    latest_avg_cpu_util=telemetry.latest_avg_cpu_util,
                    latest_avg_gpu_util=telemetry.latest_avg_gpu_util,
                    latest_avg_temp_c=telemetry.latest_avg_temp_c,
                    jetson_sample_timestamp=telemetry.jetson_sample_timestamp,
                )
                self.metrics_logger.log(metrics)
                last_metrics = metrics
                processed += 1
                if self.print_every and processed % self.print_every == 0:
                    self._print_progress(processed, metrics)
        finally:
            self.video_source.close()
