import argparse
import json
import re
import socket
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple
from urllib.request import urlopen


PROM_LINE_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([^\s]+)$')
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Triton + Jetson server metrics to Kafka.")
    parser.add_argument("--triton-metrics-url", default="http://127.0.0.1:8002/metrics")
    parser.add_argument("--kafka-bootstrap-servers", required=True)
    parser.add_argument("--topic", default="dnn_partition.server_metrics")
    parser.add_argument("--server-id", default=socket.gethostname())
    parser.add_argument("--window-s", type=float, default=1.0, help="Sampling/publish interval in seconds.")
    parser.add_argument("--client-id", default="dnn-partition-server-metrics")
    return parser.parse_args()


def fetch_prometheus_metrics(url: str) -> Dict[str, Dict[Tuple[Tuple[str, str], ...], float]]:
    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")

    metrics = defaultdict(dict)
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE_RE.match(line)
        if not match:
            continue
        metric_name, _, labels_blob, value_blob = match.groups()
        labels = {}
        if labels_blob:
            for key, value in LABEL_RE.findall(labels_blob):
                labels[key] = value.replace('\\"', '"')
        try:
            value = float(value_blob)
        except ValueError:
            continue
        metrics[metric_name][tuple(sorted(labels.items()))] = value
    return metrics


def get_metric_value(metrics: dict, metric_name: str, labels: Optional[dict] = None) -> Optional[float]:
    entries = metrics.get(metric_name, {})
    if labels is None:
        if not entries:
            return None
        return sum(entries.values())
    key = tuple(sorted(labels.items()))
    return entries.get(key)


def diff_metric(current: Optional[float], previous: Optional[float], window_s: float) -> float:
    if current is None or previous is None or window_s <= 0:
        return 0.0
    return max(0.0, (current - previous) / window_s)


def avg_duration_ms(current_sum: Optional[float], previous_sum: Optional[float], current_count: Optional[float], previous_count: Optional[float]) -> float:
    if current_sum is None or previous_sum is None or current_count is None or previous_count is None:
        return 0.0
    delta_count = current_count - previous_count
    delta_sum = current_sum - previous_sum
    if delta_count <= 0 or delta_sum < 0:
        return 0.0
    return (delta_sum / delta_count) / 1000.0


def read_jtop_server_metrics(jetson) -> dict:
    stats = getattr(jetson, "stats", {}) or {}
    gpu = getattr(jetson, "gpu", {}) or {}
    temperature = getattr(jetson, "temperature", {}) or {}
    memory = getattr(jetson, "memory", {}) or {}
    power = getattr(jetson, "power", {}) or {}

    gpu_util = _to_float(stats.get("GPU"))
    cpu_util = _avg_cpu_util(stats)
    mem_util = _memory_util_percent(stats)
    temperature_c = _avg_temperature_c(temperature if temperature else stats)
    power_w = _milli_to_watts(stats.get("power cur"))
    power_avg_w = _milli_to_watts(stats.get("power avg"))
    gpu_mem_used_mb, gpu_mem_total_mb = _gpu_memory_from_jtop(memory)
    gpu_freq_mhz = _gpu_frequency_mhz(gpu)
    gpu_temp_c = _named_temperature_c(temperature, "gpu")
    emc_util_percent = _emc_util_percent(memory)
    emc_freq_mhz = _emc_frequency_mhz(memory)
    vdd_in_power_mw = _total_power_mw(power)

    # Triton Prometheus already provides GPU memory bytes, so temperature/power come from jtop.
    return {
        "gpu_util_percent": gpu_util,
        "gpu_freq_mhz": gpu_freq_mhz,
        "gpu_temp_c": gpu_temp_c,
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "gpu_mem_total_mb": gpu_mem_total_mb,
        "cpu_util_percent": cpu_util,
        "mem_util_percent": mem_util,
        "emc_util_percent": emc_util_percent,
        "emc_freq_mhz": emc_freq_mhz,
        "temperature_c": temperature_c,
        "power_w": power_w,
        "power_avg_w": power_avg_w,
        "power_vdd_in_mw": vdd_in_power_mw,
    }


def _avg_cpu_util(stats: dict) -> Optional[float]:
    values = []
    for key, value in stats.items():
        name = str(key).replace(" ", "").upper()
        if not name.startswith("CPU"):
            continue
        suffix = name[3:]
        if suffix and not suffix.isdigit():
            continue
        parsed = _to_float(value)
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None
    return sum(values) / float(len(values))


def _avg_temperature_c(temperature_source: dict) -> Optional[float]:
    values = []
    for value in temperature_source.values():
        if isinstance(value, dict):
            temp = _to_float(value.get("temp"))
        else:
            temp = _to_float(value)
        if temp is not None and temp > -200:
            values.append(temp)
    if not values:
        return None
    return sum(values) / float(len(values))


def _memory_util_percent(stats: dict) -> Optional[float]:
    ram = _to_float(stats.get("RAM"))
    if ram is None or ram <= 0:
        return None
    # On Jetson stats this value may be reported as a ratio in [0, 1].
    return ram * 100.0 if ram <= 1.0 else ram


def _gpu_memory_from_jtop(memory: dict):
    if not isinstance(memory, dict):
        return None, None

    candidates = [
        memory.get("GPU"),
        memory.get("gpu"),
        memory.get("RAM"),
        memory.get("ram"),
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        used_mb = _memory_value_to_mb(candidate.get("used") or candidate.get("use"))
        total_mb = _memory_value_to_mb(candidate.get("tot") or candidate.get("total"))
        if used_mb is not None or total_mb is not None:
            return used_mb, total_mb

    return None, None


def _memory_value_to_mb(value) -> Optional[float]:
    parsed = _to_float(value)
    if parsed is None:
        return None
    # Heuristic: values larger than 64K are likely bytes.
    if parsed > 65536:
        return parsed / (1024.0 * 1024.0)
    # Values larger than 4096 are likely KB.
    if parsed > 4096:
        return parsed / 1024.0
    return parsed


def _gpu_frequency_mhz(gpu: dict) -> Optional[float]:
    if not isinstance(gpu, dict):
        return None

    for gpu_metrics in gpu.values():
        if not isinstance(gpu_metrics, dict):
            continue
        freq = gpu_metrics.get("freq") or {}
        parsed = _to_float(freq.get("cur"))
        if parsed is not None:
            return parsed
    return None


def _named_temperature_c(temperature: dict, name_fragment: str) -> Optional[float]:
    if not isinstance(temperature, dict):
        return None

    name_fragment = name_fragment.lower()
    for sensor_name, sensor_metrics in temperature.items():
        normalized_name = str(sensor_name).lower()
        if name_fragment not in normalized_name:
            continue
        if isinstance(sensor_metrics, dict):
            parsed = _to_float(sensor_metrics.get("temp"))
        else:
            parsed = _to_float(sensor_metrics)
        if parsed is not None and parsed > -200:
            return parsed
    return None


def _emc_util_percent(memory: dict) -> Optional[float]:
    if not isinstance(memory, dict):
        return None

    emc = memory.get("EMC") or memory.get("emc") or {}
    if not isinstance(emc, dict):
        return None
    return _to_float(emc.get("val"))


def _emc_frequency_mhz(memory: dict) -> Optional[float]:
    if not isinstance(memory, dict):
        return None

    emc = memory.get("EMC") or memory.get("emc") or {}
    if not isinstance(emc, dict):
        return None
    return _to_float(emc.get("cur"))


def _total_power_mw(power: dict) -> Optional[float]:
    if not isinstance(power, dict):
        return None

    total = power.get("tot") or {}
    if isinstance(total, dict):
        parsed = _to_float(total.get("power"))
        if parsed is not None:
            return parsed

    rails = power.get("rail") or {}
    if not isinstance(rails, dict):
        return None

    preferred_names = ("VDD_IN", "POM_5V_IN")
    for rail_name in preferred_names:
        rail = rails.get(rail_name)
        if not isinstance(rail, dict):
            continue
        parsed = _to_float(rail.get("power"))
        if parsed is not None:
            return parsed
    return None


def _to_float(value) -> Optional[float]:
    if value in (None, "OFF"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _milli_to_watts(value) -> Optional[float]:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return parsed / 1000.0


def build_payload(
    current_metrics: dict,
    previous_metrics: Optional[dict],
    system_metrics: dict,
    server_id: str,
    window_s: float,
    timestamp: float,
) -> dict:
    model_keys = set()
    for metric_name in (
        "nv_inference_request_success",
        "nv_inference_request_failure",
        "nv_inference_count",
        "nv_inference_exec_count",
        "nv_inference_pending_request_count",
    ):
        for labels_tuple in current_metrics.get(metric_name, {}):
            labels = dict(labels_tuple)
            if "model" in labels:
                model_keys.add((labels.get("model"), labels.get("version", "1")))

    previous_metrics = previous_metrics or {}
    models = []
    total_success_rps = 0.0
    total_failure_rps = 0.0
    total_inference_rps = 0.0
    total_execution_rps = 0.0
    total_pending_requests = 0

    for model_name, version in sorted(model_keys):
        labels = {"model": model_name, "version": version}

        success_now = get_metric_value(current_metrics, "nv_inference_request_success", labels)
        success_prev = get_metric_value(previous_metrics, "nv_inference_request_success", labels)
        failure_now = get_metric_value(current_metrics, "nv_inference_request_failure", labels)
        failure_prev = get_metric_value(previous_metrics, "nv_inference_request_failure", labels)
        inference_now = get_metric_value(current_metrics, "nv_inference_count", labels)
        inference_prev = get_metric_value(previous_metrics, "nv_inference_count", labels)
        execution_now = get_metric_value(current_metrics, "nv_inference_exec_count", labels)
        execution_prev = get_metric_value(previous_metrics, "nv_inference_exec_count", labels)
        pending_now = get_metric_value(current_metrics, "nv_inference_pending_request_count", labels) or 0.0

        success_rps = diff_metric(success_now, success_prev, window_s)
        failure_rps = diff_metric(failure_now, failure_prev, window_s)
        inference_rps = diff_metric(inference_now, inference_prev, window_s)
        execution_rps = diff_metric(execution_now, execution_prev, window_s)

        queue_sum_now = get_metric_value(current_metrics, "nv_inference_queue_duration_us", labels)
        queue_sum_prev = get_metric_value(previous_metrics, "nv_inference_queue_duration_us", labels)
        input_sum_now = get_metric_value(current_metrics, "nv_inference_compute_input_duration_us", labels)
        input_sum_prev = get_metric_value(previous_metrics, "nv_inference_compute_input_duration_us", labels)
        infer_sum_now = get_metric_value(current_metrics, "nv_inference_compute_infer_duration_us", labels)
        infer_sum_prev = get_metric_value(previous_metrics, "nv_inference_compute_infer_duration_us", labels)
        output_sum_now = get_metric_value(current_metrics, "nv_inference_compute_output_duration_us", labels)
        output_sum_prev = get_metric_value(previous_metrics, "nv_inference_compute_output_duration_us", labels)

        models.append(
            {
                "model_name": model_name,
                "model_version": version,
                "success_rps": success_rps,
                "failure_rps": failure_rps,
                "inference_rps": inference_rps,
                "execution_rps": execution_rps,
                "pending_requests": int(round(pending_now)),
                "avg_queue_time_ms": avg_duration_ms(queue_sum_now, queue_sum_prev, success_now, success_prev),
                "avg_compute_input_ms": avg_duration_ms(input_sum_now, input_sum_prev, success_now, success_prev),
                "avg_compute_infer_ms": avg_duration_ms(infer_sum_now, infer_sum_prev, success_now, success_prev),
                "avg_compute_output_ms": avg_duration_ms(output_sum_now, output_sum_prev, success_now, success_prev),
            }
        )

        total_success_rps += success_rps
        total_failure_rps += failure_rps
        total_inference_rps += inference_rps
        total_execution_rps += execution_rps
        total_pending_requests += int(round(pending_now))

    gpu_util_ratio = get_metric_value(current_metrics, "nv_gpu_utilization")
    gpu_mem_used_bytes = get_metric_value(current_metrics, "nv_gpu_memory_used_bytes")
    gpu_mem_total_bytes = get_metric_value(current_metrics, "nv_gpu_memory_total_bytes")
    cpu_util_ratio = get_metric_value(current_metrics, "nv_cpu_utilization")
    cpu_mem_used_bytes = get_metric_value(current_metrics, "nv_cpu_memory_used_bytes")
    cpu_mem_total_bytes = get_metric_value(current_metrics, "nv_cpu_memory_total_bytes")

    mem_util_percent = None
    if cpu_mem_used_bytes is not None and cpu_mem_total_bytes:
        mem_util_percent = (cpu_mem_used_bytes / cpu_mem_total_bytes) * 100.0
    elif system_metrics.get("mem_util_percent") is not None:
        mem_util_percent = system_metrics["mem_util_percent"]

    return {
        "timestamp": timestamp,
        "server_id": server_id,
        "metrics_window_s": window_s,
        "server": {
            "gpu_util_percent": gpu_util_ratio * 100.0 if gpu_util_ratio is not None else system_metrics.get("gpu_util_percent"),
            "gpu_freq_mhz": system_metrics.get("gpu_freq_mhz"),
            "gpu_temp_c": system_metrics.get("gpu_temp_c"),
            "gpu_mem_used_mb": (
                gpu_mem_used_bytes / (1024.0 * 1024.0)
                if gpu_mem_used_bytes is not None
                else system_metrics.get("gpu_mem_used_mb")
            ),
            "gpu_mem_total_mb": (
                gpu_mem_total_bytes / (1024.0 * 1024.0)
                if gpu_mem_total_bytes is not None
                else system_metrics.get("gpu_mem_total_mb")
            ),
            "cpu_util_percent": cpu_util_ratio * 100.0 if cpu_util_ratio is not None else system_metrics.get("cpu_util_percent"),
            "mem_util_percent": mem_util_percent,
            "emc_util_percent": system_metrics.get("emc_util_percent"),
            "emc_freq_mhz": system_metrics.get("emc_freq_mhz"),
            "temperature_c": system_metrics.get("temperature_c"),
            "power_w": system_metrics.get("power_w"),
            "power_avg_w": system_metrics.get("power_avg_w"),
            "power_vdd_in_mw": system_metrics.get("power_vdd_in_mw"),
        },
        "totals": {
            "total_rps": total_success_rps + total_failure_rps,
            "total_success_rps": total_success_rps,
            "total_failure_rps": total_failure_rps,
            "total_pending_requests": total_pending_requests,
        },
        "models": models,
    }


def main() -> None:
    args = parse_args()
    try:
        from kafka import KafkaProducer
    except ImportError as exc:
        raise SystemExit("kafka-python is required. Install it with 'pip install kafka-python'.") from exc

    try:
        from jtop import jtop
    except ImportError as exc:
        raise SystemExit("jetson-stats / jtop is required. Install it with 'pip install jetson-stats'.") from exc

    producer = KafkaProducer(
        bootstrap_servers=[part.strip() for part in args.kafka_bootstrap_servers.split(",") if part.strip()],
        client_id=args.client_id,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        linger_ms=100,
        acks=1,
        retries=3,
        max_block_ms=1000,
    )

    previous_metrics = None
    with jtop() as jetson:
        print("[server-metrics] publishing to topic={0} window={1:.2f}s server_id={2}".format(args.topic, args.window_s, args.server_id))
        try:
            while True:
                window_start = time.time()
                current_metrics = fetch_prometheus_metrics(args.triton_metrics_url)
                system_metrics = read_jtop_server_metrics(jetson)
                payload = build_payload(
                    current_metrics=current_metrics,
                    previous_metrics=previous_metrics,
                    system_metrics=system_metrics,
                    server_id=args.server_id,
                    window_s=args.window_s,
                    timestamp=window_start,
                )
                producer.send(args.topic, key=args.server_id.encode("utf-8"), value=payload)
                previous_metrics = current_metrics
                elapsed = time.time() - window_start
                sleep_time = max(0.0, args.window_s - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("[server-metrics] stopping")
        finally:
            producer.flush(timeout=5)
            producer.close(timeout=5)


if __name__ == "__main__":
    main()
