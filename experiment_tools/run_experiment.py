import argparse
import csv
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib as _toml_loader
except ImportError:
    try:
        import tomli as _toml_loader
    except ImportError:
        _toml_loader = None

try:
    from kafka import KafkaConsumer
except ImportError:  # pragma: no cover
    KafkaConsumer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run timestamp-windowed load experiments and join Kafka metrics.")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config.")
    parser.add_argument("--name", required=True, help="Experiment run name.")
    return parser.parse_args()


def load_toml(path: Path):
    if _toml_loader is None:
        raise RuntimeError("TOML support requires Python 3.11+, tomli, or tomllib.")
    with path.open("rb") as handle:
        try:
            return _toml_loader.load(handle)
        except TypeError:
            handle.seek(0)
            return _toml_loader.load(handle.read().decode("utf-8"))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scenario_load_model_name(model_name: str, mode: str, partition_point: Optional[str]) -> str:
    if mode == "split" and partition_point:
        return "{0}_tail_after_{1}".format(model_name, partition_point.replace(".", "_"))
    if mode == "full_server":
        return "{0}_full".format(model_name)
    return ""


def active_server_model_name(model_name: str, mode: str, partition_point: Optional[str]) -> str:
    if mode == "full_server":
        return "{0}_full".format(model_name)
    if mode == "split" and partition_point:
        return "{0}_tail_after_{1}".format(model_name, partition_point.replace(".", "_"))
    return ""


class LoadProcessHandle:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.ready_event = threading.Event()
        self.ready_ts: Optional[float] = None
        self.output_thread: Optional[threading.Thread] = None


class ExperimentRunner:
    def __init__(self, config: dict, experiment_name: str):
        if KafkaConsumer is None:
            raise RuntimeError("kafka-python is required. Install it with 'pip install kafka-python'.")

        self.config = config
        self.experiment_name = experiment_name
        self.experiment_cfg = config["experiment"]
        self.client_cfg = config["client"]
        self.load_cfg = config["load"]
        self.kafka_cfg = config["kafka"]
        self.output_dir = Path(config["output"]["directory"]) / experiment_name

        bootstrap_servers = [part.strip() for part in self.kafka_cfg["bootstrap_servers"].split(",") if part.strip()]
        self.consumer = KafkaConsumer(
            self.kafka_cfg["client_topic"],
            self.kafka_cfg["server_topic"],
            bootstrap_servers=bootstrap_servers,
            enable_auto_commit=False,
            auto_offset_reset="latest",
            group_id=None,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            consumer_timeout_ms=1000,
            max_poll_records=500,
        )

        self.client_records: List[dict] = []
        self.server_records: List[dict] = []
        self.latest_server_payload: Optional[dict] = None
        self.plan_rows: List[dict] = []
        self.event_rows: List[dict] = []

    def close(self) -> None:
        self.consumer.close()

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_plan()
        self._write_plan()
        print("[experiment] plan contains {0} scenario(s)".format(len(self.plan_rows)))

        for plan_row in self.plan_rows:
            event_row = self._run_scenario(plan_row)
            self.event_rows.append(event_row)
            self._write_events()

        joined_rows = self._build_joined_rows()
        write_csv(self.output_dir / "experiment_joined.csv", joined_rows)
        print("[experiment] wrote:")
        print("  {0}".format(self.output_dir / "experiment_plan.csv"))
        print("  {0}".format(self.output_dir / "experiment_events.csv"))
        print("  {0}".format(self.output_dir / "experiment_joined.csv"))

    def _build_plan(self) -> None:
        scenario_id = 0
        model_name = self.client_cfg["model_name"]
        duration_s = float(self.experiment_cfg["scenario_duration_s"])
        for scenario in self.config["scenarios"]:
            mode = scenario["mode"]
            partition_point = scenario.get("partition_point") or ""
            load_model_name = scenario.get("load_model_name")
            if load_model_name is None or str(load_model_name).strip() == "":
                load_model_name = scenario_load_model_name(model_name, mode, partition_point or None)
            for load_rps in scenario["load_rps_values"]:
                scenario_id += 1
                self.plan_rows.append(
                    {
                        "experiment_id": self.experiment_name,
                        "scenario_id": scenario_id,
                        "mode": mode,
                        "partition_point": partition_point,
                        "load_model_name": load_model_name,
                        "active_server_model_name": active_server_model_name(model_name, mode, partition_point or None),
                        "load_rps_target": float(load_rps),
                        "duration_s": duration_s,
                    }
                )

    def _write_plan(self) -> None:
        write_csv(self.output_dir / "experiment_plan.csv", self.plan_rows)

    def _write_events(self) -> None:
        write_csv(self.output_dir / "experiment_events.csv", self.event_rows)

    def _run_scenario(self, plan_row: dict) -> dict:
        scenario_id = int(plan_row["scenario_id"])
        mode = plan_row["mode"]
        partition_point = plan_row["partition_point"] or None
        load_rps = float(plan_row["load_rps_target"])
        load_model_name = plan_row["load_model_name"]
        duration_s = float(plan_row["duration_s"])

        print("")
        print(
            "[experiment] scenario={0} mode={1} partition={2} load_model={3} load_rps={4:.1f} duration={5:.1f}s".format(
                scenario_id,
                mode,
                partition_point or "full",
                load_model_name or "none",
                load_rps,
                duration_s,
            )
        )

        event = {
            "experiment_id": self.experiment_name,
            "scenario_id": scenario_id,
            "mode": mode,
            "partition_point": partition_point or "",
            "load_model_name": load_model_name,
            "load_rps_target": load_rps,
            "duration_s": duration_s,
            "mode_command_sent_ts": None,
            "mode_confirmed_ts": None,
            "load_command_sent_ts": None,
            "load_confirmed_ts": None,
            "load_stop_ts": None,
            "analysis_start_ts": None,
            "analysis_end_ts": None,
            "status": "running",
        }

        load_proc: Optional[LoadProcessHandle] = None
        try:
            event["mode_command_sent_ts"] = time.time()
            self._send_mode(mode, partition_point)
            event["mode_confirmed_ts"] = self._wait_for_mode_confirmation(mode, partition_point)

            settle_before_s = float(self.experiment_cfg.get("settle_before_s", 0.0))
            if settle_before_s > 0:
                print("[experiment] settling before load for {0:.1f}s".format(settle_before_s))
                self._poll_until(time.time() + settle_before_s)

            if load_rps > 0 and load_model_name:
                event["load_command_sent_ts"] = time.time()
                load_proc = self._start_load(load_rps, load_model_name)
                event["load_confirmed_ts"] = self._wait_for_load_confirmation(load_proc, load_model_name)
                analysis_anchor = event["load_confirmed_ts"]
            else:
                print("[experiment] no external load for this scenario")
                analysis_anchor = time.time()

            settle_after_s = float(self.experiment_cfg.get("settle_after_s", 0.0))
            if settle_after_s > 0:
                print("[experiment] settling after start for {0:.1f}s".format(settle_after_s))
                self._poll_until(time.time() + settle_after_s)

            event["analysis_start_ts"] = max(analysis_anchor, time.time())
            event["analysis_end_ts"] = event["analysis_start_ts"] + duration_s
            print(
                "[experiment] collecting window {0:.1f}s from {1:.3f} to {2:.3f}".format(
                    duration_s,
                    event["analysis_start_ts"],
                    event["analysis_end_ts"],
                )
            )
            self._poll_until(event["analysis_end_ts"])
            event["status"] = "completed"
            return event
        except Exception:
            event["status"] = "failed"
            raise
        finally:
            if load_proc is not None:
                self._stop_load(load_proc)
                event["load_stop_ts"] = time.time()
                cooldown_after_load_s = float(self.experiment_cfg.get("cooldown_after_load_s", 0.0))
                if cooldown_after_load_s > 0:
                    print("[experiment] cooling down after load for {0:.1f}s".format(cooldown_after_load_s))
                    self._poll_until(time.time() + cooldown_after_load_s)
            elif load_rps <= 0:
                event["load_stop_ts"] = event["analysis_end_ts"] or time.time()

    def _send_mode(self, mode: str, partition_point: Optional[str]) -> None:
        payload = {
            "mode": mode,
            "model_name": self.client_cfg["model_name"],
            "partition_point": partition_point,
        }
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(
                json.dumps(payload).encode("utf-8"),
                (self.client_cfg["control_host"], int(self.client_cfg["control_port"])),
            )
        finally:
            sock.close()
        print(
            "[control] sent mode={0} model={1} partition={2} to udp://{3}:{4}".format(
                mode,
                self.client_cfg["model_name"],
                partition_point or "full",
                self.client_cfg["control_host"],
                self.client_cfg["control_port"],
            )
        )

    def _start_load(self, load_rps: float, load_model_name: str) -> LoadProcessHandle:
        command = self.load_cfg["start_command"].format(
            python=sys.executable,
            server_url=self.load_cfg["server_url"],
            model_name=load_model_name,
            load_rps=load_rps,
            workers=self.load_cfg["workers"],
            processes=self.load_cfg["processes"],
        )
        print("[experiment] starting load: {0}".format(command))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        handle = LoadProcessHandle(proc)
        handle.output_thread = threading.Thread(
            target=self._stream_load_output,
            args=(handle,),
            name="experiment-load-output",
            daemon=True,
        )
        handle.output_thread.start()
        time.sleep(1.0)
        if proc.poll() is not None:
            raise RuntimeError("Load command exited immediately with code {0}".format(proc.returncode))
        return handle

    def _stream_load_output(self, handle: LoadProcessHandle) -> None:
        if handle.proc.stdout is None:
            return
        for raw_line in handle.proc.stdout:
            line = raw_line.rstrip()
            if line:
                print(line)
                if not handle.ready_event.is_set() and line.startswith("[bgload-"):
                    handle.ready_ts = time.time()
                    handle.ready_event.set()
        handle.proc.stdout.close()

    def _stop_load(self, handle: LoadProcessHandle) -> None:
        proc = handle.proc
        if proc.poll() is not None:
            return
        print("[experiment] stopping load command")
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5.0)
        if handle.output_thread is not None:
            handle.output_thread.join(timeout=1.0)

    def _step_deadline(self):
        timeout_s = float(self.experiment_cfg.get("step_timeout_s", 0.0))
        if timeout_s <= 0:
            return None
        return time.time() + timeout_s

    def _wait_for_mode_confirmation(self, mode: str, partition_point: Optional[str]) -> float:
        print("[experiment] waiting for mode confirmation")
        stable_needed = int(self.experiment_cfg.get("mode_stable_records", 3))
        fallback_grace_s = float(self.experiment_cfg.get("mode_confirmation_fallback_s", 3.0))
        stable = 0
        saw_client_record = False
        saw_matching_record = False
        deadline = self._step_deadline()
        started_at = time.time()
        while self._before_deadline(deadline):
            for item in self._poll_messages():
                if item["kind"] != "client":
                    continue
                saw_client_record = True
                payload = item["data"]
                if self._client_matches_mode(payload, mode, partition_point):
                    saw_matching_record = True
                    stable += 1
                    if stable >= stable_needed:
                        print("[experiment] mode confirmed")
                        return float(payload.get("timestamp") or time.time())
                else:
                    stable = 0
            if not saw_client_record and time.time() - started_at >= fallback_grace_s:
                print("[experiment] mode confirmation fallback: no fresh client metrics observed")
                return time.time()
            if saw_matching_record and time.time() - started_at >= fallback_grace_s:
                print("[experiment] mode confirmation fallback: observed matching client metrics but not enough stable records")
                return time.time()
        raise RuntimeError("Timed out waiting for client mode confirmation")

    def _wait_for_load_confirmation(self, load_handle: LoadProcessHandle, load_model_name: str) -> float:
        print("[experiment] waiting for load confirmation")
        fallback_grace_s = float(self.experiment_cfg.get("load_confirmation_fallback_s", 3.0))
        deadline = self._step_deadline()
        while self._before_deadline(deadline):
            for item in self._poll_messages():
                if item["kind"] != "server":
                    continue
                payload = item["data"]
                model_metrics = self._find_model_metrics(payload, load_model_name)
                if model_metrics and float(model_metrics.get("success_rps") or 0.0) > 0.0:
                    print("[experiment] load confirmed with success_rps={0:.2f}".format(float(model_metrics["success_rps"])))
                    return float(payload.get("timestamp") or time.time())
            if load_handle.proc.poll() is not None:
                raise RuntimeError("Load command exited unexpectedly with code {0}".format(load_handle.proc.returncode))
            if load_handle.ready_event.is_set():
                ready_ts = load_handle.ready_ts or time.time()
                if time.time() - ready_ts >= fallback_grace_s:
                    print(
                        "[experiment] load confirmed from background client output; "
                        "Kafka server confirmation not observed"
                    )
                    return ready_ts
        raise RuntimeError("Timed out waiting for server load confirmation for {0}".format(load_model_name))

    def _poll_until(self, end_time: float) -> None:
        while time.time() < end_time:
            self._poll_messages()

    def _poll_messages(self) -> List[dict]:
        messages = self.consumer.poll(timeout_ms=1000, max_records=500)
        items = []
        for records in messages.values():
            for record in records:
                payload = record.value
                if record.topic == self.kafka_cfg["client_topic"]:
                    self.client_records.append(payload)
                    items.append({"kind": "client", "data": payload})
                elif record.topic == self.kafka_cfg["server_topic"]:
                    self.server_records.append(payload)
                    self.latest_server_payload = payload
                    items.append({"kind": "server", "data": payload})
        return items

    def _before_deadline(self, deadline) -> bool:
        if deadline is None:
            return True
        return time.time() < deadline

    def _client_matches_mode(self, payload: dict, mode: str, partition_point: Optional[str]) -> bool:
        payload_partition = payload.get("partition_point") or None
        return (
            payload.get("mode") == mode
            and payload_partition == partition_point
            and payload.get("model_name") == self.client_cfg["model_name"]
        )

    def _find_model_metrics(self, server_payload: Optional[dict], model_name: str) -> Optional[dict]:
        if not server_payload or not model_name:
            return None
        for item in server_payload.get("models", []):
            if item.get("model_name") == model_name:
                return item
        return None

    def _latest_server_before(self, timestamp: float, start_ts: float, end_ts: float) -> Optional[dict]:
        candidate = None
        for payload in self.server_records:
            ts = float(payload.get("timestamp") or 0.0)
            if ts < start_ts:
                continue
            if ts > end_ts or ts > timestamp:
                break
            candidate = payload
        return candidate

    def _build_joined_rows(self) -> List[dict]:
        rows = []
        for event in self.event_rows:
            if event.get("status") != "completed":
                continue
            start_ts = float(event["analysis_start_ts"])
            end_ts = float(event["analysis_end_ts"])
            active_model = active_server_model_name(self.client_cfg["model_name"], event["mode"], event["partition_point"] or None)
            load_model = event["load_model_name"]
            for client_payload in self.client_records:
                ts = float(client_payload.get("timestamp") or 0.0)
                if ts < start_ts or ts > end_ts:
                    continue
                server_payload = self._latest_server_before(ts, start_ts, end_ts)
                active_metrics = self._find_model_metrics(server_payload, active_model)
                load_metrics = self._find_model_metrics(server_payload, load_model)
                rows.append(
                    {
                        "experiment_id": event["experiment_id"],
                        "scenario_id": event["scenario_id"],
                        "scenario_mode": event["mode"],
                        "scenario_partition_point": event["partition_point"],
                        "scenario_load_model_name": load_model,
                        "scenario_load_rps_target": event["load_rps_target"],
                        "scenario_analysis_start_ts": start_ts,
                        "scenario_analysis_end_ts": end_ts,
                        "client_timestamp": client_payload.get("timestamp"),
                        "client_request_id": client_payload.get("request_id"),
                        "client_frame_id": client_payload.get("frame_id"),
                        "client_mode": client_payload.get("mode"),
                        "client_model_name": client_payload.get("model_name"),
                        "client_partition_point": client_payload.get("partition_point") or "",
                        "client_processing_time": client_payload.get("client_processing_time"),
                        "client_transfer_time": client_payload.get("transfer_time"),
                        "client_server_processing_time": client_payload.get("server_processing_time"),
                        "client_e2e_latency": client_payload.get("e2e_latency"),
                        "client_loop_latency": client_payload.get("loop_latency"),
                        "client_frame_read_time": client_payload.get("frame_read_time"),
                        "client_post_e2e_overhead": client_payload.get("post_e2e_overhead"),
                        "client_bytes_sent": client_payload.get("bytes_sent"),
                        "client_bytes_received": client_payload.get("bytes_received"),
                        "client_latest_sampled_power_w": client_payload.get("latest_sampled_power_w"),
                        "client_latest_sampled_temp_c": client_payload.get("latest_sampled_temp_c"),
                        "client_latest_power_w": client_payload.get("latest_power_w"),
                        "client_latest_avg_cpu_util": client_payload.get("latest_avg_cpu_util"),
                        "client_latest_avg_gpu_util": client_payload.get("latest_avg_gpu_util"),
                        "client_latest_avg_temp_c": client_payload.get("latest_avg_temp_c"),
                        "server_timestamp": server_payload.get("timestamp") if server_payload else None,
                        "server_id": server_payload.get("server_id") if server_payload else None,
                        "server_gpu_util_percent": self._nested(server_payload, "server", "gpu_util_percent"),
                        "server_gpu_freq_mhz": self._nested(server_payload, "server", "gpu_freq_mhz"),
                        "server_gpu_temp_c": self._nested(server_payload, "server", "gpu_temp_c"),
                        "server_gpu_mem_used_mb": self._nested(server_payload, "server", "gpu_mem_used_mb"),
                        "server_gpu_mem_total_mb": self._nested(server_payload, "server", "gpu_mem_total_mb"),
                        "server_cpu_util_percent": self._nested(server_payload, "server", "cpu_util_percent"),
                        "server_mem_util_percent": self._nested(server_payload, "server", "mem_util_percent"),
                        "server_emc_util_percent": self._nested(server_payload, "server", "emc_util_percent"),
                        "server_emc_freq_mhz": self._nested(server_payload, "server", "emc_freq_mhz"),
                        "server_temperature_c": self._nested(server_payload, "server", "temperature_c"),
                        "server_power_w": self._nested(server_payload, "server", "power_w"),
                        "server_power_avg_w": self._nested(server_payload, "server", "power_avg_w"),
                        "server_power_vdd_in_mw": self._nested(server_payload, "server", "power_vdd_in_mw"),
                        "server_total_rps": self._nested(server_payload, "totals", "total_rps"),
                        "server_total_success_rps": self._nested(server_payload, "totals", "total_success_rps"),
                        "server_total_failure_rps": self._nested(server_payload, "totals", "total_failure_rps"),
                        "server_total_pending_requests": self._nested(server_payload, "totals", "total_pending_requests"),
                        "active_server_model_name": active_model,
                        "active_model_success_rps": active_metrics.get("success_rps") if active_metrics else None,
                        "active_model_failure_rps": active_metrics.get("failure_rps") if active_metrics else None,
                        "active_model_inference_rps": active_metrics.get("inference_rps") if active_metrics else None,
                        "active_model_execution_rps": active_metrics.get("execution_rps") if active_metrics else None,
                        "active_model_pending_requests": active_metrics.get("pending_requests") if active_metrics else None,
                        "active_model_avg_queue_time_ms": active_metrics.get("avg_queue_time_ms") if active_metrics else None,
                        "active_model_avg_compute_input_ms": active_metrics.get("avg_compute_input_ms") if active_metrics else None,
                        "active_model_avg_compute_infer_ms": active_metrics.get("avg_compute_infer_ms") if active_metrics else None,
                        "active_model_avg_compute_output_ms": active_metrics.get("avg_compute_output_ms") if active_metrics else None,
                        "load_model_success_rps": load_metrics.get("success_rps") if load_metrics else None,
                        "load_model_failure_rps": load_metrics.get("failure_rps") if load_metrics else None,
                        "load_model_pending_requests": load_metrics.get("pending_requests") if load_metrics else None,
                        "load_model_avg_queue_time_ms": load_metrics.get("avg_queue_time_ms") if load_metrics else None,
                        "load_model_avg_compute_input_ms": load_metrics.get("avg_compute_input_ms") if load_metrics else None,
                        "load_model_avg_compute_infer_ms": load_metrics.get("avg_compute_infer_ms") if load_metrics else None,
                        "load_model_avg_compute_output_ms": load_metrics.get("avg_compute_output_ms") if load_metrics else None,
                    }
                )
        return rows

    def _nested(self, payload: Optional[dict], section: str, key: str):
        if not payload:
            return None
        return payload.get(section, {}).get(key)


def main() -> None:
    args = parse_args()
    config = load_toml(Path(args.config))
    config["experiment"]["name"] = args.name
    runner = ExperimentRunner(config, args.name)
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
