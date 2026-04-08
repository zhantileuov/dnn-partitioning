from dataclasses import dataclass
from itertools import cycle
import json
import socket
import threading
from typing import Iterable, Optional


@dataclass(frozen=True)
class SchedulerDecision:
    mode: str
    model_name: str
    partition_point: Optional[str]


class BaseScheduler:
    def next_decision(self, frame_id: int, last_metrics=None) -> SchedulerDecision:
        raise NotImplementedError


class StaticScheduler(BaseScheduler):
    def __init__(self, mode: str, model_name: str, partition_point: Optional[str] = None):
        self.decision = SchedulerDecision(mode=mode, model_name=model_name, partition_point=partition_point)

    def next_decision(self, frame_id: int, last_metrics=None) -> SchedulerDecision:
        return self.decision


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, decisions: Iterable[SchedulerDecision]):
        self._choices = cycle(list(decisions))

    def next_decision(self, frame_id: int, last_metrics=None) -> SchedulerDecision:
        return next(self._choices)


class RemoteControlledScheduler(BaseScheduler):
    def __init__(
        self,
        initial_decision: SchedulerDecision,
        valid_models: Iterable[str],
        partition_provider,
        host: str = "0.0.0.0",
        port: int = 5055,
        poll_timeout_s: float = 0.1,
    ):
        self._decision = initial_decision
        self._valid_models = set(valid_models)
        self._partition_provider = partition_provider
        self.host = host
        self.port = port
        self.poll_timeout_s = poll_timeout_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="dnn-partition-remote-control", daemon=True)
        self._thread.start()
        print(f"[control] listening for remote commands on udp://{self.host}:{self.port}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def next_decision(self, frame_id: int, last_metrics=None) -> SchedulerDecision:
        with self._lock:
            return self._decision

    def _validate_decision(self, decision: SchedulerDecision) -> None:
        if decision.mode not in {"full_local", "full_server", "split"}:
            raise ValueError(f"Unsupported mode: {decision.mode}")
        if decision.model_name not in self._valid_models:
            raise ValueError(f"Unsupported model: {decision.model_name}")
        if decision.mode == "split":
            if not decision.partition_point:
                raise ValueError("Split mode requires partition_point")
            valid_partitions = set(self._partition_provider(decision.model_name))
            if decision.partition_point not in valid_partitions:
                raise ValueError(
                    f"Invalid partition point {decision.partition_point!r} for model {decision.model_name}"
                )
        elif decision.partition_point:
            raise ValueError("partition_point must be empty unless mode=split")

    def _apply_message(self, payload: dict) -> None:
        mode = str(payload.get("mode", "")).strip()
        model_name = str(payload.get("model_name") or self._decision.model_name).strip()
        partition_point = payload.get("partition_point")
        if partition_point is not None:
            partition_point = str(partition_point).strip() or None

        decision = SchedulerDecision(mode=mode, model_name=model_name, partition_point=partition_point)
        self._validate_decision(decision)
        with self._lock:
            if decision == self._decision:
                return
            self._decision = decision
        print(
            "[control] updated "
            f"mode={decision.mode} model={decision.model_name} partition={decision.partition_point or 'full'}"
        )

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(self.poll_timeout_s)
        try:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    continue

                try:
                    payload = json.loads(data.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("Payload must be a JSON object")
                    self._apply_message(payload)
                except Exception as exc:
                    print(f"[control] ignored invalid command: {exc}")
        finally:
            sock.close()
