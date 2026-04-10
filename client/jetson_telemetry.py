import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class JetsonTelemetrySnapshot:
    latest_sampled_power_w: Optional[float] = None
    latest_sampled_temp_c: Optional[float] = None
    latest_power_w: Optional[float] = None
    latest_avg_cpu_util: Optional[float] = None
    latest_avg_gpu_util: Optional[float] = None
    latest_avg_temp_c: Optional[float] = None
    jetson_sample_timestamp: Optional[float] = None


class NullJetsonTelemetry:
    def get_latest(self) -> JetsonTelemetrySnapshot:
        return JetsonTelemetrySnapshot()

    def close(self) -> None:
        return


class JetsonTelemetrySampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = max(0.1, float(interval_s))
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = JetsonTelemetrySnapshot()
        self._thread = threading.Thread(target=self._run, name="jetson-jtop-sampler", daemon=True)
        self._cpu_count = 0
        self._gpu_count = 0
        self._temp_count = 0
        self._cpu_sum = 0.0
        self._gpu_sum = 0.0
        self._temp_sum = 0.0
        self._started = False
        self._warned = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def get_latest(self) -> JetsonTelemetrySnapshot:
        with self._lock:
            return JetsonTelemetrySnapshot(**self._snapshot.__dict__)

    def close(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            from jtop import jtop
        except ImportError:
            self._warn_once("jtop is not installed; Jetson telemetry will stay empty.")
            return

        try:
            with jtop() as jetson:
                while not self._stop_event.is_set():
                    self._update_snapshot(jetson)
                    self._stop_event.wait(self.interval_s)
        except Exception as exc:
            self._warn_once("jtop sampler stopped: {0}".format(exc))

    def _update_snapshot(self, jetson) -> None:
        stats = getattr(jetson, "stats", {}) or {}
        power = getattr(jetson, "power", {}) or {}
        temperature = getattr(jetson, "temperature", {}) or {}

        sampled_power_w = self._extract_total_power_w(power, stats)
        avg_power_w = self._extract_avg_power_w(power, stats)
        sampled_temp_c = self._extract_avg_temperature_c(temperature)
        cpu_util = self._extract_avg_cpu_util(stats)
        gpu_util = self._extract_gpu_util(stats)

        if cpu_util is not None:
            self._cpu_sum += cpu_util
            self._cpu_count += 1
        if gpu_util is not None:
            self._gpu_sum += gpu_util
            self._gpu_count += 1
        if sampled_temp_c is not None:
            self._temp_sum += sampled_temp_c
            self._temp_count += 1

        snapshot = JetsonTelemetrySnapshot(
            latest_sampled_power_w=sampled_power_w,
            latest_sampled_temp_c=sampled_temp_c,
            latest_power_w=avg_power_w if avg_power_w is not None else sampled_power_w,
            latest_avg_cpu_util=self._cpu_sum / self._cpu_count if self._cpu_count else None,
            latest_avg_gpu_util=self._gpu_sum / self._gpu_count if self._gpu_count else None,
            latest_avg_temp_c=self._temp_sum / self._temp_count if self._temp_count else None,
            jetson_sample_timestamp=time.time(),
        )
        with self._lock:
            self._snapshot = snapshot

    def _extract_total_power_w(self, power, stats: dict) -> Optional[float]:
        value = None
        if isinstance(power, dict):
            total = power.get("tot")
            if isinstance(total, dict):
                value = total.get("power")
        elif isinstance(power, (tuple, list)) and power:
            if isinstance(power[0], dict):
                value = power[0].get("cur")
        if value is None:
            value = stats.get("power cur")
        return self._milli_to_base(value)

    def _extract_avg_power_w(self, power, stats: dict) -> Optional[float]:
        value = None
        if isinstance(power, dict):
            total = power.get("tot")
            if isinstance(total, dict):
                value = total.get("avg")
        elif isinstance(power, (tuple, list)) and power:
            if isinstance(power[0], dict):
                value = power[0].get("avg")
        if value is None:
            value = stats.get("power avg")
        return self._milli_to_base(value)

    def _extract_avg_temperature_c(self, temperature) -> Optional[float]:
        temps = []
        if isinstance(temperature, dict):
            for sensor in temperature.values():
                if isinstance(sensor, dict):
                    if sensor.get("online") is False:
                        continue
                    temp = self._to_float(sensor.get("temp"))
                else:
                    temp = self._to_float(sensor)
                if temp is not None and temp > -200:
                    temps.append(temp)
        if not temps:
            return None
        return sum(temps) / float(len(temps))

    def _extract_avg_cpu_util(self, stats: dict) -> Optional[float]:
        cpu_values = []
        for key, value in stats.items():
            normalized = str(key).replace(" ", "").upper()
            if not normalized.startswith("CPU"):
                continue
            suffix = normalized[3:]
            if suffix and not suffix.isdigit():
                continue
            parsed = self._to_float(value)
            if parsed is not None:
                cpu_values.append(parsed)
        if not cpu_values:
            return None
        return sum(cpu_values) / float(len(cpu_values))

    def _extract_gpu_util(self, stats: dict) -> Optional[float]:
        return self._to_float(stats.get("GPU"))

    def _milli_to_base(self, value) -> Optional[float]:
        parsed = self._to_float(value)
        if parsed is None:
            return None
        return parsed / 1000.0

    def _to_float(self, value) -> Optional[float]:
        if value in (None, "OFF"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print("[telemetry] {0}".format(message))


def build_jetson_telemetry_sampler(enabled: bool = True, interval_s: float = 1.0):
    if not enabled:
        return NullJetsonTelemetry()

    sampler = JetsonTelemetrySampler(interval_s=interval_s)
    sampler.start()
    return sampler
