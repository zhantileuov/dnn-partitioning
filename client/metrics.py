import atexit
import csv
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from dnn_partition.common.types import RequestMetrics


class MetricsLogger:
    def log(self, metrics: RequestMetrics) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class NullMetricsLogger:
    def log(self, metrics: RequestMetrics) -> None:
        return

    def close(self) -> None:
        return


class CsvMetricsLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RequestMetrics.__annotations__.keys()))
            writer.writeheader()

    def log(self, metrics: RequestMetrics) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RequestMetrics.__annotations__.keys()))
            writer.writerow(asdict(metrics))

    def close(self) -> None:
        return


class CompositeMetricsLogger:
    def __init__(self, loggers: Iterable[MetricsLogger]):
        self.loggers = list(loggers)

    def log(self, metrics: RequestMetrics) -> None:
        for logger in self.loggers:
            logger.log(metrics)

    def close(self) -> None:
        for logger in reversed(self.loggers):
            try:
                logger.close()
            except Exception as exc:
                print(f"[metrics] failed to close logger {logger.__class__.__name__}: {exc}")


class AsyncKafkaMetricsLogger:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        client_id: str = "dnn-partition-client",
        queue_size: int = 1000,
    ):
        try:
            from kafka import KafkaProducer
        except ImportError as exc:
            raise RuntimeError(
                "Kafka logging requires the 'kafka-python' package. Install it with 'pip install kafka-python'."
            ) from exc

        self._producer_cls = KafkaProducer
        self.bootstrap_servers = [part.strip() for part in bootstrap_servers.split(",") if part.strip()]
        if not self.bootstrap_servers:
            raise ValueError("At least one Kafka bootstrap server is required.")

        self.topic = topic
        self.client_id = client_id
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._closed = False
        self._dropped_messages = 0
        self._send_errors = 0
        self._producer = None
        self._worker = threading.Thread(target=self._run, name="kafka-metrics-publisher", daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def _build_producer(self):
        return self._producer_cls(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            linger_ms=100,
            acks=1,
            retries=3,
            max_block_ms=1000,
        )

    def _run(self) -> None:
        try:
            self._producer = self._build_producer()
        except Exception as exc:
            print(f"[metrics] Kafka producer startup failed: {exc}")
            return

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                future = self._producer.send(
                    self.topic,
                    key=payload["request_id"].encode("utf-8"),
                    value=payload,
                )
                future.add_errback(self._on_send_error)
            except Exception as exc:
                self._send_errors += 1
                if self._send_errors in (1, 10) or self._send_errors % 100 == 0:
                    print(f"[metrics] Kafka send failed ({self._send_errors} total): {exc}")
            finally:
                self._queue.task_done()

        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
            except Exception as exc:
                print(f"[metrics] Kafka flush failed: {exc}")

    def _on_send_error(self, exc: BaseException) -> None:
        self._send_errors += 1
        if self._send_errors in (1, 10) or self._send_errors % 100 == 0:
            print(f"[metrics] Kafka delivery error ({self._send_errors} total): {exc}")

    def log(self, metrics: RequestMetrics) -> None:
        if self._closed:
            return

        try:
            self._queue.put_nowait(asdict(metrics))
        except queue.Full:
            self._dropped_messages += 1
            if self._dropped_messages in (1, 10) or self._dropped_messages % 100 == 0:
                print(
                    "[metrics] Kafka queue is full; "
                    f"dropped {self._dropped_messages} metrics so far. Increase DNN_PARTITION_KAFKA_QUEUE_SIZE if needed."
                )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()
        self._worker.join(timeout=5)

        if self._producer is not None:
            try:
                self._producer.close(timeout=5)
            except Exception as exc:
                print(f"[metrics] Kafka producer close failed: {exc}")


def build_metrics_logger(
    sink: str,
    csv_path: Optional[Path],
    kafka_bootstrap_servers: str,
    kafka_topic: str,
    kafka_client_id: str,
    kafka_queue_size: int,
) -> MetricsLogger:
    sink = sink.lower()

    if sink == "none":
        return NullMetricsLogger()

    if sink == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when metrics sink is 'csv'.")
        return CsvMetricsLogger(csv_path)

    if sink == "kafka":
        return AsyncKafkaMetricsLogger(
            bootstrap_servers=kafka_bootstrap_servers,
            topic=kafka_topic,
            client_id=kafka_client_id,
            queue_size=kafka_queue_size,
        )

    if sink == "both":
        if csv_path is None:
            raise ValueError("csv_path is required when metrics sink is 'both'.")
        return CompositeMetricsLogger(
            [
                CsvMetricsLogger(csv_path),
                AsyncKafkaMetricsLogger(
                    bootstrap_servers=kafka_bootstrap_servers,
                    topic=kafka_topic,
                    client_id=kafka_client_id,
                    queue_size=kafka_queue_size,
                ),
            ]
        )

    raise ValueError(f"Unsupported metrics sink: {sink}")
