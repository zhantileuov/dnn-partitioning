import csv
from dataclasses import asdict
from pathlib import Path
from typing import Union

from dnn_partition.common.types import RequestMetrics


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
