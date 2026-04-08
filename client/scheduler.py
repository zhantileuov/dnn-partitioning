from dataclasses import dataclass
from itertools import cycle
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
