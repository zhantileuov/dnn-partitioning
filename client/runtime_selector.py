from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.common.types import ExecutionPlan

from .scheduler import BaseScheduler


class ClientRuntimeSelector:
    def __init__(self, partition_manager: PartitionManager, scheduler: BaseScheduler):
        self.partition_manager = partition_manager
        self.scheduler = scheduler
        self._partition_points_by_model = {}

    def next_plan(self, frame_id: int, last_metrics=None) -> ExecutionPlan:
        decision = self.scheduler.next_decision(frame_id, last_metrics=last_metrics)
        if decision.mode == "split":
            if not decision.partition_point:
                raise ValueError("Split mode requires a partition point")
            valid_partition_points = self._partition_points_by_model.get(decision.model_name)
            if valid_partition_points is None:
                valid_partition_points = set(self.partition_manager.list_partition_points(decision.model_name))
                self._partition_points_by_model[decision.model_name] = valid_partition_points
            if decision.partition_point not in valid_partition_points:
                raise ValueError(
                    f"Invalid partition point {decision.partition_point!r} for model {decision.model_name}"
                )
            triton_name = self.partition_manager.triton_tail_name(decision.model_name, decision.partition_point)
        elif decision.mode == "full_server":
            triton_name = self.partition_manager.triton_full_name(decision.model_name)
        elif decision.mode == "full_local":
            triton_name = None
        else:
            raise ValueError(f"Unsupported mode: {decision.mode}")

        return ExecutionPlan(
            mode=decision.mode,
            model_name=decision.model_name,
            partition_point=decision.partition_point,
            triton_model_name=triton_name,
        )
