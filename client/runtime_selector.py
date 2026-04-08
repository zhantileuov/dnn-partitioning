from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.common.types import ExecutionPlan

from .scheduler import BaseScheduler


class ClientRuntimeSelector:
    def __init__(self, partition_manager: PartitionManager, scheduler: BaseScheduler):
        self.partition_manager = partition_manager
        self.scheduler = scheduler

    def next_plan(self, frame_id: int, last_metrics=None) -> ExecutionPlan:
        decision = self.scheduler.next_decision(frame_id, last_metrics=last_metrics)
        if decision.mode == "split":
            if not decision.partition_point:
                raise ValueError("Split mode requires a partition point")
            if decision.partition_point not in self.partition_manager.list_partition_points(decision.model_name):
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
