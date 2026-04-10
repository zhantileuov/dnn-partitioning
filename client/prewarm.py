from typing import List, Optional

from dnn_partition.common.types import ExecutionPlan


def build_prewarm_plans(partition_manager, mode: str, model_name: str, partition_point: Optional[str], prewarm_mode: str) -> List[ExecutionPlan]:
    prewarm_mode = (prewarm_mode or "off").lower()
    plans = []

    if prewarm_mode == "off":
        return plans

    if prewarm_mode == "current":
        return [_make_plan(partition_manager, mode, model_name, partition_point)]

    if prewarm_mode == "all":
        plans.append(_make_plan(partition_manager, "full_local", model_name, None))
        plans.append(_make_plan(partition_manager, "full_server", model_name, None))
        for point in partition_manager.list_partition_points(model_name):
            plans.append(_make_plan(partition_manager, "split", model_name, point))
        return plans

    raise ValueError("Unsupported prewarm mode: {0}".format(prewarm_mode))


def run_prewarm(local_executor, triton_client, plans: List[ExecutionPlan], sample_frame) -> None:
    if not plans:
        return

    print("[prewarm] starting warm-up for {0} path(s)".format(len(plans)))
    x = local_executor.preprocess_frame(sample_frame)

    for index, plan in enumerate(plans, 1):
        label = "{0}/{1} mode={2} model={3} partition={4}".format(
            index,
            len(plans),
            plan.mode,
            plan.model_name,
            plan.partition_point or "full",
        )
        print("[prewarm] {0}".format(label))

        if plan.mode == "full_local":
            _ = local_executor.run_full(plan.model_name, x).detach().float().cpu().numpy()
        elif plan.mode == "full_server":
            if triton_client is None:
                raise RuntimeError("Triton client is required for full_server prewarm")
            array = x.detach().float().cpu().numpy()
            triton_client.infer(plan.triton_model_name, array)
        elif plan.mode == "split":
            if triton_client is None:
                raise RuntimeError("Triton client is required for split prewarm")
            activation = local_executor.run_prefix(plan.model_name, plan.partition_point, x)
            array = activation.detach().float().cpu().numpy()
            triton_client.infer(plan.triton_model_name, array)
        else:
            raise ValueError("Unsupported prewarm mode: {0}".format(plan.mode))

    print("[prewarm] complete")


def _make_plan(partition_manager, mode: str, model_name: str, partition_point: Optional[str]) -> ExecutionPlan:
    if mode == "split":
        if not partition_point:
            raise ValueError("Split prewarm requires partition_point")
        triton_model_name = partition_manager.triton_tail_name(model_name, partition_point)
    elif mode == "full_server":
        triton_model_name = partition_manager.triton_full_name(model_name)
    elif mode == "full_local":
        triton_model_name = None
    else:
        raise ValueError("Unsupported prewarm mode: {0}".format(mode))

    return ExecutionPlan(
        mode=mode,
        model_name=model_name,
        partition_point=partition_point,
        triton_model_name=triton_model_name,
    )
