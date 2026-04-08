
def canonical_model_name(model_name: str) -> str:
    aliases = {
        "mobilenet_v3": "mobilenet_v3_large",
        "mobilenet_v3_large": "mobilenet_v3_large",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "mobilenet_v2": "mobilenet_v2",
        "resnet18": "resnet18",
        "resnet50": "resnet50",
    }
    try:
        return aliases[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model name: {model_name}") from exc


def sanitize_partition_name(partition_point: str) -> str:
    return partition_point.replace(".", "_")


def triton_full_model_name(model_name: str) -> str:
    return f"{canonical_model_name(model_name)}_full"


def triton_tail_model_name(model_name: str, partition_point: str) -> str:
    return f"{canonical_model_name(model_name)}_tail_after_{sanitize_partition_name(partition_point)}"
