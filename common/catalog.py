from pathlib import Path


DEFAULT_CHECKPOINTS = {
    "resnet18": "resnet18_road_following.pth",
    "resnet50": "resnet50_road_following.pth",
    "mobilenet_v2": "mobilenet_v2_road_following.pth",
    "mobilenet_v3_large": "mobilenet_v3_road_following.pth",
    "mobilenet_v3_small": "mobilenet_v3_small_road_following.pth",
}


def default_checkpoint_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "models"
