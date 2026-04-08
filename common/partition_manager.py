from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torchvision

from .catalog import DEFAULT_CHECKPOINTS, default_checkpoint_root
from .naming import canonical_model_name, triton_full_model_name, triton_tail_model_name
from .torch_compat import inference_context
from .types import PartitionSpec


class PartitionManager:
    def __init__(
        self,
        checkpoint_root: Optional[Union[str, Path]] = None,
        checkpoint_map=None,
        categories: Iterable[str] = ("apex",),
    ):
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else default_checkpoint_root()
        self.checkpoint_map = dict(DEFAULT_CHECKPOINTS)
        if checkpoint_map:
            self.checkpoint_map.update(checkpoint_map)
        self.categories = tuple(categories)
        self.output_dim = 2 * len(self.categories)

    def list_models(self) -> List[str]:
        return sorted(self.checkpoint_map.keys())

    def checkpoint_path(self, model_name: str, checkpoint_path: Optional[Union[str, Path]] = None) -> Path:
        if checkpoint_path is not None:
            return Path(checkpoint_path)
        canonical_name = canonical_model_name(model_name)
        return self.checkpoint_root / self.checkpoint_map[canonical_name]

    def create_model(self, model_name: str) -> nn.Module:
        canonical_name = canonical_model_name(model_name)
        if canonical_name == "resnet18":
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.output_dim)
            return model
        if canonical_name == "resnet50":
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.output_dim)
            return model
        if canonical_name == "mobilenet_v2":
            model = torchvision.models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.output_dim)
            return model
        if canonical_name == "mobilenet_v3_large":
            model = torchvision.models.mobilenet_v3_large(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.output_dim)
            return model
        if canonical_name == "mobilenet_v3_small":
            model = torchvision.models.mobilenet_v3_small(pretrained=False)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.output_dim)
            return model
        raise ValueError(f"Unsupported model: {model_name}")

    def load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> nn.Module:
        model = self.create_model(model_name)
        path = self.checkpoint_path(model_name, checkpoint_path)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return model.eval().to(device)

    def get_partition_specs(self, model_name: str, model: Optional[nn.Module] = None) -> List[PartitionSpec]:
        model = model if model is not None else self.create_model(model_name)
        canonical_name = canonical_model_name(model_name)

        if canonical_name in ("resnet18", "resnet50"):
            specs: list[PartitionSpec] = [
                PartitionSpec("conv1", model.conv1),
                PartitionSpec("bn1", model.bn1),
                PartitionSpec("relu", model.relu),
                PartitionSpec("maxpool", model.maxpool),
            ]
            for stage_name in ("layer1", "layer2", "layer3", "layer4"):
                stage = getattr(model, stage_name)
                for block_index, block in enumerate(stage):
                    specs.append(PartitionSpec(f"{stage_name}.{block_index}", block))
            specs.extend(
                [
                    PartitionSpec("avgpool", model.avgpool),
                    PartitionSpec("flatten", nn.Flatten(1)),
                    PartitionSpec("fc", model.fc),
                ]
            )
            return specs

        if canonical_name == "mobilenet_v2":
            specs = [PartitionSpec(f"features.{index}", layer) for index, layer in enumerate(model.features)]
            specs.extend(
                [
                    PartitionSpec("adaptive_avg_pool", nn.AdaptiveAvgPool2d((1, 1))),
                    PartitionSpec("flatten", nn.Flatten(1)),
                ]
            )
            specs.extend(PartitionSpec(f"classifier.{index}", layer) for index, layer in enumerate(model.classifier))
            return specs

        if canonical_name in ("mobilenet_v3_large", "mobilenet_v3_small"):
            specs = [PartitionSpec(f"features.{index}", layer) for index, layer in enumerate(model.features)]
            specs.extend(
                [
                    PartitionSpec("avgpool", model.avgpool),
                    PartitionSpec("flatten", nn.Flatten(1)),
                ]
            )
            specs.extend(PartitionSpec(f"classifier.{index}", layer) for index, layer in enumerate(model.classifier))
            return specs

        raise ValueError(f"Unsupported model for partitioning: {model_name}")

    def list_partition_points(self, model_name: str, include_terminal: bool = False) -> List[str]:
        points = [spec.name for spec in self.get_partition_specs(model_name)]
        if include_terminal:
            return points
        return points[:-1]

    def _split_index(self, model_name: str, partition_point: str, model: nn.Module) -> int:
        names = [spec.name for spec in self.get_partition_specs(model_name, model)]
        if partition_point not in names:
            raise ValueError(f"Unknown partition point {partition_point!r} for model {model_name}")
        return names.index(partition_point) + 1

    def build_full(
        self,
        model_name: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> nn.Module:
        model = self.load_model(model_name, checkpoint_path=checkpoint_path, device=device)
        return nn.Sequential(*[spec.module for spec in self.get_partition_specs(model_name, model)]).eval().to(device)

    def build_prefix(
        self,
        model_name: str,
        partition_point: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> nn.Module:
        model = self.load_model(model_name, checkpoint_path=checkpoint_path, device=device)
        specs = self.get_partition_specs(model_name, model)
        split_index = self._split_index(model_name, partition_point, model)
        return nn.Sequential(*[spec.module for spec in specs[:split_index]]).eval().to(device)

    def build_tail(
        self,
        model_name: str,
        partition_point: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> nn.Module:
        model = self.load_model(model_name, checkpoint_path=checkpoint_path, device=device)
        specs = self.get_partition_specs(model_name, model)
        split_index = self._split_index(model_name, partition_point, model)
        return nn.Sequential(*[spec.module for spec in specs[split_index:]]).eval().to(device)

    def activation_shape(
        self,
        model_name: str,
        partition_point: str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
        input_shape=(1, 3, 224, 224),
    ) -> tuple:
        prefix = self.build_prefix(model_name, partition_point, checkpoint_path=checkpoint_path, device=device)
        x = torch.zeros(input_shape, device=device)
        with inference_context():
            y = prefix(x)
        return tuple(y.shape)

    def triton_full_name(self, model_name: str) -> str:
        return triton_full_model_name(model_name)

    def triton_tail_name(self, model_name: str, partition_point: str) -> str:
        return triton_tail_model_name(model_name, partition_point)
