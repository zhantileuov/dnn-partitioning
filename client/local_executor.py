from functools import lru_cache
from typing import Optional, Union

import cv2
import numpy as np
import torch

from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.common.torch_compat import inference_context


class LocalExecutor:
    def __init__(self, partition_manager: PartitionManager, device: Optional[Union[str, torch.device]] = None):
        self.partition_manager = partition_manager
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def preprocess_frame(self, frame_bgr: np.ndarray, input_size: int = 224) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).to(self.device).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)

    @lru_cache(maxsize=64)
    def _full_model(self, model_name: str):
        return self.partition_manager.build_full(model_name, device=self.device)

    @lru_cache(maxsize=256)
    def _prefix_model(self, model_name: str, partition_point: str):
        return self.partition_manager.build_prefix(model_name, partition_point, device=self.device)

    def run_full(self, model_name: str, x: torch.Tensor) -> torch.Tensor:
        with inference_context():
            y = self._full_model(model_name)(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return y

    def run_prefix(self, model_name: str, partition_point: str, x: torch.Tensor) -> torch.Tensor:
        with inference_context():
            y = self._prefix_model(model_name, partition_point)(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return y
