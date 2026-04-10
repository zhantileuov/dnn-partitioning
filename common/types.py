from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    module: nn.Module


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str
    model_name: str
    partition_point: Optional[str]
    triton_model_name: Optional[str]


@dataclass
class RequestMetrics:
    request_id: str
    timestamp: float
    frame_id: int
    model_name: str
    partition_point: Optional[str]
    mode: str
    client_processing_time: float
    transfer_time: float
    server_processing_time: Optional[float]
    e2e_latency: float
    loop_latency: Optional[float] = None
    frame_read_time: Optional[float] = None
    post_e2e_overhead: Optional[float] = None
    bytes_sent: int
    bytes_received: int
    latest_sampled_power_w: Optional[float] = None
    latest_sampled_temp_c: Optional[float] = None
    latest_power_w: Optional[float] = None
    latest_avg_cpu_util: Optional[float] = None
    latest_avg_gpu_util: Optional[float] = None
    latest_avg_temp_c: Optional[float] = None
    jetson_sample_timestamp: Optional[float] = None
