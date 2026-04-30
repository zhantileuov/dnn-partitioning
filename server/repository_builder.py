import argparse
from pathlib import Path
from typing import List, Optional, Union

import torch

from dnn_partition.common.naming import canonical_model_name
from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.common.torch_compat import inference_context
from dnn_partition.server.stress_router.install import install_stress_router_model


class TritonRepositoryBuilder:
    def __init__(self, partition_manager=None):
        self.partition_manager = partition_manager or PartitionManager()

    def export_onnx(self, model: torch.nn.Module, sample: torch.Tensor, out_path: Path, enable_batching: bool = False) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        dynamic_axes = None
        if enable_batching:
            dynamic_axes = {
                "input": {0: "batch"},
                "output": {0: "batch"},
            }
        with inference_context():
            torch.onnx.export(
                model.cpu(),
                sample.cpu(),
                str(out_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

    def write_config(
        self,
        model_dir: Path,
        model_name: str,
        input_shape: tuple,
        output_shape: tuple,
        max_batch_size: int = 0,
    ) -> None:
        if max_batch_size < 0:
            raise ValueError("max_batch_size must be >= 0")
        if max_batch_size > 0:
            input_shape = input_shape[1:]
            output_shape = output_shape[1:]
            batching_config = '''dynamic_batching {
  preferred_batch_size: [2, 4, 8]
  max_queue_delay_microseconds: 1000
}
'''
        else:
            batching_config = ""
        cfg = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [{', '.join(map(str, input_shape))}]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [{', '.join(map(str, output_shape))}]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
{batching_config}
'''
        (model_dir / "config.pbtxt").write_text(cfg, encoding="utf-8")

    def export_full(
        self,
        model_name: str,
        repo_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        max_batch_size: int = 0,
    ) -> Path:
        repo_dir = Path(repo_dir)
        full_model = self.partition_manager.build_full(model_name, checkpoint_path=checkpoint_path, device=device)
        x = torch.zeros((1, 3, 224, 224), device=device)
        with inference_context():
            y = full_model(x)
        triton_name = self.partition_manager.triton_full_name(model_name)
        model_dir = repo_dir / triton_name / "1"
        self.export_onnx(full_model, x, model_dir / "model.onnx", enable_batching=max_batch_size > 0)
        self.write_config(model_dir.parent, triton_name, tuple(x.shape), tuple(y.shape), max_batch_size=max_batch_size)
        return model_dir.parent

    def export_tail(
        self,
        model_name: str,
        partition_point: str,
        repo_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        max_batch_size: int = 0,
    ) -> Path:
        repo_dir = Path(repo_dir)
        prefix = self.partition_manager.build_prefix(model_name, partition_point, checkpoint_path=checkpoint_path, device=device)
        tail = self.partition_manager.build_tail(model_name, partition_point, checkpoint_path=checkpoint_path, device=device)
        x = torch.zeros((1, 3, 224, 224), device=device)
        with inference_context():
            activation = prefix(x)
            y = tail(activation)
        triton_name = self.partition_manager.triton_tail_name(model_name, partition_point)
        model_dir = repo_dir / triton_name / "1"
        self.export_onnx(tail, activation, model_dir / "model.onnx", enable_batching=max_batch_size > 0)
        self.write_config(
            model_dir.parent,
            triton_name,
            tuple(activation.shape),
            tuple(y.shape),
            max_batch_size=max_batch_size,
        )
        return model_dir.parent

    def export_all(
        self,
        model_name: str,
        repo_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        max_batch_size: int = 0,
    ) -> List[Path]:
        exported = [
            self.export_full(
                model_name,
                repo_dir,
                checkpoint_path=checkpoint_path,
                device=device,
                max_batch_size=max_batch_size,
            )
        ]
        for partition_point in self.partition_manager.list_partition_points(model_name):
            exported.append(
                self.export_tail(
                    model_name,
                    partition_point,
                    repo_dir,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    max_batch_size=max_batch_size,
                )
            )
        return exported

    def export_all_models(self, repo_dir: Union[str, Path], device: str = "cpu", max_batch_size: int = 0) -> List[Path]:
        exported = []
        for model_name in self.partition_manager.list_models():
            exported.extend(self.export_all(model_name, repo_dir, device=device, max_batch_size=max_batch_size))
        return exported

    def install_stress_router(self, repo_dir: Union[str, Path]) -> Path:
        return install_stress_router_model(repo_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full and tail ONNX models for Triton.")
    parser.add_argument("--repo-dir", required=True, help="Output Triton model repository directory.")
    parser.add_argument("--model", default="all", help="Model name or 'all'.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Enable Triton/ONNX batching with this max batch size. Default 0 disables batching.",
    )
    parser.add_argument(
        "--include-stress-router",
        action="store_true",
        help="Also install the stress_router Python backend model into the target repository.",
    )
    parser.add_argument(
        "--extra-full-model",
        action="append",
        default=[],
        metavar="MODEL",
        help=(
            "Also export a full-only model into the same repository, e.g. "
            "--extra-full-model resnet50. May be repeated. Useful for a "
            "background RPS target without exporting all of its partitions."
        ),
    )
    args = parser.parse_args()

    builder = TritonRepositoryBuilder()
    if args.model == "all":
        exported = builder.export_all_models(args.repo_dir, device=args.device, max_batch_size=args.max_batch_size)
        primary_models = set(builder.partition_manager.list_models())
    else:
        exported = builder.export_all(args.model, args.repo_dir, device=args.device, max_batch_size=args.max_batch_size)
        primary_models = {canonical_model_name(args.model)}

    extra_full_models = []
    for model_name in args.extra_full_model:
        canonical_name = canonical_model_name(model_name)
        if canonical_name in extra_full_models:
            continue
        extra_full_models.append(canonical_name)

    for model_name in extra_full_models:
        if args.model == "all" or model_name in primary_models:
            continue
        exported.append(builder.export_full(model_name, args.repo_dir, device=args.device, max_batch_size=args.max_batch_size))

    if args.include_stress_router:
        exported.append(builder.install_stress_router(args.repo_dir))
    for path in exported:
        print(path)


if __name__ == "__main__":
    main()
