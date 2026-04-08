import argparse
from pathlib import Path
from typing import List, Optional, Union

import torch

from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.common.torch_compat import inference_context


class TritonRepositoryBuilder:
    def __init__(self, partition_manager=None):
        self.partition_manager = partition_manager or PartitionManager()

    def export_onnx(self, model: torch.nn.Module, sample: torch.Tensor, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
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
                dynamic_axes=None,
            )

    def write_config(self, model_dir: Path, model_name: str, input_shape: tuple, output_shape: tuple) -> None:
        cfg = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
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
'''
        (model_dir / "config.pbtxt").write_text(cfg, encoding="utf-8")

    def export_full(self, model_name: str, repo_dir: Union[str, Path], checkpoint_path: Optional[Union[str, Path]] = None, device: str = "cpu") -> Path:
        repo_dir = Path(repo_dir)
        full_model = self.partition_manager.build_full(model_name, checkpoint_path=checkpoint_path, device=device)
        x = torch.zeros((1, 3, 224, 224), device=device)
        with inference_context():
            y = full_model(x)
        triton_name = self.partition_manager.triton_full_name(model_name)
        model_dir = repo_dir / triton_name / "1"
        self.export_onnx(full_model, x, model_dir / "model.onnx")
        self.write_config(model_dir.parent, triton_name, tuple(x.shape), tuple(y.shape))
        return model_dir.parent

    def export_tail(
        self,
        model_name: str,
        partition_point: str,
        repo_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
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
        self.export_onnx(tail, activation, model_dir / "model.onnx")
        self.write_config(model_dir.parent, triton_name, tuple(activation.shape), tuple(y.shape))
        return model_dir.parent

    def export_all(
        self,
        model_name: str,
        repo_dir: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> List[Path]:
        exported = [self.export_full(model_name, repo_dir, checkpoint_path=checkpoint_path, device=device)]
        for partition_point in self.partition_manager.list_partition_points(model_name):
            exported.append(
                self.export_tail(
                    model_name,
                    partition_point,
                    repo_dir,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
            )
        return exported

    def export_all_models(self, repo_dir: Union[str, Path], device: str = "cpu") -> List[Path]:
        exported = []
        for model_name in self.partition_manager.list_models():
            exported.extend(self.export_all(model_name, repo_dir, device=device))
        return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full and tail ONNX models for Triton.")
    parser.add_argument("--repo-dir", required=True, help="Output Triton model repository directory.")
    parser.add_argument("--model", default="all", help="Model name or 'all'.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    builder = TritonRepositoryBuilder()
    if args.model == "all":
        exported = builder.export_all_models(args.repo_dir, device=args.device)
    else:
        exported = builder.export_all(args.model, args.repo_dir, device=args.device)
    for path in exported:
        print(path)


if __name__ == "__main__":
    main()
