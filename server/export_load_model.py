import argparse
from pathlib import Path

import torch

from dnn_partition.common.partition_manager import PartitionManager
from dnn_partition.common.torch_compat import inference_context

from .repository_builder import TritonRepositoryBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one standalone full model into a Triton repository for background load generation."
    )
    parser.add_argument("--repo-dir", required=True, help="Target Triton model repository directory.")
    parser.add_argument("--model", required=True, help="Source architecture, e.g. resnet50.")
    parser.add_argument(
        "--triton-name",
        default=None,
        help="Standalone Triton model name. Default: <model>_load",
    )
    parser.add_argument("--device", default="cpu", help="Torch device used for export.")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="Enable Triton/ONNX batching with this max batch size. Default 0 disables batching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    triton_name = args.triton_name or "{0}_load".format(args.model)

    partition_manager = PartitionManager()
    builder = TritonRepositoryBuilder(partition_manager)

    model = partition_manager.build_full(args.model, device=args.device)
    x = torch.zeros((1, 3, 224, 224), device=args.device)
    with inference_context():
        y = model(x)

    model_dir = repo_dir / triton_name / "1"
    builder.export_onnx(model, x, model_dir / "model.onnx", enable_batching=args.max_batch_size > 0)
    builder.write_config(model_dir.parent, triton_name, tuple(x.shape), tuple(y.shape), max_batch_size=args.max_batch_size)
    print(model_dir.parent)


if __name__ == "__main__":
    main()
