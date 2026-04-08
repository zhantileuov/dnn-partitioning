import argparse
from pathlib import Path

from dnn_partition.common.partition_manager import PartitionManager

from .repository_builder import TritonRepositoryBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate Triton model repositories for each supported architecture."
    )
    parser.add_argument(
        "--repos-root",
        default="dnn_partition/server/repos",
        help="Root directory where per-model repositories will be created.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of models to export. Defaults to all supported models.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partition_manager = PartitionManager()
    builder = TritonRepositoryBuilder(partition_manager)
    repos_root = Path(args.repos_root)
    models = args.models or partition_manager.list_models()

    for model_name in models:
        repo_dir = repos_root / f"{model_name}_repo"
        exported = builder.export_all(model_name, repo_dir, device=args.device)
        print(f"\n[{model_name}]")
        print(f"repo: {repo_dir}")
        for path in exported:
            print(path)


if __name__ == "__main__":
    main()
