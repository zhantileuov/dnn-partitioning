import shutil
import argparse
from pathlib import Path
from typing import Union


def install_stress_router_model(repo_dir: Union[str, Path]) -> Path:
    """Copy the stress router Triton model into an existing model repository."""
    repo_dir = Path(repo_dir)
    source_dir = Path(__file__).resolve().parent / "model_repository" / "stress_router"
    target_dir = repo_dir / "stress_router"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the Triton stress router model into a model repository.")
    parser.add_argument("--repo-dir", required=True, help="Target Triton model repository directory.")
    args = parser.parse_args()
    installed = install_stress_router_model(args.repo_dir)
    print(installed)


if __name__ == "__main__":
    main()
