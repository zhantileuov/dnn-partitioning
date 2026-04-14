__all__ = ["TritonRepositoryBuilder"]


def __getattr__(name: str):
    if name == "TritonRepositoryBuilder":
        from .repository_builder import TritonRepositoryBuilder

        return TritonRepositoryBuilder
    raise AttributeError(name)
