__all__ = ["install_stress_router_model"]


def __getattr__(name: str):
    if name == "install_stress_router_model":
        from .install import install_stress_router_model

        return install_stress_router_model
    raise AttributeError(name)
