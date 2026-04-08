import contextlib

import torch


@contextlib.contextmanager
def inference_context():
    if hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    else:
        with torch.no_grad():
            yield
