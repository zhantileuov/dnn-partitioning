import time

import numpy as np

try:
    import tritonclient.grpc as grpcclient
except ImportError:  # pragma: no cover
    grpcclient = None


class TritonRequestClient:
    def __init__(self, url: str):
        if grpcclient is None:
            raise ImportError("tritonclient.grpc is required to use TritonRequestClient")
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def infer(self, model_name: str, array: np.ndarray, timeout_s: float = 30.0):
        infer_input = grpcclient.InferInput("input", list(array.shape), self._dtype_to_triton(array.dtype))
        infer_input.set_data_from_numpy(array)
        output = grpcclient.InferRequestedOutput("output")

        bytes_sent = int(array.nbytes)
        t0 = time.perf_counter()
        result = self.client.infer(
            model_name=model_name,
            inputs=[infer_input],
            outputs=[output],
            client_timeout=timeout_s,
        )
        transfer_time = time.perf_counter() - t0
        output_array = result.as_numpy("output")
        bytes_received = int(output_array.nbytes)
        return output_array, transfer_time, None, bytes_sent, bytes_received

    @staticmethod
    def _dtype_to_triton(dtype: np.dtype) -> str:
        mapping = {
            np.dtype(np.float32): "FP32",
            np.dtype(np.float16): "FP16",
            np.dtype(np.uint8): "UINT8",
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype for Triton gRPC: {dtype}")
        return mapping[dtype]
