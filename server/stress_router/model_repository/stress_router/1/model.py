import json

import numpy as np
import triton_python_backend_utils as pb_utils

from target_catalog import TARGET_MODEL_CATALOG


def _decode_target_name(tensor) -> str:
    values = tensor.as_numpy()
    if values.size != 1:
        raise pb_utils.TritonModelException("TARGET_MODEL_NAME must contain exactly one element")
    value = values.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class TritonPythonModel:
    def initialize(self, args):
        # The router exists to avoid sending very large intermediate activations
        # over the network. We cache one server-side tensor per target model so
        # each incoming tiny control request can trigger exactly one real tail
        # model execution with low wrapper overhead.
        self.model_config = json.loads(args["model_config"])
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, "STATUS")["data_type"]
        )
        self.cached_payloads = {}
        self.requested_output_names = ["output"]

        for target_name, spec in TARGET_MODEL_CATALOG.items():
            self.cached_payloads[target_name] = np.zeros(spec["shape"], dtype=spec["dtype"])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                target_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_MODEL_NAME")
                if target_tensor is None:
                    raise pb_utils.TritonModelException("Missing TARGET_MODEL_NAME input")

                target_model_name = _decode_target_name(target_tensor)
                cached_payload = self.cached_payloads.get(target_model_name)
                if cached_payload is None:
                    raise pb_utils.TritonModelException(
                        "Unknown target model {0!r}. Add it to target_catalog.py".format(target_model_name)
                    )

                # Exactly one router request maps to exactly one inner Triton
                # inference request, so wrapper RPS should track the stressed
                # tail model's RPS closely in metrics.
                inner_request = pb_utils.InferenceRequest(
                    model_name=target_model_name,
                    requested_output_names=self.requested_output_names,
                    inputs=[pb_utils.Tensor("input", cached_payload)],
                )
                inner_response = inner_request.exec()
                if inner_response.has_error():
                    raise pb_utils.TritonModelException(inner_response.error().message())

                status_tensor = pb_utils.Tensor("STATUS", np.array([0], dtype=self.output_dtype))
                responses.append(pb_utils.InferenceResponse(output_tensors=[status_tensor]))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))))
        return responses

    def finalize(self):
        self.cached_payloads = {}
