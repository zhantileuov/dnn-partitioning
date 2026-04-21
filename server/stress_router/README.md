# Triton Stress Router

This directory contains a Triton Python backend wrapper model plus helper code for stressing partitioned tail models without sending the real large intermediate activation tensor over the network.

## Why This Exists

Early split points such as `resnet18_tail_after_stem` require very large activation tensors. When a remote load generator sends those activations directly, the achieved RPS can be limited by network transfer and serialization instead of the tail model itself.

The stress router solves that by changing the request flow:

1. The remote client sends a tiny request to the Triton `stress_router` model.
2. The request contains only the target tail model name.
3. The router reuses a cached server-side tensor with the correct input shape and dtype.
4. The router issues exactly one internal Triton inference request to the selected tail model.
5. The router returns a tiny status response.

That gives a much cleaner way to stress tail-model execution capacity, especially for large early activations.

## One Request = One Tail Execution

The wrapper is intentionally strict:

- one incoming `stress_router` request
- one internal inference request to the selected tail model
- one small status response

This keeps wrapper RPS aligned as closely as possible with the stressed model's request rate in Triton metrics.

## Files

- `model_repository/stress_router/config.pbtxt`: Triton model config
- `model_repository/stress_router/1/model.py`: Python backend wrapper
- `model_repository/stress_router/1/target_catalog.py`: hardcoded mapping of target tail models to input shape and dtype
- `install.py`: helper to copy the router model into an existing Triton model repository

## Deployment

If you already build a Triton repository with the existing ONNX tail models, install the router into the same repo:

```bash
python3 -m dnn_partition.server.repository_builder \
  --repo-dir ~/dnn_partition/server/dynamic_model_repo \
  --model resnet18 \
  --device cpu \
  --include-stress-router
```

To build a partitioned repo and include ResNet-50 as a full-model RPS target
in the same repository:

```bash
python3 -m dnn_partition.server.repository_builder \
  --repo-dir ~/dnn_partition/server/dynamic_model_repo \
  --model resnet18 \
  --extra-full-model resnet50 \
  --device cpu \
  --include-stress-router
```

Or, if the repo already exists:

```bash
python3 -m dnn_partition.server.stress_router.install \
  --repo-dir ~/dnn_partition/server/dynamic_model_repo
```

Then start Triton as usual with the same repository mounted.

## Remote Stress Client

Use the dedicated client instead of the existing direct background load generator:

```bash
python controller/stress_router_client.py \
  --server-url 172.22.231.61:8001 \
  --target-model-name resnet18_tail_after_stem \
  --target-rps 200 \
  --workers 8 \
  --processes 4
```

This client sends only the target-model name to `stress_router`.

For direct full-model background load against ResNet-50, use:

```bash
python controller/background_client.py \
  --server-url 172.22.231.61:8001 \
  --model-name resnet50_full \
  --target-rps 20 \
  --workers 4 \
  --input-mode random
```

## Verification

Check Triton metrics or the Kafka server metrics stream for both models:

- `stress_router`
- the selected tail model, such as `resnet18_tail_after_stem`

Under healthy steady load, the success/request RPS for `stress_router` and the target tail model should track each other closely because each wrapper request triggers exactly one internal tail-model execution.

## Assumptions In This First Version

- the initial hardcoded catalog covers the current ResNet tail models used in this repo
- cached server-side tensors use zeros and are reused across requests
- tail models currently use FP32 inputs in this repo

## Future Improvements

- derive the mapping automatically from exported Triton metadata
- add FP16 router catalogs once FP16 tail models exist
- expose per-target router counters and richer status outputs
