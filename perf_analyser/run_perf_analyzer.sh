#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-resnet18_full}"
URL="${URL:-127.0.0.1:8001}"
PROTOCOL="${PROTOCOL:-grpc}"
INPUT_DATA="${INPUT_DATA:-random}"
CONCURRENCY_RANGE="${CONCURRENCY_RANGE:-1:16:2}"
MEASUREMENT_INTERVAL="${MEASUREMENT_INTERVAL:-5000}"
SDK_IMAGE="${SDK_IMAGE:-nvcr.io/nvidia/tritonserver:26.01-py3-sdk}"
USE_DOCKER="${USE_DOCKER:-auto}"

ARGS=(
  -m "${MODEL}"
  -i "${PROTOCOL}"
  -u "${URL}"
  --input-data "${INPUT_DATA}"
  --concurrency-range "${CONCURRENCY_RANGE}"
  --measurement-interval "${MEASUREMENT_INTERVAL}"
)

if [[ "${USE_DOCKER}" == "0" || "${USE_DOCKER}" == "false" ]]; then
  exec perf_analyzer "${ARGS[@]}"
fi

if [[ "${USE_DOCKER}" == "1" || "${USE_DOCKER}" == "true" ]]; then
  exec docker run --rm --net=host "${SDK_IMAGE}" perf_analyzer "${ARGS[@]}"
fi

if command -v perf_analyzer >/dev/null 2>&1; then
  exec perf_analyzer "${ARGS[@]}"
fi

exec docker run --rm --net=host "${SDK_IMAGE}" perf_analyzer "${ARGS[@]}"
