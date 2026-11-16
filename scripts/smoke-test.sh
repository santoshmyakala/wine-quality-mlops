#!/usr/bin/env bash
# Start the image, wait for readiness and score one sample. $IMAGE is set by the pipeline.
set -euo pipefail
: "${IMAGE:?IMAGE must be set}"
docker run -d --name wine-smoke -p 8000:8000 "$IMAGE" >/dev/null
trap 'docker rm -f wine-smoke >/dev/null' EXIT
for _ in $(seq 1 30); do curl -fsS localhost:8000/readyz >/dev/null 2>&1 && break; sleep 1; done
curl -fsS -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"samples":[{
  "fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0,"residual sugar":1.9,"chlorides":0.076,
  "free sulfur dioxide":11,"total sulfur dioxide":34,"density":0.9978,"pH":3.51,"sulphates":0.56,
  "alcohol":9.4,"is_red":true}]}'
echo
docker exec wine-smoke id -u | grep -qx 10001 && echo "runs as non-root uid 10001"
