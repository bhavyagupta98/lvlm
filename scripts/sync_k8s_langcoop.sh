#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="${NAMESPACE:-seelab}"
POD="${POD:-carla-client-dev}"
DEST_ROOT="${DEST_ROOT:-/workspace/carla_client}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing LangCoop runner bundle to ${POD}:${DEST_ROOT} in namespace ${NAMESPACE}"

kubectl -n "${NAMESPACE}" cp "${REPO_ROOT}/test_runner/" "${POD}:${DEST_ROOT}/"
kubectl -n "${NAMESPACE}" cp "${REPO_ROOT}/configs/" "${POD}:${DEST_ROOT}/"
kubectl -n "${NAMESPACE}" cp "${REPO_ROOT}/run_langcoop_test.py" "${POD}:${DEST_ROOT}/run_langcoop_test.py"
kubectl -n "${NAMESPACE}" cp "${REPO_ROOT}/run_langcoop_multiview_test.py" "${POD}:${DEST_ROOT}/run_langcoop_multiview_test.py"
kubectl -n "${NAMESPACE}" cp "${REPO_ROOT}/run_langcoop_leaderboard_test.py" "${POD}:${DEST_ROOT}/run_langcoop_leaderboard_test.py"

echo "Sync complete."
