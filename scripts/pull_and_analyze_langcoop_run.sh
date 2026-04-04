#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

NAMESPACE="${NAMESPACE:-seelab}"
POD="${POD:-carla-client-dev}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/carla_client}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-test_results_langcoop_leaderboard}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/Desktop/langcoop_runs}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_LABEL}"
LOG_FILE="${LOCAL_RUN_DIR}/pull_and_analyze.log"

mkdir -p "${LOCAL_RUN_DIR}"

exec > >(tee "${LOG_FILE}") 2>&1

echo "Pulling ${POD}:${REMOTE_ROOT}/${REMOTE_RESULTS_DIR} -> ${LOCAL_RUN_DIR}"
kubectl -n "${NAMESPACE}" cp "${POD}:${REMOTE_ROOT}/${REMOTE_RESULTS_DIR}" "${LOCAL_RUN_DIR}/"

COPIED_RUN_DIR=""
for candidate in \
  "${LOCAL_RUN_DIR}/${REMOTE_RESULTS_DIR}" \
  "${LOCAL_RUN_DIR}/$(basename "${REMOTE_RESULTS_DIR}")"; do
  if [[ -d "${candidate}" ]]; then
    COPIED_RUN_DIR="${candidate}"
    break
  fi
done

if [[ -z "${COPIED_RUN_DIR}" ]]; then
  COPIED_RUN_DIR="$(find "${LOCAL_RUN_DIR}" -maxdepth 3 -type d -name "$(basename "${REMOTE_RESULTS_DIR}")" | head -n 1 || true)"
fi

if [[ -z "${COPIED_RUN_DIR}" && -d "${LOCAL_RUN_DIR}/images" ]]; then
  COPIED_RUN_DIR="${LOCAL_RUN_DIR}"
fi

if [[ -z "${COPIED_RUN_DIR}" || ! -d "${COPIED_RUN_DIR}" ]]; then
  echo "Expected copied run directory not found under: ${LOCAL_RUN_DIR}" >&2
  echo "Available directories:" >&2
  find "${LOCAL_RUN_DIR}" -maxdepth 3 -type d | sort >&2
  exit 1
fi

echo "Stitching videos from copied image folders"
STITCH_FAILURES=0
if [[ -d "${COPIED_RUN_DIR}/images" ]]; then
  while IFS= read -r scenario_dir; do
    [[ -z "${scenario_dir}" ]] && continue
    echo "  stitching scenario: ${scenario_dir}"
    scenario_name="$(basename "${scenario_dir}")"
    scenario_video_dir="${LOCAL_RUN_DIR}/videos/${scenario_name}"
    mkdir -p "${scenario_video_dir}"
    while IFS= read -r agent_dir; do
      [[ -z "${agent_dir}" ]] && continue
      agent_name="$(basename "${agent_dir}")"
      echo "    stitching agent: ${agent_name}"
      if ! python3 "${REPO_ROOT}/stitch_frames.py" \
        --images-dir "${agent_dir}" \
        --format mp4 \
        --output-dir "${scenario_video_dir}" \
        --output-file "${agent_name}"; then
        echo "    stitch failed for ${agent_name}"
        STITCH_FAILURES=$((STITCH_FAILURES + 1))
      fi
    done < <(find "${scenario_dir}" -mindepth 1 -maxdepth 1 -type d -name "agent_*" | sort)
  done < <(find "${COPIED_RUN_DIR}/images" -mindepth 1 -maxdepth 1 -type d | sort)
fi

echo "Analyzing copied run"
ANALYZE_FAILURES=0
if ! python3 "${REPO_ROOT}/scripts/analyze_langcoop_run.py" \
  --run-dir "${COPIED_RUN_DIR}" \
  --output-json "${LOCAL_RUN_DIR}/analysis.json" \
  --output-text "${LOCAL_RUN_DIR}/analysis.txt"; then
  ANALYZE_FAILURES=$((ANALYZE_FAILURES + 1))
  echo "Analysis step failed"
fi

VIDEO_COUNT=$(find "${LOCAL_RUN_DIR}/videos" -type f \( -name "*.mp4" -o -name "*.gif" \) 2>/dev/null | wc -l | tr -d ' ')
echo "Video files created: ${VIDEO_COUNT}"
echo "Stitch failures: ${STITCH_FAILURES}"
echo "Analysis failures: ${ANALYZE_FAILURES}"

echo "Done."
echo "Run copied to: ${LOCAL_RUN_DIR}"
echo "Summary report: ${LOCAL_RUN_DIR}/analysis.txt"
echo "Log file: ${LOG_FILE}"
