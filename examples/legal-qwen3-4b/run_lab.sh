#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/workspace/aibuildai-runtime
PY="$R/lab-venv/bin/python"
REPO=/workspace/AIBuildAI/harvey-labs
PLAN=/workspace/legal-agent/lab-readiness/lab-plan.json
MODE="${1:---dry-run}"
if [[ $# -gt 0 ]]; then shift; fi
export PATH="$R/lab-tools/bin:$PATH"
export LAB_K8S_NAMESPACE="${LAB_K8S_NAMESPACE:-bryan-aibuildai-lab}"
IMAGE_REPORT=/workspace/legal-agent/lab-readiness/image.json
if [[ -z "${LAB_K8S_IMAGE:-}" && -f "$IMAGE_REPORT" ]]; then
  LAB_K8S_IMAGE="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["pinned_image"])' "$IMAGE_REPORT")"
  export LAB_K8S_IMAGE
fi
COMMON=(--repo "$REPO" --plan "$PLAN" --sandbox-backend kubernetes --output /workspace/legal-agent/lab-readiness/study)
case "$MODE" in
  --dry-run)
    "$PY" "$HERE/lab_study.py" run "${COMMON[@]}" --condition base --model qwen3-base --split smoke "$@"
    exec "$PY" "$HERE/check_lab.py" --namespace "$LAB_K8S_NAMESPACE" --fixture
    ;;
  --sandbox-check)
    exec "$PY" "$HERE/check_lab.py" --namespace "$LAB_K8S_NAMESPACE" --fixture
    ;;
  --generate|--evaluate)
    "$PY" "$HERE/check_lab.py" --namespace "$LAB_K8S_NAMESPACE" --fixture
    SCORE=()
    if [[ "$MODE" == --evaluate ]]; then
      if [[ -z "${ANTHROPIC_API_KEY:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
        echo 'Standard dual judging needs ANTHROPIC_API_KEY and OPENAI_API_KEY exported in this terminal.' >&2
        exit 1
      fi
      SCORE=(--score)
    fi
    exec "$PY" "$HERE/lab_study.py" run "${COMMON[@]}" --condition base --model qwen3-base --split smoke "$@" --execute "${SCORE[@]}"
    ;;
  *) echo 'Usage: bash run_lab.sh [--dry-run|--sandbox-check|--generate|--evaluate] [lab_study run options]' >&2; exit 2 ;;
esac
