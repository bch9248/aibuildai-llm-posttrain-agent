#!/usr/bin/env bash
# Uses only credentials inherited from the caller's terminal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/workspace/aibuildai-runtime
CONFIG="${AIBUILDAI_TASK_CONFIG:-/workspace/legal-agent/pilot-ready/agent.yaml}"
MODE="${1:-run}"
if [[ "$MODE" == --smoke ]]; then
  CONFIG=/workspace/legal-agent/smoke-test/agent.yaml
elif [[ "$MODE" != run && "$MODE" != --dry-run ]]; then
  echo 'Usage: bash run_agent.sh [--dry-run|--smoke]' >&2
  exit 2
fi
"$R/manager-venv/bin/python" "$HERE/dry_run.py" "$CONFIG"
"$R/manager-venv/bin/python" "$R/source/aibuildai-llm-posttrain-agent/examples/kubernetes-runtime/launch_controller.py" --preflight
if [[ "$MODE" == --dry-run ]]; then exit 0; fi
if [[ -z "${AIBUILDAI_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-}}}" ]]; then
  echo 'Export AIBUILDAI_API_KEY in this terminal, then rerun.' >&2
  exit 1
fi
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
echo "Launching agent with $CONFIG"
exec "$R/manager-venv/bin/python" "$R/source/aibuildai-llm-posttrain-agent/examples/kubernetes-runtime/launch_controller.py" -- run "$CONFIG"
