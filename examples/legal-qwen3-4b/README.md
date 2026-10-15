LAB is the intended target benchmark. See [LAB_READINESS.md](LAB_READINESS.md) for the evaluation audit, blockers and post-training commands. This training pilot currently uses SFT NLL as a proxy; it does not measure LAB improvement.

This directory prepares a bounded autonomous Qwen3-4B legal fine-tuning pilot for the
current GPU Pod. It reuses the persisted runtime under `/workspace/aibuildai-runtime`.
The fresh upstream agent checkout alone does not contain the Pod database/lifecycle
patches or the all-role recorded-spend cap; the launcher deliberately uses the staged,
patched runtime. The KB supplies the cards, not training rows. All 48 card hashes are
checked against the local KB checkout before launch. This pilot uses frozen local
cards/manifest and no live MCP service (`mcps: {}`), so KB hosting is not required.

Run the no-cost checks:

```bash
cd /workspace/AIBuildAI/aibuildai-llm-posttrain-agent
bash examples/legal-qwen3-4b/run_agent.sh --dry-run
```

In YOUR terminal, enter the controller key without putting it in shell history:

```bash
read -rsp 'Controller API key: ' AIBUILDAI_API_KEY; echo
export AIBUILDAI_API_KEY
bash examples/legal-qwen3-4b/run_agent.sh
```

The configured controller is `claude-opus-5`, separate from the trained
`Qwen/Qwen3-4B`. Use a controller model available to your Anthropic-compatible
provider; edit `llm.default.model` in the YAML if needed. For other providers, set
`AIBUILDAI_BASE_URL` and their model mappings as documented in the repository README.
An ordinary OpenAI API key/Responses endpoint cannot be used directly here.
The launcher only inherits your terminal credentials; it does not load credential files.

Default config: `/workspace/legal-agent/pilot-ready/agent.yaml`.
Task statement: `/workspace/legal-agent/pilot-ready/task/README.md`.
Results: `/workspace/legal-agent/pilot-ready/playground/qwen3-13-legal-pilot/`.

Defaults are at most 100 encodable rows from each of 13 sources, 100 optimizer
updates per candidate, accumulation13, sequence length2048, rank at most32, and one
candidate plus an optional second. The agent chooses the learning rate/rank and
writes/runs its sequential search. It measures an unmodified baseline, independently
scores adapters and exports a merged model. The prepared pilot exploration budget is 60 minutes;
setup/finalization can extend total elapsed time. Recorded all-role API cap is $4,
exploration admission $3, no separate per-agent cap. In-flight provider calls can overshoot caps;
GPU costs are separate. These are pilot limits, not time/cost estimates.

For a smaller first PAID runtime test, with two optimizer updates and 13 training rows:

```bash
bash examples/legal-qwen3-4b/run_agent.sh --smoke
```

The smoke config has a $4 recorded all-role cap, $1.50 exploration admission and
20-minute exploration budget. It is not a useful full training run.

Generate a distinct larger task (fresh output required) and launch it:

```bash
USE_TORCH=0 /workspace/aibuildai-runtime/training-venv/bin/python \
  examples/legal-qwen3-4b/prepare_pilot.py \
  --output /workspace/legal-agent/pilot-larger \
  --rows-per-source 1000 --validation-per-source 16 \
  --steps 1000 --minutes 240 --cost-usd 30
AIBUILDAI_TASK_CONFIG=/workspace/legal-agent/pilot-larger/agent.yaml \
  bash examples/legal-qwen3-4b/run_agent.sh
```

Existing preparation contains 13,014 SFT train and 4,047 validation rows. It examined
at most 1000 rows per selected config/split, not full datasets. Increasing the task's
row cap cannot recover data absent from preparation. `pipeline.py prepare --help`
describes fresh data preparation. No custom wrappers are developed by the agent.

The 48-card disposition is: 13 SFT sources ready, 11 evaluation-only, 6 raw-text CPT
candidates deferred, 15 needing a wrapper/holdout/objective/policy change, and 3 missing
source terms. Full IDs/reasons/revisions are in task/inventory.json and the adjacent
DATASETS.md. LexGLUE uses case_hold and unfair_tos; Lawma uses sc_adminaction. CUAD
uses the existing pinned archive adapter. CaseHOLD is not added a second time.

Frozen scoring is macro held-out teacher-forced NLL, minimized, capped at eight
encodable examples per available source. CUAD has no validation rows in the cached
preparation and is explicitly omitted from this metric. Training data are interleaved.
Official tests are excluded; document groups are checked against all cached validation
rows. This does not establish legal accuracy, LAB gains, or full-corpus decontamination.
Long prompts may be shortened and oversized answers skipped; logs record both.

This Pod currently has an A10080GB and 32GiB CPU memory. Nested bubblewrap fails, so
sandboxing is disabled and the dedicated Pod is the process boundary. Same-UID code
can access private scorer files; this is not an adversarial evaluation boundary.
Kubernetes mode has no browser workspace or web pause/resume. Use terminal logs and
`watch -n 1 nvidia-smi` to monitor. No LAB setup or paid judges are needed.

After container replacement, restore accounts using the existing
`examples/kubernetes-runtime/setup_local.sh` in the STAGED runtime, then start its
`postgres_service.py` as described in its README. The launcher preflight fails clearly
if PostgreSQL is unavailable. Do not replace the staged patched engine with the fresh
upstream checkout during this run.
