LAB is the target outcome. The current `pilot-ready/agent.yaml` still optimizes held-out
SFT NLL. It has no LAB scorer hook, so training success there cannot establish LAB gains.
This audit prepared a separate post-training LAB path, but real execution is blocked by
the current Pod's sandbox permissions. No benchmark task or paid judge has been run.

The official repository is downloaded at `/workspace/AIBuildAI/harvey-labs`, revision
`a15650fe74bb20b86f160aaabdad75b87a91b6e8` (lab-core1.2.0). The persisted Kubernetes
adaptation was restored on this revision, and its dependency lock regenerated. This is
an adapted execution backend; it is not an unmodified official Podman run.

Verified:

- 2,010 task configurations downloaded. Frozen split: 52 development tasks, 1,956
  final-test tasks and 2 excluded variants of development families. The three-task smoke
  set is a subset of development. Firm-knowledge's shared DMS stays wholly in test.
- 99 relevant offline tests pass: official rubric scoring with mock judges, path
  handling and Kubernetes sandbox contract tests. These are not performance results.
- Actual synthetic DOCX generation/extraction passes with the grader's Pandoc3.11.
- Public official AMD64 sandbox image resolved to an immutable digest in
  `/workspace/legal-agent/lab-readiness/image.json`. The image source also pins Pandoc3.11;
  the live fixture checks its running version once permissions are available.
- Live sandbox fixture stops at `PermissionError: Kubernetes identity lacks create
  pods/ in bryan-aibuildai-lab`. SelfSubjectAccessReview also denies get/delete/exec Pods
  and reading NetworkPolicies. No task Pods were created.
- Local Qwen Chat Completions startup was attempted but did not become ready after
  12 minutes; the loader was in disk sleep on the mounted volume. The audit stopped
  its endpoint/warm-reader. Actual model responses and tool-call compatibility remain
  unverified, and cold-load latency must be resolved before time-bounded evaluation.
- Native Podman is absent and `max_user_namespaces=0`; `unshare -Ur true` fails with
  ENOSPC. Installing Podman alone will not fix this current Pod's namespace restriction.

The dedicated LAB Python is `/workspace/aibuildai-runtime/lab-venv/bin/python` (3.12).
The training Python remains in its separate environment. Machine-readable evidence and
logs are in `/workspace/legal-agent/lab-readiness/`.

Repeat the no-cost dry-run (expected to return exit1 until the sandbox is available):

```bash
cd /workspace/AIBuildAI/aibuildai-llm-posttrain-agent
bash examples/legal-qwen3-4b/run_lab.sh --dry-run
```

Infrastructure gap: an administrator must provision the isolated LAB namespace and
scoped controller credentials using
`/workspace/aibuildai-runtime/source/aibuildai-llm-posttrain-agent/examples/kubernetes-runtime/manifests/lab-namespace.yaml`
and its adjacent README/create_lab_kubeconfig.py. No permission changes were attempted
in this audit. Export the issued file path as LAB_KUBECONFIG, then run:

```bash
export LAB_KUBECONFIG=/secure/path/lab-kubeconfig
bash examples/legal-qwen3-4b/run_lab.sh --sandbox-check
```

That synthetic fixture must pass document transfers, read-only inputs, DOCX creation,
Pandoc version, timeouts, no credential/token inheritance, output collection and blocked
network egress. Alternatively run the official Podman backend on a host where it works;
use the same backend/image for every compared condition.

After the fixture passes, start local Qwen in one terminal (stop it before GPU training):

```bash
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
/workspace/aibuildai-runtime/training-venv/bin/python \
  examples/legal-qwen3-4b/serve_local.py \
  --model /workspace/legal-agent/smoke-test/task/model \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c --name qwen3-base
```

The adapter uses local Chat Completions with Qwen tool-call parsing, rather than LAB's
stock OpenAI Responses adapter. The serial Transformers endpoint is for initial checks;
validate context/tool/output behavior before a larger study. It rejects context overflow.
Do not set OPENAI_BASE_URL to the local endpoint; LAB_LOCAL_BASE_URL is separate.

In a second terminal, export ANTHROPIC_API_KEY and OPENAI_API_KEY for standard paid dual
judging, then run one development task first:

```bash
bash examples/legal-qwen3-4b/run_lab.sh --evaluate \
  --condition base-scored --model qwen3-base --split smoke --limit 1
```

`--generate` instead of `--evaluate` generates deliverables without paid judging. Use a
fresh condition for a new run. To judge an already generated run without regenerating,
use the official evaluator with the `run_id` and `task` from that condition's results.json:

```bash
PATH=/workspace/aibuildai-runtime/lab-tools/bin:$PATH \
/workspace/aibuildai-runtime/lab-venv/bin/python -m lab_core.evaluation.run_eval \
  --run-id RUN_ID --task TASK_ID --judges claude-sonnet-4-6 gpt-5.5
```

Run that command from `/workspace/AIBuildAI/harvey-labs`. Direct manual judging writes
official score artifacts; it does not update the study wrapper's results.json. For its
paired-comparison utility, use conditions evaluated with `--evaluate` end to end.

For the fine-tuned condition, stop the base endpoint, serve the merged final_model path
with `--name qwen3-tuned`, and run the identical tasks/settings with a fresh condition.
An unmerged adapter can also be served with `--adapter PATH` and the same frozen base.
Deliverable model loading must be verified independently of training-reported scores.

Budget gap: the $4 cap governs training-controller agents only. It does not meter/stop
judge calls made by LAB subprocesses or GPU charges. Standard judging uses Claude
Sonnet4.6 and GPT5.5. The 3-task smoke set has234 criteria (468 judge assessments per
condition; 936 for base+tuned); all52 dev tasks have4,820 criteria (9,640 assessments per
condition). Actual billed tokens/retries determine spend. No dollar estimate or $4
end-to-end guarantee has been validated. Per-loop LAB needs explicit judge-spend and
wall-clock enforcement, not just the existing controller cap.

Research protocol: start with post-training evaluation; use a small fixed dev subset
for occasional iterations only after the pipeline works. Keep LAB task documents,
instructions, rubrics and outputs out of training, and reserve final test for the fixed
recipe. To attribute gains to extra datasets, compare the 3-source pilot and 13-source
mixture with matched processed-token budgets, base checkpoint, sequence length,
optimizer, seeds, agent tools/skills, sandbox and judges. Include unmodified Qwen as a
baseline. Report both official all-pass and criterion-pass metrics, tool failures,
missing outputs, context overflow and uncertainty. Lower SFT NLL alone is not LAB gain.

Remaining integration work: freeze a runnable LAB dev scorer, map its metric to the
training engine, serialize model serving/scoring/training on the single GPU, and enforce
judge/time budgets. Do not switch the training search's objective to an unrunnable LAB
scorer: setup would fail before useful training. The existing pilot configuration was
left at its requested 60-minute/$4 settings during this readiness audit.
