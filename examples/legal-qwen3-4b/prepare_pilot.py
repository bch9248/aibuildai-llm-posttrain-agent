#!/usr/bin/env python3
"""Generate a bounded 13-source training task from the existing prepared data."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from transformers import AutoTokenizer
from train import encode

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/workspace/aibuildai-runtime')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--prepared-data', type=Path, default=Path('/workspace/legal-agent/prepared'))
p.add_argument('--model', type=Path, default=Path('/workspace/legal-agent/smoke-test/task/model'))
p.add_argument('--controller', default='claude-opus-5')
p.add_argument('--rows-per-source', type=int, default=100)
p.add_argument('--validation-per-source', type=int, default=8)
p.add_argument('--steps', type=int, default=100)
p.add_argument('--minutes', type=int, default=120)
p.add_argument('--cost-usd', type=float, default=20)
a = p.parse_args()
if min(a.rows_per_source, a.validation_per_source, a.steps, a.minutes) < 1 or a.cost_usd <= 0:
    p.error('All limits must be positive')
subprocess.run([sys.executable, str(HERE/'make_autonomous_task.py'),
    '--prepared-data', str(a.prepared_data), '--output', str(a.output),
    '--model', str(a.model), '--agent-bin', str(RUNTIME/'manager-venv/bin/aibuildai'),
    '--controller', a.controller], check=True, stdout=subprocess.DEVNULL)
task = a.output.resolve()/'task'
manifest = json.loads((task/'manifest.json').read_text())
tokenizer = AutoTokenizer.from_pretrained(task/'model', local_files_only=True)
groups = defaultdict(list)
for line in (a.prepared_data/'sft_train.jsonl').open():
    row = json.loads(line)
    if len(groups[row['source']]) < a.rows_per_source and encode(row, tokenizer, 2048, 'sft')[0] is not None:
        groups[row['source']].append(row)
assert set(groups) == set(manifest['selected_sources']) and all(groups.values())
ordered = [rows[i] for i in range(max(map(len, groups.values())))
           for source, rows in sorted(groups.items()) if i < len(rows)]
(task/'sft_train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ordered))
counts, encoded = Counter(), []
for line in (a.prepared_data/'sft_validation.jsonl').open():
    row = json.loads(line)
    if counts[row['source']] >= a.validation_per_source:
        continue
    example, _ = encode(row, tokenizer, 2048, 'sft')
    if example is not None:
        encoded.append({**example, 'source': row['source']})
        counts[row['source']] += 1
blob = ''.join(json.dumps(row)+'\n' for row in encoded).encode()
(task/'validation-tokenized.jsonl').write_bytes(blob)
contract = json.loads((task/'validation-contract.json').read_text())
contract.update(validation_sha256=hashlib.sha256(blob).hexdigest(), validation_sources=sorted(counts),
    validation_rows=len(encoded), seq_length=2048,
    missing_validation_sources=sorted(set(groups)-set(counts)))
(task/'validation-contract.json').write_text(json.dumps(contract, indent=2))
(task/'pilot-settings.json').write_text(json.dumps({**vars(a),
    'training_rows':len(ordered), 'per_source_rows':{s:len(r) for s,r in groups.items()},
    'validation_counts':dict(counts)}, indent=2, default=str))
statement = f'''Fine-tune the supplied Qwen/Qwen3-4B model on the prepared legal SFT mixture.
This is a bounded training pilot: {len(ordered)} rows across 13 sources, at most
{a.rows_per_source} per source. Rows are interleaved so short training visits the mixture.
Read inventory.json for decisions on all 48 cards. Skip all remaining sources and
custom-wrapper development. Do not fetch more data or train on validation targets.
The original preparation capped examined rows at 1000 per config/split; this is not
training on every row of all 48 datasets. Preserve source attribution and pinned revisions.

SETUP: use the exact framework Program Python to install requirements.lock. Publish
sft_train.jsonl, model/, train.py, export_model.py, requirements.lock, manifest.json,
inventory.json, catalog.json and pilot-settings.json. Keep validation_score.py,
validation-contract.json and validation-tokenized.jsonl PRIVATE. Freeze supplied
validation_score.py unchanged: private/score.py must import it using importlib and return
its score(output_dir). Baseline submission.json is {{"kind":"base"}}. Independently score
it without training. The scorer can take up to 8 minutes. Metric macro_validation_nll
is MINIMIZED; lower is better. It contains {len(encoded)} held-out rows across
{len(counts)} sources; missing sources: {contract['missing_validation_sources']}.
Never substitute candidate-reported scores. This diagnostic is not legal accuracy or LAB.

META: implement a simple sequential Search: one coder, one GPU training Program,
independent scoring, optionally a second candidate within remaining budget. Propagate
these constraints. Serialize training, scoring and export on GPU0. Use
ScoreLaunch.measure(..., gpus=(0,)) with the producer's exact GPU indices. Give GPU work
at least 15 minutes. No paper, literature search, additional datasets or hosted-model
training-data generation. The dedicated Pod has 32GiB CPU RAM, 4 CPUs and an A10080GB;
use at most 24GiB RAM and 2 CPUs per Program. Bubblewrap is disabled because nested
namespaces fail. Private files are not isolated from malicious same-UID processes.
Do not print credentials or database URLs.

Train using supplied train.py, at most {a.steps} optimizer updates per candidate,
sequence length2048, accumulation13, seed42, LoRA rank1..32. Choose learning rate and
rank; rank16 and learning rate2e-4 are starting values. Command:
python PUBLIC/train.py --model PUBLIC/model --data WORKING_DATA --output CANDIDATE \
  --steps {a.steps} --accumulation 13 --seq-length 2048 --eval-rows 1
Create an EMPTY WORKING_DATA/sft_validation.jsonl and copy/link PUBLIC/sft_train.jsonl
into WORKING_DATA. The independent private scorer owns validation. Record actual source
coverage, processed/supervised tokens, truncation, skipped rows and optimizer settings.
Candidate output requires adapter/ and submission.json
{{"kind":"lora","adapter":"adapter"}}. Adapter files must be regular single-link files.

FINALIZER: deliver the trained adapter and independently loadable merged final_model/
using export_model.py with the local base. Select using the frozen metric and report
any regression honestly. Include measured base/candidate scores, commands, manifests,
actual source/token coverage and recorded API spend. Final artifacts must be regular
single-link files. Do not run LAB, OpenAI judges or Kubernetes task Pods; do not claim
LAB gains or full-corpus decontamination. Overlap screening only covers loaded rows.
'''
(task/'README.md').write_text(statement)
cfg = yaml.safe_load((a.output/'agent.yaml').read_text())
cfg['run']['task_name'] = 'qwen3-13-legal-pilot'
cfg['run']['budget'].update(wall_clock_minutes=a.minutes, cost_usd=a.cost_usd*.75,
    per_agent_cost_cap_usd=a.cost_usd/2, total_agent_cost_limit_usd=a.cost_usd)
shared = 'Follow the legal pilot README. Freeze the supplied validation scorer; minimize macro_validation_nll. Keep GPU work sequential. No LAB. Respect the configured budgets; in-flight calls can overshoot caps. Never print keys.'
cfg['llm']['system_instructions'] = {role:shared for role in ['setup','meta','finalizer']}
cfg['resources']['run'].update(cpu_max_cores=4, memory_max_gb=28)
cfg['resources']['work_unit'].update(cpu_max_cores=2, memory_max_gb=24)
for unit in cfg['work_units']:
    unit['budget'] = {'kind':'fixed','minutes':40 if unit['name'] in ['setup','meta','coder','training','finalizer','score_program'] else 20}
(a.output/'agent.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
print(json.dumps({'config':str(a.output/'agent.yaml'), 'training_rows':len(ordered),
    'sources':len(groups),'validation_rows':len(encoded), 'missing_validation':contract['missing_validation_sources']}, indent=2))
