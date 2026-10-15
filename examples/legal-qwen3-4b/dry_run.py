"""No API calls or optimizer steps: validate task provenance and runtime schema."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import yaml

R = Path('/workspace/aibuildai-runtime/source/aibuildai-llm-posttrain-agent')
sys.path.insert(0, str(R))
from config import AgentConfig
cfg = AgentConfig.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text()))
task = Path(cfg.run.data_root)
manifest = json.loads((task/'manifest.json').read_text())
assert not manifest['failures']
inventory = json.loads((task/'inventory.json').read_text())['datasets']
assert len(inventory) == 48
cards = Path('/workspace/AIBuildAI/aibuildai-knowledge-base/data/corpus/plugins/aibuildai-posttrain-marketplace/aibuildai-posttrain/skills/legal-dataset/references')
for d in inventory:
    card = cards/(d['id'].replace('/', '__')+'.md')
    assert hashlib.sha256(card.read_bytes()).hexdigest() == d['card_sha256'], d['id']
train = [json.loads(line) for line in (task/'sft_train.jsonl').open()]
assert set(row['source'] for row in train) == set(manifest['selected_sources'])
assert all(row['original_split'] not in ['test','ca_test'] for row in train)
assert not set(row['source'] for row in train) & {d['id'] for d in inventory if d['use']=='eval'}
# Check groups against ALL previously prepared validation rows, not just the scored subset.
validation = [json.loads(line) for line in Path('/workspace/legal-agent/prepared/sft_validation.jsonl').open()]
assert not {row['group'] for row in train} & {row['group'] for row in validation}
contract = json.loads((task/'validation-contract.json').read_text())
assert hashlib.sha256((task/'validation-tokenized.jsonl').read_bytes()).hexdigest() == contract['validation_sha256']
assert (task/'model/model.safetensors.index.json').is_file()
print(json.dumps({'schema':'PASS (patched Pod runtime)', 'card_hashes':48,
    'training_rows':len(train), 'training_sources':dict(Counter(row['source'] for row in train)),
    'validation_rows':contract['validation_rows'], 'missing_validation_sources':contract['missing_validation_sources'],
    'loaded_group_overlap':0, 'inventory_decisions':dict(Counter(d['decision'] for d in inventory)),
    'api_calls':0, 'optimizer_steps':0}, indent=2))
