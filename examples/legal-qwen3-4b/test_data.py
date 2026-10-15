import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from adapters import convert
from pipeline import combine
from recipes import decision
from train import encode
class AdapterTests(unittest.TestCase):

    def test_empty_unfair_tos_is_none(self):
        row = convert({'text': 'A fair clause.', 'labels': []}, {'id': 'coastalcph/lex_glue', 'adapter': 'classification', 'config': 'unfair_tos'})
        self.assertEqual(row['messages'][-1]['content'], 'none')

    def test_lawma_many_and_multiple_choices(self):
        row = convert({'opinion': 'Opinion', 'instruction': 'Classify', 'question': 'Agency?', 'choices': [str(i) for i in range(124)], 'answer': [116, 2]}, {'id': 'ricdomolm/lawma-tasks', 'adapter': 'mcq'})
        self.assertEqual(json.loads(row['messages'][-1]['content']), [2, 116])

    def test_bad_choice_is_rejected(self):
        with self.assertRaises(ValueError):
            convert({'context': 'Opinion', 'endings': ['A', 'B'], 'label': 2}, {'id': 'coastalcph/lex_glue', 'adapter': 'mcq'})

    def test_australian_text_never_enters_prompt(self):
        row = convert({'source': {'text': 'Grounding source'}, 'question': 'Why?', 'answer': 'SECRET ANSWER', 'text': 'Why? SECRET ANSWER'}, {'id': 'isaacus/open-australian-legal-qa', 'adapter': 'grounded_qa'})
        self.assertNotIn('SECRET ANSWER', row['messages'][0]['content'])
        self.assertIn('Grounding source', row['messages'][0]['content'])

    def test_cuad_absent_clause_kept(self):
        row = convert({'context': 'A contract', 'question': 'Termination?', 'answers': {'text': []}}, {'id': 'theatticusproject/cuad-qa', 'adapter': 'cuad'})
        self.assertEqual(row['messages'][-1]['content'], 'no such clause')

    def test_prewrapped_alpaca_text_ignored(self):
        row = convert({'instruction': 'Question', 'input': 'Facts', 'output': 'Answer', 'text': 'Question Answer'}, {'id': 'nisaar/LLAMA2_Legal_Dataset_4.4k_Instructions', 'adapter': 'sft', 'inputs': ['instruction', 'input'], 'target': 'output'})
        self.assertEqual(row['messages'][0]['content'], 'Question\n\nFacts')

    def test_chat_roles_and_file_group(self):
        spec = {'id': 'TheTokenFactory/sec-contracts-financial-extraction-instructions', 'adapter': 'chat'}
        base = {'conversations': [{'from': 'system', 'value': 'JSON only'}, {'from': 'human', 'value': 'Document'}, {'from': 'gpt', 'value': '{}'}], 'metadata': {'source_file': 'agreement1'}}
        a = convert(base, spec)
        b = convert({**base, 'conversations': [*base['conversations'][:-1], {'from': 'gpt', 'value': '{"x":1}'}]}, spec)
        self.assertEqual(a['group'], b['group'])
        self.assertEqual([m['role'] for m in a['messages']], ['system', 'user', 'assistant'])

    def test_markup_and_pdf(self):
        spec = {'id': 'chenghao/sec-material-contracts', 'adapter': 'text', 'field': 'file_content', 'markup': True}
        row = convert({'extension': 'html', 'file_content': '<p>Contract</p><script>bad</script>'}, spec)
        self.assertEqual(row['text'], 'Contract')
        self.assertIsNone(convert({'extension': 'pdf', 'file_content': 'base64'}, spec))

    def test_eval_only_even_named_train_is_reserved(self):
        status, _ = decision({'id': 'reglab/legal_rag_hallucinations', 'use': 'eval'})
        self.assertEqual(status, 'evaluation_only')

    def test_document_holdout_and_content_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            source = out / 'rows.jsonl'
            rows = [{'messages': [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': 'a'}], 'group': 'a' * 64, 'objective': 'sft', 'source': 'example', 'original_split': split} for q, split in [('train question', 'train'), ('val question', 'validation'), ('val question', 'validation')]]
            source.write_text(''.join((json.dumps(x) + '\n' for x in rows)))
            inv = [{'id': 'example', 'decision': 'candidate', 'attempts': [{'status': 'ready', 'spec': {'validation_split': 'validation'}, 'rows_path': str(source)}]}]
            with contextlib.redirect_stdout(io.StringIO()):
                combine(out, inv, argparse.Namespace(seed=42, allow_partial=False))
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['counts']['sft_train.jsonl'], 0)
            self.assertEqual(manifest['counts']['sft_validation.jsonl'], 1)
            self.assertEqual(manifest['duplicates_removed'], 1)

class LossMaskTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        snapshot = Path('/workspace/legal-agent/smoke-test/task/model')
        if not snapshot.exists():
            raise unittest.SkipTest('Run model download before tokenizer integration checks')
        cls.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    def test_assistant_only_mask_matches_qwen_template(self):
        messages = [{'role': 'user', 'content': 'Classify'}, {'role': 'assistant', 'content': 'yes'}]
        item, truncated = encode({'messages': messages}, self.tokenizer, 128, 'sft')
        full = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        self.assertEqual(item['input_ids'], full[:-1])
        self.assertFalse(truncated)
        supervised = [i for i, label in zip(item['input_ids'], item['labels']) if label != -100]
        self.assertEqual(self.tokenizer.decode(supervised), 'yes<|im_end|>')

    def test_prompt_shortened_target_preserved(self):
        target = 'Complete answer'
        item, truncated = encode({'messages': [{'role': 'user', 'content': 'law ' * 2000}, {'role': 'assistant', 'content': target}]}, self.tokenizer, 128, 'sft')
        self.assertTrue(truncated)
        self.assertLessEqual(len(item['input_ids']), 128)
        self.assertEqual(self.tokenizer.decode([x for x in item['labels'] if x != -100]), target + '<|im_end|>')

    def test_long_answer_skipped_instead_of_masked_away(self):
        item, _ = encode({'messages': [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'answer ' * 2000}]}, self.tokenizer, 128, 'sft')
        self.assertIsNone(item)

if __name__ == "__main__": unittest.main()
