#!/usr/bin/env python3
"""Prepare a bounded autonomous Qwen3-4B test using validation instead of LAB."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--agent-bin", type=Path, required=True)
    p.add_argument("--controller", default="claude-opus-5")
    args = p.parse_args()
    import yaml
    from transformers import AutoTokenizer
    from train import encode

    if args.output.exists():
        raise FileExistsError("Choose a fresh task output directory")
    manifest = json.loads((args.prepared_data / "manifest.json").read_text())
    if manifest["failures"] or len(manifest["selected_sources"]) != 13:
        raise ValueError("Require all 13 sources successfully prepared")
    task = args.output.resolve() / "task"
    task.mkdir(parents=True)
    for name in ["manifest.json", "inventory.json"]:
        shutil.copy2(args.prepared_data / name, task / name)
    for name in [
        "train.py",
        "export_model.py",
        "requirements.lock",
        "catalog.json",
        "validation_score.py",
    ]:
        shutil.copy2(HERE / name, task / name)

    # Physical regular files keep public links contained within the task folder.
    def stage_file(source: str, destination: str):
        # Reuse the mounted-volume snapshot without copying another 8GB.
        try:
            os.link(Path(source).resolve(), destination)
        except OSError:
            shutil.copy2(source, destination)
        if os.geteuid() == 0:
            os.chown(destination, 0, 0)
        Path(destination).chmod(0o444)

    shutil.copytree(
        args.model, task / "model", symlinks=False, copy_function=stage_file
    )
    tokenizer = AutoTokenizer.from_pretrained(task / "model", local_files_only=True)
    train_rows = {}
    with (args.prepared_data / "sft_train.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["source"] not in train_rows
                and encode(row, tokenizer, 1024, "sft")[0] is not None
            ):
                train_rows[row["source"]] = row
    if set(train_rows) != set(manifest["selected_sources"]):
        raise ValueError("Every source needs an encodable smoke training row")
    (task / "sft_train.jsonl").write_text(
        "".join(json.dumps(train_rows[s]) + "\n" for s in sorted(train_rows))
    )
    selected, rows = Counter(), []
    available = Counter()
    with (args.prepared_data / "sft_validation.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            available[row["source"]] += 1
            if selected[row["source"]] >= 1:
                continue
            encoded, _ = encode(row, tokenizer, 1024, "sft")
            if encoded is not None:
                rows.append({**encoded, "source": row["source"]})
                selected[row["source"]] += 1
    if len(selected) < 10:
        raise ValueError("Too few validation sources; expand the preparation")
    # The small smoke metric explicitly reports missing validation sources.
    excluded = sorted(set(manifest["selected_sources"]) - set(selected))
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (task / "validation-tokenized.jsonl").write_bytes(data)
    contract = {
        "metric_name": "macro_validation_nll",
        "metric_direction": "minimize",
        "validation_sha256": hashlib.sha256(data).hexdigest(),
        "validation_sources": sorted(selected),
        "missing_validation_sources": excluded,
        "validation_rows": len(rows),
        "seq_length": 1024,
        "model": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "training_sources": manifest["selected_sources"],
    }
    (task / "validation-contract.json").write_text(json.dumps(contract, indent=2))
    statement = f"""Run a SMALL autonomous fine-tuning runtime test of Qwen3-4B on the 13 prepared legal SFT datasets.

This task replaces the prior LAB task. LAB execution, judging, Kubernetes task Pods,
OpenAI calls and LAB claims are outside this run. The objective is lower
macro_validation_nll, a fixed held-out teacher-forced SFT diagnostic. This smoke metric
uses {len(rows)} examples across {len(selected)} sources; missing sources: {excluded}.
It is deliberately tiny, not evidence of generalization or legal accuracy.

Training data: sft_train.jsonl (13 rows, one encodable example per source), a tiny
subset of the {manifest["counts"]["sft_train.jsonl"]} prepared rows recorded in manifest.json. Preserve the document-group
holdouts, source attribution and pinned revisions. Use all 13 sources; do not fetch
additional data, inspect other benchmarks, or train on validation answers.

SETUP: install requirements.lock through the exact framework Program Python. The
tested CUDA build is pinned there. Freeze the supplied scorer unchanged by writing
private/score.py as a small importlib wrapper around this task's validation_score.py
and return its score(output_dir). Keep validation-tokenized.jsonl, validation_score.py
and validation-contract.json outside public/. Publish the training JSONL, manifest,
inventory, train.py, export_model.py, requirements.lock and local model directory.
Create baseline_attempt/submission.json containing {{"kind":"base"}}. Verify the
supplied scorer on it; do not train a baseline adapter. Record minimize direction.
The supplied scorer loads weights on the GPU and can take a few minutes; this task
explicitly allows up to 8 minutes for scoring, overriding the generic seconds-only
scorer guideline. It never trains or calls a provider. Use the same frozen scorer for
every candidate. Do not replace it with candidate-reported metrics.json.

META: the goal is proving runtime operation on a small API budget. Author the simplest
legal sequential Search, one coder, one training Program and one score; at most a
second candidate if both money and time remain. Skip reports, literature research,
extra reviewers and broad hyperparameter sweeps. Preserve framework-required gates.
Your Search must use ScoreLaunch.measure(..., gpus=(0,)) (or the producer's exact GPU
indices). GPU placements do not reserve cards, so serialize all training, scoring and
export operations. Give each GPU Program at least 8 minutes and at most 12 GiB RAM.
Only the chosen model should be exported at the end.

The agent chooses learning rate and LoRA rank (maximum 32); train at most TWO optimizer
updates per candidate for this smoke test, sequence length 1024, accumulation 13, seed42.
Use the supplied train.py (a copied working version may change sampling to ensure all
13 sources are available, but never change frozen scoring). Run with --model DATA_DIR/model
--data DATA_DIR --steps 2 --accumulation 13 --seq-length 1024 --eval-rows 1. The training
wrapper expects sft_validation.jsonl: create an EMPTY file in the candidate's working
data directory, alongside a link/copy to PUBLIC sft_train.jsonl. Private scoring is
separate. Do not give the training role the frozen validation targets. Accumulation 13
on exactly 13 training rows visits all sources in each update. This is a tiny runtime
test, not full-dataset training. Record actual rows/sources visited.

Each candidate output has adapter/ with regular adapter_config.json and
adapter_model.safetensors, plus submission.json containing
{{"kind":"lora","adapter":"adapter"}}. The frozen evaluator recomputes NLL using
the frozen local base model and this adapter; never trust submitted scores. The base
submission is for SETUP only: final delivery must include an actual trained adapter,
even if it did not improve. Report any regression rather than claiming improvement.

FINALIZER: deliver selected adapter, independently loadable final_model (merge with
export_model.py), experiment commands, measured scores, source coverage and actual
API cost. Export uses the local DATA_DIR/model; the revision argument is
1cfa9a7208912126459214e8b04321603b3df60c. Save regular single-link files under deliverable/.
Do not spend time generating a paper. This GPU Pod has only 16GiB CPU RAM and 4 CPUs.
Within-Pod bubblewrap is disabled because namespaces are unavailable; use the dedicated
Pod as the process boundary. Do not claim private scorer isolation against malicious
same-UID code. Do not print environment variables, credentials or database URLs.
"""
    (task / "README.md").write_text(statement)
    cfg = yaml.safe_load(
        subprocess.check_output([str(args.agent_bin), "config"], text=True)
    )
    cfg["run"].update(
        task_name="qwen3-13-autonomous-smoke",
        data_root=str(task),
        playground_root=str(args.output.resolve() / "playground"),
        setup_from_scratch=False,
        task_prompt=None,
    )
    cfg["run"]["budget"].update(
        wall_clock_minutes=20,
        cost_usd=1.5,
        per_agent_cost_cap_usd=0.65,
        total_agent_cost_limit_usd=4.0,
    )
    cfg["llm"]["default"].update(model=args.controller, max_thinking_tokens=None, effort="low")
    cfg["llm"]["retry_ceiling_s"] = 60
    cfg["search"] = {"kind": "meta", "input": {"report": False}}
    shared = "Follow the small autonomous SFT task. No LAB. Use the supplied frozen validation scorer and metric. Keep all GPU work sequential. Target under $5 TOTAL API usage including setup/finalization; exploration admission is only $1.50. Do not print keys."
    cfg["llm"]["system_instructions"] = {
        r: shared for r in ["setup", "meta", "finalizer"]
    }
    cfg["llm"]["system_instructions"]["setup"] += (
        " Wrap the supplied scorer unchanged; use the local pretrained model, and the exact Program Python for requirements.lock. Publish no validation targets."
    )
    cfg["resources"]["run"].update(cpu_max_cores=4, memory_max_gb=14)
    cfg["resources"]["work_unit"].update(cpu_max_cores=2, memory_max_gb=12)
    cfg["resources"]["gpu"]["min_free_mib"] = 20480
    cfg["resources"]["sandbox"]["enable"] = False
    cfg["writer"]["enable"] = False
    cfg["mcps"] = {}
    for unit in cfg["work_units"]:
        unit["retry"] = 0
        unit["budget"] = {
            "kind": "fixed",
            "minutes": 12
            if unit["name"] in ["setup", "training", "finalizer", "score_program"]
            else 8,
        }
    (args.output / "agent.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    # Launcher runs under this dedicated account; task inputs remain root-owned.
    playground = args.output / "playground"
    playground.mkdir()
    if os.geteuid() == 0:
        import pwd

        account = pwd.getpwnam("aibuildai-runner")
        os.chown(playground, account.pw_uid, account.pw_gid)
    print(
        json.dumps(
            {"config": str(args.output / "agent.yaml"), "contract": contract}, indent=2
        )
    )


if __name__ == "__main__":
    main()
