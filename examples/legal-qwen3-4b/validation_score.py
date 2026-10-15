#!/usr/bin/env python3
"""Frozen teacher-forced adapter evaluator for an autonomous SFT smoke run."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def checked_adapter(output: Path, submission: dict[str, Any]):
    if submission == {"kind": "base"}:
        return None
    if submission.get("kind") != "lora" or set(submission) != {"kind", "adapter"}:
        raise ValueError(
            'Expected {"kind":"base"} or {"kind":"lora","adapter":"adapter"}'
        )
    relative = Path(submission["adapter"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Adapter must be inside the attempt directory")
    adapter = output / relative
    if adapter.is_symlink() or not adapter.is_dir():
        raise ValueError("Adapter must be an ordinary directory")
    if not adapter.resolve().is_relative_to(output.resolve()):
        raise ValueError("Adapter escapes attempt")
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        p = adapter / name
        if p.is_symlink() or not p.is_file() or p.stat().st_nlink != 1:
            raise ValueError(f"Adapter requires a regular single-link {name}")
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    if cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM":
        raise ValueError("Only causal-LM LoRA adapters are accepted")
    if cfg.get("auto_mapping") or cfg.get("modules_to_save") or cfg.get("use_dora"):
        raise ValueError("Custom modules and nonstandard adapters are not supported")
    if not 1 <= cfg.get("r", 0) <= 32:
        raise ValueError("LoRA rank must be 1..32")
    return adapter


def score(output_dir: str):
    root = Path(__file__).resolve().parent
    output = Path(output_dir).resolve()
    submission = json.loads((output / "submission.json").read_text())
    checked_adapter(output, submission)
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(s in k.upper() for s in ["KEY", "TOKEN", "SECRET"])
    }
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(root / "validation_score.py"),
            "--evaluate",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=480,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Frozen evaluator failed: " + completed.stderr[-2000:])
    report = json.loads((output / "validation-evaluation.json").read_text())
    return {
        "score": report["macro_validation_nll"],
        "components": report["per_source_nll"],
    }


def evaluate(output: Path):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "validation-contract.json").read_text())
    data = (root / "validation-tokenized.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != cfg["validation_sha256"]:
        raise ValueError("Frozen validation data changed")
    submission = json.loads((output / "submission.json").read_text())
    adapter = checked_adapter(output, submission)
    model = AutoModelForCausalLM.from_pretrained(
        str(root / "model"),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,
    )
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    totals, counts, rows = defaultdict(float), defaultdict(int), defaultdict(int)
    with torch.inference_mode():
        for line in data.splitlines():
            row = json.loads(line)
            batch = {
                k: torch.tensor([row[k]], dtype=torch.long, device="cuda")
                for k in ["input_ids", "labels"]
            }
            loss = model(**batch).loss.item()
            n = sum(x != -100 for x in row["labels"][1:])
            if n <= 0 or not math.isfinite(loss):
                raise ValueError("Invalid validation loss/target")
            totals[row["source"]] += loss * n
            counts[row["source"]] += n
            rows[row["source"]] += 1
    if set(totals) != set(cfg["validation_sources"]):
        raise ValueError("Validation source coverage changed")
    losses = {s: totals[s] / counts[s] for s in sorted(totals)}
    report = {
        "macro_validation_nll": sum(losses.values()) / len(losses),
        "per_source_nll": losses,
        "per_source_rows": dict(rows),
        "validation_sha256": cfg["validation_sha256"],
        "meaning": "Held-out teacher-forced NLL; not LAB or legal accuracy.",
    }
    (output / "validation-evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--score", type=Path)
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.evaluate.resolve())
    elif args.score:
        print(json.dumps(score(str(args.score))))
    else:
        parser.error("Use --evaluate or --score")
