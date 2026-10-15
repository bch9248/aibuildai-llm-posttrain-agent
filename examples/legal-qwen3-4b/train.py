#!/usr/bin/env python3
"""Bounded BF16 LoRA baseline; records actual tokens, truncation, and validation NLL."""

from typing import Any
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import time


def encode(row: Any, tokenizer: Any, seq_length: Any, objective: Any):
    if objective == "cpt":
        ids = tokenizer.encode(row["text"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        return {"input_ids": ids[:seq_length], "labels": ids[:seq_length]}, len(
            ids
        ) > seq_length
    messages = row["messages"]
    # Train the final assistant answer; earlier turns supply context only.
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if len(target_ids) >= seq_length - 32:
        return None, False  # Never silently discard the entire assistant target.
    available = seq_length - len(target_ids)
    truncated = len(prompt_ids) > available
    if truncated:
        # Preserve task prefix and question/schema at the end. Explicit pilot compromise.
        head = min(available // 3, 256)
        marker = tokenizer.encode(
            "\n[Context shortened for training]\n", add_special_tokens=False
        )
        tail = available - head - len(marker)
        if tail <= 0:
            return None, False
        prompt_ids = prompt_ids[:head] + marker + prompt_ids[-tail:]
    ids = prompt_ids + target_ids
    return {
        "input_ids": ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }, truncated


def read_examples(path: Any, tokenizer: Any, args: Any, stats: Any):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            example, truncated = encode(row, tokenizer, args.seq_length, args.objective)
            if example is None or len(example["input_ids"]) < 2:
                stats["skipped_long_targets_or_empty"] += 1
                continue
            stats["truncated_prompts_or_cpt_documents"] += truncated
            example["source"] = row["source"]
            yield example


def tensor_batch(example: Any, device: Any):
    import torch

    return {
        k: torch.tensor([example[k]], dtype=torch.long, device=device)
        for k in ("input_ids", "labels")
    }


def evaluate(model: Any, tokenizer: Any, path: Any, args: Any):
    from collections import Counter
    import torch

    totals, counts, per_source = defaultdict(float), defaultdict(int), Counter()
    stats = defaultdict(int)
    model.eval()
    with torch.no_grad():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                source = row["source"]
                if args.eval_rows > 0 and per_source[source] >= args.eval_rows:
                    continue
                example, truncated = encode(
                    row, tokenizer, args.seq_length, args.objective
                )
                if example is None or len(example["input_ids"]) < 2:
                    stats["skipped_long_targets_or_empty"] += 1
                    continue
                stats["truncated_prompts_or_cpt_documents"] += truncated
                n = sum(x != -100 for x in example["labels"][1:])
                loss = model(**tensor_batch(example, "cuda")).loss.item()
                if not math.isfinite(loss):
                    raise RuntimeError("Nonfinite validation loss")
                totals[source] += loss * n
                counts[source] += n
                per_source[source] += 1
    losses = {s: totals[s] / counts[s] for s in totals}
    return {
        "macro_validation_nll": sum(losses.values()) / len(losses) if losses else None,
        "per_source_nll": losses,
        "per_source_rows": dict(per_source),
        "tokenization": dict(stats),
        "meaning": "Teacher-forced held-out data loss; not a LAB score or legal-accuracy measurement.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision")
    p.add_argument("--init-adapter", type=Path)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="Optional processed-token ceiling, shared across ablations; may overshoot by one optimizer update.",
    )
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-rows", type=int, default=8)
    p.add_argument("--objective", choices=["sft", "cpt"], default="sft")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    if (
        args.steps < 1
        or args.accumulation < 1
        or args.seq_length < 64
        or args.token_budget < 0
    ):
        p.error("Invalid training limits")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training recipe")
    set_seed(args.seed)
    revision = args.revision
    if not Path(args.model).exists() and not revision:
        revision = HfApi().model_info(args.model).sha
    config = {
        **vars(args),
        "revision": revision,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.output / "training_config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    start = time.monotonic()
    val = args.data / f"{args.objective}_validation.jsonl"
    baseline = evaluate(model, tokenizer, val, args)
    (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2))
    if args.eval_only:
        return
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank * 2,
                lora_dropout=0.05,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    trainable = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    stats = defaultdict(int)
    processed_tokens = supervised_tokens = epochs = 0
    train_path = args.data / f"{args.objective}_train.jsonl"

    def batches():
        nonlocal epochs
        rng = random.Random(args.seed)
        while True:
            examples = read_examples(train_path, tokenizer, args, stats)
            seen = 0
            while chunk := list(itertools.islice(examples, 128)):
                rng.shuffle(chunk)
                for item in chunk:
                    seen += 1
                    yield item
            if not seen:
                raise RuntimeError("No usable training examples after tokenization")
            epochs += 1

    iterator = batches()
    logpath = args.output / "training.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with logpath.open("w") as log:
        for step in range(1, args.steps + 1):
            losses = []
            for _ in range(args.accumulation):
                example = next(iterator)
                batch = tensor_batch(example, "cuda")
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / args.accumulation).backward()
                losses.append(loss.detach().item())
                processed_tokens += len(example["input_ids"])
                supervised_tokens += sum(x != -100 for x in example["labels"][1:])
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {
                "step": step,
                "loss": sum(losses) / len(losses),
                "processed_tokens": processed_tokens,
                "supervised_tokens": supervised_tokens,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if args.token_budget and processed_tokens >= args.token_budget:
                break
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    final = evaluate(model, tokenizer, val, args)
    summary = {
        "steps": step,
        "epochs_completed": epochs,
        "processed_tokens": processed_tokens,
        "supervised_tokens": supervised_tokens,
        "elapsed_seconds": time.monotonic() - start,
        "baseline": baseline,
        "final": final,
        "tokenization": dict(stats),
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "trainable_parameters": sum(v.numel() for v in trainable),
        "lab_evaluation": "pending; no LAB score inferred from this smoke run",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(
            args.output / "final_model", safe_serialization=True, max_shard_size="2GB"
        )
        tokenizer.save_pretrained(args.output / "final_model")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
