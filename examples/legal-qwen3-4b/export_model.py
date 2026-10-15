#!/usr/bin/env python3
"""Merge a saved legal LoRA adapter and verify that the standalone model reloads."""

import argparse
import gc
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Choose a new output directory; existing exports are preserved"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.save_pretrained(args.output)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.output,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    inputs = tokenizer("Legal contract", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = reloaded(**inputs).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("Reloaded model produced nonfinite logits")
    report = {
        "model": args.model,
        "revision": args.revision,
        "adapter": str(args.adapter.resolve()),
        "output": str(args.output.resolve()),
        "standalone_reload": "passed",
        "logits_shape": list(logits.shape),
    }
    (args.output / "export_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
