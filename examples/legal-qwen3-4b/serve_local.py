#!/usr/bin/env python3
"""Single GPU Qwen Chat Completions endpoint for LAB smoke tests; no vLLM required."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid


def parse_tools(text: str, definitions: list[dict]):
    allowed = {t["function"]["name"] for t in definitions}
    calls = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        value = json.loads(match.group(1))
        if value.get("name") not in allowed or not isinstance(
            value.get("arguments"), dict
        ):
            raise ValueError("Unknown tool or malformed arguments")
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:12],
                "type": "function",
                "function": {
                    "name": value["name"],
                    "arguments": json.dumps(value["arguments"]),
                },
            }
        )
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return content, calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--revision")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--name", default="qwen3-base")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--max-output-tokens", type=int, default=4096)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastapi import FastAPI, HTTPException
    from peft import PeftModel
    import uvicorn
    from huggingface_hub import HfApi

    revision = args.revision
    if not Path(args.model).exists() and revision is None:
        revision = HfApi().model_info(args.model).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    lock = threading.Lock()
    app = FastAPI()
    adapter_hash = None
    if args.adapter:
        h = hashlib.sha256()
        with (args.adapter / "adapter_model.safetensors").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        adapter_hash = h.hexdigest()
    identity = {
        "adapter_sha256": adapter_hash,
        "model": args.model,
        "revision": revision,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "served_name": args.name,
        "max_context": args.max_context,
        "max_output_tokens": args.max_output_tokens,
        "enable_thinking": False,
    }

    @app.get("/healthz")
    def health():
        return identity

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": args.name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def complete(body: dict):
        if body.get("model") != args.name:
            raise HTTPException(404, "Served model name does not match request")
        if body.get("stream"):
            raise HTTPException(
                400, "Streaming is not implemented in this smoke endpoint"
            )
        max_tokens = min(int(body.get("max_tokens", 1024)), args.max_output_tokens)
        if max_tokens < 1:
            raise HTTPException(400, "max_tokens must be positive")
        tools = body.get("tools", [])
        prompt = tokenizer.apply_chat_template(
            body["messages"],
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            "cuda"
        )
        length = inputs["input_ids"].shape[1]
        if length + max_tokens > args.max_context:
            # Never silently remove matter documents or the system's tool instructions.
            raise HTTPException(
                400,
                f"Context {length} + output {max_tokens} exceeds {args.max_context}",
            )
        temperature = float(body.get("temperature", 0))
        kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with lock, torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        generated = output[0, length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        try:
            content, calls = parse_tools(text, tools)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"Invalid model tool call: {exc}") from exc
        message = {"role": "assistant", "content": content or None}
        if calls:
            message["tool_calls"] = calls
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                    if calls
                    else "length"
                    if len(generated) >= max_tokens
                    else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": length,
                "completion_tokens": len(generated),
                "total_tokens": length + len(generated),
            },
        }

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
