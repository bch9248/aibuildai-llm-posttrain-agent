#!/usr/bin/env python3
"""One-command legal data inventory, streaming preparation, and Qwen LoRA runs."""

from typing import Any
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from adapters import convert, digest
from recipes import decision, recipes, PILOT

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "catalog.json").read_text())["datasets"]


def dump(path: Any, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def load_rows(spec: Any, split: Any):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url

    kwargs = dict(split=split, streaming=True)
    if "direct" in spec:
        cfg = spec["direct"]
        url = hf_hub_url(
            spec["id"],
            cfg["files"][split],
            repo_type="dataset",
            revision=spec["revision"],
        )
        dataset = load_dataset(cfg["builder"], data_files={split: url}, **kwargs)
        if spec["id"] == "casehold/casehold":

            def canonical(row: Any):
                return {
                    "example_id": row["Unnamed: 0"],
                    "citing_prompt": row["0"],
                    **{f"holding_{i}": row[str(i + 1)] for i in range(5)},
                    "label": row["11"],
                }

            dataset = dataset.map(canonical)
        return dataset
    if spec.get("upstream_zip"):
        import hashlib
        import zipfile
        import ijson
        import requests

        lock = spec["upstream_zip"]
        cache = HERE.parents[2] / "legal-runs" / "upstream"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "cuad-data.zip"

        def checksum(path: Path):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        if not archive.exists() or checksum(archive) != lock["sha256"]:
            temporary = archive.with_suffix(".tmp")
            with requests.get(lock["url"], stream=True, timeout=60) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if checksum(temporary) != lock["sha256"]:
                raise ValueError("CUAD upstream archive checksum mismatch")
            temporary.replace(archive)

        def cuad_rows():
            if split != "train":
                raise ValueError("This training wrapper never opens CUAD test.json")
            with (
                zipfile.ZipFile(archive) as z,
                z.open("train_separate_questions.json") as stream,
            ):
                for document in ijson.items(stream, "data.item"):
                    for paragraph in document["paragraphs"]:
                        for qa in paragraph["qas"]:
                            yield {
                                "title": document["title"],
                                "context": paragraph["context"],
                                "question": qa["question"],
                                "answers": {"text": [a["text"] for a in qa["answers"]]},
                            }

        return cuad_rows()
    if spec.get("zip_file"):
        import zipfile
        from huggingface_hub import hf_hub_download

        archive = hf_hub_download(
            spec["id"], spec["zip_file"], revision=spec["revision"], repo_type="dataset"
        )

        def archive_rows():
            with zipfile.ZipFile(archive) as z, z.open(spec["zip_member"]) as f:
                for line in f:
                    row = json.loads(line)
                    if row["subset"] == ("dev" if split == "validation" else split):
                        if isinstance(row["label"], str):
                            row["label"] = [
                                "contradiction",
                                "entailment",
                                "neutral",
                            ].index(row["label"])
                        yield row

        return archive_rows()
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]
    return load_dataset(
        spec["id"],
        spec["config"],
        revision=spec["revision"],
        trust_remote_code=False,
        **kwargs,
    )


def worker(args: Any):
    spec = json.loads(args.spec.read_text())
    start = time.monotonic()
    result = {"spec": spec, "status": "error", "rows": 0, "filtered": 0, "errors": []}
    try:
        ds = load_rows(spec, args.split)
        # Preserve explicit ClassLabel names rather than inventing mappings.
        feat = (getattr(ds, "features", None) or {}).get(
            "labels", (getattr(ds, "features", None) or {}).get("label")
        )
        feat = getattr(feat, "feature", feat)
        if hasattr(feat, "names"):
            spec["label_names"] = feat.names
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        with args.rows.open("w") as output:
            examined = 0
            for row in ds:
                examined += 1
                try:
                    item = convert(row, spec)
                    if item is None:
                        result["filtered"] += 1
                    else:
                        item.update(
                            source=spec["id"],
                            config=spec["config"],
                            revision=spec["revision"],
                            original_split=args.split,
                        )
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
                        result["rows"] += 1
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    result["errors"].append(f"{type(exc).__name__}: {exc}")
                    if len(result["errors"]) >= 10:
                        raise ValueError(
                            "Ten row conversion errors; refusing silent data loss"
                        ) from exc
                if args.max_rows > 0 and examined >= args.max_rows:
                    break
        result["examined"] = examined
        result["columns"] = list(row) if examined else []
        result["status"] = (
            "ready"
            if result["rows"] and not result["errors"]
            else "invalid_rows"
            if result["errors"]
            else "empty"
        )
    except Exception as exc:  # noqa: BLE001 -- isolate and record third-party loader/runtime failures
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    dump(args.result, result)
    return 0 if result["status"] == "ready" else 1


def inspect_source(card: Any):
    from huggingface_hub import HfApi

    try:
        if card["id"] == "harveyai/harvey-labs":
            rev = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(HERE.parents[2] / "harvey-labs"),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip()
            return {
                "status": "pinned" if rev == card["commit"] else "wrong_revision",
                "revision": rev,
            }
        info = HfApi().dataset_info(card["id"], revision=card["commit"], timeout=20)
        return {"status": "pinned", "revision": info.sha, "files": len(info.siblings)}
    except Exception as exc:  # noqa: BLE001 -- isolate and record third-party loader/runtime failures
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def prepare(args: Any, probe: Any = False):
    from concurrent.futures import ThreadPoolExecutor

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = []
    # Network metadata checks only; evaluation-only data rows are never opened.
    metadata = {}
    if probe:
        with ThreadPoolExecutor(max_workers=4) as pool:
            metadata = dict(
                zip([c["id"] for c in CATALOG], pool.map(inspect_source, CATALOG))
            )
    selected = args.datasets or (PILOT if args.profile == "pilot" else None)
    unknown = set(selected or []) - {c["id"] for c in CATALOG}
    if unknown:
        raise ValueError(f"Unknown dataset IDs: {sorted(unknown)}")
    for card in sorted(CATALOG, key=lambda c: (c["id"] not in PILOT, c["id"])):
        status, reason = decision(card, args.include_cpt)
        record = {
            "id": card["id"],
            "revision": card["commit"],
            "kind": card["kind"],
            "use": card["use"],
            "decision": status,
            "reason": reason,
            "holdout_rules": card["holdout_rules"],
            "traps": card["traps"],
            "card_sha256": card["card_sha256"],
            "attempts": [],
        }
        if probe:
            record["hub_check"] = metadata[card["id"]]
        run = (probe and card["use"] != "eval") or (
            status == "candidate" and (selected is None or card["id"] in selected)
        )
        record["selected_for_preparation"] = run and not probe
        if run:
            try:
                specs = recipes(card, args.all_configs)
                if not specs:
                    record["loader_status"] = "needs_wrapper"
                for spec in specs:
                    splits = [spec["split"]]
                    if not probe and spec.get("validation_split"):
                        splits.append(spec["validation_split"])
                    for split in splits:
                        spec_key = digest(
                            json.dumps(
                                [
                                    spec,
                                    split,
                                    args.max_rows,
                                    digest(
                                        (HERE / "adapters.py").read_text()
                                        + (HERE / "pipeline.py").read_text()
                                    ),
                                ],
                                sort_keys=True,
                            )
                        )[:20]
                        stage = out / "staging" / spec_key
                        stage.mkdir(parents=True, exist_ok=True)
                        spec_path, row_path, result_path = (
                            stage / "spec.json",
                            stage / "rows.jsonl",
                            stage / "result.json",
                        )
                        dump(spec_path, spec)
                        if not (
                            args.resume
                            and result_path.exists()
                            and json.loads(result_path.read_text()).get("status")
                            == "ready"
                            and row_path.exists()
                        ):
                            command = [
                                sys.executable,
                                str(HERE / "pipeline.py"),
                                "_worker",
                                "--spec",
                                str(spec_path),
                                "--split",
                                split,
                                "--rows",
                                str(row_path),
                                "--result",
                                str(result_path),
                                "--max-rows",
                                str(args.max_rows),
                            ]
                            with (stage / "loader.log").open("w") as log:
                                try:
                                    subprocess.run(
                                        command,
                                        stdout=log,
                                        stderr=log,
                                        timeout=args.timeout,
                                        check=False,
                                        env={
                                            **os.environ,
                                            "HF_HUB_DOWNLOAD_TIMEOUT": "20",
                                            "HF_HUB_ETAG_TIMEOUT": "20",
                                        },
                                    )
                                except subprocess.TimeoutExpired:
                                    dump(
                                        result_path,
                                        {
                                            "status": "timeout",
                                            "error": f"Loader exceeded {args.timeout}s",
                                            "spec": spec,
                                        },
                                    )
                            if not result_path.exists():
                                dump(
                                    result_path,
                                    {
                                        "status": "crashed",
                                        "error": "Worker exited without result; inspect loader.log",
                                        "spec": spec,
                                    },
                                )
                        attempt = json.loads(result_path.read_text())
                        attempt.update(
                            rows_path=str(row_path), log_path=str(stage / "loader.log")
                        )
                        record["attempts"].append(attempt)
                        print(
                            f"{card['id']} {spec['config'] or 'default'}/{split}: {attempt['status']} ({attempt.get('rows', 0)} rows)",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001 -- isolate and record third-party loader/runtime failures
                record["loader_status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
        inventory.append(record)
        dump(
            out / "inventory.json",
            {
                "mode": "probe" if probe else "prepare",
                "profile": args.profile,
                "max_rows_per_config_split": args.max_rows,
                "all_configs": args.all_configs,
                "datasets": inventory,
            },
        )
    if probe:
        return 0
    return combine(out, inventory, args)


def combine(out: Any, inventory: Any, args: Any):
    # Disk-backed dedup prevents corpus size from determining RAM use.
    dbpath = out / "dedup.sqlite"
    if dbpath.exists():
        dbpath.unlink()
    db = sqlite3.connect(dbpath)
    db.execute(
        "CREATE TABLE examples (key TEXT PRIMARY KEY, grp TEXT, split TEXT, objective TEXT, payload TEXT)"
    )
    db.execute("CREATE INDEX groups ON examples(grp)")
    db.execute("CREATE TABLE holdouts (grp TEXT PRIMARY KEY)")
    failures, duplicates = [], 0
    for rec in inventory:
        if rec["decision"] != "candidate":
            continue
        if rec.get("selected_for_preparation") and not rec["attempts"]:
            failures.append(
                {
                    "id": rec["id"],
                    "status": rec.get("loader_status", "error"),
                    "error": rec.get("error", "Selected source produced no attempts"),
                }
            )
        for attempt in rec["attempts"]:
            if attempt["status"] != "ready":
                failures.append(
                    {
                        "id": rec["id"],
                        "status": attempt["status"],
                        "error": attempt.get("error"),
                    }
                )
                continue
            spec = attempt["spec"]
            with Path(attempt["rows_path"]).open() as rows_stream:
                for line in rows_stream:
                    item = json.loads(line)
                    original = item["original_split"]
                    # Canonical input hashes also protect identical prompts across sources.
                    value = json.dumps(
                        item.get("messages", item.get("text")),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    key = digest(value)
                    is_val = original == spec.get("validation_split") or (
                        not spec.get("validation_split")
                        and int(item["group"][:8], 16) % 10 == 0
                    )
                    split = "validation" if is_val else "train"
                    if is_val:
                        db.execute(
                            "INSERT OR IGNORE INTO holdouts VALUES (?)",
                            (item["group"],),
                        )
                    cur = db.execute(
                        "INSERT OR IGNORE INTO examples VALUES (?, ?, ?, ?, ?)",
                        (
                            key,
                            item["group"],
                            split,
                            item["objective"],
                            json.dumps(item, ensure_ascii=False),
                        ),
                    )
                    duplicates += cur.rowcount == 0
    db.commit()
    # When one config/source validates a document, remove every training row of it.
    removed = db.execute(
        "DELETE FROM examples WHERE split='train' AND grp IN (SELECT grp FROM holdouts)"
    ).rowcount
    db.commit()
    counts = {}
    for objective in ("sft", "cpt"):
        for split in ("train", "validation"):
            filename = f"{objective}_{split}.jsonl"
            count = 0
            with (out / filename).open("w") as f:
                for (payload,) in db.execute(
                    "SELECT payload FROM examples WHERE objective=? AND split=? ORDER BY key",
                    (objective, split),
                ):
                    f.write(payload + "\n")
                    count += 1
            counts[filename] = count
    per_source = {}
    for payload, split in db.execute(
        "SELECT payload, split FROM examples WHERE objective='sft'"
    ):
        source = json.loads(payload)["source"]
        per_source.setdefault(source, {"train": 0, "validation": 0})[split] += 1
    db.close()
    manifest = {
        "counts": counts,
        "per_source_rows": per_source,
        "duplicates_removed": duplicates,
        "training_rows_removed_for_validation_document_overlap": removed,
        "failures": failures,
        "selected_sources": [
            r["id"]
            for r in inventory
            if r["decision"] == "candidate"
            and any(a["status"] == "ready" for a in r["attempts"])
        ],
        "seed": args.seed,
        "held_out_tests_loaded": False,
        "limits": "Bounded preparation only checks document overlap among the rows actually loaded; it does not certify full-corpus decontamination against LAB or other benchmarks.",
    }
    dump(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failures and not args.allow_partial else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "probe", "prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument(
            "--output", type=Path, default=HERE.parents[2] / "legal-runs" / "pilot"
        )
        p.add_argument("--profile", choices=["pilot", "broad"], default="pilot")
        p.add_argument("--datasets", nargs="+")
        p.add_argument(
            "--max-rows",
            type=int,
            default=1000,
            help="Per config/split source rows; -1 reads all. Probe defaults to 3.",
        )
        p.add_argument(
            "--all-configs",
            action="store_true",
            help="All Lawma/LexGLUE configs, otherwise representative configs only.",
        )
        p.add_argument("--include-cpt", action="store_true")
        p.add_argument("--timeout", type=int, default=180)
        p.add_argument("--resume", action="store_true")
        p.add_argument(
            "--allow-partial",
            action="store_true",
            help="Permit training with failed selected sources excluded and recorded.",
        )
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--steps", type=int, default=100)
        p.add_argument("--token-budget", type=int, default=0)
        p.add_argument("--seq-length", type=int, default=2048)
        p.add_argument("--model", default="Qwen/Qwen3-4B")
        p.add_argument("--model-revision", default=None)
        p.add_argument("--merge", action="store_true")
    w = sub.add_parser("_worker")
    for name in ("spec", "rows", "result"):
        w.add_argument("--" + name, type=Path, required=True)
    w.add_argument("--split", required=True)
    w.add_argument("--max-rows", type=int, required=True)
    args = parser.parse_args()
    if args.command == "_worker":
        code = worker(args)
        # Streaming libraries can retain download threads after a bounded take.
        # All output files are closed and the result is atomically published.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    if args.max_rows == 0 or args.max_rows < -1:
        parser.error("--max-rows must be positive or -1")
    if args.command == "inventory":
        dump(
            args.output / "inventory.json",
            {
                "datasets": [
                    {
                        **c,
                        "decision": decision(c, args.include_cpt)[0],
                        "reason": decision(c, args.include_cpt)[1],
                    }
                    for c in CATALOG
                ]
            },
        )
        return 0
    if args.command == "probe":
        if "--max-rows" not in sys.argv:
            args.max_rows = 3
        return prepare(args, probe=True)
    result = prepare(args)
    if result or args.command != "run":
        return result
    model_lock = HERE.parents[2] / "legal-runs" / "model-lock.json"
    revision = args.model_revision
    if revision is None and model_lock.exists():
        lock = json.loads(model_lock.read_text())
        if lock["model"] == args.model:
            revision = lock["revision"]
    if not (args.output / "sft_train.jsonl").stat().st_size:
        raise ValueError(
            "No eligible SFT training rows; inspect the inventory and selected dataset IDs"
        )
    for objective in ("cpt", "sft"):
        if objective == "cpt" and not args.include_cpt:
            continue
        if not (args.output / f"{objective}_train.jsonl").stat().st_size:
            continue
        command = [
            sys.executable,
            str(HERE / "train.py"),
            "--data",
            str(args.output),
            "--output",
            str(args.output / objective),
            "--model",
            args.model,
            "--steps",
            str(args.steps),
            "--seq-length",
            str(args.seq_length),
            "--objective",
            objective,
            "--seed",
            str(args.seed),
            "--token-budget",
            str(args.token_budget),
        ]
        if revision:
            command += ["--revision", revision]
        if objective == "sft" and (args.output / "cpt" / "adapter").exists():
            command += ["--init-adapter", str(args.output / "cpt" / "adapter")]
        if args.merge and objective == "sft":
            command.append("--merge")
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
