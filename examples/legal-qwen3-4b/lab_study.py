#!/usr/bin/env python3
"""Freeze a LAB study, run local Qwen through the official harness, compare paired scores."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

PIN = "a15650fe74bb20b86f160aaabdad75b87a91b6e8"
HERE = Path(__file__).resolve().parent


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_repo(repo: Path):
    rev = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if rev != PIN:
        raise ValueError(f"LAB must be at {PIN}, found {rev}")


def freeze(args: argparse.Namespace):
    verify_repo(args.repo)
    by_area = defaultdict(lambda: defaultdict(list))
    for file in sorted((args.repo / "tasks").rglob("task.json")):
        task = file.parent.relative_to(args.repo / "tasks").as_posix()
        parts = task.split("/")
        family = "/".join(parts[:-1]) if parts[-1].startswith("scenario-") else task
        if parts[0] == "firm-knowledge":
            family = "firm-knowledge/shared-dms"  # One shared corpus stays on one side.
        by_area[parts[0]][family].append(task)
    rng = random.Random(args.seed)
    dev, test, excluded = [], [], []
    for area, families in sorted(by_area.items()):
        names = sorted(families)
        rng.shuffle(names)
        selected = set(names[: min(2, max(0, len(names) - 1))])
        for family in names:
            tasks = sorted(families[family])
            if family in selected:
                dev.append(tasks[0])
                excluded.extend(tasks[1:])
            else:
                test.extend(tasks)
    smoke = [
        next(t for t in dev if t.startswith(area + "/"))
        for area in ("contracts", "corporate-ma", "litigation-dispute-resolution")
    ]
    hashes = {
        t: sha(args.repo / "tasks" / t / "task.json")
        for t in sorted(dev + test + excluded)
    }
    plan = {
        "lab_revision": PIN,
        "seed": args.seed,
        "splits": {
            "smoke": smoke,
            "dev": sorted(dev),
            "test": sorted(test),
            "excluded_dev_family_variants": sorted(excluded),
        },
        "task_config_hashes": hashes,
        "judges": ["claude-sonnet-4-6", "gpt-5.5"],
        "settings": {
            "max_turns": 100,
            "temperature": 0.0,
            "max_output_tokens": 4096,
            "enable_thinking": False,
            "skills": "all official defaults",
            "sandbox_image": "lab-sandbox:latest",
        },
        "conditions": ["base", "pilot", "broad"],
        "protocol": "All LAB data excluded from training. Use dev for iteration, evaluate test once after fixing the recipe. Match training token budgets, sequence length, seed, optimizer, agent tools, skills, model endpoint, judges and task IDs. Report source inclusion and actual tokens; more datasets is an experiment, not an assumed improvement.",
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    if args.plan.exists():
        raise FileExistsError(
            "Plan already exists; use a new filename to avoid changing the frozen evaluation"
        )
    args.plan.write_text(json.dumps(plan, indent=2))
    print(
        json.dumps(
            {
                "tasks": len(hashes),
                "dev": len(dev),
                "test": len(test),
                "excluded_family_variants": len(excluded),
                "smoke": smoke,
            },
            indent=2,
        )
    )


def check(args: argparse.Namespace):
    report = {
        "python": sys.version,
        "podman": shutil.which("podman"),
        "pandoc": shutil.which("pandoc"),
        "anthropic_judge_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai_judge_key": bool(os.environ.get("OPENAI_API_KEY")),
        "local_url": os.environ.get("LAB_LOCAL_BASE_URL", "http://127.0.0.1:8001/v1"),
    }
    if report["podman"]:
        result = subprocess.run(
            ["podman", "info", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        report["podman_usable"] = result.returncode == 0
        report["podman_error"] = result.stderr.strip() if result.returncode else None
    if (args.repo / "lab_core").exists():
        verify_repo(args.repo)
        report["lab_revision"] = PIN
    print(json.dumps(report, indent=2))
    return report


def run(args: argparse.Namespace):
    verify_repo(args.repo)
    plan = json.loads(args.plan.read_text())
    if plan["lab_revision"] != PIN:
        raise ValueError("Wrong plan revision")
    if not re.fullmatch("[A-Za-z0-9_-]+", args.condition):
        raise ValueError(
            "Condition name must contain letters, digits, hyphens or underscores"
        )
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    tasks = plan["splits"][args.split]
    if args.limit:
        tasks = tasks[: args.limit]
    for task in tasks:
        if (
            sha(args.repo / "tasks" / task / "task.json")
            != plan["task_config_hashes"][task]
        ):
            raise ValueError(f"Frozen task changed: {task}")
    study = args.output / args.condition
    study.mkdir(parents=True, exist_ok=True)
    settings = {
        "plan_sha256": sha(args.plan),
        "condition": args.condition,
        "split": args.split,
        "model": args.model,
        "tasks": tasks,
        "scored": args.score,
        "lab_settings": plan["settings"],
        "judges": plan["judges"],
    }
    sys.path.insert(0, str(args.repo))
    from lab_core.sandbox.kubernetes import DEFAULT_K8S_IMAGE

    settings["execution_backend"] = {
        "backend": args.sandbox_backend,
        "image": os.environ.get("LAB_K8S_IMAGE", DEFAULT_K8S_IMAGE)
        if args.sandbox_backend == "kubernetes"
        else plan["settings"]["sandbox_image"],
        "namespace": os.environ.get("LAB_K8S_NAMESPACE")
        if args.sandbox_backend == "kubernetes"
        else None,
    }
    if not args.execute:
        print(json.dumps(settings, indent=2))
        print(
            "Dry plan only. Add --execute to run; --score also invokes the paid judges."
        )
        return
    if args.score:
        if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get(
            "OPENAI_API_KEY"
        ):
            raise RuntimeError(
                "Standard LAB dual judging needs ANTHROPIC_API_KEY and OPENAI_API_KEY"
            )
        if os.environ.get("OPENAI_BASE_URL"):
            raise RuntimeError(
                "Unset OPENAI_BASE_URL for official judges; use LAB_LOCAL_BASE_URL for the Qwen endpoint"
            )
    from urllib.request import urlopen

    base_url = os.environ.get("LAB_LOCAL_BASE_URL", "http://127.0.0.1:8001/v1")
    with urlopen(base_url.rstrip("/") + "/models", timeout=20) as response:
        served = json.load(response)
    if args.model not in {r["id"] for r in served["data"]}:
        raise ValueError("Endpoint is serving a different model name")
    settings["endpoint"] = base_url
    from urllib.error import HTTPError

    try:
        with urlopen(
            base_url.removesuffix("/v1").rstrip("/") + "/healthz", timeout=20
        ) as response:
            identity = json.load(response)
        settings["endpoint_identity"] = identity
        if (
            identity.get("max_output_tokens", plan["settings"]["max_output_tokens"])
            < plan["settings"]["max_output_tokens"]
        ):
            raise ValueError(
                "Local server output cap is smaller than the frozen LAB budget"
            )
    except HTTPError as exc:
        if exc.code != 404:
            raise
        settings["endpoint_identity"] = {
            "served_name": args.model,
            "note": "External endpoint has no identity route; archive model/checkpoint hashes separately.",
        }
    settings["execution_backend"]["harness_sha256"] = hashlib.sha256(
        b"".join(
            (args.repo / name).read_bytes()
            for name in [
                "lab_core/harness/run.py",
                "lab_core/harness/tools.py",
                "lab_core/harness/agent_loop.py",
                "lab_core/sandbox/sandbox.py",
                "lab_core/sandbox/kubernetes.py",
            ]
        )
    ).hexdigest()
    previous = study / "study.json"
    if previous.exists() and json.loads(previous.read_text()) != settings:
        raise ValueError(
            "Condition settings or checkpoint identity changed; use a new condition name"
        )
    # Actual harness keeps task.json/rubrics outside the mounted workspace.
    sys.path.insert(0, str(args.repo))
    from lab_core.harness import run as harness
    from lab_adapter import LocalChatAdapter

    harness.create_adapter = lambda model, temperature=0.0, reasoning_effort=None: (
        LocalChatAdapter(model.split("/", 1)[1], temperature, reasoning_effort)
    )
    os.environ["LAB_MAX_OUTPUT_TOKENS"] = str(plan["settings"]["max_output_tokens"])
    (study / "study.json").write_text(json.dumps(settings, indent=2))
    rows = []
    for task in tasks:
        run_id = f"qwen3-study/{args.condition}/{args.split}/{hashlib.sha256(task.encode()).hexdigest()[:16]}"
        results = args.repo / "results" / run_id
        if results.exists() and not args.resume:
            raise FileExistsError(
                f"Existing run {run_id}; use a new condition or --resume"
            )
        row = {"task": task, "run_id": run_id, "status": "error"}
        try:
            if not (args.resume and (results / "metrics.json").exists()):
                parsed = harness.parser.parse_args(
                    [
                        "--model",
                        f"local/{args.model}",
                        "--task",
                        task,
                        "--run-id",
                        run_id,
                        "--sandbox-backend",
                        args.sandbox_backend,
                        "--max-turns",
                        str(plan["settings"]["max_turns"]),
                        "--temperature",
                        str(plan["settings"]["temperature"]),
                    ]
                )
                harness.main(parsed)
            if args.score and not (
                args.resume and (results / "scores_dual.json").exists()
            ):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lab_core.evaluation.run_eval",
                        "--run-id",
                        run_id,
                        "--task",
                        task,
                        "--judges",
                        *plan["judges"],
                        "--parallel",
                        "2",
                    ],
                    cwd=args.repo,
                    check=True,
                )
            row["status"] = "scored" if args.score else "generated_unscored"
            if args.score:
                scores = json.loads((results / "scores_dual.json").read_text())
                row.update(
                    all_pass=scores["dual_all_pass_rate"],
                    criterion_pass=scores["dual_criterion_pass"],
                )
            row["metrics"] = json.loads((results / "metrics.json").read_text())
            if (results / "sandbox-runtime.json").exists():
                row["sandbox_runtime"] = json.loads(
                    (results / "sandbox-runtime.json").read_text()
                )
        except Exception as exc:  # noqa: BLE001 -- isolate and record third-party loader/runtime failures
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        (study / "results.json").write_text(json.dumps(rows, indent=2))
    errors = sum(r["status"] == "error" for r in rows)
    print(json.dumps({"completed": len(rows), "errors": errors}, indent=2))
    return 1 if errors else 0


def compare(args: argparse.Namespace):
    baseline, candidate = args.baseline, args.candidate
    a_cfg, b_cfg = [
        json.loads((p / "study.json").read_text()) for p in (baseline, candidate)
    ]
    for key in ("plan_sha256", "split", "tasks", "lab_settings", "judges"):
        if a_cfg[key] != b_cfg[key]:
            raise ValueError(f"Incomparable studies: {key} differs")
    if a_cfg.get("execution_backend") != b_cfg.get("execution_backend"):
        raise ValueError("Incomparable studies: execution backend differs")
    a, b = [
        {r["task"]: r for r in json.loads((p / "results.json").read_text())}
        for p in (baseline, candidate)
    ]
    expected = a_cfg["tasks"]
    if not expected:
        raise ValueError("No tasks to compare")
    missing = [
        t
        for t in expected
        if a.get(t, {}).get("status") != "scored"
        or b.get(t, {}).get("status") != "scored"
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} tasks missing scores or failed; do not drop failures from denominator"
        )
    if a_cfg.get("execution_backend", {}).get("backend") == "kubernetes":
        for task in expected:
            image_a = a[task].get("sandbox_runtime", {}).get("image_id")
            image_b = b[task].get("sandbox_runtime", {}).get("image_id")
            if not image_a or image_a != image_b:
                raise ValueError("Missing or different Kubernetes sandbox image IDs")
    deltas = [b[t]["all_pass"] - a[t]["all_pass"] for t in expected]
    by_area = defaultdict(list)
    for task, delta in zip(expected, deltas):
        by_area[task.split("/")[0]].append(delta)
    rng = random.Random(42)
    # Cluster by family; scenarios are correlated. Firm knowledge shares one DMS.
    clusters = defaultdict(list)
    for task, delta in zip(expected, deltas):
        family = task.rsplit("/scenario-", 1)[0] if "/scenario-" in task else task
        if task.startswith("firm-knowledge/"):
            family = "firm-knowledge/shared-dms"
        clusters[family].append(delta)
    values = list(clusters.values())
    boot = []
    for _ in range(2000):
        sampled = [x for _ in values for x in rng.choice(values)]
        boot.append(sum(sampled) / len(sampled))
    boot.sort()
    result = {
        "tasks": len(expected),
        "families": len(values),
        "baseline_all_pass": sum(a[t]["all_pass"] for t in expected) / len(expected),
        "candidate_all_pass": sum(b[t]["all_pass"] for t in expected) / len(expected),
        "paired_all_pass_delta": sum(deltas) / len(deltas),
        "baseline_criterion_pass": sum(a[t]["criterion_pass"] for t in expected)
        / len(expected),
        "candidate_criterion_pass": sum(b[t]["criterion_pass"] for t in expected)
        / len(expected),
        "family_cluster_bootstrap_95pct": [boot[50], boot[1949]],
        "per_area_delta": {k: sum(v) / len(v) for k, v in by_area.items()},
        "limitation": "Small smoke/dev subsets do not establish overall LAB improvement. Examine document/tool failures as well as rubric scores.",
    }
    print(json.dumps(result, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("freeze", "check", "run"):
        q = sub.add_parser(command)
        q.add_argument("--repo", type=Path, default=HERE.parents[2] / "harvey-labs")
        q.add_argument(
            "--plan",
            type=Path,
            default=HERE.parents[2] / "legal-runs" / "lab-plan.json",
        )
        q.add_argument("--seed", type=int, default=42)
        if command == "run":
            q.add_argument(
                "--sandbox-backend", choices=["podman", "kubernetes"], default="podman"
            )
            q.add_argument("--condition", required=True)
            q.add_argument(
                "--model", required=True, help="Exact served name from /v1/models"
            )
            q.add_argument("--split", choices=["smoke", "dev", "test"], default="smoke")
            q.add_argument("--limit", type=int, default=0)
            q.add_argument(
                "--output",
                type=Path,
                default=HERE.parents[2] / "legal-runs" / "lab-study",
            )
            q.add_argument("--execute", action="store_true")
            q.add_argument("--score", action="store_true")
            q.add_argument("--resume", action="store_true")
    q = sub.add_parser("compare")
    q.add_argument("--baseline", type=Path, required=True)
    q.add_argument("--candidate", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"freeze": freeze, "check": check, "run": run, "compare": compare}[
        args.command
    ](args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
