#!/usr/bin/env python3
"""Check Kubernetes LAB permissions or run a synthetic sandbox fixture (no model API)."""

import argparse
import json
from pathlib import Path
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab-repo", type=Path, default=Path("/workspace/AIBuildAI/harvey-labs")
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--kubeconfig")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lab_repo))
    from lab_core.sandbox.kubernetes import KubernetesSandbox

    report = {"namespace": args.namespace, "paid_api_calls": 0, "live_pod_test": False}
    try:
        with tempfile.TemporaryDirectory(prefix="lab-k8s-check-") as tmp:
            root = Path(tmp)
            docs = root / "documents"
            docs.mkdir()
            (docs / "matter.txt").write_text(
                "Synthetic test matter: the amount is 123 USD."
            )
            sandbox = KubernetesSandbox(
                documents_dir=docs,
                output_dir=root / "output",
                workspace_dir=root / "workspace",
                namespace=args.namespace,
                kubeconfig=args.kubeconfig,
            )
            sandbox.preflight()
            report["permissions_and_policy"] = "passed"
            if args.fixture:
                with sandbox:
                    if (
                        sandbox.read_file("/workspace/documents/matter.txt")
                        != (docs / "matter.txt").read_bytes()
                    ):
                        raise RuntimeError("Document transfer failed")
                    blocked = sandbox.exec(
                        "echo changed >> /workspace/documents/matter.txt"
                    )
                    if blocked.ok:
                        raise RuntimeError("Document mount is writable")
                    sandbox.write_file(
                        "/workspace/output/result.md", "# Synthetic result"
                    )
                    result = sandbox.exec(
                        "python -c \"from docx import Document; d=Document(); d.add_paragraph('Synthetic result'); d.save('/workspace/output/result.docx')\""
                    )
                    if not result.ok:
                        raise RuntimeError("Document generation failed")
                    version = sandbox.exec("pandoc --version")
                    if not version.ok or not version.stdout.startswith("pandoc 3.11"):
                        raise RuntimeError("Sandbox must match the grading host's Pandoc 3.11")
                    report["sandbox_pandoc"] = version.stdout.splitlines()[0]
                    timed = sandbox.exec("sleep 10", timeout=1)
                    if not timed.timed_out:
                        raise RuntimeError("Task command timeout was not enforced")
                    isolated = sandbox.exec(
                        "python -c \"import os; print(any('API_KEY' in k or 'TOKEN' in k for k in os.environ)); print(os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'))\""
                    )
                    if isolated.stdout.strip().splitlines() != ["False", "False"]:
                        raise RuntimeError("Task contains unexpected credentials")
                    # A reachable external address would violate LAB's network=none.
                    network = sandbox.exec(
                        "python -c \"import socket; socket.create_connection(('1.1.1.1',443),timeout=2)\"",
                        timeout=5,
                    )
                    if network.ok:
                        raise RuntimeError(
                            "Task network is not isolated by the cluster CNI"
                        )
                if not (root / "output/result.docx").exists():
                    raise RuntimeError("Output collection failed")
                report["live_pod_test"] = True
                report["fixture"] = "passed"
        report["status"] = "ready"
    except Exception as exc:  # noqa: BLE001 -- preflight records SDK/admission failures without exposing credentials
        report.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
