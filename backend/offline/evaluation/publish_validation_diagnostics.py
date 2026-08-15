"""Publish sanitized, Validation-only diagnostic evidence."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from offline.config import config


PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/)")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_diagnostic(payload, expected_version):
    protocol = payload.get("protocol", {})
    if protocol.get("version") != expected_version:
        raise ValueError(f"Expected diagnostic version {expected_version}")
    if protocol.get("split") != "validation":
        raise ValueError("Diagnostic is not Validation-only")
    if protocol.get("test_accessed") is not False:
        raise ValueError("Diagnostic does not prove Test remained sealed")


def sanitize(value):
    if isinstance(value, dict):
        return {
            key: ("<DEVICE>" if key == "device" else sanitize(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and PRIVATE_PATH.search(value):
        raise ValueError("Private absolute path found in diagnostic payload")
    return value


def git_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def publish(inputs, output, commit=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    records = {}
    sources = []
    for version, path in sorted(inputs.items()):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_diagnostic(payload, version)
        records[f"v{version}"] = sanitize(payload)
        sources.append({
            "version": version,
            "file": path.name,
            "sha256": sha256(path),
        })
    result = {
        "experiment": "pytorch-baseline-v1-validation-diagnostics",
        "git_commit": commit or git_commit(),
        "selection_boundary": {
            "allowed_split": "validation",
            "test_accessed": False,
            "test_metrics_copied": False,
        },
        "sources": sources,
        "diagnostics": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    evaluation = config.TEMP_DIR / "evaluation"
    parser = argparse.ArgumentParser(
        description="Publish sanitized Validation-only diagnostics"
    )
    parser.add_argument(
        "--v2", type=Path,
        default=evaluation / "baseline_v1_validation_diagnostics_v2.json",
    )
    parser.add_argument(
        "--v3", type=Path,
        default=(
            evaluation / "baseline_v1_validation_model_diagnostics_v3.json"
        ),
    )
    parser.add_argument(
        "--v4", type=Path,
        default=(
            evaluation / "baseline_v1_validation_cross_diagnostics_v4.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "docs"
            / "results"
            / "baseline_v1_validation_diagnostics.json"
        ),
    )
    args = parser.parse_args()
    result = publish(
        {2: args.v2, 3: args.v3, 4: args.v4}, args.output
    )
    print(json.dumps(result["selection_boundary"], indent=2))
    print(f"Evidence saved to: {args.output}")


if __name__ == "__main__":
    main()
