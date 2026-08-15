import json

import pytest

from offline.evaluation.publish_validation_diagnostics import (
    publish,
    sanitize,
    validate_diagnostic,
)


def payload(version, device="cuda"):
    return {
        "protocol": {
            "version": version,
            "split": "validation",
            "test_accessed": False,
            "device": device,
        },
        "metric": 0.5,
    }


def test_validate_rejects_test_access():
    value = payload(2)
    value["protocol"]["test_accessed"] = True
    with pytest.raises(ValueError, match="Test remained sealed"):
        validate_diagnostic(value, 2)


def test_sanitize_replaces_device_and_rejects_private_path():
    assert sanitize(payload(3))["protocol"]["device"] == "<DEVICE>"
    with pytest.raises(ValueError, match="Private absolute path"):
        sanitize({"path": r"C:\\Users\\person\\artifact.json"})


def test_publish_combines_three_versions(tmp_path):
    inputs = {}
    for version in (2, 3, 4):
        path = tmp_path / f"v{version}.json"
        path.write_text(json.dumps(payload(version)), encoding="utf-8")
        inputs[version] = path
    output = tmp_path / "public.json"
    result = publish(inputs, output, commit="abc123")
    assert result["selection_boundary"]["test_accessed"] is False
    assert set(result["diagnostics"]) == {"v2", "v3", "v4"}
    assert output.exists()
