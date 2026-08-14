import pytest

import offline.evaluation.evaluate_baseline as evaluation


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"batch_size": 0}, "batch_size must be greater than zero"),
        ({"k_values": ()}, "k values must be positive"),
        ({"k_values": (0, 10)}, "k values must be positive"),
    ],
)
def test_run_baseline_rejects_invalid_configuration(arguments, message):
    with pytest.raises(ValueError, match=message):
        evaluation.run_baseline_evaluation(**arguments)


def test_run_baseline_protects_existing_final_result(tmp_path):
    output_path = tmp_path / "baseline.json"
    output_path.write_text("frozen", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluation.run_baseline_evaluation(output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == "frozen"


def test_build_manifest_records_experiment(monkeypatch):
    monkeypatch.setattr(evaluation, "git_commit", lambda: "abc123")
    manifest = evaluation.build_manifest(
        seed=42,
        retrieval={"model": {"best_checkpoint_epoch": 14}},
        ranking={"model": {"best_checkpoint_epoch": 5}},
    )

    assert manifest["experiment"] == "baseline-v0"
    assert manifest["git_commit"] == "abc123"
    assert manifest["seed"] == 42
    assert manifest["configuration"]["embedding_dimension"] > 0
    assert manifest["configuration"]["configured_default_epochs"] > 0
    assert "epochs" not in manifest["configuration"]
    assert manifest["training_selection"] == {
        "retrieval_best_checkpoint_epoch": 14,
        "ranking_best_checkpoint_epoch": 5,
    }
    assert manifest["artifact_root"] == (
        "<FUNREC_PROCESSED_DATA_PATH>/web_project"
    )
