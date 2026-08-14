from pathlib import Path

import pytest

import offline.evaluation.evaluate_ranking as evaluation


def test_run_evaluation_rejects_invalid_batch_size():
    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        evaluation.run_evaluation(batch_size=0)


def test_run_evaluation_protects_existing_result(tmp_path):
    output_path = tmp_path / "ranking.json"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluation.run_evaluation(output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == "existing"
