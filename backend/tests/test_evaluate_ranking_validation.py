import pickle

import pytest

import offline.evaluation.evaluate_ranking_validation as evaluation


def test_validation_loader_rejects_embedded_test(monkeypatch, tmp_path):
    sample_path = tmp_path / "ranking.pkl"
    with open(sample_path, "wb") as file:
        pickle.dump({"train": {}, "validation": {}, "test": {}}, file)
    monkeypatch.setattr(evaluation.config, "RANKING_TRAIN_DATA_PATH", sample_path)
    with pytest.raises(ValueError, match="exactly Train and Validation"):
        evaluation.load_validation_artifacts(tmp_path / "missing.pt")


def test_validation_evaluation_refuses_existing_output(tmp_path):
    output = tmp_path / "metrics.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluation.run_validation_evaluation(output)
