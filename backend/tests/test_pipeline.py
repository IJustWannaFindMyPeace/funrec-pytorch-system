import sys
from unittest.mock import Mock

import offline.pipeline as pipeline
from offline.feature import preprocess_ranking, preprocess_retrieval
from modeling.youtubednn import SCORING_CONTRACT_SCALED_COSINE_V2


def test_all_pipeline_uses_selection_only_and_v2_scoring(monkeypatch):
    retrieval_preprocess = Mock()
    ranking_preprocess = Mock()
    retrieval_training = Mock()
    ranking_training = Mock()
    ingest = Mock()
    deploy = Mock()

    monkeypatch.setattr(pipeline, "run_retrieval_preprocessing", retrieval_preprocess)
    monkeypatch.setattr(pipeline, "run_ranking_preprocessing", ranking_preprocess)
    monkeypatch.setattr(pipeline, "run_retrieval_training", retrieval_training)
    monkeypatch.setattr(pipeline, "run_ranking_training", ranking_training)
    monkeypatch.setattr(pipeline, "ingest_to_redis", ingest)
    monkeypatch.setattr(pipeline, "deploy_local", deploy)
    monkeypatch.setattr(sys, "argv", ["offline.pipeline", "--steps", "all"])

    pipeline.main()

    retrieval_preprocess.assert_called_once_with(selection_only=True)
    ranking_preprocess.assert_called_once_with(selection_only=True)
    retrieval_training.assert_called_once_with(
        scoring_contract=SCORING_CONTRACT_SCALED_COSINE_V2,
        logit_scale=10.0,
    )
    ranking_training.assert_called_once_with()
    ingest.assert_called_once_with(flush=False)
    deploy.assert_called_once_with()


def test_pipeline_accepts_explicit_v2_logit_scale(monkeypatch):
    training = Mock()
    monkeypatch.setattr(pipeline, "run_retrieval_training", training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "offline.pipeline",
            "--steps",
            "retrieval_training",
            "--retrieval-logit-scale",
            "7.0",
        ],
    )

    pipeline.main()

    training.assert_called_once_with(
        scoring_contract=SCORING_CONTRACT_SCALED_COSINE_V2,
        logit_scale=7.0,
    )


def test_retrieval_preprocess_cli_is_selection_only(monkeypatch):
    preprocessing = Mock()
    monkeypatch.setattr(
        preprocess_retrieval,
        "run_retrieval_preprocessing",
        preprocessing,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["offline.feature.preprocess_retrieval", "--max-seq-len", "20"],
    )

    preprocess_retrieval.main()

    preprocessing.assert_called_once_with(
        max_seq_len=20,
        selection_only=True,
    )


def test_ranking_preprocess_cli_is_selection_only(monkeypatch):
    preprocessing = Mock()
    monkeypatch.setattr(
        preprocess_ranking,
        "run_ranking_preprocessing",
        preprocessing,
    )

    preprocess_ranking.main()

    preprocessing.assert_called_once_with(selection_only=True)
