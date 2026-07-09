from rag_reliability.methods.lettucedetect.features import aggregate_token_scores


def test_aggregate_token_scores() -> None:
    assert aggregate_token_scores([{"prob": 0.2}, {"prob": 0.8}], threshold=0.5) == [
        0.8,
        0.5,
        0.5,
    ]


def test_aggregate_token_scores_empty() -> None:
    assert aggregate_token_scores([], threshold=0.5) == [0.0, 0.0, 0.0]
