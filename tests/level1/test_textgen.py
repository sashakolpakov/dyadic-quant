import json

import torch

from dyadic_quant.level1.textgen import (
    build_wikitext_prompts,
    cosine_similarity,
    edit_ratio,
    exact_match,
    merge_generations,
    token_jaccard,
)


class _StubTokenizer:
    """Reversible whitespace tokenizer for deterministic prompt tests."""

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)


def test_lexical_metrics_bounds_and_values():
    assert exact_match("hello ", "hello")
    assert not exact_match("hello", "world")
    assert edit_ratio("abc", "abc") == 1.0
    assert edit_ratio("", "") == 1.0
    assert 0.0 < edit_ratio("kitten", "sitting") < 1.0
    assert token_jaccard("a b c", "b c d") == 2 / 4
    assert token_jaccard("", "") == 1.0
    assert token_jaccard("a", "") == 0.0
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_wikitext_prompts_are_deterministic_and_evenly_spaced():
    tokenizer = _StubTokenizer()
    token_ids = torch.arange(100)
    first = build_wikitext_prompts(
        tokenizer, token_ids, count=5, prefix_tokens=4
    )
    second = build_wikitext_prompts(
        tokenizer, token_ids, count=5, prefix_tokens=4
    )
    assert first == second
    assert len(first) == 5
    assert first[0]["id"] == "wiki_0"
    # Evenly spaced windows: span = (100-4)//5 = 19.
    assert first[1]["id"] == "wiki_19"
    assert first[0]["prompt"] == "0 1 2 3"


def test_merge_generations_records_prompts_once_and_overwrites_variant(tmp_path):
    path = tmp_path / "gens.json"
    prompts = {"arc": [{"id": "q1", "prompt": "p1"}]}
    merge_generations(
        path,
        variant="bf16_source",
        prompts_by_family=prompts,
        generations_by_family={"arc": {"q1": "answer A"}},
    )
    merge_generations(
        path,
        variant="dyadic_4",
        prompts_by_family=prompts,
        generations_by_family={"arc": {"q1": "answer B"}},
    )
    document = json.loads(path.read_text())
    assert document["prompts"] == prompts
    assert set(document["generations"]) == {"bf16_source", "dyadic_4"}
    assert document["generations"]["dyadic_4"]["arc"]["q1"] == "answer B"

    # Re-running a variant overwrites rather than duplicates.
    merge_generations(
        path,
        variant="dyadic_4",
        prompts_by_family=prompts,
        generations_by_family={"arc": {"q1": "answer C"}},
    )
    document = json.loads(path.read_text())
    assert document["generations"]["dyadic_4"]["arc"]["q1"] == "answer C"
