"""Deterministic free-running text generation and lexical comparison helpers.

These utilities support comparing the *textual* output of quantized model
variants against the source checkpoint, complementing the teacher-forced
next-token agreement reported elsewhere. Two output strings may disagree
verbatim yet carry the same meaning; the lexical metrics here are cheap local
signals, while semantic equivalence (embedding cosine and an LLM judge) lives in
the experiment driver that owns the external backends.

All prompt construction is fully deterministic so that every model variant is
generated from byte-identical prompts across separate process invocations.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_arc_prompts(
    tokenizer,
    questions: list[dict[str, object]],
    *,
    count: int,
) -> list[dict[str, str]]:
    """Instruction prompts asking for a free-form ARC-Easy answer.

    The instruct chat template is applied so the prompt matches real usage of
    the model. The full free-form answer is what we later compare, not the
    likelihood-scored letter.
    """
    prompts: list[dict[str, str]] = []
    for question in questions[:count]:
        choices = "\n".join(
            f"{LETTERS[index]}. {choice}"
            for index, choice in enumerate(question["choices"])
        )
        user_message = (
            "Answer the following multiple-choice question. State the correct "
            "option and briefly explain why.\n"
            f"Question: {question['question']}\n{choices}"
        )
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append({"id": str(question["id"]), "prompt": formatted})
    return prompts


def build_wikitext_prompts(
    tokenizer,
    token_ids: torch.Tensor,
    *,
    count: int,
    prefix_tokens: int,
) -> list[dict[str, str]]:
    """Raw base-LM continuation prompts taken from evenly spaced windows.

    Continuation is a base-LM task, so no chat template is applied: each prompt
    is the decoded text of ``prefix_tokens`` consecutive audited WikiText tokens.
    Windows are spaced deterministically across the available token stream.
    """
    usable = len(token_ids) - prefix_tokens
    if usable <= 0:
        raise ValueError("token stream is too short for the requested prefix")
    span = max(1, usable // count)
    prompts: list[dict[str, str]] = []
    for index in range(count):
        offset = index * span
        if offset + prefix_tokens > len(token_ids):
            break
        window = token_ids[offset : offset + prefix_tokens]
        text = tokenizer.decode(window, skip_special_tokens=True)
        prompts.append({"id": f"wiki_{offset}", "prompt": text})
    return prompts


def generate_texts(
    model,
    tokenizer,
    device: torch.device,
    prompts: list[dict[str, str]],
    *,
    max_new_tokens: int,
) -> dict[str, str]:
    """Greedy, deterministic generation. Returns only the newly generated text.

    Prompts are already fully formatted strings (chat template applied where
    relevant), so special tokens are not re-added during tokenization.
    """
    outputs: dict[str, str] = {}
    with torch.inference_mode():
        for item in prompts:
            encoded = tokenizer(
                item["prompt"],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[0, encoded.input_ids.shape[1] :]
            outputs[item["id"]] = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
    return outputs


def merge_generations(
    path: Path,
    *,
    variant: str,
    prompts_by_family: dict[str, list[dict[str, str]]],
    generations_by_family: dict[str, dict[str, str]],
) -> None:
    """Atomically merge one variant's generations into the shared JSON file.

    The prompt text is recorded once per family (stable across variants) so the
    comparison driver can present prompts alongside outputs. Re-running a variant
    overwrites its prior generations rather than duplicating them.
    """
    if path.exists():
        document = json.loads(path.read_text())
    else:
        document = {"prompts": {}, "generations": {}}
    for family, prompts in prompts_by_family.items():
        existing = document["prompts"].get(family)
        if existing is None:
            document["prompts"][family] = prompts
        elif existing != prompts:
            raise RuntimeError(
                f"prompt set for family '{family}' differs from the recorded "
                "prompts; all variants must share identical prompts"
            )
    document["generations"][variant] = generations_by_family
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


# --- Lexical comparison metrics (cheap, local, deterministic) ----------------


def exact_match(reference: str, candidate: str) -> bool:
    return reference.strip() == candidate.strip()


def edit_ratio(reference: str, candidate: str) -> float:
    """Normalized character similarity in [0, 1] from Levenshtein distance.

    1.0 means identical strings; 0.0 means maximally different.
    """
    a, b = reference, candidate
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, substitute))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(a), len(b))


def token_jaccard(reference: str, candidate: str) -> float:
    """Jaccard overlap of whitespace-delimited token sets."""
    a = set(reference.split())
    b = set(candidate.split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
