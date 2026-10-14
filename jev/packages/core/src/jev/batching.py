"""Deterministic global optimizer windows with rank-aligned microbatch schedules.

All ranks derive the identical plan from (seed, epoch, window, world_size): a
serialized global cursor plus content-derived permutation seeds replace a shared
mutable sampler. Microbatch counts are aligned per window with zero-weight dummy
passes so every rank issues the same number of collectives per optimizer step.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jev_contracts import validate_example

from .processing import decision_messages, permute_example


@dataclass(frozen=True)
class Window:
    """One optimizer window: exactly the real decisions it contributes."""

    index: int
    ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class RankSchedule:
    """A rank's real sub-batches for one window; dummies are added at run time."""

    rank: int
    subbatches: tuple[tuple[str, ...], ...]

    @property
    def real_count(self) -> int:
        return sum(len(subbatch) for subbatch in self.subbatches)


def _seed_int(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest, 16)


def check_rows(rows: Sequence[dict]) -> dict[str, dict]:
    """Index strictly validated rows by ID; duplicates and empty inputs fail."""
    if not rows:
        raise ValueError("A training split must not be empty")
    by_id: dict[str, dict] = {}
    for row in rows:
        validated = validate_example(row)
        if validated["id"] in by_id:
            raise ValueError(f"Duplicate example ID in split: {validated['id']!r}")
        by_id[validated["id"]] = validated
    return by_id


def epoch_order(rows: Mapping[str, dict] | Sequence[dict], seed: int, epoch: int) -> list[str]:
    """Deterministic seeded order over sorted IDs; independent of dict insertion."""
    ids = sorted(rows) if isinstance(rows, Mapping) else sorted(check_rows(rows))
    if not ids:
        raise ValueError("Cannot plan an epoch without examples")
    order = list(ids)
    random.Random(_seed_int(seed, "epoch", epoch)).shuffle(order)
    return order


def plan_windows(order: Sequence[str], global_batch: int) -> list[Window]:
    """Split an epoch order into global optimizer windows; the tail stays partial.

    The final window is real data that must be trained on or explicitly dropped
    by the caller; this function never silently discards or repeats examples.
    """
    if type(global_batch) is not int or global_batch < 1:
        raise ValueError("global_batch must be a positive integer")
    if not order:
        raise ValueError("Cannot plan windows without examples")
    return [
        Window(index, tuple(order[start:start + global_batch]))
        for index, start in enumerate(range(0, len(order), global_batch))
    ]


def permutation_seed(seed: int, epoch: int, example_id: str) -> int:
    """Content-stable candidate permutation seed; resume needs no RNG replay."""
    return _seed_int(seed, "permute", epoch, example_id)


def prepared_rows(rows_by_id: Mapping[str, dict], window: Window, *, seed: int,
                  epoch: int) -> list[dict]:
    """Permute choice candidates per (seed, epoch, id); score/noul keep their order."""
    if window.size == 0:
        raise ValueError("A window must contain at least one real example")
    return [
        permute_example(rows_by_id[row_id], permutation_seed(seed, epoch, row_id))
        for row_id in window.ids
    ]


def assign_ranks(window: Window, world_size: int, micro_cap: int) -> list[list[str]]:
    """Chunked round-robin: rank r takes micro_cap ids per round, in window order.

    Round g covers window[g*world_size*micro_cap :]; within it rank r receives
    slice [r*micro_cap : r*micro_cap + micro_cap]. Empty tail slices are legal.
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if type(micro_cap) is not int or micro_cap < 1:
        raise ValueError("micro_cap must be a positive integer")
    per_rank: list[list[str]] = [[] for _ in range(world_size)]
    rounds = math.ceil(window.size / (world_size * micro_cap))
    for round_index in range(rounds):
        base = round_index * world_size * micro_cap
        for rank in range(world_size):
            start = base + rank * micro_cap
            per_rank[rank].extend(window.ids[start:start + micro_cap])
    return per_rank


def split_by_budget(ids: Sequence[str], lengths: Mapping[str, int], *, micro_cap: int,
                    token_budget: int) -> list[tuple[str, ...]]:
    """Greedy sub-batches bounded by example cap and padded-token budget.

    A single example larger than the budget forms its own sub-batch; the trainer
    must still reject inputs beyond max_length during processing.
    """
    if type(micro_cap) is not int or micro_cap < 1:
        raise ValueError("micro_cap must be a positive integer")
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError("token_budget must be a positive integer")
    subbatches: list[tuple[str, ...]] = []
    current: list[str] = []
    current_max = 0
    for row_id in ids:
        length = lengths.get(row_id)
        if type(length) is not int or length < 1:
            raise ValueError(f"Missing token length for example: {row_id!r}")
        projected = max(current_max, length) * (len(current) + 1)
        if current and (len(current) >= micro_cap or projected > token_budget):
            subbatches.append(tuple(current))
            current, current_max = [], 0
            projected = length
        current.append(row_id)
        current_max = max(current_max, length)
    if current:
        subbatches.append(tuple(current))
    return subbatches


def plan_rank_schedule(window: Window, *, world_size: int, micro_cap: int,
                       token_budget: int, lengths: Mapping[str, int]) -> list[RankSchedule]:
    """Per-rank sub-batch schedules for one window, given local example lengths."""
    schedules = []
    for rank, local_ids in enumerate(assign_ranks(window, world_size, micro_cap)):
        subbatches = split_by_budget(local_ids, lengths, micro_cap=micro_cap,
                                     token_budget=token_budget) if local_ids else []
        schedules.append(RankSchedule(rank, tuple(subbatches)))
    total = sum(schedule.real_count for schedule in schedules)
    if total != window.size:
        raise ValueError("Rank assignment lost or duplicated window examples")
    return schedules


def token_length(model: object, row: dict) -> int:
    """Unpadded prompt length via the exact rendering path used for processing.

    This must mirror prepare_batch: shared decision_messages rendering plus the
    processor's chat template. It never pads, truncates, or opens images.
    """
    messages = decision_messages(row, model.codes)
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return len(model.processor.tokenizer.encode(text, add_special_tokens=False))


def local_lengths(
    model: object, rows_by_id: Mapping[str, dict], ids: Sequence[str],
) -> dict[str, int]:
    return {row_id: token_length(model, rows_by_id[row_id]) for row_id in ids}


def window_examples(window: Window, prepared: Sequence[dict]) -> list[dict]:
    """Return the window's prepared rows in canonical window order."""
    by_id = {row["id"]: row for row in prepared}
    if len(by_id) != window.size:
        raise ValueError("Prepared rows do not cover the window exactly once")
    return [by_id[row_id] for row_id in window.ids]


@dataclass(frozen=True)
class Cursor:
    """Everything required to resume the deterministic plan, besides RNG state."""

    epoch: int
    window_index: int
    update: int
    consumed: int

    def advance(self, window: Window) -> Cursor:
        return Cursor(self.epoch, self.window_index + 1, self.update + 1,
                      self.consumed + window.size)
