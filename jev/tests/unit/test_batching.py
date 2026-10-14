from __future__ import annotations

from types import SimpleNamespace

import pytest

from jev.batching import (
    Cursor,
    Window,
    assign_ranks,
    check_rows,
    epoch_order,
    local_lengths,
    permutation_seed,
    plan_rank_schedule,
    plan_windows,
    prepared_rows,
    split_by_budget,
    token_length,
    window_examples,
)
from jev_contracts import target_distribution, validate_example

from .test_processing import FakeProcessor, make_row


def numbered(count, *, split="train", kind="choice", options=3, family="fam"):
    rows = []
    for index in range(count):
        row = make_row(kind, options)
        row["id"] = f"{split}-{index:03d}"
        row["family_id"] = f"{family}-{index % 2}"
        row["split"] = split
        rows.append(row)
    return rows


def test_check_rows_indexes_validated_rows_and_rejects_problems():
    rows = numbered(3)
    by_id = check_rows(rows)
    assert sorted(by_id) == sorted(row["id"] for row in rows)
    assert all(validate_example(row) == by_id[row["id"]] for row in rows)
    with pytest.raises(ValueError, match="empty"):
        check_rows([])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        check_rows(rows + [dict(rows[0])])
    with pytest.raises(ValueError):
        check_rows([{"id": "broken"}])


def test_epoch_order_is_content_stable_and_insertion_independent():
    rows = numbered(12)
    by_id = check_rows(rows)
    order = epoch_order(by_id, seed=3, epoch=0)
    assert sorted(order) == sorted(by_id)
    assert order == epoch_order(by_id, seed=3, epoch=0)
    assert order == epoch_order(list(reversed(rows)), seed=3, epoch=0)
    assert order != epoch_order(by_id, seed=4, epoch=0)
    assert order != epoch_order(by_id, seed=3, epoch=1)
    with pytest.raises(ValueError, match="epoch"):
        epoch_order({}, seed=0, epoch=0)


def test_plan_windows_chunks_and_keeps_the_real_tail():
    ids = [f"e{i}" for i in range(5)]
    windows = plan_windows(ids, 2)
    assert [(window.index, window.ids, window.size) for window in windows] == [
        (0, ("e0", "e1"), 2), (1, ("e2", "e3"), 2), (2, ("e4",), 1),
    ]
    assert plan_windows(["only"], 8) == [Window(0, ("only",))]
    with pytest.raises(ValueError, match="windows"):
        plan_windows([], 1)
    for bad in (0, True, 2.5):
        with pytest.raises(ValueError, match="global_batch"):
            plan_windows(["a"], bad)


def test_permutation_seed_is_content_stable():
    assert permutation_seed(1, 0, "a") == permutation_seed(1, 0, "a")
    assert permutation_seed(1, 0, "a") != permutation_seed(1, 0, "b")
    assert permutation_seed(1, 0, "a") != permutation_seed(1, 1, "a")
    assert permutation_seed(1, 0, "a") != permutation_seed(2, 0, "a")


def test_prepared_rows_permute_choice_only_and_align_targets():
    choice = make_row("choice", 4)
    choice["id"] = "c-0"
    choice["target"] = {
        "kind": "distribution", "probabilities": {str(i): (i + 1) / 10 for i in range(4)},
    }
    score = make_row("score", 3)
    score["id"] = "s-0"
    noul = make_row("noul")
    noul["id"] = "n-0"
    by_id = check_rows([choice, score, noul])
    window = Window(7, ("c-0", "s-0", "n-0"))
    prepared = prepared_rows(by_id, window, seed=5, epoch=0)
    assert [row["id"] for row in prepared] == ["c-0", "s-0", "n-0"]
    assert prepared == prepared_rows(by_id, window, seed=5, epoch=0)
    weights = choice["target"]["probabilities"]
    aligned = target_distribution(prepared[0])
    for index, option in enumerate(prepared[0]["question"]["options"]):
        assert aligned[index] == weights[option["id"]]
    assert {option["id"] for option in prepared[0]["question"]["options"]} == set(weights)
    # Some seed must move the candidates: permutation actually happens for choice rows.
    variants = {
        tuple(
            option["id"] for option in
            prepared_rows(by_id, window, seed=seed, epoch=0)[0]["question"]["options"]
        )
        for seed in range(24)
    }
    assert len(variants) > 1
    # Ordered task types never move and never leak the option id into descriptions.
    assert prepared[1]["question"]["options"] == by_id["s-0"]["question"]["options"]
    assert prepared[2]["question"]["options"] == by_id["n-0"]["question"]["options"]
    with pytest.raises(ValueError, match="at least one real"):
        prepared_rows(by_id, Window(0, ()), seed=0, epoch=0)


def test_assign_ranks_chunked_round_robin():
    ids = tuple(f"e{i}" for i in range(5))
    assert assign_ranks(Window(0, ids), 2, 2) == [["e0", "e1", "e4"], ["e2", "e3"]]
    assert assign_ranks(Window(0, ids[:2]), 4, 1) == [["e0"], ["e1"], [], []]
    schedules = assign_ranks(Window(0, ids), 3, 1)
    assert sorted(sum(schedules, [])) == sorted(ids)
    for bad_world, bad_cap in ((0, 1), (True, 1), (1, 0), (1, 2.5)):
        with pytest.raises(ValueError):
            assign_ranks(Window(0, ids), bad_world, bad_cap)


def test_split_by_budget_bounds():
    lengths = {"a": 4, "b": 2, "c": 3, "d": 9}
    assert split_by_budget(
        ("a", "b", "c", "d"), lengths, micro_cap=2, token_budget=8,
    ) == [("a", "b"), ("c",), ("d",)]
    assert split_by_budget(
        ("a", "b", "c"), {"a": 1, "b": 1, "c": 1}, micro_cap=2, token_budget=100,
    ) == [("a", "b"), ("c",)]
    assert split_by_budget(("big",), {"big": 1000}, micro_cap=4, token_budget=10) == [("big",)]
    assert split_by_budget((), {}, micro_cap=1, token_budget=1) == []
    with pytest.raises(ValueError, match="token length"):
        split_by_budget(("a",), {}, micro_cap=1, token_budget=10)
    for cap, budget in ((0, 1), (True, 1), (1, 0), (1, 1.5)):
        with pytest.raises(ValueError):
            split_by_budget(("a",), {"a": 1}, micro_cap=cap, token_budget=budget)


def test_plan_rank_schedule_conserves_every_window_example():
    ids = tuple(f"e{i}" for i in range(5))
    lengths = {name: 2 for name in ids}
    schedules = plan_rank_schedule(
        Window(3, ids), world_size=2, micro_cap=2, token_budget=64, lengths=lengths,
    )
    assert [schedule.rank for schedule in schedules] == [0, 1]
    assert schedules[0].subbatches == (("e0", "e1"), ("e4",))
    assert schedules[1].subbatches == (("e2", "e3"),)
    assert [schedule.real_count for schedule in schedules] == [3, 2]
    assert sum(schedule.real_count for schedule in schedules) == len(ids)


def test_token_length_and_local_lengths_use_the_rendering_path():
    model = SimpleNamespace(codes=["A", "B", "C"], processor=FakeProcessor())
    row = make_row(count=2)
    # The rendered prompt is not a single vocabulary entry and carries no chat
    # prefix, so the fake tokenizer reports its two fallback tokens.
    assert token_length(model, row) == 2
    by_id = check_rows(numbered(2))
    assert local_lengths(model, by_id, sorted(by_id)) == {"train-000": 2, "train-001": 2}


def test_window_examples_orders_rows_and_rejects_partial_coverage():
    by_id = check_rows(numbered(3))
    window = Window(1, ("train-001", "train-000"))
    prepared = prepared_rows(by_id, window, seed=1, epoch=0)
    assert [row["id"] for row in window_examples(window, prepared)] == ["train-001", "train-000"]
    with pytest.raises(ValueError, match="cover"):
        window_examples(window, prepared[:1])


def test_cursor_advance_serializes_the_plan():
    cursor = Cursor(0, 0, 0, 0)
    assert cursor.advance(Window(0, ("a", "b"))) == Cursor(0, 1, 1, 2)
    assert cursor == Cursor(0, 0, 0, 0)
    with pytest.raises(AttributeError):
        cursor.epoch = 1
