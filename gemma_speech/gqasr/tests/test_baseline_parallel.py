import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gqasr.baseline_parallel import run_parallel, validate_merge


def samples():
    return [
        dict(id=f"train:zh:{i}", language="zh", reference="你好", audio=f"/{i}.wav",
             stratum="han_only" if i % 2 else "han_latin")
        for i in range(10)
    ]


def test_exact_merge_restores_order():
    rows = samples()
    merged = validate_merge(rows, [[{**r, "hypothesis": "你好"} for r in rows[rank::3]]
                                   for rank in range(3)])
    assert [r["id"] for r in merged] == [r["id"] for r in rows]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reference"])
def test_invalid_merge_rejected(kind):
    rows = samples()
    result = [dict(r, hypothesis="你好") for r in rows]
    if kind == "missing":
        result.pop()
    elif kind == "duplicate":
        result.append(result[0])
    else:
        result[0]["reference"] = "changed"
    with pytest.raises(ValueError):
        validate_merge(rows, [result])


def test_parallel_driver_selects_once_and_pools_all_rows(tmp_path, monkeypatch):
    import gqasr.baseline_parallel as module
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"index_path": "unused"}))
    calls, selections = [], []

    def select(*args):
        selections.append(args)
        return samples(), {"han_only": 100, "han_latin": 100}

    class Worker:
        def __init__(self, cmd, stdout, stderr, env):
            calls.append(env["CUDA_VISIBLE_DEVICES"])
            output = Path(cmd[cmd.index("--output") + 1])
            output.mkdir()
            rows = module.read_rows(cmd[cmd.index("--samples") + 1])
            for mode in ("auto", "chinese"):
                module.write_rows(output / f"predictions-{mode}.jsonl",
                                  [{**r, "hypothesis": "你好"} for r in rows])
            (output / "provenance.json").write_text("{}")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(module, "select_training_samples", select)
    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    args = SimpleNamespace(gpus="2,5,7", samples=None, train=True, run_config=str(config),
                           per_group=5, seed=42, output=str(tmp_path / "out"), modes=["auto", "chinese"],
                           model="unused", batch_size=1, max_new_tokens=768)
    run_parallel(args)
    assert len(selections) == 1
    assert calls == ["2", "5", "7"]
    scores = json.loads((tmp_path / "out/scores.json").read_text())
    assert scores["auto"]["strict"]["utterances"] == 10
    assert scores["auto"]["by_stratum"]["han_latin"]["strict"]["utterances"] == 5
