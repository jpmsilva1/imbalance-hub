import pandas as pd

from scripts.scan import _score_tasks, build_row


def _task(i):
    series = pd.Series(range(i, i + 50), dtype=float)
    return (f"gluonts:test:{i}", "gluonts", "test", str(i), "D", "", series, False)


def test_score_tasks_parallel_matches_serial():
    tasks = [_task(i) for i in range(4)]

    serial = [build_row(*t) for t in tasks]
    parallel = _score_tasks(tasks, workers=2)

    assert parallel == serial


def test_score_tasks_empty_returns_empty():
    assert _score_tasks([], workers=4) == []
