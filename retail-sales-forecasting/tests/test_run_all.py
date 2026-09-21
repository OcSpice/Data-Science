import subprocess

import run_all


def test_pipeline_steps_are_ordered():
    assert [name for name, _ in run_all.STEPS] == [
        "flagship EDA and baseline analysis",
        "walk-forward validation",
        "advanced demand planning",
    ]


def test_main_runs_each_step_in_order(monkeypatch):
    calls = []

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_all.main()

    assert len(calls) == 3
    assert all(check is True for _, _, check in calls)
    assert [call[0][2:] for call in calls] == [
        ["-m", "src.flagship_analysis"],
        ["-m", "src.run_walk_forward"],
        ["-m", "advanced_demand_planning.src.run_planning"],
    ]
    assert all(call[1] == run_all.ROOT for call in calls)
