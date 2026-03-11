from autoresearch_rl.controller.loop import run_loop


def test_loop_runs():
    r = run_loop(max_iterations=1)
    assert r.iterations == 1
