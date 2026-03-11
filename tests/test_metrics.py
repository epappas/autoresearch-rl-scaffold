from autoresearch_rl.eval.metrics import parse_metrics


def test_parse_metrics():
    p = parse_metrics("x\nloss=2.0\nval_bpb=1.19\n")
    assert p.loss == 2.0
    assert p.val_bpb == 1.19
