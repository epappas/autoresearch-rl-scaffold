def score_from_metrics(metrics: dict) -> float:
    """Lower is better placeholder objective."""
    return float(metrics.get("val_bpb", 999.0))
