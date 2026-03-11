from __future__ import annotations

import re
from dataclasses import dataclass

VAL_BPB_RE = re.compile(r"val[_-]?bpb\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
LOSS_RE = re.compile(r"loss\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass
class ParsedMetrics:
    val_bpb: float | None = None
    loss: float | None = None


def parse_metrics(text: str) -> ParsedMetrics:
    val_bpb = None
    loss = None
    for line in text.splitlines():
        m = VAL_BPB_RE.search(line)
        if m:
            val_bpb = float(m.group(1))
        m2 = LOSS_RE.search(line)
        if m2:
            loss = float(m2.group(1))
    return ParsedMetrics(val_bpb=val_bpb, loss=loss)
