from typing import Protocol


class ProposalPolicy(Protocol):
    def propose_diff(self, state: dict) -> str: ...
