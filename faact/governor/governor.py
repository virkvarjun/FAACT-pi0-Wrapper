"""Runtime governor for chunk acceptance / rejection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np


class GovernorDecision(Enum):
    EXECUTE = "execute"
    REJECT_REPLAN = "reject_replan"
    TRUNCATE = "truncate"
    FALLBACK = "fallback"


@dataclass
class GovernorResult:
    decision: GovernorDecision
    risk_score: float
    threshold: float
    reason: str


class Governor:
    """Configurable runtime supervisor for chunk approval."""

    def __init__(
        self,
        threshold: float = 0.5,
        strategy: str = "threshold",
    ) -> None:
        self.threshold = threshold
        self.strategy = strategy

    def decide(
        self,
        risk_score: float,
        chunk_idx: int = 0,
    ) -> GovernorResult:
        """Decide whether to execute or reject the proposed chunk."""
        if self.strategy == "threshold":
            if risk_score < self.threshold:
                return GovernorResult(
                    decision=GovernorDecision.EXECUTE,
                    risk_score=risk_score,
                    threshold=self.threshold,
                    reason="below_threshold",
                )
            return GovernorResult(
                decision=GovernorDecision.REJECT_REPLAN,
                risk_score=risk_score,
                threshold=self.threshold,
                reason="above_threshold",
            )
        return GovernorResult(
            decision=GovernorDecision.EXECUTE,
            risk_score=risk_score,
            threshold=self.threshold,
            reason="unknown_strategy",
        )
