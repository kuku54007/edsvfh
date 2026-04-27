from __future__ import annotations

import numpy as np

from .config import DecisionConfig


class DecisionLayer:
    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self.high_risk_count = 0
        self.warning_count = 0

    def reset(self) -> None:
        self.high_risk_count = 0
        self.warning_count = 0

    def decide(self, risk: np.ndarray, policy_uncertainty: float, done_probability: float) -> tuple[str, str]:
        short = float(risk[0])
        mid = float(np.max(risk[:-1])) if len(risk) > 1 else short
        long = float(risk[-1])
        spread = float(abs(long - short))
        if short >= self.config.stop_threshold:
            self.high_risk_count += 1
        else:
            self.high_risk_count = 0
        if short >= self.config.warning_threshold or long >= self.config.stop_threshold:
            self.warning_count += 1
        else:
            self.warning_count = 0

        if self.high_risk_count >= self.config.confirm_count:
            return "shield", "short-horizon risk exceeded the shield threshold"
        if (
            self.warning_count >= self.config.confirm_count
            and short >= self.config.warning_threshold
            and long >= self.config.stop_threshold
            and policy_uncertainty >= self.config.warning_uncertainty
        ):
            return "shield", "multi-horizon risk indicates likely imminent failure before subgoal completion"
        if policy_uncertainty >= self.config.abstain_uncertainty or spread > self.config.spread_threshold:
            return "abstain", "uncertainty is too high for a reliable autonomous decision"
        if short < self.config.continue_threshold and long < self.config.stop_threshold:
            return "continue", "calibrated risk is below the operational threshold"
        if done_probability > 0.7:
            return "continue", "the current subgoal appears complete"
        if short >= self.config.warning_threshold or mid >= self.config.warning_threshold:
            return "watch", "near-term risk is elevated; keep monitoring while autonomy remains active"
        return "watch", "risk is elevated but not yet confirmed"
