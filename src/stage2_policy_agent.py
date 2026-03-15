"""Prototype reinforcement-learning policy agent for dynamic Stage 2 thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


cwd = Path.cwd()
if (cwd / "src").exists():
    project_root = cwd
elif (cwd.parent / "src").exists():
    project_root = cwd.parent
else:
    raise FileNotFoundError("Could not locate project root containing 'src'.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class PolicyAgent:
    """Simple tabular Q-learning agent for dynamic Z-score thresholds."""

    base_threshold: float = 2.0
    action_space: Tuple[float, ...] = (-0.5, 0.0, 0.5)
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    z_bins: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    adj_r2_bins: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    vol_bins: Tuple[float, ...] = (0.0025, 0.005, 0.01, 0.015, 0.02)

    def __post_init__(self) -> None:
        self.q_table: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _digitize(self, value: float, bins: Iterable[float]) -> int:
        """Discretize one continuous state feature."""
        return int(np.digitize([value], list(bins))[0])

    def discretize_state(self, z_score: float, adj_r2: float, volatility: float) -> Tuple[int, int, int]:
        """Map the continuous state into a discrete Q-table key."""
        return (
            self._digitize(float(z_score), self.z_bins),
            self._digitize(float(adj_r2), self.adj_r2_bins),
            self._digitize(float(volatility), self.vol_bins),
        )

    def _q_values(self, state: Tuple[int, int, int]) -> np.ndarray:
        """Return the Q-values for one state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space), dtype=float)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, int, int], explore: bool = True) -> int:
        """Choose an action index using epsilon-greedy exploration."""
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(len(self.action_space)))
        return int(np.argmax(self._q_values(state)))

    def threshold_from_action(self, action_idx: int) -> float:
        """Convert an action into an active Z-score threshold."""
        threshold = self.base_threshold + float(self.action_space[action_idx])
        return max(0.5, threshold)

    def signal_from_state(self, z_score: float, threshold: float) -> int:
        """Return trading direction from z-score and threshold."""
        if z_score <= -threshold:
            return 1
        if z_score >= threshold:
            return -1
        return 0

    def train(
        self,
        audit_df: pd.DataFrame,
        epochs: int = 8,
        vol_window: int = 20,
    ) -> pd.DataFrame:
        """Train the policy agent on an existing Stage 2 audit dataset."""
        df = audit_df.copy().sort_index()
        if "Signal_Z" not in df.columns or "Adj_R2" not in df.columns:
            raise KeyError("audit_df must contain Signal_Z and Adj_R2 columns.")

        df["Volatility"] = df["Actual_Price"].pct_change().rolling(vol_window).std()
        df["Next_Return"] = df["Actual_Price"].pct_change().shift(-1)
        train_df = df.dropna(subset=["Signal_Z", "Adj_R2", "Volatility", "Next_Return"]).copy()
        if train_df.empty:
            raise ValueError("Not enough data to train the policy agent.")

        state_keys = [
            self.discretize_state(row.Signal_Z, row.Adj_R2, row.Volatility)
            for row in train_df.itertuples()
        ]

        for _ in range(epochs):
            for idx in range(len(train_df) - 1):
                row = train_df.iloc[idx]
                state = state_keys[idx]
                next_state = state_keys[idx + 1]
                action_idx = self.select_action(state, explore=True)
                threshold = self.threshold_from_action(action_idx)
                position = self.signal_from_state(float(row["Signal_Z"]), threshold)
                reward = float(position * row["Next_Return"])

                q_values = self._q_values(state)
                next_q = np.max(self._q_values(next_state))
                td_target = reward + self.discount_factor * next_q
                q_values[action_idx] += self.learning_rate * (td_target - q_values[action_idx])

        return self.run_policy(audit_df=train_df)

    def run_policy(self, audit_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned policy to a prepared audit dataset."""
        df = audit_df.copy().sort_index()
        if "Volatility" not in df.columns:
            df["Volatility"] = df["Actual_Price"].pct_change().rolling(20).std()
        df = df.dropna(subset=["Signal_Z", "Adj_R2", "Volatility"]).copy()
        if df.empty:
            raise ValueError("Not enough prepared state data to run the policy.")

        policy_rows = []
        equity = 1.0
        for row in df.itertuples():
            state = self.discretize_state(row.Signal_Z, row.Adj_R2, row.Volatility)
            action_idx = self.select_action(state, explore=False)
            threshold = self.threshold_from_action(action_idx)
            position = self.signal_from_state(float(row.Signal_Z), threshold)
            pnl = float(position * getattr(row, "Next_Return", 0.0))
            equity *= 1.0 + pnl
            policy_rows.append(
                {
                    "Date": row.Index,
                    "Dynamic_Threshold": threshold,
                    "Action_Adjustment": self.action_space[action_idx],
                    "Policy_Position": position,
                    "Reward": pnl,
                    "Policy_Equity_Curve": equity,
                }
            )

        return pd.DataFrame(policy_rows).set_index("Date")
