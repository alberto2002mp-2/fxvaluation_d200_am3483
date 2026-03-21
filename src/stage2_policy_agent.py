"""Policy-learning overlay for dynamic Stage 2 signal thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class PolicyAgent:
    """Train a tabular Q-learning policy over Stage 2 signal states.

    The agent learns whether to tighten or relax the absolute z-score trigger
    based on the current valuation gap, model fit quality, and realized FX
    volatility, which gives the final audit layer a deterministic adaptive
    policy overlay on top of the stacked ensemble.
    """

    base_threshold: float = 2.0
    action_space: Tuple[float, ...] = (-0.5, 0.0, 0.5)
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    random_state: int = 42
    z_bins: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    adj_r2_bins: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    vol_bins: Tuple[float, ...] = (0.0025, 0.005, 0.01, 0.015, 0.02)

    def __post_init__(self) -> None:
        """Initialize the Q-table and deterministic random generator.

        Returns:
            None. The method mutates the dataclass instance in place.
        """
        self.q_table: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.rng = np.random.default_rng(self.random_state)

    def _digitize(self, value: float, bins: Iterable[float]) -> int:
        """Bucket a continuous state variable for tabular reinforcement learning.

        Args:
            value: Continuous feature value from the current market state.
            bins: Ordered bin edges used to discretize the feature.

        Returns:
            The integer bucket index used as part of the Q-table key.
        """
        return int(np.digitize([value], list(bins))[0])

    def discretize_state(self, z_score: float, adj_r2: float, volatility: float) -> Tuple[int, int, int]:
        """Map the continuous market state into a discrete Q-learning key.

        Args:
            z_score: Standardized valuation-gap signal.
            adj_r2: Rolling adjusted R-squared from the underlying Stage 2 model.
            volatility: Realized spot-return volatility used as a risk proxy.

        Returns:
            A three-dimensional discrete state key for the tabular policy.
        """
        return (
            self._digitize(float(z_score), self.z_bins),
            self._digitize(float(adj_r2), self.adj_r2_bins),
            self._digitize(float(volatility), self.vol_bins),
        )

    def _q_values(self, state: Tuple[int, int, int]) -> np.ndarray:
        """Return the action-value vector for one discretized state.

        Args:
            state: Discrete Q-table key built from the market state.

        Returns:
            A dense NumPy vector of action values for the supplied state.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space), dtype=float)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, int, int], explore: bool = True) -> int:
        """Choose an action with epsilon-greedy exploration.

        Args:
            state: Discrete state key used to retrieve Q-values.
            explore: Whether to allow exploratory actions during training.

        Returns:
            The integer index of the selected threshold-adjustment action.
        """
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.action_space)))
        return int(np.argmax(self._q_values(state)))

    def threshold_from_action(self, action_idx: int) -> float:
        """Translate an action into the operative z-score trigger.

        Args:
            action_idx: Index of the selected threshold adjustment.

        Returns:
            The non-negative absolute trigger used for signal generation.
        """
        threshold = self.base_threshold + float(self.action_space[action_idx])
        return max(0.5, threshold)

    def signal_from_state(self, z_score: float, threshold: float) -> int:
        """Convert the valuation gap into a trading stance.

        Args:
            z_score: Standardized valuation-gap signal.
            threshold: Active absolute trigger selected by the policy.

        Returns:
            ``1`` for long, ``-1`` for short, and ``0`` for neutral.
        """
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
        """Fit the Q-learning policy on a prepared stacked-audit dataset.

        The reward function uses next-day spot returns so the agent learns a
        threshold policy that balances responsiveness against noise in the
        valuation signal.

        Args:
            audit_df: Stage 2 audit dataset containing ``Signal_Z``, ``Adj_R2``,
                and ``Actual_Price`` columns.
            epochs: Number of passes through the training sequence.
            vol_window: Rolling window used to estimate realized volatility.

        Returns:
            A policy-run DataFrame containing daily thresholds, positions, and
            the resulting equity curve.
        """
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
        """Run the learned policy through a prepared audit dataset.

        Args:
            audit_df: Dataset containing the discretized state inputs and, when
                available, next-period returns for realized policy rewards.

        Returns:
            A date-indexed DataFrame with dynamic thresholds, positions, rewards,
            and cumulative policy equity.
        """
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
