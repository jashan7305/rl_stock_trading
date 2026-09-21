"""
Custom Gymnasium trading environment for the RL stock-trading project.

Actions:
    0 = HOLD
    1 = BUY  -> target position = 100% stock
    2 = SELL -> target position = 0% stock

The environment is intentionally simple and compatible with
Stable-Baselines3 DQN and A2C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Single-stock trading environment.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix used as observations. These should normally be
        scaled using a scaler fitted on the training set.
    prices : pd.Series or array-like
        ORIGINAL, unscaled close prices corresponding to `features`.
    initial_capital : float
        Starting portfolio value.
    transaction_cost : float
        Proportional transaction cost. 0.001 = 0.1%.
    """

    metadata = {"render_modes": []}

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        features: pd.DataFrame,
        prices: pd.Series | np.ndarray,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
    ):
        super().__init__()

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        self.dates = features.index.copy()
        features = features.reset_index(drop=True)
        prices = pd.Series(prices).reset_index(drop=True)

        if len(features) != len(prices):
            raise ValueError(
                "features and prices must contain the same number of rows."
            )

        if len(features) < 2:
            raise ValueError("TradingEnv requires at least 2 observations.")

        if not np.isfinite(features.to_numpy(dtype=np.float32)).all():
            raise ValueError("features contain NaN or infinite values.")

        if not np.isfinite(prices.to_numpy(dtype=np.float64)).all():
            raise ValueError("prices contain NaN or infinite values.")

        if (prices <= 0).any():
            raise ValueError("All prices must be greater than zero.")

        if initial_capital <= 0:
            raise ValueError("initial_capital must be greater than zero.")

        if transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative.")

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)

        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)

        self.n_features = self.features.shape[1]

        # 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)

        # Market features + current portfolio position.
        # Position is 0.0 (cash) or 1.0 (fully invested).
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 1,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital
        self.trade_count = 0

        self.portfolio_history: list[float] = []
        self.position_history: list[float] = []
        self.action_history: list[int] = []
        self.date_history: list = []

    def _get_observation(self) -> np.ndarray:
        """Return current market features plus current position."""
        market_features = self.features.iloc[self.current_step].to_numpy(
            dtype=np.float32
        )

        return np.concatenate(
            [
                market_features,
                np.array([self.position], dtype=np.float32),
            ]
        )

    def _get_info(
        self,
        action: int | None = None,
        trade_executed: bool = False,
    ) -> dict:

        return {
            "step": self.current_step,
            "price": float(self.prices.iloc[self.current_step]),
            "position": float(self.position),
            "portfolio_value": float(self.portfolio_value),
            "initial_capital": float(self.initial_capital),
            "total_return": float(
                self.portfolio_value / self.initial_capital - 1.0
            ),
            "trade_count": int(self.trade_count),
            "action": None if action is None else int(action),
            "trade_executed": bool(trade_executed),
            "date": self.dates[self.current_step],
        }

    def reset(self, *, seed=None, options=None):
        """Reset the environment to the beginning of the dataset."""
        super().reset(seed=seed)

        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital
        self.trade_count = 0

        self.portfolio_history = [self.portfolio_value]
        self.position_history = [self.position]
        self.action_history = []
        self.date_history = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one trading decision.

        The agent observes features at day t and chooses a position.
        The reward is based on the portfolio movement from t to t+1.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Convert discrete action to target portfolio position.
        old_position = self.position

        if action == self.BUY:
            target_position = 1.0
        elif action == self.SELL:
            target_position = 0.0
        else:  # HOLD
            target_position = old_position

        trade_executed = target_position != old_position

        if trade_executed:
            self.trade_count += 1

        self.position = target_position

        # We need a next price because the action taken at t affects
        # the portfolio during the interval t -> t+1.
        current_price = float(self.prices.iloc[self.current_step])
        next_price = float(self.prices.iloc[self.current_step + 1])

        price_return = next_price / current_price - 1.0

        # Transaction cost is charged when changing position.
        turnover = abs(target_position - old_position)
        cost_multiplier = 1.0 - self.transaction_cost * turnover

        value_after_trade = self.portfolio_value * cost_multiplier

        # Only the invested portion participates in the stock return.
        new_portfolio_value = value_after_trade * (
            1.0 + target_position * price_return
        )

        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_value = float(new_portfolio_value)

        # Log portfolio return is a stable RL reward.
        reward = float(
            np.log(
                self.portfolio_value / self.previous_portfolio_value
            )
        )

        self.portfolio_history.append(self.portfolio_value)
        self.position_history.append(self.position)
        self.action_history.append(action)

        self.current_step += 1

        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        observation = self._get_observation()

        info = self._get_info(
            action=action,
            trade_executed=trade_executed,
        )

        info["price_return"] = float(price_return)
        info["reward"] = float(reward)
        info["turnover"] = float(turnover)

        return observation, reward, terminated, truncated, info

    def get_portfolio_history(self) -> pd.DataFrame:
        """Return the portfolio history as a DataFrame."""
        n = len(self.portfolio_history)

        return pd.DataFrame(
            {
                "Portfolio_Value": self.portfolio_history,
                "Position": self.position_history,
            }
        ).iloc[:n]

    def render(self):
        """Simple text rendering for debugging."""
        print(
            f"Step: {self.current_step:>5} | "
            f"Price: {self.prices.iloc[self.current_step]:>10.2f} | "
            f"Position: {self.position:.0f} | "
            f"Portfolio: ₹{self.portfolio_value:,.2f}"
        )

    def close(self):
        """No external resources need to be closed."""
        pass
