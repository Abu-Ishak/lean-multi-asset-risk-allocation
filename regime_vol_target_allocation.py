"""QuantConnect LEAN Multi-Asset Regime & Volatility-Targeted Portfolio Algorithm.

Constructs a diversified institutional multi-asset allocation across US Large-Cap
Equities (SPY), Growth Equities (QQQ), Intermediate Treasuries (IEF), Long-Duration
Treasuries (TLT), Gold (GLD), and Short-Term T-Bills (SHY). Combines regime
filtering, rolling minimum-variance risk weighting, annualized volatility scaling
(10% target), and a drawdown circuit breaker.
"""

from AlgorithmImports import *
import numpy as np


class RegimeVolTargetAllocation(QCAlgorithm):
    """Multi-asset minimum-variance and volatility-targeted LEAN strategy."""

    def initialize(self) -> None:
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(1_000_000)
        self.set_benchmark("SPY")

        self.risk_assets = ["SPY", "QQQ", "IEF", "TLT", "GLD"]
        self.defensive_asset = "SHY"
        self.all_tickers = self.risk_assets + [self.defensive_asset]

        self.symbols = {}
        self.sma_200 = {}
        for ticker in self.all_tickers:
            eq = self.add_equity(ticker, Resolution.DAILY)
            eq.set_leverage(1.0)
            self.symbols[ticker] = eq.symbol
            self.sma_200[ticker] = self.sma(eq.symbol, 200, Resolution.DAILY)

        self.lookback_days = 63
        self.target_vol = 0.10
        self.max_asset_weight = 0.40
        self.rebalance_buffer = 0.0025
        self.peak_equity = 1_000_000.0
        self.max_drawdown_limit = 0.08
        self.set_warm_up(210, Resolution.DAILY)

        self.schedule.on(
            self.date_rules.month_start("SPY"),
            self.time_rules.after_market_open("SPY", 30),
            self.rebalance_portfolio,
        )

    def on_data(self, slice: Slice) -> None:
        if self.is_warming_up:
            return
        equity = self.portfolio.total_portfolio_value
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown >= self.max_drawdown_limit:
            # Drawdown circuit breaker: cut gross risk exposure in half
            for ticker in self.risk_assets:
                sym = self.symbols[ticker]
                current_w = self.portfolio[sym].holdings_value / equity
                if current_w > 0.02:
                    self.set_holdings(sym, current_w * 0.5)

    def rebalance_portfolio(self) -> None:
        if self.is_warming_up:
            return

        history = self.history(
            [self.symbols[t] for t in self.risk_assets],
            self.lookback_days + 1,
            Resolution.DAILY,
        )
        if history.empty:
            return

        closes = history["close"].unstack(level=0)
        returns = closes.pct_change().dropna()
        if len(returns) < 30:
            return

        # Ledoit-style shrinkage toward diagonal covariance for stability
        sample_cov = returns.cov().values * 252.0
        diag_cov = np.diag(np.diag(sample_cov))
        cov = 0.80 * sample_cov + 0.20 * diag_cov + np.eye(len(self.risk_assets)) * 1e-6

        inv_cov = np.linalg.inv(cov)
        ones = np.ones(len(self.risk_assets))
        raw_w = inv_cov @ ones
        raw_w = np.clip(raw_w, 0.0, None)
        weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(len(self.risk_assets)) / len(self.risk_assets)
        weights = np.minimum(weights, self.max_asset_weight)
        weights /= weights.sum()

        # Regime filter: halve risk-asset weights trading below 200-day SMA
        for idx, ticker in enumerate(self.risk_assets):
            sym = self.symbols[ticker]
            if self.securities[sym].price < self.sma_200[ticker].current.value:
                weights[idx] *= 0.50

        # Volatility targeting: scale portfolio to 10% annualized vol cap
        port_vol = float(np.sqrt(weights.T @ cov @ weights))
        vol_scalar = min(1.0, self.target_vol / max(port_vol, 1e-4))
        target_weights = {t: float(weights[i] * vol_scalar) for i, t in enumerate(self.risk_assets)}
        target_weights[self.defensive_asset] = max(0.0, 1.0 - sum(target_weights.values()))

        total_val = self.portfolio.total_portfolio_value
        for ticker, target_w in target_weights.items():
            sym = self.symbols[ticker]
            curr_w = self.portfolio[sym].holdings_value / total_val
            if abs(target_w - curr_w) >= self.rebalance_buffer:
                self.set_holdings(sym, target_w)
