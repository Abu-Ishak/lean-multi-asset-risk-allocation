"""QuantConnect LEAN Algorithm Framework: Modular Alpha, Covariance & Risk Model.

Demonstrates institutional separation of concerns using LEAN's AlphaModel,
PortfolioConstructionModel, and RiskManagementModel base classes across a
cross-asset ETF universe (SPY, QQQ, IEF, TLT, GLD, SHY).
"""

from datetime import timedelta
from AlgorithmImports import *
import numpy as np


class CrossAssetMomentumAlphaModel(AlphaModel):
    """Generates risk-adjusted trend insights from 63D/200D dual moving averages."""

    def __init__(self, fast: int = 63, slow: int = 200) -> None:
        self.fast = fast
        self.slow = slow
        self.indicators = {}

    def on_securities_changed(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        for security in changes.added_securities:
            sym = security.symbol
            self.indicators[sym] = {
                "fast": algorithm.sma(sym, self.fast, Resolution.DAILY),
                "slow": algorithm.sma(sym, self.slow, Resolution.DAILY),
            }

    def update(self, algorithm: QCAlgorithm, data: Slice) -> list[Insight]:
        insights: list[Insight] = []
        for sym, ind in self.indicators.items():
            if not (ind["fast"].is_ready and ind["slow"].is_ready):
                continue
            direction = (
                InsightDirection.UP
                if ind["fast"].current.value > ind["slow"].current.value
                else InsightDirection.FLAT
            )
            insights.append(Insight.price(sym, timedelta(days=21), direction))
        return insights


class MinimumVariancePortfolioConstructionModel(PortfolioConstructionModel):
    """Computes constrained minimum-variance target weights across active insights."""

    def __init__(self, lookback: int = 63, target_vol: float = 0.10) -> None:
        super().__init__()
        self.lookback = lookback
        self.target_vol = target_vol

    def create_targets(self, algorithm: QCAlgorithm, insights: list[Insight]) -> list[PortfolioTarget]:
        active = [i.symbol for i in insights if i.direction == InsightDirection.UP]
        if not active:
            return [PortfolioTarget(i.symbol, 0.0) for i in insights]

        history = algorithm.history(active, self.lookback + 1, Resolution.DAILY)
        if history.empty:
            return []

        returns = history["close"].unstack(level=0).pct_change().dropna()
        if len(returns) < 25:
            return []

        cov = returns.cov().values * 252.0 + np.eye(len(active)) * 1e-6
        ones = np.ones(len(active))
        raw_w = np.clip(np.linalg.inv(cov) @ ones, 0.0, None)
        weights = raw_w / raw_w.sum() if raw_w.sum() > 0 else ones / len(active)
        port_vol = float(np.sqrt(weights.T @ cov @ weights))
        vol_scale = min(1.0, self.target_vol / max(port_vol, 1e-4))

        targets = [PortfolioTarget(sym, float(weights[idx] * vol_scale)) for idx, sym in enumerate(active)]
        for insight in insights:
            if insight.symbol not in active:
                targets.append(PortfolioTarget(insight.symbol, 0.0))
        return targets


class DrawdownCircuitBreakerRiskModel(RiskManagementModel):
    """Cuts portfolio exposure when peak-to-trough portfolio drawdown exceeds limit."""

    def __init__(self, max_drawdown: float = 0.08) -> None:
        self.max_drawdown = max_drawdown
        self.peak_value = 0.0

    def manage_risk(self, algorithm: QCAlgorithm, targets: list[PortfolioTarget]) -> list[PortfolioTarget]:
        equity = algorithm.portfolio.total_portfolio_value
        self.peak_value = max(self.peak_value, equity)
        drawdown = (self.peak_value - equity) / max(self.peak_value, 1.0)
        if drawdown < self.max_drawdown:
            return targets
        return [PortfolioTarget(t.symbol, t.quantity * 0.5) for t in targets]


class ModularFactorFrameworkAlgorithm(QCAlgorithm):
    """Framework entrypoint wiring Alpha, Minimum-Variance Construction & Risk."""

    def initialize(self) -> None:
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(1_000_000)
        symbols = [Symbol.create(t, SecurityType.EQUITY, Market.USA) for t in ["SPY", "QQQ", "IEF", "TLT", "GLD", "SHY"]]
        self.set_universe_selection(ManualUniverseSelectionModel(symbols))
        self.set_alpha(CrossAssetMomentumAlphaModel())
        self.set_portfolio_construction(MinimumVariancePortfolioConstructionModel())
        self.set_risk_management(DrawdownCircuitBreakerRiskModel(max_drawdown=0.08))
        self.set_execution(ImmediateExecutionModel())
