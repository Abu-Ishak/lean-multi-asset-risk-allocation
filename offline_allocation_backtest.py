"""Standalone Multi-Regime Simulation & Tearsheet Generator for LEAN Allocation Logic.

Tests the exact Minimum-Variance + Volatility-Targeting + 200D Regime Filter logic
implemented in `regime_vol_target_allocation.py` against a Static 60/40 Benchmark
and an Equal-Weight Portfolio across 1,500 simulated trading days spanning four
macro regimes (Expansion, Rate-Hike Shock, Liquidity Drawdown, Recovery), net of
10 bps one-way rebalance transaction costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ASSETS = ["SPY", "QQQ", "IEF", "TLT", "GLD", "SHY"]


def simulate_multi_regime_returns(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 1,500 days of cross-asset daily returns across 4 macro regimes."""
    rng = np.random.default_rng(seed)
    regimes = [
        # (days, annualized means [SPY, QQQ, IEF, TLT, GLD, SHY], annualized vols, eq_bond_corr)
        (450, [0.14, 0.19, 0.035, 0.04, 0.07, 0.025], [0.14, 0.19, 0.055, 0.11, 0.13, 0.012], -0.25),
        (320, [-0.14, -0.22, -0.07, -0.16, 0.08, 0.045], [0.22, 0.28, 0.095, 0.18, 0.15, 0.014], 0.52),
        (250, [-0.24, -0.28, 0.09, 0.15, 0.12, 0.040], [0.29, 0.34, 0.085, 0.17, 0.17, 0.012], -0.45),
        (480, [0.16, 0.21, 0.045, 0.05, 0.09, 0.042], [0.15, 0.20, 0.060, 0.12, 0.13, 0.012], -0.15),
    ]
    blocks = []
    labels = []
    for idx, (n_days, mu_ann, vol_ann, eq_bond_corr) in enumerate(regimes):
        corr = np.eye(6)
        corr[0, 1] = corr[1, 0] = 0.88
        corr[2, 3] = corr[3, 2] = 0.86
        for e in (0, 1):
            for b in (2, 3):
                corr[e, b] = corr[b, e] = eq_bond_corr
        corr[0, 4] = corr[4, 0] = 0.10
        corr[2, 4] = corr[4, 2] = 0.25
        vols = np.array(vol_ann) / np.sqrt(252.0)
        cov = np.outer(vols, vols) * corr
        mu_daily = np.array(mu_ann) / 252.0
        shocks = rng.multivariate_normal(mu_daily, cov, size=n_days)
        blocks.append(shocks)
        labels.append(np.full(n_days, idx))
    return np.vstack(blocks), np.concatenate(labels)


def run_allocation_backtest(returns: np.ndarray, cost_bps: float = 10.0) -> Dict[str, np.ndarray]:
    """Run Regime Vol-Targeted Min-Variance vs. Static 60/40 and Equal-Weight."""
    n_days, n_assets = returns.shape
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    lookback = 63
    warmup = 200
    target_vol = 0.10
    cost_rate = cost_bps / 10_000.0

    strat_ret = np.zeros(n_days - warmup)
    b6040_ret = np.zeros(n_days - warmup)
    eqw_ret = np.zeros(n_days - warmup)

    w_strat = np.array([0.16, 0.10, 0.28, 0.16, 0.15, 0.15])
    w_6040 = np.array([0.45, 0.15, 0.30, 0.10, 0.00, 0.00])
    w_eqw = np.ones(n_assets) / n_assets
    total_turnover = 0.0

    for t in range(warmup, n_days):
        idx = t - warmup
        if idx % 21 == 0:
            win = returns[t - lookback : t, :5]
            sample_cov = np.cov(win, rowvar=False) * 252.0
            cov = 0.80 * sample_cov + 0.20 * np.diag(np.diag(sample_cov)) + np.eye(5) * 1e-6
            raw = np.clip(np.linalg.inv(cov) @ np.ones(5), 0.0, None)
            w_risk = raw / raw.sum() if raw.sum() > 0 else np.ones(5) / 5.0
            w_risk = np.minimum(w_risk, 0.40)
            w_risk /= w_risk.sum()

            sma_200 = np.mean(prices[t - warmup : t, :5], axis=0)
            below_trend = prices[t - 1, :5] < sma_200
            w_risk[below_trend] *= 0.50

            port_vol = float(np.sqrt(w_risk.T @ cov @ w_risk))
            vol_scale = min(1.0, target_vol / max(port_vol, 1e-4))
            new_w = np.zeros(6)
            new_w[:5] = w_risk * vol_scale
            new_w[5] = max(0.0, 1.0 - new_w[:5].sum())

            turnover = float(np.sum(np.abs(new_w - w_strat)))
            total_turnover += turnover
            tc = turnover * cost_rate
            w_strat = new_w
        else:
            tc = 0.0

        strat_ret[idx] = float(w_strat @ returns[t]) - tc
        b6040_ret[idx] = float(w_6040 @ returns[t])
        eqw_ret[idx] = float(w_eqw @ returns[t])

    return {
        "Regime Vol-Target MinVar": strat_ret,
        "Static 60/40 Benchmark": b6040_ret,
        "Equal-Weight (1/N)": eqw_ret,
        "annual_turnover": np.array([total_turnover / ((n_days - warmup) / 252.0)]),
    }


def compute_stats(r: np.ndarray, bench: np.ndarray) -> Dict[str, float]:
    equity = np.cumprod(1.0 + r)
    n_years = len(r) / 252.0
    cagr = float(equity[-1] ** (1.0 / n_years) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(252.0))
    sharpe = float((np.mean(r) * 252.0) / max(vol, 1e-8))
    downside = r[r < 0.0]
    down_dev = float(np.std(downside, ddof=1) * np.sqrt(252.0)) if len(downside) > 1 else vol
    sortino = float((np.mean(r) * 252.0) / max(down_dev, 1e-8))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    beta = float(np.cov(r, bench)[0, 1] / max(np.var(bench), 1e-12))
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": max_dd, "Calmar": calmar, "Beta": beta}


def build_tearsheet_svg(res: Dict[str, np.ndarray], out_path: Path) -> None:
    """Create 3-panel SVG tearsheet: Cumulative Equity, Rolling 63D Vol, Drawdown."""
    series_meta = [
        ("Regime Vol-Target MinVar", "#34d399", 2.3),
        ("Static 60/40 Benchmark", "#f59e0b", 1.8),
        ("Equal-Weight (1/N)", "#38bdf8", 1.5),
    ]
    n = len(res["Regime Vol-Target MinVar"])
    step = max(1, n // 260)
    xs_idx = np.arange(0, n, step)

    def map_x(i: int) -> float:
        return 76.0 + (i / max(n - 1, 1)) * 845.0

    equities = {k: np.cumprod(1.0 + res[k]) * 100.0 for k, _, _ in series_meta}
    eq_min = min(float(np.min(v)) for v in equities.values()) * 0.97
    eq_max = max(float(np.max(v)) for v in equities.values()) * 1.03

    def poly_points(vals: np.ndarray, y_top: float, y_bot: float, vmin: float, vmax: float) -> str:
        pts = []
        for i in xs_idx:
            px = map_x(int(i))
            py = y_bot - ((float(vals[i]) - vmin) / max(vmax - vmin, 1e-6)) * (y_bot - y_top)
            pts.append(f"{px:.1f},{py:.1f}")
        return " ".join(pts)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 620" width="960" height="620">',
        '<defs><style>',
        '.bg{fill:#0b132b}.panel{fill:#131f3d;stroke:#24345c;stroke-width:1}',
        '.title{fill:#f8fafc;font:700 17px Segoe UI,Arial,sans-serif}',
        '.sub{fill:#94a3b8;font:12px Segoe UI,Arial,sans-serif}',
        '.lbl{fill:#e2e8f0;font:700 12px Segoe UI,Arial,sans-serif}',
        '.ax{fill:#94a3b8;font:10.5px Consolas,monospace}',
        '.grid{stroke:#1e293b;stroke-width:1;stroke-dasharray:3 3}',
        '</style></defs>',
        '<rect class="bg" width="960" height="620" rx="10"/>',
        '<text x="24" y="32" class="title">LEAN MULTI-ASSET PORTFOLIO SIMULATION (NET OF 10 BPS REBALANCE COSTS)</text>',
        '<text x="24" y="52" class="sub">Calibrated Multi-Regime Daily Simulation (1,300 OOS Days: Bull, Rate Shock, Crisis, Recovery)  |  Target Vol: 10.0%</text>',
        '<rect class="panel" x="24" y="68" width="912" height="190" rx="8"/>',
        '<text x="40" y="90" class="lbl">1. CUMULATIVE WEALTH INDEX (BASE = 100)</text>',
    ]

    for y_val in np.linspace(eq_min, eq_max, 4):
        py = 242.0 - ((y_val - eq_min) / max(eq_max - eq_min, 1e-6)) * 136.0
        svg.append(f'<line x1="76" y1="{py:.1f}" x2="921" y2="{py:.1f}" class="grid"/>')
        svg.append(f'<text x="68" y="{py + 4:.1f}" text-anchor="end" class="ax">{y_val:.0f}</text>')

    for idx, (name, col, sw) in enumerate(series_meta):
        pts = poly_points(equities[name], 106.0, 242.0, eq_min, eq_max)
        svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="{pts}"/>')
        lx = 470 + idx * 155
        svg.append(f'<line x1="{lx}" y1="86" x2="{lx + 16}" y2="86" stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<text x="{lx + 22}" y="90" class="ax">{name.split()[0]} {name.split()[1]}</text>')

    # Panel 2: Rolling 63D Realized Volatility
    svg.append('<rect class="panel" x="24" y="272" width="912" height="156" rx="8"/>')
    svg.append('<text x="40" y="294" class="lbl">2. ROLLING 63-DAY ANNUALIZED VOLATILITY vs. 10.0% TARGET CAP</text>')
    rolls = {}
    for name, _, _ in series_meta[:2]:
        r = res[name]
        rv = np.array([np.std(r[max(0, i - 62) : i + 1], ddof=1) * np.sqrt(252.0) * 100.0 if i > 5 else 10.0 for i in range(n)])
        rolls[name] = rv
    v_max = max(float(np.max(rolls["Static 60/40 Benchmark"])), 18.0) * 1.05
    t_py = 414.0 - (10.0 / v_max) * 106.0
    svg.append(f'<line x1="76" y1="{t_py:.1f}" x2="921" y2="{t_py:.1f}" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="5 3"/>')
    svg.append(f'<text x="68" y="{t_py + 4:.1f}" text-anchor="end" class="ax">10%</text>')
    for name, col, sw in series_meta[:2]:
        pts = poly_points(rolls[name], 308.0, 414.0, 0.0, v_max)
        svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="{pts}"/>')

    # Panel 3: Underwater Drawdown
    svg.append('<rect class="panel" x="24" y="442" width="912" height="158" rx="8"/>')
    svg.append('<text x="40" y="464" class="lbl">3. UNDERWATER DRAWDOWN COMPARISON (%)</text>')
    dds = {}
    for name, _, _ in series_meta:
        eq = equities[name]
        dds[name] = (eq / np.maximum.accumulate(eq) - 1.0) * 100.0
    dd_min = min(float(np.min(v)) for v in dds.values()) * 1.08
    for lvl in [0.0, dd_min * 0.5, dd_min]:
        py = 584.0 - ((lvl - dd_min) / max(0.0 - dd_min, 1e-6)) * 104.0
        svg.append(f'<line x1="76" y1="{py:.1f}" x2="921" y2="{py:.1f}" class="grid"/>')
        svg.append(f'<text x="68" y="{py + 4:.1f}" text-anchor="end" class="ax">{lvl:.1f}%</text>')
    for name, col, sw in series_meta:
        pts = poly_points(dds[name], 480.0, 584.0, dd_min, 0.0)
        svg.append(f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="{pts}"/>')

    svg.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    returns, _ = simulate_multi_regime_returns(seed=42)
    res = run_allocation_backtest(returns, cost_bps=10.0)
    bench = res["Static 60/40 Benchmark"]
    print("=== LEAN MULTI-ASSET ALLOCATION SIMULATION (NET 10 BPS) ===")
    print(f"Annualized Two-Way Portfolio Turnover: {res['annual_turnover'][0]:.2%}\n")
    for name in ["Regime Vol-Target MinVar", "Static 60/40 Benchmark", "Equal-Weight (1/N)"]:
        s = compute_stats(res[name], bench)
        print(
            f"{name:26s} | CAGR: {s['CAGR']:6.2%} | Vol: {s['Vol']:6.2%} | "
            f"Sharpe: {s['Sharpe']:5.2f} | Sortino: {s['Sortino']:5.2f} | "
            f"MaxDD: {s['MaxDD']:7.2%} | Calmar: {s['Calmar']:5.2f} | Beta: {s['Beta']:4.2f}"
        )

    out_svg = Path(__file__).resolve().parent / "assets" / "lean_allocation_tearsheet.svg"
    build_tearsheet_svg(res, out_svg)
    print(f"\nGenerated SVG tearsheet: {out_svg}")


if __name__ == "__main__":
    main()
