"""Long-only portfolio baseline with past-only covariance and trading costs."""
import numpy as np
import pandas as pd
import cvxpy as cp


def minimum_variance(covariance, diagonal_budget=None):
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2
    if diagonal_budget is not None:
        covariance = covariance + np.diag(diagonal_budget)
    # Scale objective for numerical stability; optimum is unchanged.
    scale = max(np.trace(covariance) / len(covariance), 1e-12)
    weights = cp.Variable(len(covariance))
    problem = cp.Problem(cp.Minimize(cp.quad_form(weights, cp.psd_wrap(covariance / scale))),
                         [weights >= 0, cp.sum(weights) == 1])
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise RuntimeError(f"Portfolio optimization failed: {problem.status}")
    result = np.maximum(weights.value, 0)
    return result / result.sum()


def backtest(returns, test_index, qhat, calibration_scores, lookback=252,
             cost_bps=10, robustness=1.0):
    """Rebalance first trading day of each month, before that day's return.

    Covariance uses only dates strictly before the decision date.
    Turnover is sum(abs(target - drifted weights)); cost_bps is per traded dollar.
    Initial allocation from cash is charged 100% turnover.
    """
    if lookback < 2 or cost_bps < 0 or robustness < 0:
        raise ValueError("Invalid backtest configuration")
    cols = returns.columns
    q = qhat.reindex(cols).to_numpy()
    if not np.isfinite(q).all():
        raise ValueError("Conformal budgets must be finite")
    # Matched total budget control: distribute budget by calibration residual MSE.
    variance = calibration_scores.reindex(columns=cols).pow(2).mean().to_numpy()
    if variance.sum() <= 0:
        raise ValueError("Calibration residual scores have zero variance")
    budgets = {
        "MV": None,
        "Robust CP": robustness * q ** 2,
        "Robust control": robustness * (q ** 2).sum() * variance / variance.sum(),
    }
    names = ["1/N", *budgets]
    held = {name: np.zeros(len(cols)) for name in names}
    records, allocations = [], []
    previous_month = None
    for date in test_index:
        realized = returns.loc[date, cols].to_numpy()
        rebalance = previous_month != date.to_period("M")
        if rebalance:
            history = returns.loc[returns.index < date].tail(lookback)
            if len(history) != lookback:
                raise ValueError(f"Not enough past observations at {date}")
            covariance = history.cov().to_numpy()
            targets = {"1/N": np.repeat(1 / len(cols), len(cols))}
            targets.update({name: minimum_variance(covariance, budget)
                            for name, budget in budgets.items()})
        for name in names:
            target = targets[name] if rebalance else held[name]
            turnover = np.abs(target - held[name]).sum() if rebalance else 0.0
            cost = cost_bps / 10000 * turnover
            gross_return = float(target @ realized)
            net_return = (1 - cost) * (1 + gross_return) - 1
            held[name] = target * (1 + realized) / (1 + gross_return)
            records.append({"Date": date, "Strategy": name, "Gross": gross_return,
                            "Net": net_return, "Turnover": turnover, "Cost": cost})
            if rebalance:
                allocations.append({"Date": date, "Strategy": name, **dict(zip(cols, target))})
        previous_month = date.to_period("M")
    return pd.DataFrame(records), pd.DataFrame(allocations)


def performance(records):
    rows = []
    for name, frame in records.groupby("Strategy", sort=False):
        net = frame.set_index("Date")["Net"]
        wealth = (1 + net).cumprod()
        peak = wealth.cummax().clip(lower=1.0)
        volatility = net.std(ddof=1) * np.sqrt(252)
        rows.append({"Strategy": name,
                     "CAGR (%)": (wealth.iloc[-1] ** (252 / len(net)) - 1) * 100,
                     "Volatility (%)": volatility * 100,
                     "Sharpe (rf=0)": net.mean() * 252 / volatility if volatility > 0 else np.nan,
                     "Max drawdown (%)": (wealth / peak - 1).min() * 100,
                     "Annual turnover (x)": frame.Turnover.sum() * 252 / len(net),
                     "Total return (%)": (wealth.iloc[-1] - 1) * 100})
    return pd.DataFrame(rows).set_index("Strategy")
