# %% [markdown]
# # DSS for Risk — baseline dự báo và conformal
# Dữ liệu → lợi suất → đặc trưng trễ → Ridge → split conformal → đánh giá.
# Đây là tầng dự báo của Bước 1. Robust minimum variance và backtest danh mục
# sẽ được bổ sung sau khi định nghĩa cách liên kết bất định với tối ưu.
# Các khoảng hiện tại có độ phủ biên riêng từng ETF, không phải độ phủ đồng thời.

# %%
from pathlib import Path
import math

import numpy as np
import pandas as pd
import yfinance as yf
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Tất cả lợi suất và sai số được lưu dưới dạng thập phân; chỉ nhân 100 khi hiển thị.
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EEM"]
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"  # yfinance không bao gồm ngày kết thúc.
N_LAGS = 5
RIDGE_ALPHA = 1.0
MIS_COVERAGE = 0.10
DATA_DIR = Path("data")
CACHE_PATH = DATA_DIR / "etf_adjusted_close_2010_2025.csv"
REFRESH_DATA = False

assert 0 < MIS_COVERAGE < 1
assert N_LAGS >= 1

# %% [markdown]
# ## 1. Tải và kiểm tra giá điều chỉnh
# Lưu cache để những lần chạy sau dùng cùng một bản dữ liệu. Đặt REFRESH_DATA=True
# khi muốn tải lại. Nếu có giá thiếu, dừng để kiểm tra thay vì tự điền giá.
# Thư viện cần có: numpy, pandas, yfinance, scikit-learn, IPython.

# %%
if CACHE_PATH.exists() and not REFRESH_DATA:
    prices = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
else:
    raw_data = yf.download(
        TICKERS, start=START_DATE, end=END_DATE,
        interval="1d", auto_adjust=True, progress=False,
    )
    if raw_data.empty:
        raise ValueError("Không tải được dữ liệu ETF. Kiểm tra kết nối và nguồn dữ liệu.")
    prices = raw_data["Close"].copy()

missing_tickers = set(TICKERS) - set(prices.columns)
if missing_tickers:
    raise ValueError(f"Thiếu tài sản: {sorted(missing_tickers)}")

prices = prices.loc[:, TICKERS].sort_index()
prices.index = pd.DatetimeIndex(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices = prices.loc[
    (prices.index >= pd.Timestamp(START_DATE))
    & (prices.index < pd.Timestamp(END_DATE))
]
if prices.empty or prices.index.has_duplicates:
    raise ValueError("Dữ liệu rỗng hoặc ngày giao dịch bị trùng.")
if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
    raise ValueError("Giá chứa giá trị thiếu, không hữu hạn hoặc không dương.")

if REFRESH_DATA or not CACHE_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(CACHE_PATH)

print(f"Giá: {prices.shape}; {prices.index.min().date()} → {prices.index.max().date()}")
display(prices.head())

# %% [markdown]
# ## 2. Lợi suất đơn và thống kê mô tả

# %%
returns = prices.pct_change(fill_method=None).iloc[1:].copy()
if not np.isfinite(returns.to_numpy()).all():
    raise ValueError("Lợi suất có giá trị không hữu hạn.")

summary = returns.agg(["mean", "std", "min", "max"]).T * 100
summary.columns = ["Trung bình ngày (%)", "Độ lệch chuẩn ngày (%)",
                   "Ngày giảm mạnh nhất (%)", "Ngày tăng mạnh nhất (%)"]
display(summary.round(3))

# %% [markdown]
# ## 3. Đặc trưng và chia dữ liệu theo thời gian
# Mỗi ngày t dùng lợi suất t−1 đến t−5 của 5 ETF để dự báo lợi suất ngày t.
# Train: 2010–2018; calibration: 2019; test: 2020–2025.
# Các ngày đầu mỗi tập vẫn được phép dùng lịch sử của tập trước đó.

# %%
features = pd.concat(
    [returns.shift(lag).add_suffix(f"_lag{lag}") for lag in range(1, N_LAGS + 1)],
    axis=1,
)
valid_rows = features.notna().all(axis=1)
features = features.loc[valid_rows].copy()
targets = returns.loc[valid_rows].copy()

X_train, y_train = features.loc["2010":"2018"], targets.loc["2010":"2018"]
X_cal, y_cal = features.loc["2019":"2019"], targets.loc["2019":"2019"]
X_test, y_test = features.loc["2020":"2025"], targets.loc["2020":"2025"]

split_rows = []
for name, X_part, y_part in [
    ("Train", X_train, y_train), ("Calibration", X_cal, y_cal),
    ("Test", X_test, y_test),
]:
    assert len(X_part) > 0, f"Tập {name} rỗng."
    assert X_part.index.equals(y_part.index)
    split_rows.append({"Tập": name, "Số ngày": len(X_part),
                       "Bắt đầu": X_part.index.min().date(),
                       "Kết thúc": X_part.index.max().date()})
assert X_train.index.max() < X_cal.index.min()
assert X_cal.index.max() < X_test.index.min()
display(pd.DataFrame(split_rows).set_index("Tập"))

# %% [markdown]
# ## 4. Mô hình Ridge cố định
# Scaler và Ridge chỉ học trên train. Không chọn tham số bằng kết quả test.

# %%
model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
model.fit(X_train, y_train)

def predict_returns(X):
    """Giữ đúng ngày và thứ tự tài sản khi chuyển dự báo sang DataFrame."""
    return pd.DataFrame(model.predict(X), index=X.index, columns=targets.columns)

pred_cal = predict_returns(X_cal)
pred_test = predict_returns(X_test)

# %% [markdown]
# ## 5. Split conformal với sai số tuyệt đối
# Với n mẫu calibration, lấy phần tử thứ ceil((n+1)(1−alpha)) sau khi sắp xếp.
# Nếu vị trí vượt n, ngưỡng là vô hạn theo quy tắc hiệu chỉnh hữu hạn mẫu.
# Ngưỡng cố định riêng từng tài sản; khoảng bằng dự báo ± ngưỡng.
# Bảo đảm lý thuyết cần tính exchangeability; không mặc nhiên áp dụng cho chuỗi
# tài chính có phụ thuộc thời gian và thay đổi phân phối. Báo cáo độ phủ thực nghiệm.

# %%
def conformal_threshold(scores, alpha):
    """Ngưỡng hữu hạn mẫu cho các cột điểm sai số không âm."""
    if not 0 < alpha < 1:
        raise ValueError("alpha phải nằm trong (0, 1).")
    if scores.empty or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Điểm calibration rỗng hoặc không hữu hạn.")
    if (scores < 0).any().any():
        raise ValueError("Điểm sai số phải không âm.")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        return pd.Series(np.inf, index=scores.columns)
    return pd.Series(np.sort(scores.to_numpy(), axis=0)[rank - 1], index=scores.columns)

cal_scores = (y_cal - pred_cal).abs()
qhat = conformal_threshold(cal_scores, MIS_COVERAGE)
lower_test = pred_test.subtract(qhat, axis="columns")
upper_test = pred_test.add(qhat, axis="columns")
print(f"Mức phủ mục tiêu từng ETF: {100 * (1 - MIS_COVERAGE):.0f}%")
display(qhat.mul(100).rename("Ngưỡng (điểm phần trăm)").round(3).to_frame())

# %% [markdown]
# ## 6. Đánh giá trên test
# MAE và độ rộng tính bằng điểm phần trăm; độ phủ tính bằng % số ngày.
# So sánh dự báo Ridge với dự báo bằng 0 và trung bình train.

# %%
covered_test = y_test.ge(lower_test) & y_test.le(upper_test)
interval_results = pd.DataFrame({
    "Độ phủ test (%)": covered_test.mean() * 100,
    "Độ rộng TB (điểm phần trăm)": (upper_test - lower_test).mean() * 100,
    "Số ngày bỏ sót": (~covered_test).sum(),
})
coverage_by_year = covered_test.groupby(covered_test.index.year).mean() * 100
coverage_by_year.index.name = "Năm"
mae_results = pd.DataFrame({
    "Ridge": (y_test - pred_test).abs().mean(),
    "Dự báo bằng 0": y_test.abs().mean(),
    "Trung bình train": y_test.subtract(y_train.mean(), axis="columns").abs().mean(),
}) * 100

display(interval_results.round(3))
display(coverage_by_year.round(2))
display(mae_results.round(3))

spy_intervals = pd.DataFrame({
    "Thực tế (%)": y_test["SPY"] * 100,
    "Dự báo (%)": pred_test["SPY"] * 100,
    "Đầu mút dưới (%)": lower_test["SPY"] * 100,
    "Đầu mút trên (%)": upper_test["SPY"] * 100,
})
display(spy_intervals.head().round(3))

# %% [markdown]
# ## 7. Robust minimum variance: định nghĩa baseline
# Long-only, tổng tỷ trọng bằng 1. Ước lượng covariance S từ 252 ngày trước đó.
# Tập bất định: {S + diag(d): 0 ≤ d_i ≤ lambda * q_i²}.
# Min_w max_d wᵀ(S + diag(d))w tương đương min_w wᵀS w + lambda Σ q_i² w_i².
# q_i là ngưỡng conformal hiệu chỉnh năm 2019, cố định trong giai đoạn test.
# Đây là lựa chọn mô hình hóa để thử nghiệm: conformal không bảo đảm covariance
# thực nằm trong tập này. Khoảng lợi suất không tự chuyển thành vùng tin cậy covariance.
# Dự báo Ridge được dùng để tạo sai số conformal, không đi vào mục tiêu minimum variance.
# Thêm robust control có cùng tổng ngân sách nhưng phân bổ theo MSE calibration,
# nhằm phân biệt lợi ích của conformal với hiệu ứng regularization thông thường.
# Cần cvxpy; nếu thiếu chạy %pip install cvxpy rồi restart kernel.

# %%
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

# %% [markdown]
# ## 8. Backtest có chi phí giao dịch
# Tái cân bằng đầu mỗi tháng, dùng thông tin đến ngày trước đó; nhận lợi suất ngày t
# sau khi giao dịch. Các tỷ trọng tự trôi giữa các lần tái cân bằng.
# Chi phí 10 bps trên mỗi đơn vị mua/bán, turnover = tổng |thay đổi tỷ trọng|.
# Tính cả chi phí phân bổ ban đầu từ tiền mặt. Đây là giả định thực thi ở giá đóng cửa
# ngày trước đó; chưa mô hình hóa trượt giá hoặc khoảng chờ thực thi.
# lambda=1 được cố định trước test, không tối ưu bằng kết quả test.

# %%
COV_LOOKBACK = 252
TRADING_COST_BPS = 10
ROBUSTNESS = 1.0
records, allocations = backtest(
    returns, y_test.index, qhat, cal_scores,
    lookback=COV_LOOKBACK, cost_bps=TRADING_COST_BPS, robustness=ROBUSTNESS,
)
portfolio_results = performance(records)
display(portfolio_results.round(3))
display(allocations.head(8).round(4))

# %%
import matplotlib.pyplot as plt
net_returns = records.pivot(index="Date", columns="Strategy", values="Net")
wealth = (1 + net_returns).cumprod()
drawdown = wealth / wealth.cummax().clip(lower=1) - 1
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
wealth.plot(ax=axes[0], title="Portfolio wealth after transaction costs (initial wealth = 1)")
axes[0].set_ylabel("Wealth")
(drawdown * 100).plot(ax=axes[1], title="Drawdown")
axes[1].set_ylabel("Drawdown (%)")
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()

# %%
RESULT_DIR = Path("results_baseline")
RESULT_DIR.mkdir(exist_ok=True)
interval_results.to_csv(RESULT_DIR / "conformal_coverage.csv")
coverage_by_year.to_csv(RESULT_DIR / "coverage_by_year.csv")
mae_results.to_csv(RESULT_DIR / "forecast_mae.csv")
portfolio_results.to_csv(RESULT_DIR / "portfolio_metrics.csv")
records.to_csv(RESULT_DIR / "daily_backtest.csv", index=False)
allocations.to_csv(RESULT_DIR / "monthly_weights.csv", index=False)
fig.savefig(RESULT_DIR / "wealth_drawdown.png", dpi=160)
print("Đã lưu bảng và biểu đồ tại:", RESULT_DIR.resolve())

# %% [markdown]
# ## 9. Cách đọc kết quả cho báo cáo
# So sánh Robust CP với MV về biến động, drawdown, Sharpe sau phí và turnover.
# So sánh thêm Robust control: nếu tương đương, chưa có bằng chứng lợi ích riêng
# của conformal. Sharpe dùng lãi suất phi rủi ro bằng 0; CAGR annualize theo 252 ngày.
# Đây là kiểm tra khả thi trên 5 ETF, chưa chứng minh ý nghĩa thống kê hay tổng quát.
# Calibration 2019 có thể không đại diện cho 2020–2025. Chưa có conformal đa biến
# hoặc regime-adaptive; đây là các phần nghiên cứu tiếp theo.
