# %% [markdown]
# # DSS for Risk — baseline dự báo và conformal
# Dữ liệu → lợi suất → đặc trưng trễ → Ridge → split conformal → đánh giá.
# Bản mở rộng Bước 1: Ridge, MLP PyTorch, GARCH arch và robust MV.
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
# ## 7. MLP bằng PyTorch — dự báo lợi suất
# MLP nhỏ 25→32→16→5, Adam, MSE; không tìm tham số trên test.
# Dùng 2018 làm validation nội bộ, 2010–2017 để chọn số epoch (early stopping).
# Sau đó huấn luyện lại từ đầu trên toàn bộ train 2010–2018 với số epoch đã chọn.
# Scaler đầu vào và đầu ra đều chỉ học trên dữ liệu huấn luyện tương ứng.
# MLP là baseline học sâu, chưa phải LSTM/PatchTST. Seed cố định không đảm bảo
# kết quả giống tuyệt đối giữa mọi phiên bản thư viện/phần cứng.

# %%
import copy
import json
import platform
import random
import warnings
from importlib.metadata import version
import torch
from torch import nn
from arch import arch_model

SEED = 42
MAX_EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
torch.set_num_threads(2)

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

def new_mlp():
    seed_everything()
    return nn.Sequential(nn.Linear(features.shape[1], 32), nn.ReLU(),
                         nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, len(TICKERS)))

def tensor(values):
    return torch.tensor(np.asarray(values), dtype=torch.float32)

def train_mlp(X_fit, y_fit, epochs, X_val=None, y_val=None):
    sx, sy = StandardScaler(), StandardScaler()
    x = tensor(sx.fit_transform(X_fit))
    y = tensor(sy.fit_transform(y_fit))
    net = new_mlp()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale = np.inf, 0, 0
    logs = []
    if X_val is not None:
        xv, yv = tensor(sx.transform(X_val)), tensor(sy.transform(y_val))
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        optimizer.step()
        row = {"Epoch": epoch, "Train MSE standardized": float(loss.detach())}
        if X_val is not None:
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(xv), yv))
            row["Validation MSE standardized"] = val_loss
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
        logs.append(row)
        if X_val is not None and stale >= PATIENCE:
            break
    return net, sx, sy, best_epoch if X_val is not None else epochs, pd.DataFrame(logs)

inner_mask = X_train.index.year < 2018
_, _, _, selected_epochs, training_log = train_mlp(
    X_train.loc[inner_mask], y_train.loc[inner_mask], MAX_EPOCHS,
    X_train.loc[~inner_mask], y_train.loc[~inner_mask],
)
mlp, mlp_x_scaler, mlp_y_scaler, _, _ = train_mlp(X_train, y_train, selected_epochs)
mlp.eval()

def predict_mlp(X):
    with torch.no_grad():
        prediction = mlp(tensor(mlp_x_scaler.transform(X))).numpy()
    return pd.DataFrame(mlp_y_scaler.inverse_transform(prediction), index=X.index, columns=TICKERS)

mlp_cal, mlp_test = predict_mlp(X_cal), predict_mlp(X_test)
mlp_scores = (y_cal - mlp_cal).abs()
mlp_qhat = conformal_threshold(mlp_scores, MIS_COVERAGE)
print("MLP epochs selected using 2018 validation:", selected_epochs)
display(training_log.tail())

# %% [markdown]
# ## 8. GARCH(1,1) bằng arch — dự báo biến động
# Mỗi ETF có mô hình ConstantMean + GARCH(1,1), phân phối Normal, fit 2010–2018.
# Không refit trên calibration/test. Hệ số cố định; trạng thái variance cập nhật
# sau khi quan sát lợi suất mỗi ngày: h[t+1]=omega+a*(r[t]-mu)^2+b*h[t].
# h[t] được tạo TRƯỚC khi quan sát r[t]. Fit với lợi suất nhân 100 để ổn định số học;
# đổi variance về đơn vị thập phân bằng chia 10000. Đây không phải DCC-GARCH.

# %%
def fit_garch_and_forecast(train_returns, future_returns):
    sigma = pd.DataFrame(index=future_returns.index, columns=train_returns.columns, dtype=float)
    parameter_rows = []
    states = {}
    for asset in train_returns.columns:
        fitted = arch_model(train_returns[asset] * 100, mean="Constant", vol="GARCH",
                            p=1, q=1, dist="normal", rescale=False).fit(disp="off")
        if fitted.convergence_flag != 0:
            raise RuntimeError(f"GARCH did not converge for {asset}")
        p = fitted.params
        h = float(fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        initial_h = h
        for date, observed in future_returns[asset].items():
            if not np.isfinite(h) or h <= 0:
                raise ValueError(f"Invalid GARCH forecast for {asset} at {date}")
            sigma.loc[date, asset] = np.sqrt(h) / 100
            h = float(p['omega'] + p['alpha[1]'] * (observed * 100 - p['mu']) ** 2 + p['beta[1]'] * h)
        states[asset] = {"mu": float(p['mu']), "omega": float(p['omega']),
                         "a": float(p['alpha[1]']), "b": float(p['beta[1]']), "initial_h": initial_h}
        parameter_rows.append({"Asset": asset, **p.to_dict(), "Persistence": p['alpha[1]']+p['beta[1]'],
                               "Convergence flag": fitted.convergence_flag})
    return sigma, pd.DataFrame(parameter_rows).set_index("Asset"), states

future_returns = returns.loc["2019":"2025"]
garch_sigma, garch_parameters, garch_states = fit_garch_and_forecast(
    returns.loc["2010":"2018"], future_returns)
sigma_cal, sigma_test = garch_sigma.loc[y_cal.index], garch_sigma.loc[y_test.index]
display(garch_parameters.round(5))

# %% [markdown]
# ## 9. Hai loại conformal và đối chứng Gaussian GARCH
# Giữ split conformal tuyệt đối cho Ridge/MLP. Thêm split conformal chuẩn hóa theo
# sigma GARCH: score=|r-MLP|/sigma; ngưỡng chuẩn hóa vẫn cố định trên calibration.
# Khoảng MLP-GARCH CP: MLP ± q_normalized*sigma[t]. Đây chưa phải regime-adaptive.
# GARCH Gaussian: mu_train ± 1.645*sigma[t]; khoảng tham số, không phải conformal.
# Đánh giá variance bằng QLIKE và MSE với proxy (r-mu_train)^2; proxy nhiễu,
# không phải realized volatility intraday. QLIKE càng thấp càng tốt.

# %%
normalized_scores = mlp_scores / sigma_cal
normalized_qhat = conformal_threshold(normalized_scores, MIS_COVERAGE)
dynamic_width_cal = sigma_cal.multiply(normalized_qhat, axis='columns')
dynamic_width_test = sigma_test.multiply(normalized_qhat, axis='columns')
garch_mean = garch_parameters['mu'].reindex(TICKERS) / 100
garch_test_mean = pd.DataFrame(np.tile(garch_mean.to_numpy(), (len(y_test), 1)),
                               index=y_test.index, columns=TICKERS)
intervals = {
    'Ridge CP': (pred_test, pd.DataFrame(np.tile(qhat.to_numpy(), (len(y_test), 1)), index=y_test.index, columns=TICKERS)),
    'MLP CP': (mlp_test, pd.DataFrame(np.tile(mlp_qhat.to_numpy(), (len(y_test), 1)), index=y_test.index, columns=TICKERS)),
    'MLP GARCH CP': (mlp_test, dynamic_width_test),
    'GARCH Gaussian': (garch_test_mean, sigma_test * 1.6448536269514722),
}
coverage_rows, yearly_rows = [], []
for name, (center, width) in intervals.items():
    covered = y_test.ge(center - width) & y_test.le(center + width)
    for asset in TICKERS:
        coverage_rows.append({'Method': name, 'Asset': asset,
                             'Coverage (%)': covered[asset].mean()*100,
                             'Mean width (percentage points)': 2*width[asset].mean()*100})
    for year, row in (covered.groupby(covered.index.year).mean()*100).iterrows():
        yearly_rows.append({'Method': name, 'Year': year, **row.to_dict()})
all_coverage = pd.DataFrame(coverage_rows)
all_yearly = pd.DataFrame(yearly_rows)
all_mae = pd.DataFrame({
    'Ridge': (y_test-pred_test).abs().mean(), 'MLP': (y_test-mlp_test).abs().mean(),
    'Zero': y_test.abs().mean(), 'Train mean': y_test.subtract(y_train.mean(), axis='columns').abs().mean(),
}) * 100
variance_proxy = y_test.subtract(garch_mean, axis='columns').pow(2)
variance_predictions = {
    'GARCH': sigma_test.pow(2),
    'Rolling 252': returns.rolling(252).var().shift(1).loc[y_test.index],
}
vol_rows=[]
for name, variance_pred in variance_predictions.items():
    assert (variance_pred > 0).all().all()
    for asset in TICKERS:
        v, proxy = variance_pred[asset], variance_proxy[asset]
        vol_rows.append({'Method': name, 'Asset': asset,
                        'QLIKE': float((np.log(v)+proxy/v).mean()),
                        'Variance MSE': float((proxy-v).pow(2).mean())})
volatility_results = pd.DataFrame(vol_rows)
display(all_mae.round(4))
display(all_coverage.round(3))
display(volatility_results)

# %% [markdown]
# ## 10. Danh mục — tách đóng góp của MLP, GARCH và conformal
# Cùng long-only, covariance/correlation rolling 252 ngày, lịch tháng và phí 10 bps.
# GARCH covariance: D_sigma[t] Corr_past D_sigma[t] (không phải DCC).
# Robust covariance: S_base + diag(d), 0≤d≤width[t]^2; nghiệm dùng upper budget.
# Chiều rộng khoảng lợi suất KHÔNG phải vùng tin cậy covariance: liên kết này
# vẫn là giả định baseline, cần nêu rõ khi báo cáo với thầy.
# Mỗi phiên bản CP có robust control cùng tổng ngân sách, phân bổ theo MSE
# (hoặc MSE chuẩn hóa nhân sigma²) để kiểm tra hiệu ứng regularization.

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


def extended_backtest(returns, index, sigma, width, mlp_q, ridge_q,
                      mlp_cal_scores, normalized_cal_scores, cost_bps=10):
    names = ['1/N', 'MV', 'Ridge CP MV', 'MLP CP MV', 'MLP control MV',
             'GARCH MV', 'MLP GARCH CP MV', 'GARCH control MV']
    held = {name: np.zeros(len(TICKERS)) for name in names}
    mse = mlp_cal_scores.pow(2).mean().reindex(TICKERS).to_numpy()
    normalized_mse = normalized_cal_scores.pow(2).mean().reindex(TICKERS).to_numpy()
    rows, weight_rows = [], []
    month = None
    for date in index:
        rebalance = date.to_period('M') != month
        if rebalance:
            history = returns.loc[returns.index < date, TICKERS].tail(252)
            assert len(history) == 252
            sample_cov = history.cov().to_numpy()
            daily_sigma = sigma.loc[date, TICKERS].to_numpy()
            garch_cov = np.diag(daily_sigma) @ history.corr().to_numpy() @ np.diag(daily_sigma)
            ridge_budget = ridge_q.reindex(TICKERS).to_numpy()**2
            mlp_budget = mlp_q.reindex(TICKERS).to_numpy()**2
            dynamic_budget = width.loc[date, TICKERS].to_numpy()**2
            dynamic_control = normalized_mse * daily_sigma**2
            dynamic_control *= dynamic_budget.sum() / dynamic_control.sum()
            targets = {
                '1/N': np.ones(len(TICKERS))/len(TICKERS),
                'MV': minimum_variance(sample_cov),
                'Ridge CP MV': minimum_variance(sample_cov, ridge_budget),
                'MLP CP MV': minimum_variance(sample_cov, mlp_budget),
                'MLP control MV': minimum_variance(sample_cov, mlp_budget.sum()*mse/mse.sum()),
                'GARCH MV': minimum_variance(garch_cov),
                'MLP GARCH CP MV': minimum_variance(garch_cov, dynamic_budget),
                'GARCH control MV': minimum_variance(garch_cov, dynamic_control),
            }
        realized = returns.loc[date, TICKERS].to_numpy()
        for name in names:
            target = targets[name] if rebalance else held[name]
            turnover = np.abs(target-held[name]).sum() if rebalance else 0.0
            cost = cost_bps/10000*turnover
            gross = float(target @ realized)
            net = (1-cost)*(1+gross)-1
            held[name] = target*(1+realized)/(1+gross)
            rows.append({'Date': date, 'Strategy': name, 'Gross': gross, 'Net': net,
                         'Turnover': turnover, 'Cost': cost})
            if rebalance:
                weight_rows.append({'Date': date, 'Strategy': name, **dict(zip(TICKERS, target))})
        month=date.to_period('M')
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)

extended_records, extended_weights = extended_backtest(
    returns, y_test.index, sigma_test, dynamic_width_test,
    mlp_qhat, qhat, mlp_scores, normalized_scores)
extended_metrics = performance(extended_records)
display(extended_metrics.round(3))

# %% [markdown]
# ## 11. Lưu kết quả và cấu hình tái lập
# Kết quả mới vào results_torch_garch; không ghi đè results_baseline.
# Chưa có kiểm định thống kê, nhiều seed, regime detection hoặc tái cân bằng
# theo mức bất định. Không kết luận mô hình mới có hiệu quả trước khi đọc số liệu.

# %%
import matplotlib.pyplot as plt
RESULT_DIR = Path('results_torch_garch')
RESULT_DIR.mkdir(exist_ok=True)
all_mae.to_csv(RESULT_DIR/'forecast_mae.csv')
all_coverage.to_csv(RESULT_DIR/'conformal_coverage.csv', index=False)
all_yearly.to_csv(RESULT_DIR/'coverage_by_year.csv', index=False)
volatility_results.to_csv(RESULT_DIR/'volatility_metrics.csv', index=False)
garch_parameters.to_csv(RESULT_DIR/'garch_parameters.csv')
training_log.to_csv(RESULT_DIR/'mlp_training_log.csv', index=False)
extended_metrics.to_csv(RESULT_DIR/'portfolio_metrics.csv')
extended_records.to_csv(RESULT_DIR/'daily_backtest.csv', index=False)
extended_weights.to_csv(RESULT_DIR/'monthly_weights.csv', index=False)
sigma_test.to_csv(RESULT_DIR/'garch_sigma_test.csv')
predictions = pd.concat({'Actual':y_test,'Ridge':pred_test,'MLP':mlp_test},axis=1)
predictions.to_csv(RESULT_DIR/'test_predictions.csv')
widths = pd.concat({name:w for name,(_,w) in intervals.items()},axis=1)
widths.to_csv(RESULT_DIR/'interval_halfwidths.csv')
config = {'seed':SEED, 'selected_epochs':selected_epochs, 'max_epochs':MAX_EPOCHS,
          'patience':PATIENCE,'learning_rate':LEARNING_RATE,'weight_decay':WEIGHT_DECAY,
          'alpha':MIS_COVERAGE,'lambda':1.0,'lookback':252,'cost_bps':10,
          'garch_fit_period':'2010-2018','garch_refit':False,
          'python':platform.python_version(),
          'packages':{p:version(p) for p in ['torch','arch','numpy','pandas','scikit-learn','cvxpy','yfinance']}}
(RESULT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
torch.save({'state_dict':mlp.state_dict(),'input_mean':mlp_x_scaler.mean_.tolist(),
            'input_scale':mlp_x_scaler.scale_.tolist(),'target_mean':mlp_y_scaler.mean_.tolist(),
            'target_scale':mlp_y_scaler.scale_.tolist(),'features':features.columns.tolist(),
            'assets':TICKERS,'config':config}, RESULT_DIR/'mlp_checkpoint.pt')
wealth = (1+extended_records.pivot(index='Date',columns='Strategy',values='Net')).cumprod()
fig, axes = plt.subplots(2,1,figsize=(12,9),sharex=True)
wealth.plot(ax=axes[0],title='Wealth after transaction costs')
(100*(wealth/wealth.cummax().clip(lower=1)-1)).plot(ax=axes[1],title='Drawdown (%)',legend=False)
for ax in axes:
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(RESULT_DIR/'wealth_drawdown.png',dpi=160)
if 'agg' not in plt.get_backend().lower():
    plt.show()
print('Results saved to:', RESULT_DIR.resolve())

# %% [markdown]
# ## Tài liệu triển khai
# PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
# arch forecasting: https://arch.readthedocs.io/en/latest/univariate/forecasting.html
# GARCH chỉ dự báo variance; không so MAE lợi suất của nó với mạng MLP.
# Gaussian GARCH là đối chứng khoảng tham số; MLP-GARCH CP vẫn dùng split conformal.
