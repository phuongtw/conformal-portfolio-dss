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
# ## Phần tiếp theo của Bước 1
# Định nghĩa robust minimum variance và cách conformal tham gia bài toán;
# thêm baseline 1/N, minimum variance thường và backtest có chi phí giao dịch.
# Chưa dùng các kết quả dự báo này để kết luận hiệu quả của danh mục.
