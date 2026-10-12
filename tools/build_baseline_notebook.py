"""Build a standalone notebook, embedding the portfolio helper source."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'archive/01_data_clean.py').read_text(encoding='utf-8')
source = source.split('# %% [markdown]\n# ## Phần tiếp theo')[0]
helper = (ROOT / 'src/portfolio_baseline.py').read_text(encoding='utf-8')
source += '''# %% [markdown]
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
'''
source += helper + '\n'
source += '''# %% [markdown]
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
'''
ast.parse(source)
(ROOT / 'archive/02_baseline.py').write_text(source, encoding='utf-8')
cells = []
for chunk in source.split('# %%')[1:]:
    markdown = chunk.startswith(' [markdown]')
    lines = chunk.splitlines()[1:]
    if markdown:
        text = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line.startswith('#') else line for line in lines)
        cell = dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))
    else:
        text = '\n'.join(lines) + '\n'
        ast.parse(text)
        cell = dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True))
    cell['id'] = f'baseline-{len(cells):02d}'
    cells.append(cell)
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
(ROOT / 'archive/02_baseline.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Built 02_baseline.ipynb and 02_baseline.py')
