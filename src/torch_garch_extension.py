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
from portfolio_baseline import minimum_variance, performance

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
