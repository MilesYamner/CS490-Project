import pandas as pd
import numpy as np
import random
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from properscoring import crps_ensemble
from tqdm import tqdm

log_returns = pd.read_csv("data/daily_log_returns.csv", index_col=0, parse_dates=True)
macro_data = pd.read_csv("data/macro_close_adj.csv", index_col=0, parse_dates=True)
log_returns, macro_data = log_returns.align(macro_data, join="inner", axis=0)

context_len = 25
pred_len = 10
num_trials = 500
num_samples = 50
crps_scores, mae_scores = [], []
all_preds, all_actuals = [], []
valid_windows = []

for stock in log_returns.columns:
    series = log_returns[stock]
    for i in range(len(series) - context_len - pred_len):
        window = series.iloc[i : i + context_len + pred_len]
        macro_window = macro_data.iloc[i : i + context_len + pred_len]
        if not window.isna().any() and not macro_window.isna().any().any():
            valid_windows.append((stock, window.index[:context_len], window.index[context_len:]))

random.shuffle(valid_windows)
valid_windows = valid_windows[:num_trials]

for stock, context_dates, target_dates in tqdm(valid_windows):
    series = log_returns[stock]
    context = series.loc[context_dates]
    target = series.loc[target_dates]

    try:
        model = ARIMA(context.values, order=(1, 0, 0))  # AR(1)
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=pred_len)
        mean_forecast = forecast_result.predicted_mean
        std_forecast = forecast_result.se_mean
    except:
        continue

    samples = np.random.normal(loc=mean_forecast[:, None], scale=std_forecast[:, None], size=(pred_len, num_samples))
    actuals = target.values

    for t in range(pred_len):
        crps_t = crps_ensemble(np.array([actuals[t]]), samples[t].reshape(1, -1))[0]
        mae_t = np.abs(actuals[t] - mean_forecast[t])
        crps_scores.append(crps_t)
        mae_scores.append(mae_t)
        all_preds.append(mean_forecast[t])
        all_actuals.append(actuals[t])

def compute_coverage(actuals, preds, stds, alpha=0.20):
    z = 1.28
    lower = np.array(preds) - z * stds
    upper = np.array(preds) + z * stds
    covered = (np.array(actuals) >= lower) & (np.array(actuals) <= upper)
    return np.mean(covered)

avg_std = np.std(np.array(all_actuals) - np.array(all_preds)) + 1e-4
coverage = compute_coverage(all_actuals, all_preds, avg_std)

print("\n=== AR(1) Baseline (ARIMA) Evaluation ===")
print(f"Mean CRPS: {np.mean(crps_scores):.4f}")
print(f"Mean MAE:  {np.mean(mae_scores):.4f}")
print(f"Min CRPS:  {np.min(crps_scores)} ")
print(f"Min MAE:  {np.min(mae_scores)}")
print(f"Std Dev MAE: {np.std(mae_scores):.4f}")
print(f"80% Empirical Coverage: {coverage:.2%}")
