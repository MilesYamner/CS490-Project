import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import functools
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from properscoring import crps_ensemble

torch.serialization.add_safe_globals([functools.partial, getattr])

log_returns = pd.read_csv("data/daily_log_returns.csv", index_col=0, parse_dates=True)
macro_data = pd.read_csv("data/macro_close_adj.csv", index_col=0, parse_dates=True)
log_returns, macro_data = log_returns.align(macro_data, join="inner", axis=0)

prediction_length = 10
context_length = 10
max_lag = 10
history_length = context_length + max_lag

stock = random.choice(log_returns.columns)
target_series = log_returns[stock].dropna()

if len(target_series) < (history_length + prediction_length):
    raise ValueError("Not enough data for selected stock.")

target_window = target_series[-(history_length + prediction_length):]
macro_window = macro_data.loc[target_window.index]
train_target = target_window[:-prediction_length]
train_macro = macro_window[:-prediction_length]
forecast_start = target_window.index[-prediction_length]

test_instance = {
    "start": train_target.index[0],
    "target": train_target.values.tolist(),
    "feat_dynamic_real": train_macro.T.values,
}

test_ds = ListDataset([test_instance], freq="D")
grouper_test = MultivariateGrouper(max_target_dim=1)
test_ds = grouper_test(test_ds)

predictor = torch.load("models/cnf_predictor.pth", map_location=torch.device("cpu"), weights_only=False)
forecast_it = predictor.predict(test_ds)
forecast = list(forecast_it)[0]
forecast_dates = pd.date_range(start=forecast_start, periods=prediction_length, freq="D")
actual_holdout = target_window[-prediction_length:].values

predicted = forecast.mean.squeeze()
lower_quantile = forecast.quantile(0.1).squeeze()
upper_quantile = forecast.quantile(0.9).squeeze()

err_low = predicted - lower_quantile
err_high = upper_quantile - predicted
yerr = [err_low, err_high]

plt.figure(figsize=(12, 6))
plt.plot(train_target.index, train_target.values, color="black", label="Training Data")
plt.errorbar(forecast_dates, predicted, yerr=yerr, capsize=3, fmt='o', color="black", linestyle="--", label="Forecast (Predicted)")
plt.plot(forecast_dates, actual_holdout, color="blue", marker="x", linestyle="--", label="Actual (Holdout)")
plt.title(f"Forecast vs Actual for {stock} (Start: {forecast_start.strftime('%Y-%m-%d')})")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

samples = np.squeeze(forecast.samples, axis=2).T  # shape: (prediction_length, num_samples)
actual_np = np.array(actual_holdout).astype(np.float64)

try:
    crps_values = crps_ensemble(actual_np, samples)
    mean_crps = crps_values.mean()
    print(f"\nMean CRPS over prediction window: {mean_crps:.4f}")
except Exception as e:
    print("CRPS computation failed:", e)

covered = (actual_np >= lower_quantile) & (actual_np <= upper_quantile)
coverage = np.mean(covered)
print(f"80% Empirical Coverage: {coverage:.2%}")
