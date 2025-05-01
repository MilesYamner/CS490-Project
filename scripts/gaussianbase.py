import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from properscoring import crps_ensemble
from tqdm import tqdm
import random


lags = [1, 2, 3]
pred_len = 10
context_len = 25
num_trials = 500
hidden_dim = 64
batch_size = 32
epochs = 10
num_samples = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_returns = pd.read_csv("data/daily_log_returns.csv", index_col=0, parse_dates=True)
macro_data = pd.read_csv("data/macro_close_adj.csv", index_col=0, parse_dates=True)
log_returns, macro_data = log_returns.align(macro_data, join="inner", axis=0)
X_train, Y_train = [], []

for stock in log_returns.columns:
    series = log_returns[stock]
    macro = macro_data.values
    for t in range(max(lags), len(series) - pred_len):
        x_lags = [series.shift(l)[t] for l in lags]
        x_macro = macro[t]
        y = series.iloc[t+1:t+1+pred_len].values
        if not np.isnan(x_lags).any() and not np.isnan(x_macro).any() and not np.isnan(y).any():
            X_train.append(np.concatenate([x_lags, x_macro]))
            Y_train.append(y)

scaler = StandardScaler()
X_train = scaler.fit_transform(np.array(X_train))
Y_train = np.array(Y_train)
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)

class GaussianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_logstd = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, pred_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1]
        mean = self.fc_mean(out)
        log_std = self.fc_logstd(out)
        return mean, torch.exp(log_std)

input_dim = X_train.shape[1]
model = GaussianLSTM(input_dim, hidden_dim, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def gaussian_nll(mean, std, target):
    return 0.5 * torch.log(2 * torch.pi * std**2) + ((target - mean)**2) / (2 * std**2)

model.train()

for epoch in range(epochs):
    perm = torch.randperm(X_tensor.size(0))
    total_loss = 0
    for i in range(0, X_tensor.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X_tensor[idx], Y_tensor[idx]
        optimizer.zero_grad()
        mean, std = model(xb)
        loss = gaussian_nll(mean, std, yb).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / (len(X_tensor)//batch_size):.4f}")

valid_windows = []
for stock in log_returns.columns:
    series = log_returns[stock]
    for i in range(len(series) - context_len - pred_len):
        window = series.iloc[i:i + context_len + pred_len]
        macro_window = macro_data.iloc[i:i + context_len + pred_len]
        if not window.isna().any() and not macro_window.isna().any().any():
            valid_windows.append((stock, window.index[:context_len], window.index[context_len:]))

random.shuffle(valid_windows)
valid_windows = valid_windows[:num_trials]

model.eval()
mae_scores, crps_scores, all_preds, all_actuals = [], [], [], []

for stock, context_dates, target_dates in tqdm(valid_windows):
    series = log_returns[stock]
    macro = macro_data

    last_date = context_dates[-1]
    try:
        x_lags = [series.shift(l).loc[last_date] for l in lags]
        x_macro = macro.loc[last_date].values
    except KeyError:
        continue

    if np.isnan(x_lags).any() or np.isnan(x_macro).any():
        continue

    x_input = np.concatenate([x_lags, x_macro]).reshape(1, -1)
    x_std = scaler.transform(x_input)
    x_tensor = torch.tensor(x_std, dtype=torch.float32).to(device)

    with torch.no_grad():
        mean, std = model(x_tensor)
        mean = mean.cpu().numpy().flatten()
        std = std.cpu().numpy().flatten()

    samples = np.random.normal(loc=mean[:, None], scale=std[:, None], size=(pred_len, num_samples))
    actuals = np.array([series.loc[d] for d in target_dates])

    for t in range(pred_len):
        crps_t = crps_ensemble(np.array([actuals[t]]), samples[t].reshape(1, -1))[0]
        mae_t = np.abs(actuals[t] - mean[t])
        crps_scores.append(crps_t)
        mae_scores.append(mae_t)
        all_actuals.append(actuals[t])
        all_preds.append(mean[t])

def compute_coverage(actuals, preds, stds, alpha=0.20):
    z = 1.28 
    lower = np.array(preds) - z * stds
    upper = np.array(preds) + z * stds
    covered = (np.array(actuals) >= lower) & (np.array(actuals) <= upper)
    return np.mean(covered)

std_for_coverage = np.std(np.array(all_actuals) - np.array(all_preds)) + 1e-4
coverage = compute_coverage(all_actuals, all_preds, std_for_coverage)

print("\n=== Gaussian LSTM Baseline Evaluation ===")
print(f"Mean CRPS: {np.mean(crps_scores):.4f}")
print(f"Mean MAE:  {np.mean(mae_scores):.4f}")
print(f"N:         {np.min(crps_scores)} scores")
print(f"Std Dev CRPS: {np.std(crps_scores):.4f}")
print(f"Std Dev MAE: {np.std(mae_scores):.4f}")
print(f"80% Empirical Coverage: {coverage:.2%}")
