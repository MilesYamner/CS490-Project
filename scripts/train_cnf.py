import pandas as pd
import numpy as np
import torch
from scripts.model.tempflow.tempflow_estimator import TempFlowEstimator
from scripts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

log_returns = pd.read_csv("data/daily_log_returns.csv", index_col=0, parse_dates=True)
macro_data = pd.read_csv("data/macro_close_adj.csv", index_col=0, parse_dates=True)
log_returns.fillna(0)
macro_data.fillna(0)

if 'Ticker' in macro_data.columns or 'Date' in macro_data.columns:
    macro_data = macro_data.drop(columns=['Ticker', 'Date'], errors='ignore')

log_returns, macro_data = log_returns.align(macro_data, join="inner", axis=0)

time_feat_fns = time_features_from_frequency_str("1D")
time_feats = np.stack([fn(log_returns.index) for fn in time_feat_fns])

macro_values = macro_data.T.values
combined_features = np.concatenate([macro_values, time_feats], axis=0)
stock_to_id = {stock: i for i, stock in enumerate(log_returns.columns)}

# Build training entries for all assets
training_entries = []
for stock in log_returns.columns:
    series = log_returns[stock].dropna()

    if len(series) != macro_data.shape[0]:
        continue

    entry = {
        "start": pd.Timestamp(series.index[0], freq="D"),
        "target": series.values.tolist(),  # flat 1D list
        "feat_dynamic_real": combined_features.tolist(),
        "static_categorical_features": [stock_to_id[stock]],
    }
    training_entries.append(entry)

# Build ListDataset for GluonTS
training_data = ListDataset(training_entries, freq="1D")

grouper_train = MultivariateGrouper(max_target_dim=1)
training_data = grouper_train(training_data)
print(f"Total time points: {len(training_entries) * log_returns.shape[0]}")

device = "cuda" if torch.cuda.is_available() else "cpu"
total_input_dim = len(list(range(1, 11))) + combined_features.shape[0]

estimator = TempFlowEstimator(
    input_size=total_input_dim-2,  # dynamic feature dim
    freq="1D",
    prediction_length=10,
    target_dim=1,
    cardinality=[len(stock_to_id)],  # Asset ID cardinality
    trainer=Trainer(
        epochs=15,
        batch_size=40,
        num_batches_per_epoch=56,
        device=device,
        callback=lambda epoch, loss: print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    ),
    lags_seq=list(range(1, 11)),
    conditioning_length=macro_data.shape[1],
)

# print(f"Features Shape Before RNN: {combined_features.shape}")
predictor = estimator.train(training_data)
torch.save(predictor, "models/cnf_predictor.pth")
print("CNF Model Trained and Saved!")
