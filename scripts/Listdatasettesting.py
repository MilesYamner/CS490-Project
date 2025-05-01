import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset

log_returns = pd.read_csv("data/daily_log_returns.csv", index_col=0, parse_dates=True)
macro_data = pd.read_csv("data/macro_close_adj.csv", index_col=0, parse_dates=True)

if 'Ticker' in macro_data.columns or 'Date' in macro_data.columns:
    macro_data = macro_data.drop(columns=['Ticker', 'Date'], errors='ignore')

log_returns, macro_data = log_returns.align(macro_data, join="inner", axis=0)
target_series = log_returns.iloc[:, 0] 
target_values = target_series.values.reshape(-1, 1) 
macro_values = macro_data.T.values 

training_data = ListDataset(
    [
        {
            "start": log_returns.index[0],
            "target": target_values.T[0],
            "feat_dynamic_real": macro_values,
        }
    ],
    freq="1D",
)
