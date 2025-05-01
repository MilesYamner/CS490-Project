import pandas as pd
import numpy as np

sp500_data = pd.read_csv("data/sp500_prices.csv", header=[0, 1], index_col=0, parse_dates=True)
close = sp500_data.xs("Close", axis=1, level=1)  # Close prices only
daily_log_returns = np.log(close / close.shift(1)).iloc[1:]

daily_log_returns.to_csv("data/daily_log_returns.csv")

macro_data = pd.read_csv("data/macro_indicators.csv", index_col=0)
macro_data.index = pd.to_datetime(macro_data.index, errors="coerce")
macro_data = macro_data[macro_data.index.notna()]

macro_close = macro_data.iloc[:, [0, 1]]
macro_close.columns = ['IRX', 'VIX']
macro_close = macro_close.loc["2019-01-03":]
macro_close.index = pd.to_datetime(macro_close.index)
daily_log_returns.index = pd.to_datetime(daily_log_returns.index)

assert daily_log_returns.index.equals(macro_close.index), "Date indices do not match"

macro_close.to_csv("data/macro_close_adj.csv")

print("Data Preprocessing Complete!")
