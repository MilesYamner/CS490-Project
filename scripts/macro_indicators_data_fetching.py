import yfinance as yf
import pandas as pd

macro_tickers = ['^VIX', "^IRX"]

macro_data = yf.download(macro_tickers, start="2019-01-01", end="2024-01-01")

macro_data.to_csv("data/macro_indicators.csv")
