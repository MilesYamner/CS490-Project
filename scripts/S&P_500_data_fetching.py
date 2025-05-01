import yfinance as yf
import pandas as pd

wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_tickers = pd.read_html(wiki_url)[0]["Symbol"].tolist()

delisted  = ['SOLV','GEV','SW','BRK.B','BF.B']
for ticker in delisted:
    print(ticker)
    sp500_tickers.remove(ticker)
# print(sp500_tickers)

sp500_data = yf.download(sp500_tickers, start="2019-01-01", end="2024-01-01", group_by="ticker")
sp500_data.to_csv("data/sp500_prices.csv")
