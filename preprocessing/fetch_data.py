import yfinance as yf
import pandas as pd
import numpy as np
import os

# 3 companies - using Indian stocks from Yahoo Finance
# TCS, Infosys, Reliance
tickers = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]

# Download 5 years of data
raw_data = {}

for ticker in tickers:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start="2019-01-01", end="2024-01-01")
    raw_data[ticker] = df[["Close"]]  # We only need closing price
    print(f"{ticker} downloaded! Shape: {df.shape}")

# Combine all into one DataFrame
combined = pd.DataFrame()

for ticker, df in raw_data.items():
    combined[ticker] = df["Close"]

# Drop rows where any value is missing
combined.dropna(inplace=True)

print("\nRaw Data Sample:")
print(combined.head())

# Normalize the data (Min-Max Normalization between 0 and 1)
normalized = (combined - combined.min()) / (combined.max() - combined.min())

print("\nNormalized Data Sample:")
print(normalized.head())

# Save both to the data/ folder
os.makedirs("../data", exist_ok=True)
combined.to_csv("../data/raw_stock_data.csv")
normalized.to_csv("../data/normalized_stock_data.csv")

print("\nFiles saved to data/ folder!")
print("raw_stock_data.csv")
print("normalized_stock_data.csv")

## ✅ Step 3 — Run the File

##In the terminal, type:
##cd preprocessing
##python fetch_data.py