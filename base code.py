#These are the libraries you can use.  You may add any libraries directy related to threading if this is a direction
#you wish to go (this is not from the course, so it's entirely on you if you wish to use threading).  Any
#further libraries you wish to use you must email me, james@uwaterloo.ca, for permission.

from IPython.display import display, Math, Latex

import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# Load the tickers
df = pd.read_csv('Tickers_Example.csv', header=None) # test with Tickers_Example.csv?
ticker_lst = list(df.iloc[:,0])

# Filter out tickers that don't exist; include only valid tickers
valid_tickers_lst = []
for i in range(len(ticker_lst)):
    ticker_data = yf.Ticker(ticker_lst[i]) # call up data
    country = ticker_data.info.get('country') 
    currency = ticker_data.info.get('currency')
    if country in {'Canada','United States'} and currency in {'CAD','USD'}: # filters us and canadian tickers and listed stocks
        valid_tickers_lst.append(ticker_lst[i])

print(valid_tickers_lst)

# Apply a filter to remove all stocks with volumes below 5000 (in the given period)
# period in which we look at the volumes 
volume_start_date = '2024-10-01'
volume_end_date = '2025-09-30' 
filtered_lst = []
for i in range(len(valid_tickers_lst)): # goes through every ticker to filter
    data = yf.download(
        tickers=valid_tickers_lst[i],
        start=volume_start_date,
        end=volume_end_date
    )
    volume_data = data[['Volume']].dropna() # volume data

    keep_months = pd.DataFrame() # df of all the months with more than 18 trading days
    volume_data['Month'] = volume_data.index.to_period('M') # create new column of index only by (YYYY-MM)
    grouped_month_index = volume_data.groupby(['Month']) # group data by month
    for month, group in grouped_month_index:
        if len(group) >= 18: # if the month has more than 18 trading days
            keep_months = pd.concat([keep_months, group]) # add data of months with more than 18 trading days
    average_daily_volume = keep_months['Volume'][valid_tickers_lst[i]].mean() # calculate average daily volume
    if average_daily_volume >= 5000: # determine if above or below 5000 shares
        filtered_lst.append(valid_tickers_lst[i]) # add to filtered list if greater or equal to 5000 shares

print(filtered_lst)

# Get the daily data over chosen timeframe (2025-10-24 to 2025-10-31 for testing?)
start_date = '2025-10-24' # change to Nov 21 2025
end_date = '2025-10-31' # change to Nov 28 2025

daily_data = yf.download(
    tickers=filtered_lst,
    start=start_date,
    end=end_date)

# Get/calculate volatility (std), beta, market cap, and sectors

metrics_df = pd.DataFrame(columns=['Ticker', 'Volatility', 'Beta', 'MarketCap', 'Sector'])

# ---- Market index for beta ----
market_index = "^GSPC"
market_hist = yf.download(market_index, start=start_date, end=end_date)
market_prices = market_hist["Adj Close"].dropna()
market_returns = market_prices.pct_change().dropna()

# ---- Loop through filtered tickers ----
for ticker in filtered_lst:

    try:
        prices = daily_data['Adj Close'][ticker]
    except KeyError:
        prices = daily_data['Close'][ticker]

    prices = prices.dropna()
    if prices.empty:
        continue

    returns = prices.pct_change().dropna()
    if returns.empty:
        continue

    volatility = returns.std(ddof=0)     # daily std

    aligned = pd.concat([returns, market_returns], axis=1, join="inner")
    aligned.columns = ["Stock", "Market"]

    # covariance(stock, market)
    cov_sm = aligned.cov(ddof=0).iloc[0,1]

    # variance(market)
    var_market = aligned["Market"].var(ddof=0)

    if var_market == 0 or pd.isna(cov_sm):
        beta = np.nan
    else:
        beta = cov_sm / var_market

    info = yf.Ticker(ticker).info
    mcap = info.get("marketCap", np.nan)
    sector = info.get("sector", "Unknown")

    metrics_df.loc[len(metrics_df)] = [
        ticker,
        volatility,
        beta,
        mcap,
        sector
    ]

metrics_df = metrics_df.dropna(subset=['Volatility', 'MarketCap']).reset_index(drop=True)

print(metrics_df)

# Use the weighted scoring algorithm in the doc to provide a score /100 per stock; take from google docs
# Ammar

scored_df = metrics_df.copy()

# Volatility scoring function
def vol_points(v):
    # v is daily volatility (e.g., 0.02 = 2%)
    if v < 0.02:
        return 45
    elif v < 0.03:
        return 40
    elif v < 0.04:
        return 35
    elif v < 0.05:
        return 25
    elif v < 0.06:
        return 15
    else:
        return 5

# Beta scoring function
def beta_points(b):
    if pd.isna(b):
        return 20  # neutral if missing
    if b < 0.6:
        return 35
    elif b < 0.9:
        return 25
    elif b < 1.1:
        return 20
    elif b < 1.3:
        return 10
    else:
        return 0

# Market cap scoring function
def cap_points(m):
    # thresholds in dollars
    if m > 200e9:
        return 20
    elif m >= 50e9:
        return 16
    elif m >= 10e9:
        return 12
    elif m >= 2e9:
        return 8
    else:
        return 4

# Apply scoring table
scored_df['VolPts']  = scored_df['Volatility'].apply(vol_points)
scored_df['BetaPts'] = scored_df['Beta'].apply(beta_points)
scored_df['CapPts']  = scored_df['MarketCap'].apply(cap_points)

# Final score out of 100
scored_df['Score'] = scored_df['VolPts'] + scored_df['BetaPts'] + scored_df['CapPts']

print(scored_df[['Ticker','Volatility','Beta','MarketCap','Score']].head())

# After scoring, put all stocks in lists based on sector
# Ammar

sector_dict = {}   # dictionary: sector → list of tickers

for _, row in scored_df.iterrows():
    sector = row['Sector']
    ticker = row['Ticker']
    
    # create key if doesn't exist
    if sector not in sector_dict:
        sector_dict[sector] = []
    
    # append ticker
    sector_dict[sector].append(ticker)

# Preview the grouping
for sec, tics in sector_dict.items():
    print(f"{sec}: {tics}")

# Take the top 5 from each sector (based on their score /100) and put them in a new dataframe
# Wendi

# Then return the top 25
# Wendi

# Make the LAST ONE A SMALL CAP GUARANTEED
# Wendi

# Use minimum variance portfolio optimization to determine optimal weights per stock

# Calculate the shares per stock based on weightings and transaction costs
# Max

# Determine what will actually be the transaction cost (What is lower)\
#Max
# Subtract that from the amount allocated to the company, and determine new total shares
#Max

# Verify that we meet all the constraints (one small cap, no more than 40% in one sector, etc.)
#Max

# Load final portfolio
# All of us!

#WHAT WE MUST DO
# for each subpart of our to do list, write a 3-5 bullet-point explanation of why we are doing specific things
# basing it on real financial data and sources
# map each ticker to the sector that they are from in a dictionary
# make sure last is a small cap, its okay if multiple fit this description but last one MUST be small cap
# maybe find other ways to pick stocks if necessary
# max wants 25 stocks
