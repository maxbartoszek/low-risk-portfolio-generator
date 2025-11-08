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
from datetime import datetime

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

# Use the weighted scoring algorithm in the doc to provide a score /100 per stock

# After scoring, put all stocks in lists based on sector

# Take the top 5 from each sector (based on their score /100) and put them in a new dataframe

# Then return the top 25

# Use minimum variance portfolio optimization to determine optimal weights per stock

# Calculate the shares per stock based on weightings and transaction costs

# Determine what will actually be the transaction cost (What is lower)
# Subtract that from the amount allocated to the company, and determine new total shares

# Verify that we meet all the constraints (one small cap, no more than 40% in one sector, etc.)

# Load final portfolio
