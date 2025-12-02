# risk-free-portfolio
# Low-Risk Portfolio Generator

A minimum-variance portfolio optimizer built in Python to construct low-volatility equity portfolios.

This project implements an automated portfolio construction pipeline that uses historical market data, custom weighted scoring algorithms, and convex optimization to build portfolios targeting minimum variance. Our approach delivered a portfolio with a 1.64 Sharpe and 0.65% daily volatility (10.35% annualized) using a csv file of random tickers. 

Note: No portfolio can ever be 'risk-free.' This is a low-volatility (or low-risk) portfolio generator that aims to just preserve money as much as possible.

# What This Model Does

This pipeline generates a minimum-variance portfolio using North American equities. It includes:

1. Data Ingestion & Cleaning
* Downloads historical prices using yfinance
* Filters for US and Canadian stocks only
* Applies liquidity constraints (at least 5,000 daily volume)
* Handles missing values and delisted tickers
* Filters out invalid or empty data

2. Stability Factor Computation
For each ticker, the model computes:

|     Factor     |             Purpose                 |
|----------------|-------------------------------------|
|   Volatility   |      Measures day-to-day risk       |
|      Beta      |      Measures market sensitivity    |
|   Market Cap   |      Stability & size proxy         |

4. Custom Scoring for Low Risk
* Low volatility → higher score
* Low beta → higher score
* Large market cap → higher score

4. Stock Selection
* Selects the top 5 stocks per sector to achieve sector diversification
* Chooses top 25 stocks overall from that pool
* Enforces at least one small-cap and one large-cap stock

5. Minimum-Variance Portfolio Optimization 
Uses CVXPY to solve optimization problems.
Subject to:
* w ≥ 0 (long-only, no short selling)
* ∑𝑤 = 1 (weights sum to 100%)
* wi ≤ 15% (max 15% in any one stock)
* wi ≥ 1 / 2n (requires each stock in final portfolio is weighted at least 1 divided by number of stocks in portfolio times 2)
* Sector exposure ≤ 40% (diversification across different sectors)
* 10-25 stocks held (for diversification) 

This produces a true minimum-variance portfolio.

6. Portfolio Construction & Reporting
* Converts optimal weights to actual shares (fractional allowed)
* Accounts for transaction fees ($2.15 USD or $0.001 per share, whichever's lower)
* Handles USD/CAD conversions
* Generates a CSV with the tickers in final portfolio and share quantities

# Requirements

* yfinance
* pandas
* numpy_financial
* datetimes
* cvxpy
* matplot.lib (for mvp optimization visualization)

# Usage
1. Prepare a CSV file with one column filled with tickers (as many as you want)
2. Update the CSV file name in the code
3. Run the notebook/script
4. Open the generated Low_Risk_Portfolio_Summary.csv

# Runtime
Expected Runtime is 1-6 minutes depending on the number of tickers. In testing:
* 40 tickers took 24 seconds
* 200 tickers took 1 minute 17 seconds
* 621 tickers took 5 minutes 58 seconds
