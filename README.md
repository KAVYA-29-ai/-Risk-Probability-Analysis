Stock Portfolio Risk Probability Analysis

This project develops a probabilistic model to estimate the likelihood that a stock portfolio’s value will drop below a certain threshold within a specified period. Using historical stock price data, market volatility measures, and Monte Carlo simulations, it helps investors quantify and manage financial risk.



Final.ipynb
Final.ipynb_
Stock Portfolio Risk Probability Analysis

This project develops a probabilistic model to estimate the likelihood that a stock portfolio’s value will drop below a certain threshold within a specified period. Using historical stock price data, market volatility measures, and Monte Carlo simulations, it helps investors quantify and manage financial risk.


[ ]
# --- Step 0: Install dependencies
!pip install yfinance numpy pandas matplotlib scikit-learn joblib --quiet

[ ]

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# Define 5 stocks for training
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

…                print(f"❌ No data for {symbol}")

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")

    return stock_data

# Execute Step 1
stock_data = fetch_stock_data(stocks)
print("Data collection completed!")
STEP 1: DATA COLLECTION
Fetching data for AAPL...
✅ AAPL: 1255 records
Fetching data for MSFT...
✅ MSFT: 1255 records
Fetching data for GOOGL...
✅ GOOGL: 1255 records
Fetching data for AMZN...
✅ AMZN: 1255 records
Fetching data for TSLA...
✅ TSLA: 1255 records
Data collection completed!

[ ]
# Step 2: Data Preprocessing - Clean and align data
def preprocess_data(stock_data):
    """Step 2: Clean and align all stock data"""
    print("\nSTEP 2: DATA PREPROCESSING")

    # Create combined DataFrame with all closing prices
    closes_df = pd.DataFrame()

    for symbol, data in stock_data.items():
        closes_df[symbol] = data['Close']
…
# Execute Step 2
closes_df = preprocess_data(stock_data)

STEP 2: DATA PREPROCESSING
Data shape after cleaning: (1255, 5)
First 5 rows:
                                 AAPL        MSFT      GOOGL        AMZN  \
Date                                                                       
2020-11-18 00:00:00-05:00  114.896561  202.918716  86.435844  155.272995   
2020-11-19 00:00:00-05:00  115.490395  204.206940  87.326202  155.850998   
2020-11-20 00:00:00-05:00  114.224907  202.255417  86.224304  154.970001   
2020-11-23 00:00:00-05:00  110.827560  201.986237  85.786316  154.919495   
2020-11-24 00:00:00-05:00  112.112511  205.591232  87.590874  155.903000   

                                 TSLA  
Date                                   
2020-11-18 00:00:00-05:00  162.213333  
2020-11-19 00:00:00-05:00  166.423340  
2020-11-20 00:00:00-05:00  163.203339  
2020-11-23 00:00:00-05:00  173.949997  
2020-11-24 00:00:00-05:00  185.126663  
Data preprocessing completed!

[ ]
# Step 3: Volatility Analysis - Calculate returns, variance, covariance
def calculate_volatility_metrics(closes_df):
    """Step 3: Calculate returns, variance, covariance"""
    print("\nSTEP 3: VOLATILITY ANALYSIS")

    # Calculate daily returns
    returns_df = closes_df.pct_change().dropna()

    # Calculate individual stock volatilities (annualized)
    daily_volatilities = returns_df. std()
…    print("Volatility analysis completed!")

    return returns_df, annual_volatilities, annual_cov, corr_matrix

# Execute Step 3
returns_df, volatilities, cov_matrix, corr_matrix = calculate_volatility_metrics(closes_df)

STEP 3: VOLATILITY ANALYSIS
Daily Volatilities:
  AAPL: 0.0177
  MSFT: 0.0162
  GOOGL: 0.0195
  AMZN: 0.0221
  TSLA: 0.0385

Annual Volatilities:
  AAPL: 0.2813
  MSFT: 0.2566
  GOOGL: 0.3094
  AMZN: 0.3505
  TSLA: 0.6118

Correlation Matrix:
           AAPL      MSFT     GOOGL      AMZN      TSLA
AAPL   1.000000  0.635645  0.577953  0.559246  0.489243
MSFT   0.635645  1.000000  0.657107  0.658274  0.420873
GOOGL  0.577953  0.657107  1.000000  0.619595  0.408563
AMZN   0.559246  0.658274  0.619595  1.000000  0.444035
TSLA   0.489243  0.420873  0.408563  0.444035  1.000000
Volatility analysis completed!

[ ]
# Step 4: Risk Metrics - Compute VaR and Conditional VaR
def calculate_risk_metrics(returns_df, confidence_level=0.95):
    """Step 4: Calculate VaR and CVaR"""
    print("\nSTEP 4: RISK METRICS CALCULATION")

    # Portfolio returns (equal weighted for training)
    portfolio_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weight for 5 stocks
    portfolio_returns = returns_df.dot(portfolio_weights)

    # Historical VaR (95% confidence)
…    print(f"  Annual Mean Return: {portfolio_mean:.4f}")
    print(f"  Annual Volatility: {portfolio_std:.4f}")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")

    print("Risk metrics calculation completed!")

    return risk_metrics, portfolio_returns

# Execute Step 4
risk_metrics, portfolio_returns = calculate_risk_metrics(returns_df)

STEP 4: RISK METRICS CALCULATION
Portfolio Risk Metrics:
  Historical VaR (95%): -0.0303
  Conditional VaR (95%): -0.0398
  Annual Mean Return: 0.2457
  Annual Volatility: 0.2859
  Sharpe Ratio: 0.8593
Risk metrics calculation completed!

[ ]
# Step 5: Simulation - Run Monte Carlo simulations
def monte_carlo_simulation(returns_df, num_simulations=10000, days=252):
    """Step 5: Monte Carlo simulation for portfolio outcomes"""
    print("\nSTEP 5: MONTE CARLO SIMULATION")

    # Calculate parameters from historical data
    mean_returns = returns_df.mean().values  # Daily mean returns
    cov_matrix = returns_df.cov().values     # Daily covariance matrix

    # Cholesky decomposition for correlated random numbers
…# Execute Step 5 (with fewer simulations for speed)
simulation_results = monte_carlo_simulation(returns_df, num_simulations=5000, days=252)

STEP 5: MONTE CARLO SIMULATION
  Running simulation 0/5000
  Running simulation 1000/5000
  Running simulation 2000/5000
  Running simulation 3000/5000
  Running simulation 4000/5000
Generated 5000 simulations for 252 days
Monte Carlo simulation completed!

[ ]
# Step 6: Probability Calculation - Estimate loss probabilities
def calculate_loss_probabilities(simulation_results):
    """Step 6: Calculate probability of losses exceeding thresholds"""
    print("\nSTEP 6: PROBABILITY CALCULATION")

    # Get final portfolio values from simulations
    final_values = simulation_results[:, -1]
    initial_value = 100

    # Calculate various loss probabilities
…    return loss_probabilities

# Execute Step 6
loss_probs = calculate_loss_probabilities(simulation_results)

STEP 6: PROBABILITY CALCULATION
Portfolio Loss Probabilities (1 Year):
  Probability of >10% loss: 0.1396 (13.96%)
  Probability of >20% loss: 0.0646 (6.46%)
  Probability of >30% loss: 0.0260 (2.60%)
  Expected portfolio value: $127.58
  Worst-case (5th percentile): $76.80
  Best-case (95th percentile): $196.06
Probability calculation completed!

[ ]
# Step 7: Scenario Testing - Model different market conditions
def scenario_analysis(returns_df, simulation_results):
    """Step 7: Analyze risk under different market scenarios"""
    print("\nSTEP 7: SCENARIO ANALYSIS")

    scenarios = {
        'normal_market': {'return_multiplier': 1.0, 'vol_multiplier': 1.0},
        'bear_market': {'return_multiplier': 0.5, 'vol_multiplier': 1.8},
        'bull_market': {'return_multiplier': 1.5, 'vol_multiplier': 0.7},
        'high_volatility': {'return_multiplier': 0.8, 'vol_multiplier': 2.2}
…
    return scenario_results

# Execute Step 7
scenario_results = scenario_analysis(returns_df, simulation_results)

STEP 7: SCENARIO ANALYSIS
  Analyzing normal_market...
  Analyzing bear_market...
  Analyzing bull_market...
  Analyzing high_volatility...

Scenario Analysis Results:
  Normal Market:
    VaR (95%): -0.0303
    Annual Return: 0.2457
    Annual Volatility: 0.2859

  Bear Market:
    VaR (95%): -0.0152
    Annual Return: 0.1228
    Annual Volatility: 0.1429

  Bull Market:
    VaR (95%): -0.0455
    Annual Return: 0.3685
    Annual Volatility: 0.4288

  High Volatility:
    VaR (95%): -0.0243
    Annual Return: 0.1965
    Annual Volatility: 0.2287

Scenario analysis completed!

[ ]
# Step 8: Visualization - Plot risk distributions and probability curves
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(simulation_results, portfolio_returns, loss_probabilities):
    """Step 8: Create comprehensive visualizations"""
    print("\nSTEP 8: VISUALIZATION")

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
…fig = create_visualizations(simulation_results, portfolio_returns, loss_probs)


[ ]
# Check current portfolio status
print("CURRENT PORTFOLIO ANALYSIS")
print("=" * 40)

# Current prices check
def check_current_prices(stocks):
    print("Current Stock Prices:")
    print("-" * 25)

    for stock in stocks:
…    total_return = (closes_df[stock].iloc[-1] / closes_df[stock].iloc[0] - 1) * 100
    print(f"  {stock}: {total_return:+.2f}%")
CURRENT PORTFOLIO ANALYSIS
========================================
Current Stock Prices:
-------------------------
  AAPL: $267.46
  MSFT: $507.49
  GOOGL: $285.02
  AMZN: $232.87
  TSLA: $408.92

Portfolio Performance Summary:
Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA
Training Period: 1254 trading days
Last Data Date: 2025-11-17

Individual Stock Performance:
-----------------------------------
  AAPL: +132.78%
  MSFT: +150.10%
  GOOGL: +229.75%
  AMZN: +49.97%
  TSLA: +152.09%

[ ]
# Step 9: Save and Download Model Files
def save_and_download_model(returns_df, risk_metrics, loss_probabilities, scenario_results):
    """Step 9: Save all model parameters and download for deployment"""
    print("\nSTEP 9: SAVE AND DOWNLOAD TRAINED MODEL")

    # Calculate and save key parameters
    model_parameters = {
        'mean_returns': returns_df.mean().tolist(),
        'covariance_matrix': returns_df.cov().values.tolist(),
        'volatilities': (returns_df.std() * np.sqrt(252)).tolist(),
…
# Execute Step 9
trained_model = save_and_download_model(returns_df, risk_metrics, loss_probs, scenario_results)

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📋 NEXT STEPS FOR HUGGING FACE DEPLOYMENT:")


Double-click (or enter) to edit


[ ]
# FIXED STEP 9: Create Hugging Face Compatible Model
import json
import numpy as np
import pandas as pd

def create_huggingface_compatible_model(returns_df, risk_metrics, loss_probabilities, scenario_results):
    """Step 9: Create model that works on Hugging Face"""
    print("\nSTEP 9: CREATING HUGGING FACE COMPATIBLE MODEL")

    # Convert all numpy values to Python native types
…print("="*60)

**📊 Portfolio Risk Analysis Report Stocks Analyzed: AAPL, MSFT, GOOGL, AMZN, TSLA Analysis Period: Last 5 Years Portfolio Weighting: Equal Weight (20% Each)

Data Collection Summary
Historical price data of 5 major technology stocks was downloaded using yfinance for the past 5 years.

Collected columns:

Open

High

Low

Close

Volume

All data was successfully aligned and cleaned to form a unified dataset with no missing values.

Data Preprocessing
A combined dataframe of closing prices was created.

Rows with any missing values were removed.

Final shape: (Number of trading days × 5 stocks)

This ensures all stocks are compared on identical trading days.

Volatility & Correlation Analysis
Daily returns were calculated using percentage change.

3.1 Daily Volatility (Standard Deviation)

Measures day-to-day risk of each stock.

3.2 Annualized Volatility

Annual Volatility = Daily Volatility × √252 This converts daily risk into yearly scale.

3.3 Covariance Matrix

Indicates how two stocks move together. Used later in Monte Carlo simulations.

3.4 Correlation Matrix

Shows the strength of relationship between stock movements:

+1 → move together

–1 → move opposite

0 → no connection

This helps understand diversification benefits.

Portfolio Risk Metrics (VaR & CVaR)
Using equal weights for all 5 stocks:

4.1 Value at Risk (VaR 95%)

Tells the maximum expected loss on a bad day (95% confidence).

Example: If VaR = –0.0125 → Highest expected daily loss = 1.25%

4.2 Conditional VaR (CVaR)

Average loss during worst days beyond VaR.

More realistic risk measure.

4.3 Annual Return & Annual Volatility

Average yearly return

Average yearly risk

4.4 Sharpe Ratio

= Annual Return / Annual Volatility Shows risk-adjusted performance. Higher = better portfolio.

Monte Carlo Simulation (5000 Simulations)
Purpose: Predict future portfolio values using statistical modelling.

Process:

Calculate mean returns & covariance

Generate correlated random returns

Simulate 252 trading days

Repeat 5000 times

Each simulation gives one possible future.

Output:

Distribution of ending portfolio values

Best, worst, expected scenarios

This gives a forward-looking risk view.

Loss Probability Analysis
Based on simulation final values:

Loss Threshold Probability

10% Loss X% 20% Loss X% 30% Loss X%

(Model will fill actual values from simulation.)

Also calculated:

Expected portfolio value

Median value

Worst 5th percentile

Best 95th percentile

This gives a clear idea of risk-return.

Scenario Stress Testing
Simulated performance under different market regimes:

Scenarios:

Normal Market

Bear Market (Returns ↓, Volatility ↑)

Bull Market (Returns ↑, Volatility ↓)

High Volatility Market (Both uncertainty & risk ↑↑)

For each:

VaR

Annual Return

Annual Volatility

This helps understand how portfolio responds to real-life shocks.

Visualizations
Generated 4 key plots:

1️⃣ Monte Carlo Simulation Paths

Shows 100 random future return paths.

2️⃣ Final Value Distribution

Histogram of all final portfolio values.

3️⃣ Historical Returns Distribution

Displays VaR visually.

4️⃣ Loss Probability Bar Chart

Shows probability of 10%, 20%, 30% losses.

Also saved as: portfolio_risk_analysis.png

Current Portfolio Status
Latest stock prices fetched

Total returns of individual stocks calculated

Last data date shown

Portfolio summary printed

This gives real-time market update.

📌 Final Summary

Your model provides a full professional-grade analysis:

✔ Historical data processing ✔ Risk metrics (VaR, CVaR) ✔ Monte Carlo forecasting ✔ Loss probability modelling ✔ Stress-testing ✔ Rich visualizations ✔ Real-time portfolio check

This is equivalent to an investment bank level risk model.**

Colab paid products - Cancel contracts here
Chat

New Conversation

🤓 Explain a complex thing

Explain Artificial Intelligence so that I can explain it to my six-year-old child.


🧠 Get suggestions and create new ideas

Please give me the best 10 travel ideas around the world


💭 Translate, summarize, fix grammar and more…

Translate "I love you" French


GPT-4o Mini
Hello, how can I help you today?
GPT-4o Mini
coin image
5
Upgrade




Stock Portfolio Risk Probability Analysis




Powered by AITOPIA 
Chat
Ask
Search
Write
Image
ChatFile
Vision
Full Page

