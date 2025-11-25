# --- Step 0: Install dependencies
!pip install yfinance numpy pandas matplotlib scikit-learn joblib --quiet

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# Define 5 stocks for training
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

def fetch_stock_data(symbols, period="5y"):
    """Step 1: Collect historical data for 5 stocks using yfinance"""
    print("STEP 1: DATA COLLECTION")

    stock_data = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")

        try:
            # Download historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if not df.empty:
                # Keep only necessary columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                stock_data[symbol] = df
                print(f"✅ {symbol}: {len(df)} records")
            else:
                print(f"❌ No data for {symbol}")

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")

    return stock_data

# Execute Step 1
stock_data = fetch_stock_data(stocks)


# Step 2: Data Preprocessing - Clean and align data
def preprocess_data(stock_data):
    """Step 2: Clean and align all stock data"""
    print("\nSTEP 2: DATA PREPROCESSING")

    # Create combined DataFrame with all closing prices
    closes_df = pd.DataFrame()

    for symbol, data in stock_data.items():
        closes_df[symbol] = data['Close']

    # Remove any missing values
    closes_df = closes_df.dropna()

    print(f"Data shape after cleaning: {closes_df.shape}")
    print("First 5 rows:")
    print(closes_df.head())
    print("Data preprocessing completed!")

    return closes_df

# Execute Step 2
closes_df = preprocess_data(stock_data)
print("Data collection completed!")


# Step 3: Volatility Analysis - Calculate returns, variance, covariance
def calculate_volatility_metrics(closes_df):
    """Step 3: Calculate returns, variance, covariance"""
    print("\nSTEP 3: VOLATILITY ANALYSIS")

    # Calculate daily returns
    returns_df = closes_df.pct_change().dropna()

    # Calculate individual stock volatilities (annualized)
    daily_volatilities = returns_df. std()
    annual_volatilities = daily_volatilities * np.sqrt(252)

    # Calculate covariance matrix (annualized)
    daily_cov = returns_df.cov()
    annual_cov = daily_cov * 252

    # Calculate correlation matrix
    corr_matrix = returns_df.corr()

    print("Daily Volatilities:")
    for stock, vol in daily_volatilities.items():
        print(f"  {stock}: {vol:.4f}")

    print("\nAnnual Volatilities:")
    for stock, vol in annual_volatilities.items():
        print(f"  {stock}: {vol:.4f}")

    print("\nCorrelation Matrix:")
    print(corr_matrix)

    print("Volatility analysis completed!")

    return returns_df, annual_volatilities, annual_cov, corr_matrix

# Execute Step 3
returns_df, volatilities, cov_matrix, corr_matrix = calculate_volatility_metrics(closes_df)

# Step 4: Risk Metrics - Compute VaR and Conditional VaR
def calculate_risk_metrics(returns_df, confidence_level=0.95):
    """Step 4: Calculate VaR and CVaR"""
    print("\nSTEP 4: RISK METRICS CALCULATION")

    # Portfolio returns (equal weighted for training)
    portfolio_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weight for 5 stocks
    portfolio_returns = returns_df.dot(portfolio_weights)

    # Historical VaR (95% confidence)
    var_historical = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    # Conditional VaR (Expected Shortfall)
    cvar = portfolio_returns[portfolio_returns <= var_historical].mean()

    # Portfolio statistics
    portfolio_mean = portfolio_returns.mean() * 252  # Annualized
    portfolio_std = portfolio_returns.std() * np.sqrt(252)  # Annualized

    risk_metrics = {
        'historical_var_95': var_historical,
        'cvar_95': cvar,
        'annual_mean_return': portfolio_mean,
        'annual_volatility': portfolio_std,
        'sharpe_ratio': portfolio_mean / portfolio_std if portfolio_std > 0 else 0
    }

    print("Portfolio Risk Metrics:")
    print(f"  Historical VaR (95%): {var_historical:.4f}")
    print(f"  Conditional VaR (95%): {cvar:.4f}")
    print(f"  Annual Mean Return: {portfolio_mean:.4f}")
    print(f"  Annual Volatility: {portfolio_std:.4f}")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")

    print("Risk metrics calculation completed!")

    return risk_metrics, portfolio_returns

# Execute Step 4
risk_metrics, portfolio_returns = calculate_risk_metrics(returns_df)

# Step 5: Simulation - Run Monte Carlo simulations
def monte_carlo_simulation(returns_df, num_simulations=10000, days=252):
    """Step 5: Monte Carlo simulation for portfolio outcomes"""
    print("\nSTEP 5: MONTE CARLO SIMULATION")

    # Calculate parameters from historical data
    mean_returns = returns_df.mean().values  # Daily mean returns
    cov_matrix = returns_df.cov().values     # Daily covariance matrix

    # Cholesky decomposition for correlated random numbers
    try:
        L = np.linalg.cholesky(cov_matrix)
    except:
        # If Cholesky fails, use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # Generate simulation results
    simulation_results = []

    for i in range(num_simulations):
        if i % 1000 == 0:
            print(f"  Running simulation {i}/{num_simulations}")

        # Generate correlated random numbers
        random_numbers = np.random.normal(0, 1, size=(days, len(mean_returns)))
        correlated_randoms = random_numbers @ L.T

        # Apply mean returns
        daily_returns = correlated_randoms + mean_returns

        # Calculate portfolio path (starting with $100)
        portfolio_value = 100
        portfolio_path = [portfolio_value]

        for day in range(days):
            # Equal weights portfolio return
            daily_portfolio_return = np.dot(daily_returns[day], [0.2, 0.2, 0.2, 0.2, 0.2])
            portfolio_value *= (1 + daily_portfolio_return)
            portfolio_path.append(portfolio_value)

        simulation_results.append(portfolio_path)

    simulation_results = np.array(simulation_results)

    print(f"Generated {num_simulations} simulations for {days} days")
    print("Monte Carlo simulation completed!")

    return simulation_results

# Execute Step 5 (with fewer simulations for speed)
simulation_results = monte_carlo_simulation(returns_df, num_simulations=5000, days=252)


# Step 6: Probability Calculation - Estimate loss probabilities
def calculate_loss_probabilities(simulation_results):
    """Step 6: Calculate probability of losses exceeding thresholds"""
    print("\nSTEP 6: PROBABILITY CALCULATION")

    # Get final portfolio values from simulations
    final_values = simulation_results[:, -1]
    initial_value = 100

    # Calculate various loss probabilities
    loss_probabilities = {
        'probability_10_percent_loss': np.mean(final_values < 90),   # 10% loss
        'probability_20_percent_loss': np.mean(final_values < 80),   # 20% loss
        'probability_30_percent_loss': np.mean(final_values < 70),   # 30% loss
        'expected_portfolio_value': np.mean(final_values),
        'median_portfolio_value': np.median(final_values),
        'worst_case_5th_percentile': np.percentile(final_values, 5),
        'best_case_95th_percentile': np.percentile(final_values, 95)
    }

    print("Portfolio Loss Probabilities (1 Year):")
    print(f"  Probability of >10% loss: {loss_probabilities['probability_10_percent_loss']:.4f} ({loss_probabilities['probability_10_percent_loss']*100:.2f}%)")
    print(f"  Probability of >20% loss: {loss_probabilities['probability_20_percent_loss']:.4f} ({loss_probabilities['probability_20_percent_loss']*100:.2f}%)")
    print(f"  Probability of >30% loss: {loss_probabilities['probability_30_percent_loss']:.4f} ({loss_probabilities['probability_30_percent_loss']*100:.2f}%)")
    print(f"  Expected portfolio value: ${loss_probabilities['expected_portfolio_value']:.2f}")
    print(f"  Worst-case (5th percentile): ${loss_probabilities['worst_case_5th_percentile']:.2f}")
    print(f"  Best-case (95th percentile): ${loss_probabilities['best_case_95th_percentile']:.2f}")

    print("Probability calculation completed!")

    return loss_probabilities

# Execute Step 6
loss_probs = calculate_loss_probabilities(simulation_results)

# Step 7: Scenario Testing - Model different market conditions
def scenario_analysis(returns_df, simulation_results):
    """Step 7: Analyze risk under different market scenarios"""
    print("\nSTEP 7: SCENARIO ANALYSIS")

    scenarios = {
        'normal_market': {'return_multiplier': 1.0, 'vol_multiplier': 1.0},
        'bear_market': {'return_multiplier': 0.5, 'vol_multiplier': 1.8},
        'bull_market': {'return_multiplier': 1.5, 'vol_multiplier': 0.7},
        'high_volatility': {'return_multiplier': 0.8, 'vol_multiplier': 2.2}
    }

    scenario_results = {}

    for scenario_name, params in scenarios.items():
        print(f"  Analyzing {scenario_name}...")

        # Adjust returns and volatility
        adjusted_returns = returns_df * params['return_multiplier']
        adjusted_volatility = adjusted_returns.std() * params['vol_multiplier']

        # Calculate scenario-specific metrics
        portfolio_returns_scenario = adjusted_returns.dot([0.2, 0.2, 0.2, 0.2, 0.2])
        var_scenario = np.percentile(portfolio_returns_scenario, 5)

        scenario_results[scenario_name] = {
            'var_95': var_scenario,
            'annual_return': portfolio_returns_scenario.mean() * 252,
            'annual_volatility': portfolio_returns_scenario.std() * np.sqrt(252)
        }

    print("\nScenario Analysis Results:")
    for scenario, metrics in scenario_results.items():
        print(f"  {scenario.replace('_', ' ').title()}:")
        print(f"    VaR (95%): {metrics['var_95']:.4f}")
        print(f"    Annual Return: {metrics['annual_return']:.4f}")
        print(f"    Annual Volatility: {metrics['annual_volatility']:.4f}")
        print()

    print("Scenario analysis completed!")

    return scenario_results

# Execute Step 7
scenario_results = scenario_analysis(returns_df, simulation_results)


# Step 8: Visualization - Plot risk distributions and probability curves
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(simulation_results, portfolio_returns, loss_probabilities):
    """Step 8: Create comprehensive visualizations"""
    print("\nSTEP 8: VISUALIZATION")

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Monte Carlo Simulation Paths
    num_paths_to_plot = 100
    for i in range(min(num_paths_to_plot, len(simulation_results))):
        axes[0, 0].plot(simulation_results[i], alpha=0.1, color='blue')

    axes[0, 0].set_title('Monte Carlo Simulation Paths')
    axes[0, 0].set_xlabel('Trading Days')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)

    # Plot 2: Final Value Distribution
    final_values = simulation_results[:, -1]
    axes[0, 1].hist(final_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(loss_probabilities['expected_portfolio_value'], color='red', linestyle='--', label=f"Mean: ${loss_probabilities['expected_portfolio_value']:.2f}")
    axes[0, 1].axvline(loss_probabilities['worst_case_5th_percentile'], color='orange', linestyle='--', label=f"5th %ile: ${loss_probabilities['worst_case_5th_percentile']:.2f}")
    axes[0, 1].set_title('Distribution of Final Portfolio Values')
    axes[0, 1].set_xlabel('Portfolio Value ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Historical Returns Distribution
    axes[1, 0].hist(portfolio_returns, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(risk_metrics['historical_var_95'], color='red', linestyle='--', label=f"VaR 95%: {risk_metrics['historical_var_95']:.4f}")
    axes[1, 0].set_title('Historical Portfolio Returns Distribution')
    axes[1, 0].set_xlabel('Daily Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Loss Probability Bar Chart
    loss_categories = ['>10% Loss', '>20% Loss', '>30% Loss']
    loss_probs_values = [
        loss_probabilities['probability_10_percent_loss'] * 100,
        loss_probabilities['probability_20_percent_loss'] * 100,
        loss_probabilities['probability_30_percent_loss'] * 100
    ]

    bars = axes[1, 1].bar(loss_categories, loss_probs_values, color=['yellow', 'orange', 'red'])
    axes[1, 1].set_title('Portfolio Loss Probabilities')
    axes[1, 1].set_ylabel('Probability (%)')

    # Add value labels on bars
    for bar, value in zip(bars, loss_probs_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')

    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('portfolio_risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Visualizations created and saved!")

    return fig

# Execute Step 8
fig = create_visualizations(simulation_results, portfolio_returns, loss_probs)

# Check current portfolio status
print("CURRENT PORTFOLIO ANALYSIS")
print("=" * 40)

# Current prices check
def check_current_prices(stocks):
    print("Current Stock Prices:")
    print("-" * 25)

    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            current_data = ticker.history(period="1d")
            current_price = current_data['Close'].iloc[-1]
            print(f"  {stock}: ${current_price:.2f}")
        except:
            print(f"  {stock}: Price unavailable")

check_current_prices(stocks)

# Portfolio performance summary
print(f"\nPortfolio Performance Summary:")
print(f"Stocks: {', '.join(stocks)}")
print(f"Training Period: {len(returns_df)} trading days")
print(f"Last Data Date: {closes_df.index[-1].strftime('%Y-%m-%d')}")

# Individual stock performance
print(f"\nIndividual Stock Performance:")
print("-" * 35)
for stock in stocks:
    total_return = (closes_df[stock].iloc[-1] / closes_df[stock].iloc[0] - 1) * 100
    print(f"  {stock}: {total_return:+.2f}%")


# Step 9: Save and Download Model Files
def save_and_download_model(returns_df, risk_metrics, loss_probabilities, scenario_results):
    """Step 9: Save all model parameters and download for deployment"""
    print("\nSTEP 9: SAVE AND DOWNLOAD TRAINED MODEL")

    # Calculate and save key parameters
    model_parameters = {
        'mean_returns': returns_df.mean().tolist(),
        'covariance_matrix': returns_df.cov().values.tolist(),
        'volatilities': (returns_df.std() * np.sqrt(252)).tolist(),
        'risk_metrics': risk_metrics,
        'loss_probabilities': loss_probabilities,
        'scenario_results': scenario_results,
        'tickers': stocks,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'data_period': '5 years'
    }

    # Save parameters to files
    with open('portfolio_risk_model.pkl', 'wb') as f:
        pickle.dump(model_parameters, f)

    # Also save as JSON for easy reading
    import json
    json_ready_params = model_parameters.copy()
    json_ready_params['covariance_matrix'] = returns_df.cov().values.tolist()

    with open('model_parameters.json', 'w') as f:
        json.dump(json_ready_params, f, indent=2)

    print("✅ Model parameters saved:")
    print(f"  - portfolio_risk_model.pkl (for deployment)")
    print(f"  - model_parameters.json (for reference)")
    print(f"  - Trained on {len(returns_df)} days of data")
    print(f"  - Stocks: {', '.join(stocks)}")

    # DOWNLOAD FILES - Colab specific
    from google.colab import files

    print("\n📥 DOWNLOADING MODEL FILES...")
    try:
        files.download('portfolio_risk_model.pkl')
        print("✅ portfolio_risk_model.pkl downloaded!")

        files.download('model_parameters.json')
        print("✅ model_parameters.json downloaded!")

        # Also download the visualization
        files.download('portfolio_risk_analysis.png')
        print("✅ portfolio_risk_analysis.png downloaded!")

    except Exception as e:
        print(f"❌ Download error: {e}")
        print("📁 Manual download steps:")
        print("   1. Left sidebar mein 'Files' icon click karo")
        print("   2. portfolio_risk_model.pkl file par right-click karo")
        print("   3. 'Download' select karo")
        print("   4. Same for model_parameters.json")

    return model_parameters

# Execute Step 9
trained_model = save_and_download_model(returns_df, risk_metrics, loss_probs, scenario_results)

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📋 NEXT STEPS FOR HUGGING FACE DEPLOYMENT:")


# FIXED STEP 9: Create Hugging Face Compatible Model
import json
import numpy as np
import pandas as pd

def create_huggingface_compatible_model(returns_df, risk_metrics, loss_probabilities, scenario_results):
    """Step 9: Create model that works on Hugging Face"""
    print("\nSTEP 9: CREATING HUGGING FACE COMPATIBLE MODEL")

    # Convert all numpy values to Python native types
    model_parameters = {
        'mean_returns': [float(x) for x in returns_df.mean().values],
        'covariance_matrix': returns_df.cov().values.tolist(),
        'volatilities': [float(x) for x in (returns_df.std() * np.sqrt(252)).values],
        'risk_metrics': {
            'historical_var_95': float(risk_metrics['historical_var_95']),
            'cvar_95': float(risk_metrics['cvar_95']),
            'annual_mean_return': float(risk_metrics['annual_mean_return']),
            'annual_volatility': float(risk_metrics['annual_volatility']),
            'sharpe_ratio': float(risk_metrics['sharpe_ratio'])
        },
        'loss_probabilities': {
            'probability_10_percent_loss': float(loss_probs['probability_10_percent_loss']),
            'probability_20_percent_loss': float(loss_probs['probability_20_percent_loss']),
            'probability_30_percent_loss': float(loss_probs['probability_30_percent_loss']),
            'expected_portfolio_value': float(loss_probs['expected_portfolio_value']),
            'median_portfolio_value': float(loss_probs['median_portfolio_value']),
            'worst_case_5th_percentile': float(loss_probs['worst_case_5th_percentile']),
            'best_case_95th_percentile': float(loss_probs['best_case_95th_percentile'])
        },
        'scenario_results': {
            'normal_market': {
                'var_95': float(scenario_results['normal_market']['var_95']),
                'annual_return': float(scenario_results['normal_market']['annual_return']),
                'annual_volatility': float(scenario_results['normal_market']['annual_volatility'])
            },
            'bear_market': {
                'var_95': float(scenario_results['bear_market']['var_95']),
                'annual_return': float(scenario_results['bear_market']['annual_return']),
                'annual_volatility': float(scenario_results['bear_market']['annual_volatility'])
            },
            'bull_market': {
                'var_95': float(scenario_results['bull_market']['var_95']),
                'annual_return': float(scenario_results['bull_market']['annual_return']),
                'annual_volatility': float(scenario_results['bull_market']['annual_volatility'])
            },
            'high_volatility': {
                'var_95': float(scenario_results['high_volatility']['var_95']),
                'annual_return': float(scenario_results['high_volatility']['annual_return']),
                'annual_volatility': float(scenario_results['high_volatility']['annual_volatility'])
            }
        },
        'tickers': stocks,
        'training_date': datetime.now().strftime("%Y-%m-%d"),
        'data_period': '5 years'
    }

    # Save as JSON (Hugging Face compatible)
    with open('portfolio_risk_model.json', 'w') as f:
        json.dump(model_parameters, f, indent=2, ensure_ascii=False)

    # Also save as pickle with protocol=4 for compatibility
    with open('portfolio_risk_model.pkl', 'wb') as f:
        pickle.dump(model_parameters, f, protocol=4)

    print("✅ Hugging Face compatible model created!")
    print("📁 Files saved:")

    # Download files
    from google.colab import files
    files.download('portfolio_risk_model.json')
    files.download('portfolio_risk_model.pkl')

    return model_parameters

# Execute FIXED Step 9
fixed_model = create_huggingface_compatible_model(returns_df, risk_metrics, loss_probs, scenario_results)

print("\n" + "="*60)
print("🎉 HUGGING FACE COMPATIBLE MODEL CREATED!")
print("="*60)
