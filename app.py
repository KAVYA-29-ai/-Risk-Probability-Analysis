#app for gardio hugging face

# =====================================================
# 📊 QuantRisk Pro - Portfolio Risk Analyzer (Gradio)
# Works for any 1–5 stock combination
# =====================================================

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- Safe unpickler (for numpy version mismatch)
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.multiarray" or module == "numpy.core.multiarray":
            return np.core.multiarray.scalar
        if module == "numpy":
            return getattr(np, name)
        return super().find_class(module, name)

# --- Model loader
def load_model():
    try:
        with open('portfolio_risk_model.pkl', 'rb') as f:
            model = SafeUnpickler(f).load()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        return create_model_from_scratch()

# --- Create model from scratch if missing
def create_model_from_scratch():
    print("🔄 Creating model from scratch...")
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    try:
        data = yf.download(stocks, period='5y')['Adj Close']
        returns = data.pct_change().dropna()

        mean_returns = returns.mean().tolist()
        cov_matrix = returns.cov().values.tolist()
        volatilities = (returns.std() * np.sqrt(252)).tolist()
        portfolio_returns = returns.dot(np.ones(len(stocks)) / len(stocks))

        model = {
            'mean_returns': mean_returns,
            'covariance_matrix': cov_matrix,
            'volatilities': volatilities,
            'risk_metrics': {
                'historical_var_95': float(np.percentile(portfolio_returns, 5)),
                'cvar_95': float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()),
                'annual_mean_return': float(portfolio_returns.mean() * 252),
                'annual_volatility': float(portfolio_returns.std() * np.sqrt(252)),
                'sharpe_ratio': float((portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)))
            },
            'tickers': stocks,
            'training_date': datetime.now().strftime("%Y-%m-%d"),
            'data_period': '5 years'
        }
        print("✅ Model created from scratch!")
        return model
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return None

# =====================================================
# 🎯 Gradio App Class
# =====================================================
class GradioRiskApp:
    def __init__(self):
        self.model = load_model()
        self.available_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        print(f"🎯 Available stocks: {self.available_stocks}")

    # --- Simulation Runner
    def run_monte_carlo(self, selected_stocks, days=252, simulations=5000, weights=None):
        """Run Monte Carlo simulation for 1–5 stocks"""
        try:
            if self.model is None or len(selected_stocks) == 0:
                print("❌ No model or stocks selected")
                return None

            print(f"🎯 Running Monte Carlo for {len(selected_stocks)} stocks: {selected_stocks}")

            mean_returns = np.array(self.model['mean_returns'])
            cov_matrix = np.array(self.model['covariance_matrix'])

            stock_indices = [self.available_stocks.index(s) for s in selected_stocks]
            mean_returns = mean_returns[stock_indices]
            cov_matrix = cov_matrix[np.ix_(stock_indices, stock_indices)]

            cov_matrix = self.make_positive_definite(cov_matrix)
            np.random.seed(42)
            L = np.linalg.cholesky(cov_matrix)

            if weights is None:
                weights = np.ones(len(selected_stocks)) / len(selected_stocks)

            simulation_results = []
            for i in range(simulations):
                random_numbers = np.random.normal(0, 1, size=(days, len(selected_stocks)))
                correlated_returns = random_numbers @ L.T + mean_returns

                portfolio_value = 100.0
                portfolio_path = [portfolio_value]

                for day in range(days):
                    daily_return = np.dot(correlated_returns[day], weights)
                    portfolio_value *= (1 + daily_return)
                    portfolio_path.append(portfolio_value)

                simulation_results.append(portfolio_path)

            print(f"✅ Simulation completed: {len(simulation_results)} paths")
            return np.array(simulation_results)

        except Exception as e:
            print(f"❌ Simulation error: {str(e)}")
            return None

    # --- Ensure covariance matrix stability
    def make_positive_definite(self, matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        if min_eig < 0:
            matrix -= 10 * min_eig * np.eye(*matrix.shape)
        return matrix

    # --- Calculate metrics
    def calculate_metrics(self, simulation_results):
        if simulation_results is None:
            return None
        final_values = simulation_results[:, -1]
        metrics = {
            'expected_value': float(np.mean(final_values)),
            'prob_10_loss': float(np.mean(final_values < 90)),
            'var_95': float(np.percentile(final_values, 5)),
            'best_case': float(np.percentile(final_values, 95)),
            'worst_case': float(np.percentile(final_values, 5))
        }
        return metrics

    # --- Plots
    def create_simulation_plot(self, simulation_results):
        if simulation_results is None:
            return None
        fig = go.Figure()
        for i in range(min(50, len(simulation_results))):
            fig.add_trace(go.Scatter(y=simulation_results[i],
                                     mode='lines', line=dict(width=1, color='lightblue'),
                                     opacity=0.1, showlegend=False))
        mean_path = np.mean(simulation_results, axis=0)
        fig.add_trace(go.Scatter(y=mean_path, mode='lines',
                                 line=dict(width=3, color='red'), name='Average Path'))
        fig.update_layout(title="📈 Monte Carlo Simulation Paths",
                          xaxis_title="Trading Days",
                          yaxis_title="Portfolio Value ($)", height=400)
        return fig

    def create_distribution_plot(self, simulation_results, metrics):
        if simulation_results is None:
            return None
        final_values = simulation_results[:, -1]
        fig = px.histogram(x=final_values, nbins=50,
                           title="📊 Portfolio Value Distribution",
                           color_discrete_sequence=['#667eea'])
        if metrics:
            fig.add_vline(x=metrics['expected_value'], line_dash="dash",
                          line_color="red", annotation_text=f"Mean: ${metrics['expected_value']:.2f}")
            fig.add_vline(x=metrics['var_95'], line_dash="dash",
                          line_color="orange", annotation_text=f"5% VaR: ${metrics['var_95']:.2f}")
        fig.update_layout(height=400, showlegend=False)
        return fig

    def create_risk_gauge(self, metrics):
        if metrics is None:
            return None
        risk_prob = metrics['prob_10_loss'] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of >10% Loss"},
            gauge={'axis': {'range': [0, 50]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 15], 'color': "lightgreen"},
                             {'range': [15, 30], 'color': "yellow"},
                             {'range': [30, 50], 'color': "red"}]}))
        fig.update_layout(height=300)
        return fig

    # --- Main Analysis Function (Fixed)
    def analyze_portfolio(self, selected_stocks, simulation_days, num_simulations):
        print(f"🔍 Analyzing: {selected_stocks}")

        if not selected_stocks:
            return self.create_error_html("Please select at least one stock."), None, None, None

        weights = np.ones(len(selected_stocks)) / len(selected_stocks)
        simulations = self.run_monte_carlo(selected_stocks, simulation_days, int(num_simulations), weights=weights)
        if simulations is None:
            return self.create_error_html("Simulation failed"), None, None, None

        metrics = self.calculate_metrics(simulations)
        if metrics is None:
            return self.create_error_html("Metrics calculation failed"), None, None, None

        summary = self.create_summary_html(metrics, selected_stocks, simulation_days, num_simulations)
        sim_plot = self.create_simulation_plot(simulations)
        dist_plot = self.create_distribution_plot(simulations, metrics)
        gauge = self.create_risk_gauge(metrics)

        return summary, sim_plot, dist_plot, gauge

    # --- HTML renderers
    def create_error_html(self, message):
        return f"""
        <div style='text-align: center; padding: 40px; background: #f8d7da; border-radius: 10px;'>
            <h3 style='color: #721c24;'>❌ Error</h3>
            <p>{message}</p>
        </div>
        """

    def create_summary_html(self, metrics, stocks, days, sims):
        risk_level = "LOW" if metrics['prob_10_loss'] < 0.1 else "MEDIUM" if metrics['prob_10_loss'] < 0.2 else "HIGH"
        risk_color = "#28a745" if risk_level == "LOW" else "#ffc107" if risk_level == "MEDIUM" else "#dc3545"

        return f"""
        <div style="padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px; color: white;">
            <h2 style="text-align: center;">📊 Risk Analysis Report</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 15px 0;">
                <h3>Portfolio: {', '.join(stocks)}</h3>
                <p>Equal weights | Period: {days} days | Simulations: {sims}</p>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                    <h4>Expected Value</h4>
                    <p style="font-size: 24px; font-weight: bold;">${metrics['expected_value']:.2f}</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                    <h4>Risk Level</h4>
                    <p style="font-size: 24px; font-weight: bold; color: {risk_color};">{risk_level}</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                    <h4>10% Loss Probability</h4>
                    <p style="font-size: 24px; font-weight: bold;">{metrics['prob_10_loss']*100:.1f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;">
                    <h4>Worst Case (5%)</h4>
                    <p style="font-size: 24px; font-weight: bold;">${metrics['var_95']:.2f}</p>
                </div>
            </div>
        </div>
        """

# =====================================================
# 🚀 Gradio UI
# =====================================================
app = GradioRiskApp()

with gr.Blocks(theme=gr.themes.Soft(), title="QuantRisk Pro") as demo:
    gr.Markdown("# 📊 QuantRisk Pro - Portfolio Risk Analyzer")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Configuration")
            stocks = gr.CheckboxGroup(
                choices=app.available_stocks,
                value=app.available_stocks,
                label="Select Stocks (1–5 Allowed)"
            )
            days = gr.Slider(30, 500, 252, label="Time Horizon (Days)")
            sims = gr.Dropdown([1000, 2500, 5000, 10000], value=5000, label="Simulations")
            btn = gr.Button("🚀 Analyze", variant="primary")

        with gr.Column(scale=2):
            summary = gr.HTML()
            with gr.Row():
                gauge = gr.Plot()
                sim_plot = gr.Plot()
            dist_plot = gr.Plot()

    btn.click(
        fn=app.analyze_portfolio,
        inputs=[stocks, days, sims],
        outputs=[summary, sim_plot, dist_plot, gauge]
    )

if __name__ == "__main__":
    demo.launch()
