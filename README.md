# RL Stock Trading Lab 📈

A reinforcement learning-based stock trading system that trains deep learning models to make buy/hold/sell decisions on Indian stocks. This project implements **DQN** (Deep Q-Network) and **A2C** (Actor-Critic) algorithms using Stable-Baselines3 to optimize trading strategies.

## Overview

This project demonstrates how reinforcement learning can be applied to algorithmic stock trading:

- **Two Agents**: DQN and A2C trained independently on each stock
- **Two Stocks**: HDFC Bank (HDFCBANK.NS) and Infosys (INFY.NS)
- **Custom Environment**: Gymnasium-compatible trading environment with realistic transaction costs
- **Interactive Dashboard**: Streamlit app for exploring model performance and backtesting results
- **Complete ML Pipeline**: Data collection, feature engineering, model training, hyperparameter tuning, and analysis

## Project Structure

```
rl_stock_trading/
├── get_data.ipynb                  # Download historical stock data from Yahoo Finance
├── train_hdfc_models.ipynb         # Train DQN & A2C on HDFC Bank stock
├── train_infosys_models.ipynb      # Train DQN & A2C on Infosys stock
├── analysis_hdfc.ipynb             # Analyze & visualize HDFC Bank model performance
├── analysis_infosys.ipynb          # Analyze & visualize Infosys model performance
├── app.py                          # Interactive Streamlit dashboard
├── environment.py                  # Custom Gymnasium trading environment
├── pyproject.toml                  # Project metadata & dependencies
├── requirements.txt                # Package dependencies
├── data/
│   ├── hdfc/
│   │   └── HDFCBANK.csv           # Historical HDFC Bank price data
│   └── infosys/
│       └── INFY.csv               # Historical Infosys price data
├── models/
│   ├── hdfc/
│   │   ├── dqn_tuning_results.csv # Hyperparameter tuning results (DQN)
│   │   └── a2c_tuning_results.csv # Hyperparameter tuning results (A2C)
│   └── infosys/
│       ├── dqn_tuning_results.csv
│       └── a2c_tuning_results.csv
└── submissions/                    # Directory for model artifacts

```

## Key Components

### 1. **Trading Environment** (`environment.py`)

A custom Gymnasium environment that simulates trading with:

- **State Space**: Market features (technical indicators) + current portfolio position
- **Action Space**: 3 discrete actions
  - `0`: HOLD (maintain current position)
  - `1`: BUY (fully enter the market)
  - `2`: SELL (fully exit the market)
- **Reward Signal**: Portfolio value changes with transaction costs
- **Realistic Constraints**:
  - Proportional transaction costs (0.1% by default)
  - Validated feature data (no NaN/infinite values)
  - Proper train/validation/test split handling

### 2. **Feature Engineering**

The following technical indicators are computed for each stock:

| Feature | Description |
|---------|-------------|
| `Return` | Daily percentage change in close price |
| `Volume_Change` | Daily percentage change in volume |
| `Close_SMA20` | Close price ratio to 20-day SMA |
| `Close_SMA50` | Close price ratio to 50-day SMA |
| `EMA_Ratio` | Ratio of 12-day EMA to 26-day EMA |
| `Volatility_20` | 20-day rolling standard deviation of returns |

All features are standardized using a scaler fitted on training data only.

### 3. **Model Training Pipeline**

**Workflow** (`train_*.ipynb`):

1. **Data Loading**: Load historical data and prepare features
2. **Data Splitting**: Chronological split (70% train, 15% val, 15% test)
3. **Hyperparameter Tuning**: Grid search over key hyperparameters
4. **Model Training**: Train DQN and A2C on training data
5. **Validation**: Evaluate models on validation set
6. **Backtesting**: Test final models on holdout test set
7. **Results Export**: Save tuning results and trained models

**Hyperparameters Tuned**:
- Learning rate
- Network architecture (layer sizes)
- Exploration parameters (epsilon decay, exploration fraction)
- Batch sizes and update frequencies

### 4. **Analysis & Visualization** (`analysis_*.ipynb`)

Each analysis notebook provides:

- **Equity Curves**: Portfolio value over time (train/val/test)
- **Drawdown Analysis**: Maximum loss from peak
- **Trade Statistics**: Total trades, win rate, Sharpe ratio
- **Indicator Analysis**: Price vs technical indicators
- **Performance Comparison**: DQN vs A2C vs Buy-and-Hold
- **Feature Importance**: Correlation analysis

### 5. **Interactive Dashboard** (`app.py`)

A Streamlit application featuring:

- **Model Selection**: Choose between stocks and algorithms (DQN/A2C)
- **Performance Metrics**: Total return, Sharpe ratio, max drawdown
- **Interactive Charts**: Plotly visualizations of equity curves and trades
- **Backtest Analysis**: Detailed performance breakdown by time period
- **Trade Details**: Full transaction history with entry/exit prices
- **Dark Theme**: Professional dark UI optimized for financial data

**Run the dashboard**:
```bash
streamlit run app.py
```

## Installation

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Setup

1. **Clone/navigate to the project**:
   ```bash
   cd /Users/divy/College/RL/Project/rl_stock_trading
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Download Data
Open and run `get_data.ipynb` to download historical price data from Yahoo Finance:
- Downloads 15+ years of daily OHLCV data for HDFC Bank and Infosys
- Saves data to `data/hdfc/` and `data/infosys/`

### 2. Train Models
Run either notebook to train models:
- `train_hdfc_models.ipynb` - Train on HDFC Bank stock
- `train_infosys_models.ipynb` - Train on Infosys stock

Each notebook:
- Performs hyperparameter tuning
- Trains both DQN and A2C agents
- Backtests on holdout test data
- Exports results and saves trained models

### 3. Analyze Results
Review performance with:
- `analysis_hdfc.ipynb` - HDFC Bank analysis
- `analysis_infosys.ipynb` - Infosys analysis

### 4. Explore Interactive Dashboard
Launch the Streamlit app to explore models interactively:
```bash
streamlit run app.py
```

## Key Metrics & Definitions

| Metric | Definition |
|--------|-----------|
| **Total Return** | (Final Portfolio Value - Initial Capital) / Initial Capital |
| **Sharpe Ratio** | Mean daily return / Std dev of daily returns (annualized) |
| **Max Drawdown** | Maximum loss from previous peak to trough |
| **Win Rate** | Percentage of profitable trades |
| **Trade Count** | Total number of transactions executed |
| **Transaction Cost** | Proportional cost charged on each trade (0.1%) |

## Important Notes

### Data & Feature Handling
- All features are **standardized only on training data**
- Scaler is fit on training set and applied to validation/test sets to avoid data leakage
- NaN values are removed during feature engineering
- Data is chronologically split (not random) to respect temporal dependency

### Model Architecture
- **DQN**: 2-3 fully connected layers (typically 256-512 units)
- **A2C**: Actor-Critic with separate policy and value networks
- Both use ReLU activations and experience replay/trajectory buffers

### Trading Constraints
- **Discrete Actions**: No partial positions (either 0% or 100% invested)
- **Transaction Costs**: 0.1% applied on every trade (realistic for retail trading)
- **No Shorting**: Models can only go long or hold cash
- **Initial Capital**: $100,000 USD equivalent

## Performance Expectations

Expected annual Sharpe ratios after training:
- **HDFC Bank**: 0.5-1.2 (market-dependent)
- **Infosys**: 0.4-1.0 (market-dependent)

*Note: Past performance does not guarantee future results. These models are for educational purposes.*

## Technologies & Libraries

| Technology | Purpose |
|-----------|---------|
| **Gymnasium** | RL environment framework |
| **Stable-Baselines3** | DQN & A2C implementations |
| **PyTorch** | Deep learning backend |
| **Pandas/NumPy** | Data manipulation |
| **Scikit-learn** | Feature scaling |
| **Plotly** | Interactive visualizations |
| **Streamlit** | Interactive web dashboard |
| **yfinance** | Stock data download |

## Project Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  1. get_data.ipynb: Download OHLCV from Yahoo Finance          │
│     HDFC Bank, Infosys (15+ years daily)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. train_*_models.ipynb: Feature Engineering & Preprocessing   │
│     • Compute 6 technical indicators                            │
│     • Chronological 70/15/15 split                              │
│     • Standardize features (fit on train only)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Hyperparameter Tuning                                        │
│     • Grid search over learning rates, architectures            │
│     • Validate on validation set                                │
│     • Export tuning results to CSV                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Train Final Models (DQN & A2C)                              │
│     • Training on train set                                     │
│     • Save trained models (.zip)                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Backtest on Test Set                                        │
│     • Evaluate on unseen test data (15%)                        │
│     • Calculate performance metrics                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. analysis_*.ipynb: Visualize & Analyze                       │
│     • Equity curves, drawdowns, trades                          │
│     • Compare DQN vs A2C vs Buy-and-Hold                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. app.py: Interactive Streamlit Dashboard                     │
│     • Explore models, view backtest results                     │
│     • Compare performance across stocks & algorithms            │
└─────────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: "Model file not found"
- Ensure you've run the training notebooks to generate models
- Models are saved in `models/hdfc/` and `models/infosys/` directories

### Issue: "Data file not found"
- Run `get_data.ipynb` to download historical price data
- Data should be in `data/hdfc/` and `data/infosys/`

### Issue: "Insufficient data points"
- The environment requires at least 2 observations
- Ensure historical data covers sufficient time period

### Issue: Streamlit app is slow
- App caches data loading and preprocessing
- First run may take longer; subsequent runs use cache

## Future Improvements

- [ ] Add support for intraday trading (1-hour, 15-min data)
- [ ] Implement position sizing (gradual entry/exit)
- [ ] Add short selling capability
- [ ] Portfolio optimization across multiple stocks
- [ ] Online learning / continual model updates
- [ ] More sophisticated reward shaping
- [ ] Transformer-based architecture for temporal modeling
- [ ] Live trading paper-trading simulation

## Disclaimer

⚠️ **This project is for educational purposes only.** 

- These models are trained on historical data and past performance does not guarantee future results
- Actual stock trading involves real financial risk
- Do not use these models for real-money trading without extensive additional validation
- Always conduct your own due diligence and consult financial advisors

## License

This project is provided as-is for educational use.

## Contact & Questions

For questions about the project structure, models, or analysis, refer to the individual notebook cells and code comments.

---

**Last Updated**: 2026-09-23  
**Python Version**: 3.11+
