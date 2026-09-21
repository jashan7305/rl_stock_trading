# app.py

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN, A2C

from environment import TradingEnv


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RL Trading Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b1120;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f8fafc !important;
    }

    /* Normal text */
    p, label, .stMarkdown {
        color: #cbd5e1;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #334155;
        background: #1e293b;
        color: #f8fafc;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #64748b;
        background: #334155;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #111827;
        border-color: #334155;
    }

    /* Number inputs */
    div[data-baseweb="input"] {
        background-color: #111827;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Custom hero */
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.95),
                rgba(15, 23, 42, 0.98)
            );
        border: 1px solid #1e293b;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 0;
    }

    .badge {
        display: inline-block;
        padding: 5px 11px;
        margin-right: 7px;
        border-radius: 999px;
        background: #1e293b;
        color: #cbd5e1;
        font-size: 0.8rem;
        border: 1px solid #334155;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

FEATURES = [
    "Return",
    "Volume_Change",
    "Close_SMA20",
    "Close_SMA50",
    "EMA_Ratio",
    "Volatility_20",
]


# ============================================================
# STOCK CONFIGURATION
# ============================================================

STOCKS = {
    "HDFC Bank": {
        "ticker": "HDFCBANK.NS",
        "csv": BASE_DIR / "data" / "hdfc" / "HDFCBANK.csv",
        "dqn": BASE_DIR / "models" / "hdfc" / "dqn_hdfc_final.zip",
        "a2c": BASE_DIR / "models" / "hdfc" / "a2c_hdfc_final.zip",
    },
    "Infosys": {
        "ticker": "INFY.NS",
        "csv": BASE_DIR / "data" / "infosys" / "INFY.csv",
        "dqn": BASE_DIR / "models" / "infosys" / "dqn_infosys_final.zip",
        "a2c": BASE_DIR / "models" / "infosys" / "a2c_infosys_final.zip",
    },
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_raw_data(csv_path):
    """
    Load the original OHLCV CSV.
    """

    df = pd.read_csv(csv_path)

    # Handle Date column
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # Remove accidental duplicate dates
    df = df.drop_duplicates(subset="Date").reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@st.cache_data
def prepare_dataset(csv_path):
    """
    Reproduce the same feature engineering and chronological
    train/validation/test split used during model training.

    IMPORTANT:
    The scaler is fitted ONLY on the training data.
    """

    df = load_raw_data(csv_path).copy()

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df["Return"] = df["Close"].pct_change()

    df["Volume_Change"] = df["Volume"].pct_change()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    df["EMA_12"] = df["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    df["EMA_26"] = df["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["Volatility_20"] = (
        df["Return"]
        .rolling(window=20)
        .std()
    )

    df["Close_SMA20"] = (
        df["Close"] / df["SMA_20"]
    ) - 1

    df["Close_SMA50"] = (
        df["Close"] / df["SMA_50"]
    ) - 1

    df["EMA_Ratio"] = (
        df["EMA_12"] / df["EMA_26"]
    ) - 1

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=FEATURES
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(train_df[FEATURES])

    train_df[FEATURES] = scaler.transform(
        train_df[FEATURES]
    )

    validation_df[FEATURES] = scaler.transform(
        validation_df[FEATURES]
    )

    test_df[FEATURES] = scaler.transform(
        test_df[FEATURES]
    )

    # --------------------------------------------------------
    # Create environment-ready objects
    # --------------------------------------------------------

    X_train = train_df[FEATURES].copy()
    X_validation = validation_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    Y_train = df.loc[
        train_df.index,
        "Close",
    ].copy()

    Y_validation = df.loc[
        validation_df.index,
        "Close",
    ].copy()

    Y_test = df.loc[
        test_df.index,
        "Close",
    ].copy()

    # Preserve Date as index
    X_train.index = df.loc[
        train_df.index,
        "Date",
    ]

    X_validation.index = df.loc[
        validation_df.index,
        "Date",
    ]

    X_test.index = df.loc[
        test_df.index,
        "Date",
    ]

    Y_train.index = df.loc[
        train_df.index,
        "Date",
    ]

    Y_validation.index = df.loc[
        validation_df.index,
        "Date",
    ]

    Y_test.index = df.loc[
        test_df.index,
        "Date",
    ]

    return {
        "raw": df,
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_validation": Y_validation,
        "Y_test": Y_test,
        "scaler": scaler,
    }


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models(dqn_path, a2c_path):

    dqn_model = DQN.load(dqn_path)

    a2c_model = A2C.load(a2c_path)

    return dqn_model, a2c_model


# ============================================================
# BUY & HOLD
# ============================================================

def evaluate_buy_and_hold(
    prices,
    initial_capital=INITIAL_CAPITAL,
):

    prices = pd.Series(prices)

    initial_price = float(
        prices.iloc[0]
    )

    shares = (
        initial_capital /
        initial_price
    )

    portfolio_values = (
        shares *
        prices.to_numpy(dtype=float)
    )

    dates = list(prices.index)

    return {
        "portfolio_values": portfolio_values,
        "dates": dates,
        "actions": [1] + [0] * (
            len(prices) - 1
        ),
        "initial_capital": initial_capital,
        "final_portfolio": float(
            portfolio_values[-1]
        ),
        "total_return": (
            portfolio_values[-1] /
            initial_capital
        ) - 1,
        "trade_count": 1,
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    features,
    prices,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost=TRANSACTION_COST,
):

    env = TradingEnv(
        features=features,
        prices=prices,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    obs, info = env.reset()

    portfolio_values = [
        initial_capital
    ]

    dates = [
        info["date"]
    ]

    actions = []

    terminated = False
    truncated = False

    while not (
        terminated or truncated
    ):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        actions.append(action)

        portfolio_values.append(
            info["portfolio_value"]
        )

        dates.append(
            info["date"]
        )

    return {
        "env": env,
        "portfolio_values": np.asarray(
            portfolio_values,
            dtype=float,
        ),
        "dates": dates,
        "actions": actions,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    portfolio_values,
    initial_capital=INITIAL_CAPITAL,
):

    portfolio_values = np.asarray(
        portfolio_values,
        dtype=float,
    )

    daily_returns = (
        portfolio_values[1:] /
        portfolio_values[:-1]
    ) - 1

    total_return = (
        portfolio_values[-1] /
        initial_capital
    ) - 1

    running_max = np.maximum.accumulate(
        portfolio_values
    )

    drawdowns = (
        portfolio_values /
        running_max
    ) - 1

    max_drawdown = float(
        drawdowns.min()
    )

    if (
        len(daily_returns) < 2
        or daily_returns.std(ddof=1) == 0
    ):
        sharpe_ratio = 0.0

    else:
        sharpe_ratio = (
            daily_returns.mean() /
            daily_returns.std(ddof=1)
        ) * np.sqrt(252)

    return {
        "Initial Capital": float(
            initial_capital
        ),
        "Final Portfolio": float(
            portfolio_values[-1]
        ),
        "Total Return": float(
            total_return
        ),
        "Max Drawdown": float(
            max_drawdown
        ),
        "Sharpe Ratio": float(
            sharpe_ratio
        ),
    }


# ============================================================
# ACTION ANALYSIS
# ============================================================

def action_counts(actions):

    actions = np.asarray(actions)

    return {
        "HOLD": int(
            np.sum(actions == 0)
        ),
        "BUY": int(
            np.sum(actions == 1)
        ),
        "SELL": int(
            np.sum(actions == 2)
        ),
    }


# ============================================================
# PORTFOLIO CHART
# ============================================================

def create_portfolio_chart(
    results,
    company_name,
):

    fig = go.Figure()

    for strategy, result in results.items():

        fig.add_trace(
            go.Scatter(
                x=result["dates"],
                y=result["portfolio_values"],
                mode="lines",
                name=strategy,
                line=dict(
                    width=2.5,
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Portfolio: ₹%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=INITIAL_CAPITAL,
        line_dash="dash",
        line_width=1,
        annotation_text="Initial Capital",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"{company_name} — Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_dark",
        height=520,
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


# ============================================================
# DRAWDOWN CHART
# ============================================================

def create_drawdown_chart(
    results,
    company_name,
):

    fig = go.Figure()

    for strategy, result in results.items():

        portfolio = np.asarray(
            result["portfolio_values"],
            dtype=float,
        )

        running_max = np.maximum.accumulate(
            portfolio
        )

        drawdown = (
            portfolio /
            running_max
        ) - 1

        fig.add_trace(
            go.Scatter(
                x=result["dates"],
                y=drawdown * 100,
                mode="lines",
                name=strategy,
                line=dict(
                    width=2,
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Drawdown: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
    )

    fig.update_layout(
        title=f"{company_name} — Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
    )

    return fig


# ============================================================
# ACTION CHART
# ============================================================

def create_action_chart(
    actions,
    strategy,
):

    counts = action_counts(actions)

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                text=list(counts.values()),
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=f"{strategy} — Actions",
        xaxis_title="Action",
        yaxis_title="Number of Steps",
        template="plotly_dark",
        height=350,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
    )

    return fig


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    # 📈 RL Trading Lab

    **Historical backtesting of Deep Reinforcement Learning trading agents
    against a Buy & Hold benchmark.**

    `DQN` &nbsp; `A2C` &nbsp; `Gymnasium` &nbsp;
    `Stable-Baselines3` &nbsp; `Indian Equities`
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Backtest Settings")

    company = st.selectbox(
        "Select stock",
        list(STOCKS.keys()),
    )

    config = STOCKS[company]

    st.divider()

    st.markdown("### Portfolio")

    initial_capital = st.number_input(
        "Initial capital (₹)",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
    )

    transaction_cost = st.number_input(
        "Transaction cost",
        min_value=0.0,
        max_value=0.05,
        value=0.001,
        step=0.0005,
        format="%.4f",
        help="0.001 = 0.1% per position change.",
    )

    st.divider()

    st.markdown("### Test Window")

    st.caption(
        "The app only evaluates the unseen 15% test set."
    )

    try:

        dataset = prepare_dataset(
            config["csv"]
        )

        test_prices = dataset["Y_test"]

        test_start = test_prices.index[0]
        test_end = test_prices.index[-1]

        st.write(
            f"**Test period**  \n"
            f"{test_start:%d %b %Y} → "
            f"{test_end:%d %b %Y}"
        )

        max_days = len(test_prices)

        num_days = st.slider(
            "Trading days",
            min_value=10,
            max_value=max_days,
            value=min(120, max_days),
            step=10,
        )

        valid_start_dates = test_prices.index[
            :max_days - num_days + 1
        ]

        selected_start = st.selectbox(
            "Start date",
            valid_start_dates,
            format_func=lambda x: x.strftime(
                "%d %b %Y"
            ),
        )

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

        st.stop()

    st.divider()

    run_backtest = st.button(
        "▶ Run Backtest",
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="small-muted">
        Models are evaluated deterministically on historical
        unseen test data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

dataset = prepare_dataset(
    config["csv"]
)

test_features = dataset["X_test"]
test_prices = dataset["Y_test"]


# ============================================================
# WINDOW SELECTION
# ============================================================

start_position = test_prices.index.get_loc(
    selected_start
)

window_positions = range(
    start_position,
    start_position + num_days,
)

window_dates = test_prices.index[
    start_position:
    start_position + num_days
]

window_features = test_features.loc[
    window_dates
]

window_prices = test_prices.loc[
    window_dates
]


# ============================================================
# LOAD MODELS
# ============================================================

try:

    dqn_model, a2c_model = load_models(
        str(config["dqn"]),
        str(config["a2c"]),
    )

except Exception as e:

    st.error(
        f"Could not load trained models: {e}"
    )

    st.stop()


# ============================================================
# RUN BACKTEST
# ============================================================

if run_backtest or "backtest_results" not in st.session_state:

    with st.spinner(
        "Running DQN, A2C and Buy & Hold backtests..."
    ):

        dqn_result = evaluate_model(
            dqn_model,
            window_features,
            window_prices,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )

        a2c_result = evaluate_model(
            a2c_model,
            window_features,
            window_prices,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )

        buy_hold_result = evaluate_buy_and_hold(
            window_prices,
            initial_capital=initial_capital,
        )

        results = {
            "DQN": dqn_result,
            "A2C": a2c_result,
            "Buy & Hold": buy_hold_result,
        }

        st.session_state.backtest_results = results

        st.session_state.backtest_config = {
            "company": company,
            "start": selected_start,
            "days": num_days,
            "initial_capital": initial_capital,
            "transaction_cost": transaction_cost,
        }


else:

    results = st.session_state.backtest_results


# ============================================================
# BACKTEST INFO
# ============================================================

current_config = st.session_state.backtest_config

st.markdown(
    '<div class="section-title">Backtest Overview</div>',
    unsafe_allow_html=True,
)

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "Stock",
        current_config["company"],
    )

with info2:
    st.metric(
        "Start",
        current_config["start"].strftime(
            "%d %b %Y"
        ),
    )

with info3:
    st.metric(
        "Trading Days",
        current_config["days"],
    )

with info4:
    st.metric(
        "Initial Capital",
        f"₹{current_config['initial_capital']:,.0f}",
    )


st.caption(
    f"Window: "
    f"{window_dates[0]:%d %b %Y} → "
    f"{window_dates[-1]:%d %b %Y} "
    f"({len(window_dates)} trading days)"
)


# ============================================================
# CALCULATE METRICS
# ============================================================

metrics = {}

for strategy, result in results.items():

    metrics[strategy] = calculate_metrics(
        result["portfolio_values"],
        current_config["initial_capital"],
    )

    if strategy == "Buy & Hold":

        metrics[strategy]["Trades"] = (
            result["trade_count"]
        )

    else:

        metrics[strategy]["Trades"] = (
            result["env"].trade_count
        )


# ============================================================
# SUMMARY CARDS
# ============================================================

st.markdown(
    '<div class="section-title">Performance Summary</div>',
    unsafe_allow_html=True,
)


# ---- DQN ----

dqn_metrics = metrics["DQN"]

a2c_metrics = metrics["A2C"]

bh_metrics = metrics["Buy & Hold"]


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("### 🤖 DQN")

    st.metric(
        "Final Portfolio",
        f"₹{dqn_metrics['Final Portfolio']:,.0f}",
    )

    st.metric(
        "Total Return",
        f"{dqn_metrics['Total Return'] * 100:.2f}%",
    )

    st.metric(
        "Max Drawdown",
        f"{dqn_metrics['Max Drawdown'] * 100:.2f}%",
    )

    st.metric(
        "Sharpe Ratio",
        f"{dqn_metrics['Sharpe Ratio']:.3f}",
    )

    st.caption(
        f"Trades: {dqn_metrics['Trades']}"
    )


with col2:

    st.markdown("### 🧠 A2C")

    st.metric(
        "Final Portfolio",
        f"₹{a2c_metrics['Final Portfolio']:,.0f}",
    )

    st.metric(
        "Total Return",
        f"{a2c_metrics['Total Return'] * 100:.2f}%",
    )

    st.metric(
        "Max Drawdown",
        f"{a2c_metrics['Max Drawdown'] * 100:.2f}%",
    )

    st.metric(
        "Sharpe Ratio",
        f"{a2c_metrics['Sharpe Ratio']:.3f}",
    )

    st.caption(
        f"Trades: {a2c_metrics['Trades']}"
    )


with col3:

    st.markdown("### 📊 Buy & Hold")

    st.metric(
        "Final Portfolio",
        f"₹{bh_metrics['Final Portfolio']:,.0f}",
    )

    st.metric(
        "Total Return",
        f"{bh_metrics['Total Return'] * 100:.2f}%",
    )

    st.metric(
        "Max Drawdown",
        f"{bh_metrics['Max Drawdown'] * 100:.2f}%",
    )

    st.metric(
        "Sharpe Ratio",
        f"{bh_metrics['Sharpe Ratio']:.3f}",
    )

    st.caption(
        f"Trades: {bh_metrics['Trades']}"
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

st.markdown(
    '<div class="section-title">Strategy Comparison</div>',
    unsafe_allow_html=True,
)

comparison_df = pd.DataFrame(
    [
        {
            "Strategy": strategy,
            "Initial Capital": metrics[strategy][
                "Initial Capital"
            ],
            "Final Portfolio": metrics[strategy][
                "Final Portfolio"
            ],
            "Total Return": metrics[strategy][
                "Total Return"
            ],
            "Max Drawdown": metrics[strategy][
                "Max Drawdown"
            ],
            "Sharpe Ratio": metrics[strategy][
                "Sharpe Ratio"
            ],
            "Trades": metrics[strategy][
                "Trades"
            ],
        }
        for strategy in [
            "DQN",
            "A2C",
            "Buy & Hold",
        ]
    ]
)

display_df = comparison_df.copy()

display_df["Initial Capital"] = (
    display_df["Initial Capital"]
    .map(lambda x: f"₹{x:,.0f}")
)

display_df["Final Portfolio"] = (
    display_df["Final Portfolio"]
    .map(lambda x: f"₹{x:,.2f}")
)

display_df["Total Return"] = (
    display_df["Total Return"]
    .map(lambda x: f"{x * 100:.2f}%")
)

display_df["Max Drawdown"] = (
    display_df["Max Drawdown"]
    .map(lambda x: f"{x * 100:.2f}%")
)

display_df["Sharpe Ratio"] = (
    display_df["Sharpe Ratio"]
    .map(lambda x: f"{x:.3f}")
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PORTFOLIO CHART
# ============================================================

st.markdown(
    '<div class="section-title">Portfolio Performance</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    create_portfolio_chart(
        results,
        company,
    ),
    use_container_width=True,
)


# ============================================================
# DRAWDOWN
# ============================================================

st.markdown(
    '<div class="section-title">Risk Analysis</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    create_drawdown_chart(
        results,
        company,
    ),
    use_container_width=True,
)


# ============================================================
# ACTION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Trading Activity</div>',
    unsafe_allow_html=True,
)

action_col1, action_col2 = st.columns(2)

with action_col1:

    st.plotly_chart(
        create_action_chart(
            results["DQN"]["actions"],
            "DQN",
        ),
        use_container_width=True,
    )

with action_col2:

    st.plotly_chart(
        create_action_chart(
            results["A2C"]["actions"],
            "A2C",
        ),
        use_container_width=True,
    )


# ============================================================
# POSITION / TRADE DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">Model Details</div>',
    unsafe_allow_html=True,
)

tab_dqn, tab_a2c = st.tabs(
    ["🤖 DQN", "🧠 A2C"]
)


with tab_dqn:

    dqn_env = results["DQN"]["env"]

    position_history = getattr(
        dqn_env,
        "position_history",
        [],
    )

    st.write(
        f"**Total trades:** "
        f"{results['DQN']['env'].trade_count}"
    )

    dqn_actions = results["DQN"]["actions"]

    dqn_action_df = pd.DataFrame(
        {
            "Date": results["DQN"]["dates"][1:],
            "Action": [
                {
                    0: "HOLD",
                    1: "BUY",
                    2: "SELL",
                }.get(a, "UNKNOWN")
                for a in dqn_actions
            ],
        }
    )

    st.dataframe(
        dqn_action_df,
        use_container_width=True,
        hide_index=True,
    )


with tab_a2c:

    a2c_env = results["A2C"]["env"]

    st.write(
        f"**Total trades:** "
        f"{results['A2C']['env'].trade_count}"
    )

    a2c_actions = results["A2C"]["actions"]

    a2c_action_df = pd.DataFrame(
        {
            "Date": results["A2C"]["dates"][1:],
            "Action": [
                {
                    0: "HOLD",
                    1: "BUY",
                    2: "SELL",
                }.get(a, "UNKNOWN")
                for a in a2c_actions
            ],
        }
    )

    st.dataframe(
        a2c_action_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">Export</div>',
    unsafe_allow_html=True,
)

download_df = comparison_df.copy()

download_csv = download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Performance CSV",
    data=download_csv,
    file_name=(
        f"{company.lower().replace(' ', '_')}"
        "_backtest_results.csv"
    ),
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 10px;
    ">
        Reinforcement Learning for Stock Trading
        · DQN vs A2C vs Buy & Hold
        <br>
        Historical backtesting only · Not financial advice
    </div>
    """,
    unsafe_allow_html=True,
)