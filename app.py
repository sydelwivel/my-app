import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# ML imports
import pmdarima as pm
from sklearn.metrics import mean_squared_error, r2_score

# --- Custom Light Orange Theme ---
THEMES = {
    "light_orange": {
        "primary_bg": "#FFF8F0",      # Very light, creamy orange
        "secondary_bg": "#FAF6F0",    # Slightly off-white complement
        "surface_bg": "#FFF3E0",      # Light orange for cards/containers
        "accent_primary": "#FB8C00",      # Vibrant orange for accents/buttons (Amber 700)
        "accent_secondary": "#FFA726",    # Lighter orange accent (Amber 500)
        "text_primary": "#1A202C",      # Dark charcoal text for readability
        "text_secondary": "#4A5568",    # Muted dark text for labels
        "text_muted": "#A0AEC0",      # Light grey for less important text
        "border_color": "#E0D8CC",      # Soft, warm border color
        "success_color": "#2E7D32",     # A deeper green that pairs well with orange
        "warning_color": "#FFC107",     # Standard warning amber/yellow
        "error_color": "#D32F2F",       # A deeper red for better contrast
        "hover_bg": "#FFE0B2",        # Light amber for hover effects
        "chart_grid": "#E0E0E0",      # Standard light grid lines
    }
}

# --- Constants using the selected theme ---
COLORS = {}

# Predefined stock lists by sector (EXPANDED)
STOCKS_BY_SECTOR = {
    "Nifty 50 Leaders": ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS", "HINDUNILVR.NS"],
    "Banking Sector": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS", "AUROPHARMA.NS"],
    "Information Technology": ["WIPRO.NS", "TECHM.NS", "HCLTECH.NS", "INFY.NS", "TCS.NS"],
    "Automotive": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "Consumer Goods": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "ASIANPAINT.NS"], # New Sector
    "Healthcare": ["APOLLOHOSP.NS", "DIVISLAB.NS", "DRL.NS"] # New Sector, distinct from Pharma
}

# Sample Portfolio Data (kept as is, but users can upload their own)
SAMPLE_PORTFOLIO_DATA = pd.DataFrame({
    'Stock': ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
              'SUNPHARMA.NS', 'WIPRO.NS', 'MARUTI.NS'],
    'Shares': [50, 100, 75, 80, 120, 60, 150, 40],
    'Purchase_Price': [2450.0, 1320.0, 3580.0, 1680.0, 920.0, 1150.0, 480.0, 9200.0],
    'Sector': ['Energy', 'IT', 'IT', 'Banking', 'Banking', 'Pharma', 'IT', 'Auto']
})

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="FinInsight Hub",
    page_icon=" ", # Reverting to standard emoji for broader compatibility with page_icon argument
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------- Professional Theme Application --------------------
def apply_professional_theme():
    """Applies the static light orange theme."""
    global COLORS
    COLORS.update(THEMES["light_orange"])

    st.markdown(f"""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');


    /* Professional color palette variables (dynamically set) */
    :root {{
        --primary-bg: {COLORS['primary_bg']};
        --secondary-bg: {COLORS['secondary_bg']};
        --surface-bg: {COLORS['surface_bg']};
        --accent-primary: {COLORS['accent_primary']};
        --accent-secondary: {COLORS['accent_secondary']};
        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
        --text-muted: {COLORS['text_muted']};
        --border-color: {COLORS['border_color']};
        --success-color: {COLORS['success_color']};
        --warning_color: {COLORS['warning_color']};
        --error_color: {COLORS['error_color']};
        --hover_bg: {COLORS['hover_bg']};
        --chart-grid: {COLORS['chart_grid']};
    }}

    /* Global app styling */
    .stApp {{
        background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Sidebar styling */
    [data-testid="stSidebarContent"] {{
        background: linear-gradient(180deg, var(--surface-bg) 0%, var(--secondary_bg) 100%);
        border-right: 2px solid var(--border-color);
        color: var(--text-primary);
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface-bg) 0%, var(--secondary_bg) 100%);
        border-right: 2px solid var(--border-color);
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown h5,
    [data-testid="stSidebar"] .stMarkdown h6,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stFileUploader,
    [data-testid="stSidebar"] .stButton > button {{
        color: var(--text-primary) !important;
    }}
    /* Specific text elements in sidebar like selectbox options */
    .stSelectbox div[data-baseweb="select"] {{
        background-color: var(--surface-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div[role="button"] {{
        color: var(--text-primary) !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div[role="listbox"] {{
        background-color: var(--secondary_bg) !important;
        color: var(--text-primary) !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div[role="listbox"] > div > div {{
        color: var(--text-primary) !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div > div:hover {{ /* Target hover for option text */
        background-color: var(--hover_bg) !important;
    }}
    /* Ensure the actual options background changes on hover/focus */
    .stSelectbox div[data-baseweb="popover"] div[role="option"]:hover {{
        background-color: var(--hover_bg) !important;
    }}


    /* Enhanced metric containers */
    div[data-testid="metric-container"] {{
        background: linear-gradient(145deg, var(--surface-bg), var(--hover_bg));
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }}

    div[data-testid="metric-container"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(251, 140, 0, 0.3); /* Accent color based hover shadow */
        border-color: var(--accent-primary);
    }}

    /* Metric labels */
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] .metric-label {{
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    /* Metric values (Ruppes numbers etc.) - Vibrant and Visible */
    div[data-testid="metric-container"] div[data-testid="metric-value"],
    div[data-testid="metric-container"] .metric-value {{
        color: var(--accent-primary) !important; /* Use a vibrant accent color for values */
        font-weight: 800 !important; /* Make it bolder */
        font-size: 1.75rem !important; /* Slightly larger */
    }}

    /* Delta styling */
    div[data-testid="metric-container"] [data-testid="metric-delta"] {{
        font-weight: 600 !important;
        font-size: 0.95rem !important; /* Slightly larger delta text */
    }}
    div[data-testid="metric-container"] [data-testid="metric-delta"] svg {{
        color: var(--text-primary) !important; /* Ensure delta arrow color is visible */
    }}

    /* Professional tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: var(--surface-bg);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--border-color); /* Add border to tab list */
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        border: none;
        color: var(--text-secondary); /* Muted for inactive tabs */
        font-weight: 600;
        font-size: 0.9rem; /* Slightly larger font */
        padding: 12px 24px;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white !important; /* White for active tab text */
        box-shadow: 0 4px 12px rgba(251, 140, 0, 0.4); /* Orange accent shadow */
    }}

    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
        background: var(--hover_bg);
        color: var(--text-primary); /* Primary text color on hover */
    }}

    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }}

    /* Main titles with Playfair Display */
    .main-title {{
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
    }}

    h1 {{ font-size: 2.5rem !important; }} /* Larger for prominence */
    h2 {{ font-size: 2rem !important; }}
    h3 {{ font-size: 1.75rem !important; }}
    h4 {{ font-size: 1.5rem !important; }}

    /* General paragraph text */
    p {{
        color: var(--text-primary); /* Default paragraph text to primary color */
    }}

    /* Enhanced form controls */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stSlider .stSlider-thumb, /* Slider thumb */
    .stSlider .stSlider-track, /* Slider track */
    .stRadio > label, /* Radio button labels */
    .stTextarea > div > textarea
    {{
        background: var(--surface-bg) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }}

    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stSlider .stSlider-thumb:focus,
    .stSlider .stSlider-track:focus,
    .stRadio > label:has(input:checked) /* Check for checked radio input within label */
    {{
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(251, 140, 0, 0.2) !important; /* Soft orange focus ring */
    }}

    /* Specific styling for radio button indicators */
    .stRadio div[role="radiogroup"] label span p {{
        color: var(--text-primary) !important; /* Ensure radio button text is visible */
    }}
    .stRadio div[role="radiogroup"] label > div > div {{
        border: 2px solid var(--border-color) !important; /* Radio circle border */
    }}
    .stRadio div[role="radiogroup"] label > div > div > div {{
        background-color: var(--accent-primary) !important; /* Radio circle fill */
    }}

    /* File uploader styling */
    .stFileUploader > div {{
        background: var(--surface-bg);
        border: 2px dashed var(--border_color);
        border-radius: 16px;
        transition: all 0.2s ease;
    }}

    .stFileUploader > div:hover {{
        border-color: var(--accent-primary);
        background: var(--hover_bg);
    }}

    /* Professional alert styling */
    .stSuccess {{
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.1), rgba(46, 125, 50, 0.05));
        border: 1px solid rgba(46, 125, 50, 0.3);
        border-radius: 12px;
        color: var(--success_color);
    }}

    .stWarning {{
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 12px;
        color: var(--warning_color);
    }}

    .stError {{
        background: linear-gradient(135deg, rgba(211, 47, 47, 0.1), rgba(211, 47, 47, 0.05));
        border: 1px solid rgba(211, 47, 47, 0.3);
        border-radius: 12px;
        color: var(--error_color);
    }}

    .stInfo {{
        background: linear-gradient(135deg, rgba(255, 167, 38, 0.1), rgba(255, 167, 38, 0.05));
        border: 1px solid rgba(255, 167, 38, 0.3);
        border-radius: 12px;
        color: var(--accent-secondary);
    }}

    /* Enhanced dataframe styling */
    .stDataFrame {{
        background: var(--surface-bg);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }}

    /* Button enhancements */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
        box-shadow: 0 4px 12px rgba(251, 140, 0, 0.3); /* Button shadow */
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(251, 140, 0, 0.5); /* Stronger shadow on hover */
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--primary-bg);
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--border_color);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-muted);
    }}

    /* Plotly chart customization for backgrounds */
    .js-plotly-plot {{
        background-color: transparent !important;
    }}
    .main-svg {{
        background-color: transparent !important;
    }}

    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)


# Apply the single theme at the start of the script
apply_professional_theme()

# -------------------- Professional Header --------------------
st.markdown(f"""
<div style="
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 2rem;
    background: linear-gradient(135deg, {COLORS['accent_primary']}1A, {COLORS['accent_secondary']}1A);
    border-radius: 16px;
    margin: -1rem -1rem 2rem -1rem;
    padding-left: 1rem;
    padding-right: 1rem;
">
    <h1 style="
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.05em;
        font-family: 'Playfair Display', serif; /* Applied Playfair Display here */
    "> FinInsight Hub</h1>
    <p style="
        color: var(--text-secondary);
        font-size: 1.25rem;
        margin-top: 1rem;
        font-weight: 500;
    ">Navigate Your Investments with Precision</p>
</div>
""", unsafe_allow_html=True)


# -------------------- Utility Functions --------------------
def get_next_trading_days(start_date, num_days):
    """Calculates the next 'num_days' trading days from a start_date."""
    trading_days = []
    current_date = start_date
    while len(trading_days) < num_days:
        if current_date.weekday() < 5:  # Monday to Friday
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    return trading_days


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_price(ticker, period="5d"):
    """Fetches stock price data for a given ticker and period."""
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return None

        # Ensure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Convert to a timezone-naive DatetimeIndex if it's tz-aware
        # Convert to Asia/Kolkata first, then localize to None
        if data.index.tz is not None:
            data.index = data.index.tz_convert('Asia/Kolkata').tz_localize(None)
        else:
            # If naive, assume UTC and convert to Asia/Kolkata, then localize to None
            # This handles cases where yfinance might return naive datetimes on some systems/versions
            data.index = data.index.tz_localize('UTC', errors='coerce').tz_convert('Asia/Kolkata').tz_localize(None)
            # 'errors=coerce' will turn invalid parsing into NaT, good for robustness

        # Normalize the index to ensure all times are midnight
        data.index = data.index.normalize()

        return data
    except Exception as e:
        # st.error(f"Error fetching data for {ticker}: {e}") # Suppress this in utility to avoid flooding UI
        return None


# -------------------- ML Prediction Function (ARIMA) --------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def train_and_predict_stock_price(ticker, days_to_predict=1, look_back_days=120):
    """
    Trains an ARIMA model using auto_arima to predict stock prices
    and provides evaluation metrics.
    """
    # Fetch slightly more data than look_back_days to allow for differencing and test set
    df = fetch_stock_price(ticker, period=f"{look_back_days + 30}d")
    if df is None or len(df) < look_back_days + 10:
        return None, None, f"Not enough historical data for robust ARIMA prediction ({ticker}). Minimum recommended data is about {look_back_days + 10} days. Found {len(df) if df is not None else 0} days."

    series = df['Close'].copy()
    # Pandas FutureWarning fix: use .ffill() and .bfill() directly
    # Ensure all days are present, fill missing with previous valid observation
    series = series.asfreq('D').ffill().bfill()
    if series.empty or len(series) < 30: # Minimum data for auto_arima to function
        return None, None, f"Preprocessed data series for {ticker} is too short for ARIMA (less than 30 points)."

    # Determine train/test split. Ensure test set has at least 10 points or is equal to prediction horizon.
    test_set_size = max(days_to_predict, 10)
    train_size = len(series) - test_set_size
    if train_size <= 0:
        return None, None, f"Not enough data for a meaningful train/test split for {ticker}. Need at least {test_set_size + 1} points."

    train_data, test_data = series[0:train_size], series[train_size:]
    if train_data.empty:
        return None, None, f"Training data for {ticker} is empty."

    try:
        # UserWarning: stepwise model cannot be fit in parallel (n_jobs=1). Falling back to stepwise parameter search.
        # This is expected for stepwise=True, so n_jobs can be left or removed.
        model = pm.auto_arima(train_data,
                              start_p=1, start_q=1,
                              test='adf',        # Use ADF test to determine differencing
                              max_p=5, max_q=5, m=1, # m=1 for non-seasonal
                              d=None, seasonal=False, # Let auto_arima determine 'd'
                              trace=False, error_action='ignore',
                              suppress_warnings=True, stepwise=True,
                              information_criterion='aic', random_state=42, n_jobs=-1)

        # Get the actual differencing order determined by auto_arima
        d_order = model.order[1]

        # Generate historical predictions on training data
        # Adjust start index for in-sample predictions based on differencing order (d)
        start_in_sample_idx = d_order # Start predictions from index 'd'
        
        historical_predicted_values = pd.Series([], dtype='float64') # Initialize as empty
        if start_in_sample_idx < len(train_data): # Ensure there's data to predict for
            historical_predictions_raw = model.predict_in_sample(start=start_in_sample_idx, end=len(train_data)-1)
            # Align index - important! The predicted_in_sample corresponds to the actual points.
            historical_predicted_values = pd.Series(historical_predictions_raw, index=train_data.index[start_in_sample_idx:])
        else:
            # This case means train_data is too short for even one in-sample prediction given 'd_order'
            st.warning(f"No in-sample predictions generated for {ticker} due to short training data relative to differencing order ({d_order}).")


        # Evaluate model performance on a small test set (if test_data has enough points)
        rmse, r2 = np.nan, np.nan
        if len(test_data) > 0:
            test_predictions_series = model.predict(n_periods=len(test_data))
            # Align test predictions to the test_data index
            test_predictions_series.index = test_data.index
            rmse = np.sqrt(mean_squared_error(test_data, test_predictions_series))
            r2 = r2_score(test_data, test_predictions_series)
            metrics = {"RMSE": rmse, "R2 Score": r2, "ARIMA Order": model.order}
        else:
            metrics = {"RMSE": "N/A", "R2 Score": "N/A", "ARIMA Order": model.order}
            # st.warning(f"Not enough test data for {ticker} to calculate robust test set metrics.")


        # Generate future forecasts with confidence intervals
        future_forecast_values, conf_int = model.predict(n_periods=days_to_predict, return_conf_int=True)
        last_train_date = train_data.index[-1].date()
        future_trading_dates_naive = get_next_trading_days(last_train_date + timedelta(days=1), days_to_predict)
        
        future_predictions_series_aligned = pd.Series(future_forecast_values,
                                                      index=pd.to_datetime(future_trading_dates_naive).normalize())
        
        conf_int_df = pd.DataFrame(conf_int, columns=['lower_ci', 'upper_ci'],
                                   index=pd.to_datetime(future_trading_dates_naive).normalize())

        # Combine historical and future predictions for plotting
        df_close_for_plot = df['Close'].copy()
        # Create a combined index for all relevant dates: historical actuals, historical predictions, future predictions
        all_dates_union = df_close_for_plot.index.union(historical_predicted_values.index).union(future_predictions_series_aligned.index)
        plot_data_for_chart = pd.DataFrame(index=all_dates_union)
        plot_data_for_chart['Close'] = df_close_for_plot
        plot_data_for_chart['Predicted_Close'] = np.nan
        plot_data_for_chart.loc[historical_predicted_values.index, 'Predicted_Close'] = historical_predicted_values.values
        plot_data_for_chart.loc[future_predictions_series_aligned.index, 'Predicted_Close'] = future_predictions_series_aligned.values
        plot_data_for_chart['Date_Original'] = plot_data_for_chart.index

        # Add confidence intervals to plot data (only for future predictions)
        plot_data_for_chart['lower_ci'] = np.nan
        plot_data_for_chart['upper_ci'] = np.nan
        plot_data_for_chart.loc[conf_int_df.index, ['lower_ci', 'upper_ci']] = conf_int_df[['lower_ci', 'upper_ci']].values

        next_day_prediction = future_predictions_series_aligned.iloc[0] if not future_predictions_series_aligned.empty else None

        return next_day_prediction, metrics, plot_data_for_chart
    except Exception as e:
        return None, None, f"ARIMA model training failed for {ticker}: {e}. Try reducing lookback period or prediction horizon."


# -------------------- Enhanced Chart Plotting --------------------
def plot_professional_stock_chart(ticker, prediction_df=None):
    df = fetch_stock_price(ticker, period="1y") # Always fetch 1 year for base chart
    if df is not None and not df.empty:
        df_plot = df.copy()
        fig = go.Figure()

        # Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'], high=df_plot['High'],
            low=df_plot['Low'], close=df_plot['Close'],
            name=f'{ticker.replace(".NS", "")} Candlestick',
            increasing_line_color=COLORS['success_color'],
            decreasing_line_color=COLORS['error_color'],
            increasing_fillcolor='rgba(46, 125, 50, 0.3)',
            decreasing_fillcolor='rgba(211, 47, 47, 0.3)'
        ))

        if prediction_df is not None and isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
            # Ensure correct index type and timezone for prediction_df
            if not isinstance(prediction_df.index, pd.DatetimeIndex):
                prediction_df.index = pd.to_datetime(prediction_df.index)
            if prediction_df.index.tz is not None:
                prediction_df.index = prediction_df.index.tz_localize(None)
            prediction_df.index = prediction_df.index.normalize()
            prediction_df['Date_Original'] = prediction_df.index

            last_actual_date = df_plot.index.max()
            historical_preds = prediction_df[prediction_df['Date_Original'] <= last_actual_date].dropna(subset=['Predicted_Close'])
            future_preds = prediction_df[prediction_df['Date_Original'] > last_actual_date].dropna(subset=['Predicted_Close'])

            # Plot historical predictions on training data
            if not historical_preds.empty:
                fig.add_trace(go.Scatter(
                    x=historical_preds['Date_Original'], y=historical_preds['Predicted_Close'],
                    mode='lines', name='ARIMA Predicted (Historical)',
                    line=dict(color=COLORS['accent_secondary'], width=2, dash='solid'),
                    hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: ₹%{y:.2f}<extra></extra>'
                ))

            # Plot future predictions and link them
            if not future_preds.empty:
                # Connect last historical prediction/actual to first future prediction
                connect_x = []
                connect_y = []
                if not historical_preds.empty:
                    connect_x.append(historical_preds['Date_Original'].iloc[-1])
                    connect_y.append(historical_preds['Predicted_Close'].iloc[-1])
                elif not df_plot.empty: # Fallback to last actual close if no historical predictions
                    connect_x.append(df_plot.index[-1])
                    connect_y.append(df_plot['Close'].iloc[-1])

                if connect_x and not future_preds.empty: # Only add if there's a point to connect from and to
                    connect_x.append(future_preds['Date_Original'].iloc[0])
                    connect_y.append(future_preds['Predicted_Close'].iloc[0])
                    fig.add_trace(go.Scatter(
                        x=connect_x, y=connect_y,
                        mode='lines', line=dict(color=COLORS['accent_primary'], width=3, dash='dot'),
                        showlegend=False, name='Prediction Link', # Added name for clarity in hover
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: ₹%{y:.2f}<extra></extra>'
                    ))

                # Plot future predictions
                fig.add_trace(go.Scatter(
                    x=future_preds['Date_Original'], y=future_preds['Predicted_Close'],
                    mode='lines+markers', name='ARIMA Future Prediction',
                    line=dict(color=COLORS['accent_primary'], width=3, dash='dash'),
                    marker=dict(size=8, symbol='star-diamond', color=COLORS['accent_primary']),
                    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Predicted</b>: ₹%{y:.2f}<extra></extra>'
                ))
                
                # Add Confidence Intervals (shaded area)
                if 'lower_ci' in future_preds.columns and 'upper_ci' in future_preds.columns:
                    # Combine upper and lower CI points to form a filled area
                    ci_x = future_preds['Date_Original'].tolist() + future_preds['Date_Original'].tolist()[::-1]
                    ci_y = future_preds['upper_ci'].tolist() + future_preds['lower_ci'].tolist()[::-1]
                    fig.add_trace(go.Scatter(
                        x=ci_x,
                        y=ci_y,
                        fill='toself',
                        fillcolor='rgba(251,140,0,0.15)', # Accent primary with transparency
                        line=dict(color='rgba(255,255,255,0)'), # Invisible line
                        hoverinfo="skip", # Don't show hover for this trace
                        name='Confidence Interval',
                        showlegend=True,
                    ))


        fig.update_layout(
            title={
                'text': f'{ticker.replace(".NS", "")} - Price Analysis & ARIMA Prediction',
                'x': 0.5, 'xanchor': 'center',
                'font': {'size': 20, 'color': COLORS['text_primary']}
            },
            xaxis_title='Trading Period', yaxis_title='Price (INR)',
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_primary'], family='Inter'),
            xaxis=dict(gridcolor=COLORS['chart_grid'], showgrid=True, linecolor=COLORS['border_color']),
            yaxis=dict(gridcolor=COLORS['chart_grid'], showgrid=True, linecolor=COLORS['border_color']),
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(x=0, y=1.0, xanchor='left', font=dict(color=COLORS['text_primary']))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Market data unavailable for {ticker}. Please check the ticker symbol.")


# -------------------- Professional Portfolio Analysis --------------------
def create_professional_portfolio_analysis(portfolio_df):
    """Generates portfolio analysis charts and updated DataFrame."""
    current_values = []
    # Using st.progress for better UX during portfolio calculation
    progress_text = "Calculating portfolio current values..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, row in portfolio_df.iterrows():
        current_data = fetch_stock_price(row['Stock'], period="1d") # Fetch just 1 day for current price
        if current_data is not None and not current_data.empty:
            current_price = current_data['Close'].iloc[-1]
            current_values.append(current_price * row['Shares'])
        else:
            current_values.append(row['Purchase_Price'] * row['Shares']) # Fallback to purchase value
            # st.warning(f"Could not get live price for {row['Stock']}. Using purchase price for portfolio value calculation.") # Suppress to avoid flooding
        my_bar.progress((i + 1) / len(portfolio_df), text=f"Processing {row['Stock']}...")
    my_bar.empty() # Remove progress bar after completion

    portfolio_df['Current_Value'] = current_values
    sector_values = portfolio_df.groupby('Sector')['Current_Value'].sum().reset_index()
    pie_colors = [COLORS['accent_primary'], COLORS['accent_secondary'], COLORS['success_color'],
                  COLORS['warning_color'], COLORS['error_color'], '#8b5cf6', '#a855f7', '#ec4899']

    fig = px.pie(
        sector_values, values='Current_Value', names='Sector',
        title="Portfolio Sector Allocation Analysis",
        color_discrete_sequence=pie_colors
    )
    fig.update_layout(
        template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_primary'], family='Inter'),
        title=dict(x=0.5, font=dict(size=18)), showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05,
                    font=dict(color=COLORS['text_primary']))
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color=COLORS['primary_bg'], width=1)))

    return fig, portfolio_df


# -------------------- Professional Sidebar --------------------
with st.sidebar:
    st.markdown("### Dashboard Controls")
    st.markdown("---")
    st.markdown("#### Portfolio Data Management")
    uploaded_file = st.file_uploader(
        "Upload Portfolio CSV", type=['csv'],
        help="Required columns: Stock, Shares, Purchase_Price, Sector (e.g., RELIANCE.NS,100,2500,Energy)"
    )

    portfolio_df = None
    if uploaded_file:
        try:
            temp_df = pd.read_csv(uploaded_file)
            # Standardize Stock tickers to uppercase
            if 'Stock' in temp_df.columns:
                temp_df['Stock'] = temp_df['Stock'].astype(str).str.upper().str.strip()
            temp_df['Shares'] = pd.to_numeric(temp_df['Shares'], errors='coerce')
            temp_df['Purchase_Price'] = pd.to_numeric(temp_df['Purchase_Price'], errors='coerce')
            temp_df['Sector'] = temp_df['Sector'].astype(str).str.strip() # Ensure sector is string

            initial_rows = len(temp_df)
            temp_df.dropna(subset=['Stock', 'Shares', 'Purchase_Price', 'Sector'], inplace=True)
            if len(temp_df) < initial_rows:
                st.warning(f"Removed {initial_rows - len(temp_df)} row(s) due to missing or invalid data in uploaded CSV.")
            if temp_df.empty:
                st.error("Uploaded CSV is empty or invalid after cleaning. Using sample data.")
                portfolio_df = SAMPLE_PORTFOLIO_DATA
            else:
                portfolio_df = temp_df
                st.success("Portfolio loaded successfully from CSV.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}. Please ensure correct format (Stock, Shares, Purchase_Price, Sector). Using sample data.")
            portfolio_df = SAMPLE_PORTFOLIO_DATA
    else:
        st.info("Using sample portfolio. Upload your CSV for personalized analysis.")
        portfolio_df = SAMPLE_PORTFOLIO_DATA

    st.markdown("---")
    st.markdown("#### Portfolio Overview")
    if not portfolio_df.empty:
        total_stocks = len(portfolio_df)
        total_sectors = portfolio_df['Sector'].nunique()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Holdings", total_stocks)
        with col2:
            st.metric("Sectors", total_sectors)
    else:
        st.warning("No portfolio data available.")

    st.markdown("---")
    st.markdown("#### Custom Watchlist")
    custom_watchlist_input = st.text_area(
        "Enter tickers (one per line, e.g., RELIANCE.NS, TSLA). Max 10 tickers for performance.",
        value="RELIANCE.NS\nINFY.NS\nGOOGL\nMSFT", # Default watchlist for demonstration
        height=150,
        key="custom_watchlist_area"
    )
    # Parse the watchlist input and limit to 10 for performance
    custom_watchlist_tickers = [
        t.strip().upper() for t in custom_watchlist_input.split('\n') if t.strip()
    ][:10] # Limit to first 10 tickers
    if len(custom_watchlist_tickers) > 10: # Only show warning if user entered more than 10
        st.warning("Watchlist limited to first 10 tickers for optimal performance.")


    st.markdown("---")
    st.markdown("#### Cache Management")
    if st.button("Refresh All Market Data (Clear Cache)"):
        st.cache_data.clear()
        st.rerun() # Rerun to force data refetch
        st.success("Cached market data cleared. Data will be refetched on demand.")


# -------------------- Main Navigation Tabs --------------------
# Reverting to emoji icons for tab labels as Font Awesome HTML is not directly supported by st.tabs
live_portfolio_tab, holdings_tab, analysis_tab, charts_tab, ml_tab = st.tabs([
    " Live Market", # Original emoji
    " My Holdings", # Original emoji
    " Portfolio Analysis", # Original emoji
    " Stock Charts", # Original emoji
    " ML Prediction" # Original emoji
])

# -------------------- Live Portfolio Tab --------------------
with live_portfolio_tab:
    st.markdown(
        """<h2 class="main-title">Live Market Overview</h2>""",
        unsafe_allow_html=True
    )

    if custom_watchlist_tickers:
        st.subheader("Selected Equities Performance")
        
        # --- MODIFIED LAYOUT ---
        # Create as many columns as there are tickers to display them in a single horizontal row.
        cols = st.columns(len(custom_watchlist_tickers))
        
        for i, ticker in enumerate(custom_watchlist_tickers):
            with cols[i]:  # Use the i-th column for the i-th ticker
                with st.spinner(f"Fetching {ticker}..."):
                    df = fetch_stock_price(ticker, period="2d")  # Fetch 2 days to get prev close
                    if df is not None and not df.empty and len(df) >= 2:
                        latest_close = df['Close'].iloc[-1]
                        prev_close = df['Close'].iloc[-2]
                        delta = latest_close - prev_close
                        delta_pct = (delta / prev_close) * 100 if prev_close != 0 else 0
                        st.metric(
                            label=ticker.replace('.NS', ''),
                            value=f"₹{latest_close:,.2f}",
                            delta=f"{delta:+.2f} ({delta_pct:+.2f}%)"
                        )
                    else:
                        st.warning(f"Data N/A for {ticker}")
        # --- END OF MODIFICATION ---

    else:
        st.info("Add stocks to your custom watchlist in the sidebar to see their live performance here.")

    st.markdown("### Predefined Sector Performance (Nifty 50 Leaders)")
    # Show Nifty 50 Leaders by default if no custom watchlist, or in addition
    nifty_leaders = STOCKS_BY_SECTOR["Nifty 50 Leaders"]
    cols_per_row = 3
    num_cols = len(nifty_leaders)
    if num_cols > 0:
        for i in range(0, num_cols, cols_per_row):
            cols = st.columns(min(cols_per_row, num_cols - i))
            for j, ticker in enumerate(nifty_leaders[i : i + cols_per_row]):
                with cols[j]:
                    with st.spinner(f"Fetching {ticker}..."):
                        df = fetch_stock_price(ticker, period="2d")
                        if df is not None and not df.empty and len(df) >=2:
                            latest_close = df['Close'].iloc[-1]
                            prev_close = df['Close'].iloc[-2]
                            delta = latest_close - prev_close
                            delta_pct = (delta / prev_close) * 100 if prev_close != 0 else 0
                            st.metric(
                                label=ticker.replace('.NS', ''),
                                value=f"₹{latest_close:,.2f}",
                                delta=f"{delta:+.2f} ({delta_pct:+.2f}%)"
                            )
                        else:
                            st.warning(f"Data N/A for {ticker}")
    else:
        st.warning("No predefined Nifty 50 Leaders found.")


# -------------------- Holdings Tab --------------------
with holdings_tab:
    st.markdown(
        """<h2 class="main-title">Portfolio Holdings Management</h2>""",
        unsafe_allow_html=True
    )
    if not portfolio_df.empty:
        # Calculate totals dynamically. Use a map to cache current prices if fetching many times
        current_prices_cache = {}
        total_investment = 0
        total_current_value = 0
        
        # Using st.spinner for overall calculation
        with st.spinner("Calculating portfolio totals..."):
            for _, row in portfolio_df.iterrows():
                ticker = row['Stock']
                if ticker not in current_prices_cache:
                    current_data = fetch_stock_price(ticker, period="1d")
                    if current_data is not None and not current_data.empty:
                        current_prices_cache[ticker] = current_data['Close'].iloc[-1]
                    else:
                        current_prices_cache[ticker] = row['Purchase_Price'] # Fallback
                        # st.warning(f"Could not get live price for {ticker}. Using purchase price for portfolio value calculation.")

                invested_value = row['Purchase_Price'] * row['Shares']
                total_investment += invested_value
                total_current_value += current_prices_cache[ticker] * row['Shares']
                
        total_pnl = total_current_value - total_investment
        pnl_percentage = (total_pnl / total_investment) * 100 if total_investment > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Investment", f"₹{total_investment:,.0f}")
        with col2:
            st.metric("Current Value", f"₹{total_current_value:,.0f}")
        with col3:
            st.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{pnl_percentage:+.1f}%")
        with col4:
            st.metric("Total Holdings", f"{len(portfolio_df)} positions")

        st.markdown("### Detailed Holdings Analysis")
        enhanced_portfolio = portfolio_df.copy()
        data_rows = []
        # Re-iterate, using the cached prices for detailed view
        for _, row in enhanced_portfolio.iterrows():
            ticker = row['Stock']
            invested_value = row['Purchase_Price'] * row['Shares']
            
            current_price = current_prices_cache.get(ticker, row['Purchase_Price']) # Get from cache or fallback
            current_value = current_price * row['Shares']
            pnl = current_value - invested_value
            pnl_pct = (pnl / invested_value) * 100 if invested_value != 0 else 0.0
            
            if current_price == row['Purchase_Price'] and ticker not in current_prices_cache:
                 st.warning(f"Live data for {ticker} unavailable. P&L calculated based on purchase price.")
            
            data_rows.append([row['Stock'], row['Sector'], row['Shares'], row['Purchase_Price'], current_price, invested_value, current_value, pnl, pnl_pct])

        display_df = pd.DataFrame(data_rows, columns=['Stock', 'Sector', 'Shares', 'Purchase_Price', 'Current_Price', 'Invested_Value', 'Current_Value', 'P&L_Amount', 'P&L_%'])

        st.dataframe(
            display_df.style.format({
                'Purchase_Price': "₹{:,.2f}", 'Current_Price': "₹{:,.2f}",
                'Invested_Value': "₹{:,.0f}", 'Current_Value': "₹{:,.0f}",
                'P&L_Amount': "₹{:,.0f}", 'P&L_%': "{:+.1f}%"
            }),
            use_container_width=True, hide_index=True
        )

        st.markdown("### Performance Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Top Performers (by P&L %)")
            # Filter out NaNs if any P&L_% couldn't be calculated
            top_performers = display_df.dropna(subset=['P&L_%']).nlargest(3, 'P&L_%')
            if not top_performers.empty:
                for _, row in top_performers.iterrows():
                    st.success(f"**{row['Stock'].replace('.NS','')}**: {row['P&L_%']:+.1f}%")
            else:
                st.info("No top performers found.")
        with col2:
            st.markdown("#### Underperformers (by P&L %)")
            underperformers = display_df.dropna(subset=['P&L_%']).nsmallest(3, 'P&L_%')
            if not underperformers.empty:
                for _, row in underperformers.iterrows():
                    st.error(f"**{row['Stock'].replace('.NS','')}**: {row['P&L_%']:+.1f}%")
            else:
                st.info("No underperformers found.")
    else:
        st.info("Upload your portfolio CSV to view holdings. A sample portfolio is currently loaded.")

# -------------------- Analysis Tab --------------------
with analysis_tab:
    st.markdown(
        """<h2 class="main-title">Advanced Portfolio Analytics</h2>""",
        unsafe_allow_html=True
    )
    if not portfolio_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            pie_fig, updated_portfolio = create_professional_portfolio_analysis(portfolio_df.copy())
            st.plotly_chart(pie_fig, use_container_width=True)
        with col2:
            stock_fig = px.bar(
                updated_portfolio, x='Stock', y='Current_Value', color='Sector',
                title="Individual Stock Value Distribution",
                color_discrete_sequence=[COLORS['accent_primary'], COLORS['accent_secondary'], COLORS['success_color'], COLORS['warning_color'], '#8b5cf6']
            )
            stock_fig.update_layout(
                template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_primary'], family='Inter'),
                xaxis=dict(tickangle=45, title='Stock Symbol'), yaxis=dict(title='Portfolio Value (₹)'),
                title=dict(x=0.5, font=dict(size=16)), legend=dict(font=dict(color=COLORS['text_primary']))
            )
            st.plotly_chart(stock_fig, use_container_width=True)
    else:
        st.info("No portfolio data available for advanced analytics. Upload your CSV or use the sample data.")

# -------------------- Charts Tab --------------------
with charts_tab:
    st.markdown(
        """<h2 class="main-title">Professional Stock Analysis & Technicals</h2>""",
        unsafe_allow_html=True
    )

    st.markdown("### Select Stock for Chart Analysis")
    chart_choice = st.radio(
        "Choose stock input method:",
        ("Predefined List", "Enter Custom Ticker"),
        key="chart_input_method",
        horizontal=True
    )

    selected_stock_chart = None
    if chart_choice == "Predefined List":
        col1, col2 = st.columns(2)
        with col1:
            selected_sector = st.selectbox("Select Sector", list(STOCKS_BY_SECTOR.keys()), key="chart_sector_select")
        with col2:
            selected_stock_chart = st.selectbox("Select Stock", STOCKS_BY_SECTOR[selected_sector], key="chart_stock_select")
    else: # Enter Custom Ticker
        custom_ticker_input = st.text_input(
            "Enter Yahoo Finance Ticker Symbol (e.g., RELIANCE.NS, GOOGL, AAPL)",
            placeholder="RELIANCE.NS",
            key="custom_chart_ticker_input"
        ).upper() # Convert to uppercase for consistency

        if custom_ticker_input:
            selected_stock_chart = custom_ticker_input
        else:
            st.info("Please enter a stock ticker to view its chart.")

    if selected_stock_chart:
        # Fetch data for the selected stock for chart and technicals
        with st.spinner(f"Fetching data for {selected_stock_chart} and preparing chart..."):
            df_chart_data = fetch_stock_price(selected_stock_chart, period="1y")

        if df_chart_data is not None and not df_chart_data.empty:
            st.markdown(f"### Displaying Chart for **{selected_stock_chart.replace('.NS', '')}**")
            
            # Attempt to get prediction data for chart overlay
            # Note: ARIMA can be slow, so give user feedback
            with st.spinner(f"Generating ARIMA prediction for {selected_stock_chart} chart overlay..."):
                _, _, prediction_plot_df = train_and_predict_stock_price(selected_stock_chart, 1, 120)
            
            if isinstance(prediction_plot_df, pd.DataFrame):
                plot_professional_stock_chart(selected_stock_chart, prediction_plot_df)
            else:
                # If prediction fails, plot without prediction line
                plot_professional_stock_chart(selected_stock_chart, None)
                if isinstance(prediction_plot_df, str): # Check if it's an error message string from ARIMA
                     st.warning(f"ARIMA prediction for chart overlay not available: {prediction_plot_df}")

            st.markdown("### Technical Analysis Indicators")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(df_chart_data) >= 5: # Need at least 5 data points for 5-day SMA
                    sma_5 = df_chart_data['Close'].rolling(window=5, min_periods=1).mean().iloc[-1]
                    st.metric("5-Day Moving Average", f"₹{sma_5:.2f}")
                else:
                    st.metric("5-Day Moving Average", "N/A (requires 5+ days data)")
            with col2:
                # Ensure enough data for percentage change (at least 2 points)
                if len(df_chart_data) > 1:
                    volatility = df_chart_data['Close'].pct_change().std() * 100
                    st.metric("Price Volatility", f"{volatility:.2f}%")
                else:
                    st.metric("Price Volatility", "N/A (requires 2+ days data)")
            with col3:
                if 'Volume' in df_chart_data.columns and not df_chart_data['Volume'].empty and len(df_chart_data) > 0:
                    volume_avg = df_chart_data['Volume'].mean()
                    st.metric("Average Volume", f"{volume_avg / 1000000:.1f}M")
                else:
                    st.metric("Average Volume", "N/A")
        else:
            st.error(f"Could not retrieve market data for **{selected_stock_chart}**. Please check the ticker symbol and try again.")
    else:
        st.info("Please select or enter a stock ticker to view its chart and technicals.")


# -------------------- ML Prediction Tab --------------------
with ml_tab:
    st.markdown(
        """<h2 class="main-title">ARIMA Model Stock Price Projections</h2>""",
        unsafe_allow_html=True
    )
    st.info("This model provides predictive insights into future closing prices using historical data. Please note that accuracy can vary, and these predictions do not constitute financial advice.")

    ml_choice = st.radio(
        "Choose stock input method:",
        ("Predefined List", "Enter Custom Ticker"),
        key="ml_input_method",
        horizontal=True
    )

    selected_ml_stock = None
    if ml_choice == "Predefined List":
        col1, col2 = st.columns(2)
        with col1:
            selected_ml_sector = st.selectbox("Select Sector for ML", list(STOCKS_BY_SECTOR.keys()), key="ml_sector_select")
        with col2:
            selected_ml_stock = st.selectbox("Select Stock for Prediction", STOCKS_BY_SECTOR[selected_ml_sector], key="ml_stock_select")
    else: # Enter Custom Ticker
        custom_ml_ticker_input = st.text_input(
            "Enter Yahoo Finance Ticker Symbol (e.g., RELIANCE.NS, GOOGL, AAPL)",
            placeholder="INFY.NS",
            key="custom_ml_ticker_input"
        ).upper()

        if custom_ml_ticker_input:
            selected_ml_stock = custom_ml_ticker_input
        else:
            st.info("Please enter a stock ticker to generate predictions.")

    prediction_horizon = st.slider("Prediction Horizon (Trading Days)", 1, 5, 1, key="ml_prediction_horizon")
    lookback_period = st.slider("Lookback Period for Training (Days)", 60, 252, 120, key="ml_lookback_period") # Max 1 year of trading days

    if st.button("Generate Prediction", key="ml_predict_button") and selected_ml_stock:
        with st.spinner(f"Training ARIMA model for {selected_ml_stock} with {lookback_period} days data and predicting {prediction_horizon} days..."):
            next_day_pred, metrics, prediction_plot_df = train_and_predict_stock_price(selected_ml_stock, prediction_horizon, lookback_period)

        st.markdown(f"### Prediction for {selected_ml_stock.replace('.NS', '')}")
        if next_day_pred is not None:
            current_price_data = fetch_stock_price(selected_ml_stock, period="1d")
            current_close = current_price_data['Close'].iloc[-1] if current_price_data is not None and not current_price_data.empty else None
            delta_str = "N/A"
            if current_close:
                # Compare next day prediction to current actual close
                pred_delta = next_day_pred - current_close
                pred_delta_pct = (pred_delta / current_close) * 100 if current_close != 0 else 0
                delta_str = f"₹{pred_delta:+.2f} ({pred_delta_pct:+.2f}%)"

            st.subheader(f"Forecast for the next {prediction_horizon} Trading Day(s)")
            if isinstance(prediction_plot_df, pd.DataFrame) and 'Predicted_Close' in prediction_plot_df.columns:
                # Filter for only future predictions for the table display
                # CORRECTED LINE: Use .date() to get just the date part for comparison
                future_preds_display = prediction_plot_df[prediction_plot_df.index.date > datetime.now().date()].copy()
                if not future_preds_display.empty:
                    # Prepare for display, including date and value, and CI if available
                    display_cols = ['Predicted_Close']
                    if 'lower_ci' in future_preds_display.columns and 'upper_ci' in future_preds_display.columns:
                        future_preds_display['Confidence Interval (95%)'] = future_preds_display.apply(
                            lambda row: f"[{row['lower_ci']:.2f}, {row['upper_ci']:.2f}]", axis=1
                        )
                        display_cols.append('Confidence Interval (95%)')

                    st.dataframe(
                        future_preds_display[display_cols].style.format({"Predicted_Close": "₹{:,.2f}"}),
                        use_container_width=True,
                        hide_index=False # Show date index
                    )
                    # Also show the most immediate next day's prediction in a metric
                    first_future_date = future_preds_display.index[0].strftime('%B %d, %Y')
                    first_future_pred_value = future_preds_display['Predicted_Close'].iloc[0]
                    st.metric(label=f"Predicted Close on {first_future_date}", value=f"₹{first_future_pred_value:,.2f}", delta=delta_str)
                else:
                    st.info("No future predictions available for display (check prediction horizon or data availability).")
            else:
                # Fallback metric if prediction_plot_df isn't a DataFrame
                st.metric(label=f"Predicted Close on Next Trading Day", value=f"₹{next_day_pred:,.2f}", delta=delta_str)

            st.markdown("### Model Performance Metrics (on historical test data)")
            # Check if metrics are actually calculated (not NaN for RMSE/R2)
            if metrics and not (pd.isna(metrics['RMSE']) or pd.isna(metrics['R2 Score'])):
                col_rmse, col_r2, col_order = st.columns(3)
                col_rmse.metric("RMSE", f"{metrics['RMSE']:.2f}")
                col_r2.metric("R² Score", f"{metrics['R2 Score']:.2f}")
                col_order.metric("ARIMA Order", f"{metrics['ARIMA Order']}")
                st.markdown(f"<small style='color: var(--text-muted);'>RMSE measures the average magnitude of the errors in the test set. R² indicates how well the model explains the variability of the test data (1.0 is a perfect fit).</small>", unsafe_allow_html=True)
            else:
                st.warning("Performance metrics are N/A, likely due to insufficient test data for evaluation or model failure.")

        else:
            st.error(f"Could not generate prediction for **{selected_ml_stock}**. The stock may lack sufficient historical data or the model failed to train: {prediction_plot_df}") # prediction_plot_df here contains the error message

    elif not selected_ml_stock:
        st.info("Enter a ticker or select from the list to generate a prediction.")

# -------------------- Professional Footer --------------------
st.markdown("---")
st.markdown(f"""
<div style="
    text-align: center;
    padding: 2.5rem 0;
    color: var(--text-muted);
    background: linear-gradient(135deg, {COLORS['accent_primary']}0D, {COLORS['accent_secondary']}0D);
    border-radius: 16px;
    margin-top: 2rem;
">
    <h4 style="
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif; /* Applied Playfair Display here */
    ">FinInsight Hub</h4>
    <p style="font-size: 1.0rem; margin-bottom: 0.5rem; font-weight: 500;">
        Comprehensive Investment Insights at Your Fingertips
    </p>
    <p style="font-size: 0.9rem; color: var(--text-muted);">
        © {datetime.now().year} FinInsight Hub. All Rights Reserved.
    </p>
</div>
""", unsafe_allow_html=True)