import streamlit as st
import yfinance as yf
import pandas as pd
import ta  # For technical analysis indicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Helper Functions ---

def get_nifty200_tickers():
    # **IMPORTANT:** Replace this with your actual method of fetching Nifty 200 tickers
    # This is a placeholder list.
    return [
        "ADANIENT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
        "BHARTIARTL.NS", "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
        "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS",
        "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS",
        "SHREECEM.NS", "SUNPHARMA.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
        "ULTRACEMCO.NS", "WIPRO.NS" # Add the rest of the Nifty 200 tickers
    ]

def fetch_data(tickers, period="1y", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, group_by="ticker")
    return data

def calculate_indicators(df):
    if df.empty:
        return df
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    return df

def check_upper_band_touch(df):
    if df.empty or df.iloc[-1]['Close'] is None or df.iloc[-1]['BB_upper'] is None:
        return False
    return df.iloc[-1]['Close'] >= df.iloc[-1]['BB_upper']

def get_top_gainers_losers(data):
    latest_data = {ticker: df.iloc[-1]['Close'] for ticker, df in data.items() if not df.empty and 'Close' in df.columns and not df.iloc[-1].isnull().any()}
    previous_data = {ticker: df.iloc[-2]['Close'] for ticker, df in data.items() if len(df) >= 2 and 'Close' in df.columns and not df.iloc[-2].isnull().any()}

    if not latest_data or not previous_data:
        return pd.DataFrame(), pd.DataFrame()

    gainers_data = []
    losers_data = []

    for ticker, latest_price in latest_data.items():
        if ticker in previous_data and previous_data[ticker] != 0:
            change_percentage = ((latest_price - previous_data[ticker]) / previous_data[ticker]) * 100
            gainers_data.append({'Ticker': ticker, 'Change (%)': change_percentage})
            losers_data.append({'Ticker': ticker, 'Change (%)': change_percentage})

    gainers_df = pd.DataFrame(gainers_data).sort_values(by='Change (%)', ascending=False).head(10)
    losers_df = pd.DataFrame(losers_data).sort_values(by='Change (%)', ascending=True).head(10)

    return gainers_df, losers_df

def plot_candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_indicators(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[0] * len(df), line=dict(color='black', width=0.5), name='Zero Line (MACD)'), row=2, col=1)

    fig.update_layout(title=f'{ticker} Price and MACD', xaxis_title='Date', yaxis_title='Price (Top), MACD (Bottom)')
    return fig

def display_security_info(ticker):
    stock_data = yf.download(ticker, period="1y", interval="1d")
    if not stock_data.empty:
        st.subheader(f"Security: {ticker.split('.')[0]}") # Display without '.NS'
        latest_info = stock_data.iloc[-1]
        st.metric("Latest Price", f"{latest_info['Close']:.2f} INR", f"{latest_info['Close'] - stock_data.iloc[-2]['Close']:.2f} ({((latest_info['Close'] - stock_data.iloc[-2]['Close']) / stock_data.iloc[-2]['Close']) * 100:.2f}%)" if len(stock_data) > 1 else None)

        col1, col2, col3 = st.columns(3)
        col1.metric("High", f"{latest_info['High']:.2f} INR")
        col2.metric("Low", f"{latest_info['Low']:.2f} INR")
        col3.metric("Volume", f"{int(latest_info['Volume']):,}")

        st.subheader("Candlestick Chart")
        st.plotly_chart(plot_candlestick(stock_data, ticker), use_container_width=True)

        st.subheader("Technical Indicators (Price & MACD)")
        indicators_df = calculate_indicators(stock_data.copy())
        st.plotly_chart(plot_indicators(indicators_df, ticker), use_container_width=True)

        st.subheader("Historical Data")
        st.dataframe(stock_data.tail(10)) # Show last 10 days of data
    else:
        st.error(f"Could not retrieve data for {ticker}")

# --- Main Streamlit App ---

st.title("Interactive Nifty 200 Stock Market Dashboard")

nifty200_tickers = get_nifty200_tickers()
nifty200_data = fetch_data(nifty200_tickers, period="3mo", interval="1d") # Adjust period as needed

if nifty200_data:
    st.subheader("Nifty 200 Overview")
    # **TODO:** Implement display of key Nifty 200 index information here
    # You might need to fetch data for the ^NSEI ticker for the index itself.
    nifty_index_data = yf.download("^NSEI", period="1d", interval="1d")
    if not nifty_index_data.empty:
        st.metric("Nifty 200", f"{nifty_index_data['Close'].iloc[-1]:.2f}", f"{nifty_index_data['Close'].iloc[-1] - nifty_index_data['Close'].iloc[-2]:.2f} ({((nifty_index_data['Close'].iloc[-1] - nifty_index_data['Close'].iloc[-2]) / nifty_index_data['Close'].iloc[-2]) * 100:.2f}%)" if len(nifty_index_data) > 1 else None)
    else:
        st.warning("Could not fetch Nifty 200 index data.")

    gainers_df, losers_df = get_top_gainers_losers(nifty200_data)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Gainers")
        st.dataframe(gainers_df)

    with col2:
        st.subheader("Top 10 Losers")
        st.dataframe(losers_df)

    st.subheader("Nifty 200 Stocks Meeting Criteria")
    meeting_criteria_stocks = []
    for ticker, df in nifty200_data.items():
        if not df.empty:
            indicators = calculate_indicators(df.copy())
            if not indicators.empty and indicators.iloc[-1]['MACD'] > 0 and indicators.iloc[-1]['RSI'] > 50 and check_upper_band_touch(df.copy()) and indicators.iloc[-1]['EMA_20'] > indicators.iloc[-1]['SMA_20']: # Using SMA_20 as a proxy
                meeting_criteria_stocks.append(ticker)

    if meeting_criteria_stocks:
        st.write("Stocks meeting the criteria (MACD > 0, RSI > 50, touching upper Bollinger Band, EMA > 20):")
        st.write(meeting_criteria_stocks)
    else:
        st.info("No stocks currently meet all the specified criteria.")

    st.sidebar.header("Security Analysis")
    selected_ticker = st.sidebar.selectbox("Choose a Nifty 200 Stock", [""] + [t.split('.')[0] for t in nifty200_tickers]) # Display without '.NS'

    if selected_ticker:
        display_security_info(selected_ticker + ".NS")

else:
    st.error("Failed to fetch data for Nifty 200 stocks.")
