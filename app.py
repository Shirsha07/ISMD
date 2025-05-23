import streamlit as st
import yfinance as yf
import pandas as pd
import ta  # For technical analysis indicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Helper Functions ---

def get_nifty200_tickers():
    return [
        "ADANIENSOL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "ALKEM.NS",
        "AMBUJACEM.NS", "APLLTD.NS", "APOLLOHOSP.NS", "AUROPHARMA.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
        "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHARTIARTL.NS",
        "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "CANBK.NS",
        "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COLPAL.NS", "CONCOR.NS",
        "CROMPTON.NS", "DABUR.NS", "DALBHARAT.NS", "DIVISLAB.NS", "DLF.NS",
        "DMART.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS", "GLAND.NS",
        "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAVELLS.NS",
        "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
        "HINDALCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS",
        "ICICIPRULI.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IGL.NS", "INDIGO.NS",
        "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IRCTC.NS", "ITC.NS",
        "JINDALSTEL.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", "KANSAINER.NS", "KOTAKBANK.NS",
        "L&TFH.NS", "LALPATHLAB.NS", "LICHSGFIN.NS", "LT.NS", "LTIM.NS", "LUPIN.NS",
        "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS",
        "MCX.NS", "METROBRAND.NS", "MGL.NS", "MINDTREE.NS", "MPL.NS", "MRF.NS",
        "MUTHOOTFIN.NS", "NAM-INDIA.NS", "NATCOPHARM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS",
        "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS",
        "ONGC.NS", "PAGEIND.NS", "PEL.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS",
        "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "PRESTIGE.NS", "PTC.NS", "RAMCOCEM.NS",
        "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
        "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SRTRANSFIN.NS", "STARHEALTH.NS",
        "SUNPHARMA.NS", "SUNTV.NS", "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACONSUM.NS",
        "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
        "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS",
        "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "WIPRO.NS", "YESBANK.NS",
        "ZYDUSLIFE.NS"
    ]

def fetch_data(tickers, period="3mo", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=True, threads=True)
    result = {}
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                result[ticker] = data[ticker]
            else:
                result[ticker] = data
        except KeyError:
            print(f"Data for {ticker} not found or could not be fetched.")
    return result
def process_nifty_data(nifty200_data):
    processed_data = {}
    for ticker, data_item in nifty200_data.items():
        if isinstance(data_item, pd.DataFrame) and not data_item.empty:
            processed_data[ticker] = data_item
        else:
            print(f"Warning: No valid DataFrame for ticker {ticker}")
            processed_data[ticker] = pd.DataFrame()
    return processed_data

nifty200_tickers = get_nifty200_tickers()
nifty200_data = fetch_data(nifty200_tickers, period="3mo", interval="1d") # Adjust period as needed
nifty200_data = process_nifty_data(nifty200_data)

print(f"Type of nifty200_data before the if statement: {type(nifty200_data)}")
if isinstance(nifty200_data, dict):
    print("nifty200_data is a dictionary.")
    for key, value in nifty200_data.items():
        print(f"  Key: {key}, Type: {type(value)}")
elif isinstance(nifty200_data, pd.DataFrame):
    print("nifty200_data is a DataFrame.")
else:
    print(f"nifty200_data is of an unexpected type: {type(nifty200_data)}")

if bool(nifty200_data): # Explicitly check if the dictionary is non-empty
    st.subheader("Nifty 200 Overview")
    # ... rest of your code
else:
    st.error("Failed to fetch data for Nifty 200 stocks.")
print(f"Type of nifty200_data before the if statement: {type(nifty200_data)}")
if isinstance(nifty200_data, dict):
    print("nifty200_data is a dictionary.")
    for key, value in nifty200_data.items():
        print(f"  Key: {key}, Type: {type(value)}")
elif isinstance(nifty200_data, pd.DataFrame):
    print("nifty200_data is a DataFrame.")
else:
    print(f"nifty200_data is of an unexpected type: {type(nifty200_data)}")

if bool(nifty200_data): # Explicitly check if the dictionary is non-empty
    st.subheader("Nifty 200 Overview")
    # ... rest of your code
else:
    st.error("Failed to fetch data for Nifty 200 stocks.")
print(f"Type of nifty200_data before the if statement: {type(nifty200_data)}")
if isinstance(nifty200_data, dict):
    print("nifty200_data is a dictionary.")
    for key, value in nifty200_data.items():
        print(f"  Key: {key}, Type: {type(value)}")
elif isinstance(nifty200_data, pd.DataFrame):
    print("nifty200_data is a DataFrame.")
else:
    print(f"nifty200_data is of an unexpected type: {type(nifty200_data)}")

if bool(nifty200_data): # Explicitly check if the dictionary is non-empty
    st.subheader("Nifty 200 Overview")
    # ... rest of your code
else:
    st.error("Failed to fetch data for Nifty 200 stocks.")


def calculate_indicators(df):
    if df.empty or 'Close' not in df.columns:
        return df
    try:
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
    except Exception as e:
        print(f"Error calculating indicators: {e}")
    return df

def check_upper_band_touch(df):
    if df.empty or 'Close' not in df.columns or 'BB_upper' not in df.columns or df['Close'].isnull().any() or df['BB_upper'].isnull().any():
        return False
    return df['Close'].iloc[-1] >= df['BB_upper'].iloc[-1]

def get_top_gainers_losers(data):
    latest_data = {}
    previous_data = {}
    for ticker, item in data.items():
        if isinstance(item, pd.DataFrame) and not item.empty and 'Close' in item.columns and len(item) >= 2:
            close_series = item['Close'].dropna()
            if len(close_series) >= 2:
                latest_data[ticker] = close_series.iloc[-1]
                previous_data[ticker] = close_series.iloc[-2]

    if not latest_data or not previous_data:
        return pd.DataFrame(), pd.DataFrame()

    gainers_data = []
    losers_data = []

    for ticker, latest_price in latest_data.items():
        if ticker in previous_data and previous_data[ticker] is not None and previous_data[ticker] != 0 and latest_price is not None:
            change_percentage = ((latest_price - previous_data[ticker]) / previous_data[ticker]) * 100
            gainers_data.append({'Ticker': ticker, 'Change (%)': change_percentage})
            losers_data.append({'Ticker': ticker, 'Change (%)': change_percentage})

    gainers_df = pd.DataFrame(gainers_data).sort_values(by='Change (%)', ascending=False).head(10)
    losers_df = pd.DataFrame(losers_data).sort_values(by='Change (%)', ascending=True).head(10)

    return gainers_df, losers_df

def plot_candlestick(df, ticker):
    if df.empty or 'Open' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        return go.Figure()
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_indicators(df, ticker):
    if df.empty or 'Close' not in df.columns or 'MACD' not in df.columns:
        return make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
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
    if isinstance(stock_data, pd.DataFrame) and not stock_data.empty and 'Close' in stock_data.columns:
        st.subheader(f"Security: {ticker.split('.')[0]}") # Display without '.NS'
        latest_info = stock_data.iloc[-1]
        previous_info = stock_data.iloc[-2] if len(stock_data) > 1 else None
        change = latest_info['Close'] - previous_info['Close'] if previous_info is not None else 0
        change_percent = (change / previous_info['Close']) * 100 if previous_info is not None and previous_info['Close'] != 0 else 0
        st.metric("Latest Price", f"{latest_info['Close'].iloc[-1]:.2f} INR", f"{change:.2f} ({change_percent:.2f}%)" if previous_info is not None else None)

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
        st.error(f"Could not retrieve valid data for {ticker}")

# --- Main Streamlit App ---

st.title("Interactive Nifty 200 Stock Market Dashboard")

nifty200_tickers = get_nifty200_tickers()
nifty200_data = fetch_data(nifty200_tickers, period="3mo", interval="1d") # Adjust period as needed

# Explicitly convert each item in nifty200_data to a DataFrame
for ticker, data_item in nifty200_data.items():
    if isinstance(data_item, pd.Series):
        nifty200_data[ticker] = pd.DataFrame(data_item)
    elif not isinstance(data_item, pd.DataFrame) and data_item is not None:
        try:
            nifty200_data[ticker] = pd.DataFrame(data_item)
        except Exception as e:
            print(f"Error converting data for {ticker}: {e}")
            nifty200_data[ticker] = pd.DataFrame() # Set to empty DataFrame on error
    elif data_item is None:
        nifty200_data[ticker] = pd.DataFrame() # Set to empty DataFrame if None

print(f"Type of nifty200_data before the if statement: {type(nifty200_data)}")
if isinstance(nifty200_data, dict):
    print("nifty200_data is a dictionary.")
    for key, value in nifty200_data.items():
        print(f"  Key: {key}, Type: {type(value)}")
elif isinstance(nifty200_data, pd.DataFrame):
    print("nifty200_data is a DataFrame.")
else:
    print(f"nifty200_data is of an unexpected type: {type(nifty200_data)}")

if bool(nifty200_data): # Explicitly check if the dictionary is non-empty
    st.subheader("Nifty 200 Overview")
    nifty_index_data = yf.download("^NSEI", period="1d", interval="1d")
    if isinstance(nifty_index_data, pd.DataFrame) and not nifty_index_data.empty and 'Close' in nifty_index_data.columns:
        latest_nifty = nifty_index_data['Close'].iloc[-1]
        previous_nifty = nifty_index_data['Close'].iloc[-2] if len(nifty_index_data) > 1 else None

        change_nifty = (latest_nifty - previous_nifty) if previous_nifty is not None else 0
        change_percent_nifty = ((change_nifty / previous_nifty) * 100) if previous_nifty is not None and previous_nifty != 0 else 0

        latest_nifty_value = latest_nifty if isinstance(latest_nifty, (int, float)) else latest_nifty.iloc[0] if not nifty_index_data['Close'].empty else None
        change_nifty_value = change_nifty if isinstance(change_nifty, (int, float)) else change_nifty.iloc[0] if previous_nifty is not None and not nifty_index_data['Close'].empty else None
        change_percent_nifty_value = change_percent_nifty if isinstance(change_percent_nifty, (int, float)) else change_percent_nifty.iloc[0] if previous_nifty is not None and previous_nifty != 0 and not nifty_index_data['Close'].empty else None

        if latest_nifty_value is not None:
            st.metric(
                "Nifty 200",
                f"{latest_nifty_value:.2f}",
                f"{change_nifty_value:.2f} ({change_percent_nifty_value:.2f}%)" if previous_nifty is not None and latest_nifty_value is not None else None,
            )
        else:
            st.warning("Could not display Nifty 200 overview.")
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
        if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df.columns and 'MACD' in df.columns and 'RSI' in df.columns and 'EMA_20' in df.columns and 'SMA_20' in df.columns and 'BB_upper' in df.columns:
            indicators = calculate_indicators(df.copy())
            if not indicators.empty and len(indicators) > 0 and \
               indicators['MACD'].iloc[-1] > 0 and \
               indicators['RSI'].iloc[-1] > 50 and \
               check_upper_band_touch(df.copy()) and \
               (
                   'EMA_20' in indicators.columns and
                   'SMA_20' in indicators.columns and
                   indicators['EMA_20'].iloc[-1] > indicators['SMA_20'].iloc[-1]
               ):
                meeting_criteria_stocks.append(ticker)
    if meeting_criteria_stocks:
        st.write("Stocks meeting the criteria (MACD > 0, RSI > 50, touching upper Bollinger Band, EMA > SMA 20):")
        st.write(meeting_criteria_stocks)
    else:
        st.info("No stocks currently meet all the specified criteria based on the last available data.")

    st.sidebar.header("Security Analysis")
    selected_ticker = st.sidebar.selectbox("Choose a Nifty 200 Stock", [""] + [t.split('.')[0] for t in nifty200_tickers]) # Display without '.NS'

    if selected_ticker:
        display_security_info(selected_ticker + ".NS")

else:
    st.error("Failed to fetch data for Nifty 200 stocks.")
