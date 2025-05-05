import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Helper Functions ---

def get_nifty200_tickers():
    return [ "ADANIENSOL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "ALKEM.NS",
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
        "ZYDUSLIFE.NS" ]

def fetch_data(tickers, period="3mo", interval="1d"):
    return yf.download(tickers, period=period, interval=interval, group_by="ticker")

def calculate_indicators(df):
    if df.empty or 'Close' not in df.columns:
        return df
    try:
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
    except Exception as e:
        st.warning(f"Indicator Error: {e}")
    return df

def get_top_gainers_losers(data):
    latest_data, previous_data = {}, {}
    for ticker, item in data.items():
        if isinstance(item, pd.DataFrame) and 'Close' in item.columns and len(item) >= 2:
            latest_data[ticker] = item['Close'].iloc[-1]
            previous_data[ticker] = item['Close'].iloc[-2]

    gainers = [{'Ticker': t, 'Change (%)': ((latest_data[t] - previous_data[t]) / previous_data[t]) * 100}
               for t in latest_data if previous_data[t] != 0]
    gainers_df = pd.DataFrame(gainers).sort_values("Change (%)", ascending=False)
    losers_df = gainers_df.sort_values("Change (%)", ascending=True)

    return gainers_df.head(10), losers_df.head(10)

def plot_indicators(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=2, col=1)
    fig.update_layout(title=f"{ticker} Candlestick + MACD", xaxis_title="Date")
    return fig

def display_security_info(ticker):
    st.subheader(f"Security: {ticker.replace('.NS', '')}")
    stock_data = yf.download(ticker, period="1y", interval="1d")
    if stock_data.empty:
        st.warning("No data found.")
        return

    latest = stock_data.iloc[-1]
    previous = stock_data.iloc[-2] if len(stock_data) > 1 else None

    change = latest['Close'] - previous['Close'] if previous is not None else 0
    change_percent = (change / previous['Close']) * 100 if previous is not None else 0
    st.metric("Latest Price", f"{latest['Close']:.2f} INR", f"{change:.2f} ({change_percent:.2f}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{latest['High']:.2f}")
    col2.metric("Low", f"{latest['Low']:.2f}")
    col3.metric("Volume", f"{int(latest['Volume']):,}")

    df = calculate_indicators(stock_data)
    fig = plot_indicators(df, ticker)
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit App ---

st.set_page_config(page_title="Nifty 200 Dashboard", layout="wide")
st.title("📊 Nifty 200 Stock Market Dashboard")

tickers = get_nifty200_tickers()

option = st.selectbox("Select a Nifty 200 Stock", tickers)

if option:
    display_security_info(option)

st.markdown("---")
st.subheader("📈 Top Gainers & Losers")

with st.spinner("Fetching data..."):
    full_data = {ticker: yf.download(ticker, period="5d", interval="1d") for ticker in tickers[:50]}
    gainers, losers = get_top_gainers_losers(full_data)

    col1, col2 = st.columns(2)
    with col1:
        st.write("📈 Top Gainers")
        st.dataframe(gainers)
    with col2:
        st.write("📉 Top Losers")
        st.dataframe(losers)

