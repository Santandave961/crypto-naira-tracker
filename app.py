import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Crypto vs Naira Tracker", layout="wide")


# ── Generate synthetic historical data ───────────────────────────────────────
@st.cache_resource
def generate_data():
    np.random.seed(42)
    days = 365
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(days)]

    # Naira/USD rate (gradual depreciation from ~1500 to ~1600)
    naira_usd = 1500 + np.cumsum(np.random.normal(0.3, 2.5, days))
    naira_usd = np.clip(naira_usd, 1400, 1700)

    # BTC in USD (volatile, trending up)
    btc_usd = 42000 + np.cumsum(np.random.normal(10, 800, days))
    btc_usd = np.clip(btc_usd, 30000, 75000)

    # ETH in USD
    eth_usd = 2200 + np.cumsum(np.random.normal(2, 60, days))
    eth_usd = np.clip(eth_usd, 1500, 4000)

    # BNB in USD
    bnb_usd = 300 + np.cumsum(np.random.normal(0.5, 8, days))
    bnb_usd = np.clip(bnb_usd, 200, 600)

    # USDT (stable, slight peg drift)
    usdt_usd = 1 + np.random.normal(0, 0.002, days)
    usdt_usd = np.clip(usdt_usd, 0.995, 1.005)

    df = pd.DataFrame({
        "date":      dates,
        "naira_usd": naira_usd.round(2),
        "btc_usd":   btc_usd.round(2),
        "eth_usd":   eth_usd.round(2),
        "bnb_usd":   bnb_usd.round(2),
        "usdt_usd":  usdt_usd.round(4),
    })

    # Convert crypto prices to NGN
    df["btc_ngn"]  = (df["btc_usd"]  * df["naira_usd"]).round(2)
    df["eth_ngn"]  = (df["eth_usd"]  * df["naira_usd"]).round(2)
    df["bnb_ngn"]  = (df["bnb_usd"]  * df["naira_usd"]).round(2)
    df["usdt_ngn"] = (df["usdt_usd"] * df["naira_usd"]).round(2)

    # Returns
    for col in ["btc_usd","eth_usd","bnb_usd","naira_usd"]:
        df[col+"_return"] = df[col].pct_change() * 100

    df["date"] = pd.to_datetime(df["date"])
    return df


df = generate_data()


# ── Forecasting function (Linear Regression on time index) ───────────────────
def forecast_price(series, forecast_days=30):
    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    future_X   = np.arange(n, n + forecast_days).reshape(-1, 1)
    future_y   = model.predict(future_X)
    train_pred = model.predict(X)
    mae  = mean_absolute_error(y, train_pred)
    rmse = np.sqrt(mean_squared_error(y, train_pred))
    r2   = r2_score(y, train_pred)
    return future_y, mae, rmse, r2


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Crypto vs Naira Tracker")
st.caption("BTC | ETH | BNB | USDT — Performance vs Nigerian Naira")
st.markdown("Track how major cryptocurrencies perform against the Nigerian Naira with trend analysis and 30-day price forecasting.")
st.divider()

# ── Latest Prices KPIs ────────────────────────────────────────────────────────
st.subheader("Latest Prices")
latest = df.iloc[-1]
prev   = df.iloc[-2]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("USD/NGN",  f"N{latest['naira_usd']:,.0f}",
          f"{latest['naira_usd']-prev['naira_usd']:+.1f}")
k2.metric("BTC/NGN",  f"N{latest['btc_ngn']/1e6:.2f}M",
          f"{(latest['btc_usd']-prev['btc_usd'])/prev['btc_usd']*100:+.2f}%")
k3.metric("ETH/NGN",  f"N{latest['eth_ngn']/1e6:.2f}M",
          f"{(latest['eth_usd']-prev['eth_usd'])/prev['eth_usd']*100:+.2f}%")
k4.metric("BNB/NGN",  f"N{latest['bnb_ngn']:,.0f}",
          f"{(latest['bnb_usd']-prev['bnb_usd'])/prev['bnb_usd']*100:+.2f}%")
k5.metric("USDT/NGN", f"N{latest['usdt_ngn']:,.1f}",
          f"{(latest['usdt_usd']-prev['usdt_usd'])/prev['usdt_usd']*100:+.4f}%")

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
st.subheader("Time Range")
period = st.select_slider(
    "Select Period",
    options=["1 Month","3 Months","6 Months","1 Year"],
    value="6 Months"
)
period_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
fdf = df.tail(period_map[period]).copy()

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("Price Charts")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Naira/USD", "BTC", "ETH & BNB", "Comparison", "Correlation"
])

with tab1:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(fdf["date"], fdf["naira_usd"], color="#e74c3c", linewidth=2)
    ax.fill_between(fdf["date"], fdf["naira_usd"], alpha=0.15, color="#e74c3c")
    ax.set_ylabel("NGN per USD")
    ax.set_title("USD/NGN Exchange Rate")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Naira depreciation stats
    dep = ((fdf["naira_usd"].iloc[-1] - fdf["naira_usd"].iloc[0]) / fdf["naira_usd"].iloc[0]) * 100
    s1, s2, s3 = st.columns(3)
    s1.metric("Period Start",       f"N{fdf['naira_usd'].iloc[0]:,.0f}")
    s2.metric("Period End",         f"N{fdf['naira_usd'].iloc[-1]:,.0f}")
    s3.metric("Naira Depreciation", f"{dep:+.2f}%")

with tab2:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))

    ax1.plot(fdf["date"], fdf["btc_usd"]/1000, color="#f39c12", linewidth=2)
    ax1.fill_between(fdf["date"], fdf["btc_usd"]/1000, alpha=0.15, color="#f39c12")
    ax1.set_ylabel("Price (USD Thousands)")
    ax1.set_title("BTC Price in USD")
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2.plot(fdf["date"], fdf["btc_ngn"]/1e6, color="#e67e22", linewidth=2)
    ax2.fill_between(fdf["date"], fdf["btc_ngn"]/1e6, alpha=0.15, color="#e67e22")
    ax2.set_ylabel("Price (NGN Millions)")
    ax2.set_title("BTC Price in NGN")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    btc_ret = ((fdf["btc_usd"].iloc[-1] - fdf["btc_usd"].iloc[0]) / fdf["btc_usd"].iloc[0]) * 100
    b1, b2, b3 = st.columns(3)
    b1.metric("BTC Start (USD)", f"${fdf['btc_usd'].iloc[0]:,.0f}")
    b2.metric("BTC End (USD)",   f"${fdf['btc_usd'].iloc[-1]:,.0f}")
    b3.metric("BTC Return",      f"{btc_ret:+.2f}%")

with tab3:
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    axes[0,0].plot(fdf["date"], fdf["eth_usd"], color="#3498db", linewidth=1.5)
    axes[0,0].set_title("ETH/USD"); axes[0,0].set_ylabel("USD")

    axes[0,1].plot(fdf["date"], fdf["eth_ngn"]/1e6, color="#2980b9", linewidth=1.5)
    axes[0,1].set_title("ETH/NGN"); axes[0,1].set_ylabel("NGN Millions")

    axes[1,0].plot(fdf["date"], fdf["bnb_usd"], color="#2ecc71", linewidth=1.5)
    axes[1,0].set_title("BNB/USD"); axes[1,0].set_ylabel("USD")

    axes[1,1].plot(fdf["date"], fdf["bnb_ngn"], color="#27ae60", linewidth=1.5)
    axes[1,1].set_title("BNB/NGN"); axes[1,1].set_ylabel("NGN")

    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.markdown("#### Normalized Performance Comparison (Base 100)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label, color in [
        ("btc_usd",  "BTC",        "#f39c12"),
        ("eth_usd",  "ETH",        "#3498db"),
        ("bnb_usd",  "BNB",        "#2ecc71"),
        ("naira_usd","USD/NGN",    "#e74c3c"),
    ]:
        normalized = (fdf[col] / fdf[col].iloc[0]) * 100
        ax.plot(fdf["date"], normalized, label=label, linewidth=2, color=color)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Normalized Price (Base = 100)")
    ax.set_title("BTC vs ETH vs BNB vs Naira Depreciation")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Returns Summary")
    returns_data = {}
    for col, label in [("btc_usd","BTC"),("eth_usd","ETH"),("bnb_usd","BNB"),("naira_usd","NGN/USD")]:
        ret = ((fdf[col].iloc[-1] - fdf[col].iloc[0]) / fdf[col].iloc[0]) * 100
        returns_data[label] = ret

    ret_df = pd.DataFrame(list(returns_data.items()), columns=["Asset","Return (%)"])
    ret_df["Return (%)"] = ret_df["Return (%)"].round(2)

    fig, ax = plt.subplots(figsize=(7, 3))
    colors  = ["#2ecc71" if v > 0 else "#e74c3c" for v in ret_df["Return (%)"]]
    ax.bar(ret_df["Asset"], ret_df["Return (%)"], color=colors, width=0.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Asset Returns over {period}")
    for i, val in enumerate(ret_df["Return (%)"]):
        ax.text(i, val + (0.3 if val >= 0 else -0.8), f"{val:+.1f}%",
                ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab5:
    st.markdown("#### Correlation Matrix")
    corr_df = fdf[["btc_usd","eth_usd","bnb_usd","usdt_usd","naira_usd"]].copy()
    corr_df.columns = ["BTC","ETH","BNB","USDT","NGN/USD"]
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Price Correlation Matrix")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(corr.iloc[i,j]) < 0.7 else "white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Forecasting ───────────────────────────────────────────────────────────────
st.subheader("30-Day Price Forecast")
st.markdown("Linear Regression trend forecasting on the last 180 days of price data.")

forecast_asset = st.selectbox("Select Asset to Forecast",
                               ["BTC/USD","ETH/USD","BNB/USD","USD/NGN"])

asset_map = {
    "BTC/USD":  ("btc_usd",  "$",  1),
    "ETH/USD":  ("eth_usd",  "$",  1),
    "BNB/USD":  ("bnb_usd",  "$",  1),
    "USD/NGN":  ("naira_usd","N",  1),
}

col, prefix, div = asset_map[forecast_asset]
series = df[col].tail(180)
future_prices, mae, rmse, r2 = forecast_price(series)

future_dates = [df["date"].iloc[-1] + timedelta(days=i+1) for i in range(30)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df["date"].tail(180), series.values / div,
        color="#3498db", linewidth=2, label="Historical")
ax.plot(future_dates, future_prices / div,
        color="#e74c3c", linewidth=2, linestyle="--", label="Forecast")
ax.axvline(df["date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
ax.fill_between(future_dates,
                (future_prices * 0.95) / div,
                (future_prices * 1.05) / div,
                alpha=0.2, color="#e74c3c", label="95% Confidence Band")
ax.set_ylabel(f"Price ({prefix})")
ax.set_title(f"{forecast_asset} — 30-Day Forecast")
ax.legend()
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

f1, f2, f3 = st.columns(3)
f1.metric("Forecast Start", f"{prefix}{future_prices[0]/div:,.2f}")
f2.metric("Forecast End",   f"{prefix}{future_prices[-1]/div:,.2f}")
f3.metric("Trend",
          f"{((future_prices[-1]-future_prices[0])/future_prices[0])*100:+.2f}%")

st.markdown("**Model Metrics**")
m1, m2, m3 = st.columns(3)
m1.metric("MAE",      f"{prefix}{mae/div:,.2f}")
m2.metric("RMSE",     f"{prefix}{rmse/div:,.2f}")
m3.metric("R2 Score", f"{r2:.4f}")

st.divider()

# ── NGN Purchasing Power Calculator ──────────────────────────────────────────
st.subheader("NGN Crypto Calculator")
st.markdown("How much crypto can you buy with Nigerian Naira?")

ngn_amount = st.number_input("Enter NGN Amount", min_value=1000, max_value=100000000,
                              value=100000, step=1000)

cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("BTC you can buy",  f"{ngn_amount/latest['btc_ngn']:.8f} BTC")
cc2.metric("ETH you can buy",  f"{ngn_amount/latest['eth_ngn']:.6f} ETH")
cc3.metric("BNB you can buy",  f"{ngn_amount/latest['bnb_ngn']:.4f} BNB")
cc4.metric("USDT you can buy", f"{ngn_amount/latest['usdt_ngn']:.2f} USDT")

st.divider()

# ── Technical Details ─────────────────────────────────────────────────────────
with st.expander("Technical Details"):
    st.markdown(
        "- **Forecasting:** Linear Regression on time index\n"
        "- **Forecast Horizon:** 30 days\n"
        "- **Training Window:** Last 180 days\n"
        "- **Confidence Band:** +/- 5% of forecast\n"
        "- **Assets:** BTC, ETH, BNB, USDT vs NGN\n"
        "- **Data:** Synthetic historical prices (365 days)\n"
        "- **Exchange Rate:** Simulated USD/NGN depreciation trend\n"
        "- **Note:** For production use, replace with live data from CoinGecko API or yfinance"
    )