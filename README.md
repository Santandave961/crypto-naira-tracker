# Crypto vs Naira Tracker

A data science web application that tracks BTC, ETH, BNB, and USDT performance against the Nigerian Naira with trend analysis, correlation insights, and 30-day price forecasting.

Built with Python and Streamlit, deployed on Streamlit Community Cloud.

---

## Live Demo

[Click here to view the app](https://share.streamlit.io)

---

## Overview

This project explores the relationship between major cryptocurrencies and the Nigerian Naira, giving Nigerian investors and analysts a clear picture of how crypto assets perform relative to Naira depreciation over time.

---

## Features

- Latest prices for BTC, ETH, BNB, USDT in NGN
- Naira/USD exchange rate trend and depreciation stats
- BTC price in both USD and NGN
- ETH and BNB price charts
- Normalized performance comparison (Base 100)
- Returns summary across all assets
- Correlation matrix between all assets
- 30-day price forecasting with confidence band
- NGN Crypto Calculator — how much crypto can you buy with your Naira

---

## Forecast Model

| Metric | Value |
|--------|-------|
| Algorithm | Linear Regression |
| Training Window | Last 180 days |
| Forecast Horizon | 30 days |
| Confidence Band | +/- 5% of forecast |
| Assets | BTC, ETH, BNB, USD/NGN |

---

## Tech Stack

- **Language:** Python 3
- **Framework:** Streamlit
- **ML Library:** scikit-learn
- **Algorithm:** Linear Regression (time series forecasting)
- **Data Processing:** pandas, NumPy
- **Visualisation:** Matplotlib

---

## Project Structure

```
crypto-naira-tracker/
    app.py              # Main Streamlit application
    requirements.txt    # Python dependencies
    README.md           # Project documentation
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Santandave961/crypto-naira-tracker.git
cd crypto-naira-tracker
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How It Works

1. **Data** - 365 days of synthetic price data for BTC, ETH, BNB, USDT, and USD/NGN
2. **NGN Conversion** - All crypto prices converted to NGN using the simulated exchange rate
3. **Normalization** - Assets normalized to Base 100 for fair performance comparison
4. **Forecasting** - Linear Regression trained on last 180 days predicts next 30 days
5. **Calculator** - NGN amount divided by latest NGN price of each asset

---

## Key Insights

- BTC and ETH are highly correlated in price movement
- Naira depreciation amplifies crypto gains for Nigerian holders
- USDT acts as a stable store of value against Naira depreciation
- BNB shows lower volatility compared to BTC and ETH

---

## Note on Data

This app currently uses synthetic historical price data for demonstration purposes. For production use, replace the data generation function with live API calls from CoinGecko or yfinance.

---

## Author

**Okparaji Wisdom**
Data Science Student | Fintech Portfolio Builder

- GitHub: [@Santandave961](https://github.com/Santandave961)
- LinkedIn: [Connect with me](https://linkedin.com)

---

## License

MIT License - feel free to use and modify this project.
