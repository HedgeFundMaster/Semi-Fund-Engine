
# 🏦 Semi-Liquid Fund Selection Engine

A 5,000-line full-stack scoring engine for institutional-grade semi-liquid alternative funds, built with Python, Streamlit, and the Google Sheets API.

## 🚀 Overview

The Semi-Fund Engine is a dynamic, interactive dashboard that:
- Scores semi-liquid alternative funds (1Y+, 3Y+, 5Y+) across multiple performance and risk metrics
- Applies adaptive weight normalization for missing data
- Penalizes funds with misleadingly high Sharpe/Sortino ratios
- Categorizes and ranks funds by asset class (e.g., Private Credit, Infrastructure)
- Integrates directly with Google Sheets for real-time data ingestion and scoring
- Visualizes results with Plotly for investment team decision-making

## 📊 Key Features
- 🔍 **Composite Scoring:** Weighted by age group, with logic for 1Y, 3Y, and 5Y funds
- 🧠 **Sharpe Integrity Check:** Flags smoothed NAV distortions in interval funds
- 📈 **Quartile Penalization:** Applies 2022 drawdown penalties and Std Dev bonuses
- 🔗 **Google Sheets API Integration:** Live sync with fund-level data
- 📊 **Category Rankings:** Top 10 funds by strategy with tiering engine (T1, T2, T3)
- 🛠️ **Fully Customizable:** Easily extendable with Streamlit components and modular design

## ⚙️ Tech Stack
- `Python`, `pandas`, `NumPy`
- `Streamlit` for UI
- `Plotly` for visuals
- `Google Sheets API` for fund data ingestion
- `YAML` config and `.env` support for modular logic

## 🧠 Sample Use Cases
- Internal investment committee reviews
- Fund-of-funds screening
- SMA allocation benchmarking
- Client-ready visual analytics
## by: Brendan O'Sullivan | Boston College | 



This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
