# ─── 0) STREAMLIT CONFIG & LIBRARIES ───────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="Fund Dashboard", layout="wide")

st.write("✅ App code loaded")    # ← add this, redeploy, and look for it in the UI

import logging
logging.basicConfig(level=logging.DEBUG)

import os, json
import seaborn as sns
import pandas as pd
import gspread
import matplotlib.pyplot as plt
import numpy as np
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ─── 1) AUTHENTICATION & CONFIGURATION ─────────────────────────────────────
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

SHEET_ID = "1p7rZ4sX3uKcSpRy5hHfau0isszyHSDQfjcKgNBH29XI"

# ─── 1A) COMPOSITE SCORE CONFIGURATION (Locked Framework) ─────────────────────────────────────
METRIC_WEIGHTS = {
    "sharpe_composite": 0.35,
    "sortino_composite": 0.25,
    "delta": 0.10,
    "total_return": 0.05, 
    "aum": 0.05,
    "expense": 0.05
}
INTEGRITY_PENALTY_THRESHOLD = 4.5 # Sharpe > 4.5 flagged 
INTEGRITY_PENALTY_AMOUNT = 0.25 # subtract if flagged

# ─── 2) AUTHENTICATION FUNCTION ────────────────────────────────────────────
@st.cache_resource
def get_gspread_client():
    """Get authenticated Google Sheets client with caching"""
    try:
        # 1a) Local dev: load key from file
        if os.path.exists("gcp_key.json"):
            with open("gcp_key.json","r") as fp:
                creds_dict = json.load(fp)
        # 1b) Cloud: load from secrets.toml
        else:
            creds_dict = {
                "type":                        st.secrets["type"],
                "project_id":                  st.secrets["project_id"],
                "private_key_id":              st.secrets["private_key_id"],
                "private_key":                 st.secrets["private_key"],
                "client_email":                st.secrets["client_email"],
                "client_id":                   st.secrets["client_id"],
                "auth_uri":                    st.secrets["auth_uri"],
                "token_uri":                   st.secrets["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
                "client_x509_cert_url":        st.secrets["client_x509_cert_url"],
            }

        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"🚨 Authentication failed: {str(e)}")
        st.info("💡 Please check your Google Sheets credentials and try refreshing the page.")
        st.stop()

# ─── 2A) DATA LOADING: TAB-BASED FUND DATA  ──────────────────────────────────
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_inception_group(tab_keyword: str) -> pd.DataFrame:
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID)
    ws = next((w for w in sheet.worksheets() if tab_keyword in w.title), None)
    if ws is None:
        st.error(f"⚠️ No worksheet with '{tab_keyword}' in the title.")
        st.stop()
    
    # Get all values
    values = ws.get_all_values()
    if not values:
        st.error(f"⚠️ Worksheet '{ws.title}' is empty!")
        st.stop()
    
    # Find header row
    header, *data = values
    df = pd.DataFrame(data, columns=header)
    df = df.map(lambda v: v.strip() if isinstance(v, str) else v)      
    for col in df.columns: 
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    df['Inception Date'] = pd.to_datetime(df['Inception Date'], errors='coerce')
    df = df.dropna(subset=['Ticker', 'Fund', 'Inception Date'])
    return df

# ─── 3) SCORING IMPLEMENTATIONS ──────────────────────────────────────────────────
def calculate_scores_1y(df: pd.DataFrame):
    df = df.copy()
    df['Delta'] = df['Total Return'] - df.groupby("Category")['Total Return'].transform("mean")
    df["sharpe_composite"] = df ['Sharpe (1Y)']
    df['sortino_composite'] = df['Sortino (1Y)']
    df['aum_score'] = (np.log1p(df['AUM']) - np.log1p(df['AUM']).min()) /\
                        (np.log1p(df['AUM']).max() - np.log1p(df['AUM']).min())
    df['expense_score'] = 1 - ((df['Net Expense']) - df['Net Expense'].min()) /\
                            (df['Net Expense'].max() - df['Net Expense'].min())
    
    df['Score'] = (
        METRIC_WEIGHTS['sharpe_composite'] * df['sharpe_composite'] +
        METRIC_WEIGHTS['sortino_composite'] * df['sortino_composite'] +
        METRIC_WEIGHTS['delta'] * df['Delta'] +
        METRIC_WEIGHTS['total_return'] * df['Total Return'] +
        METRIC_WEIGHTS['aum'] * df['aum_score'] +
        METRIC_WEIGHTS['expense'] * df['expense_score']
    )
    if 'Sharpe (1Y)' in df:
        df.loc[df['Sharpe (1Y)'] > INTEGRITY_PENALTY_THRESHOLD, 'Score'] -= INTEGRITY_PENALTY_AMOUNT 

    if 'Std Dev (1Y)' in df:
        q3 = df['Std Dev (1Y)'].quantile(0.75)
        df.loc[df['Std Dev (1Y)'] > q3, 'Score'] -= 0.1

    if 'VaR' in df:
        q3 = df['VaR'].quantile(0.75)
        df.loc[df['VaR'] > q3, 'Score'] -= 0.1 
    
    return df, None 

def calculate_scores_3y(df: pd.DataFrame):
    df = df.copy()
    df['Delta'] = df['Total Return (3Y)'] - df.groupby("Category")['Total Return (3Y)'].transform("mean")   
    df['sharpe_composite'] = 0.5 * df['Sharpe (3Y)'] + 0.5 * df['Sharpe (1Y)']
    df['sortino_composite'] = 0.5 * df['Sortino (3Y)'] + 0.5 * df['Sortino (1Y)']
    df['aum_score'] = (np.log1p(df['AUM']) - np.log1p(df['AUM']).min()) / \
                      (np.log1p(df['AUM']).max() - np.log1p(df['AUM']).min())
    df['expense_score'] = 1 - ((df['Net Expense'] - df['Net Expense'].min()) / \
                             (df['Net Expense'].max() - df['Net Expense'].min()))

    df['Score'] = (
        METRIC_WEIGHTS['sharpe_composite'] * df['sharpe_composite'] +
        METRIC_WEIGHTS['sortino_composite'] * df['sortino_composite'] +
        METRIC_WEIGHTS['delta'] * df['Delta'] +
        METRIC_WEIGHTS['total_return'] * df['Total Return (3Y)'] +
        METRIC_WEIGHTS['aum'] * df['aum_score'] +
        METRIC_WEIGHTS['expense'] * df['expense_score']
    )

    # Penalty: High Sharpe
    df.loc[df['Sharpe (3Y)'] > INTEGRITY_PENALTY_THRESHOLD, 'Score'] -= INTEGRITY_PENALTY_AMOUNT

    # Penalty: Std Dev (3Y)
    if 'Std Dev (3Y)' in df.columns:
        q3 = df['Std Dev (3Y)'].quantile(0.75)
        df.loc[df['Std Dev (3Y)'] >= q3, 'Score'] -= 0.1

    # Penalty: 2022 Return
    if '2022 Return' in df.columns:
        q1 = df['2022 Return'].quantile(0.25)
        df.loc[df['2022 Return'] <= q1, 'Score'] -= 0.1

    return df, None

def calculate_scores_5y(df: pd.DataFrame):
    df = df.copy()
    df['Delta'] = df['Total Return (5Y)'] - df.groupby("Category")['Total Return (5Y)'].transform("mean")

    # Composite Sharpe + Sortino weighting
    df['sharpe_composite'] = (
        0.50 * df['Sharpe (5Y)'] +
        0.30 * df['Sharpe (3Y)'] +
        0.20 * df['Sharpe (1Y)']
    )

    df['sortino_composite'] = (
        0.50 * df['Sortino (5Y)'] +
        0.30 * df['Sortino (3Y)'] +
        0.20 * df['Sortino (1Y)']
    )

    # Log-normalized AUM
    df['aum_score'] = (np.log1p(df['AUM']) - np.log1p(df['AUM']).min()) / \
                      (np.log1p(df['AUM']).max() - np.log1p(df['AUM']).min())
    df['expense_score'] = 1 - ((df['Net Expense'] - df['Net Expense'].min()) /
                               (df['Net Expense'].max() - df['Net Expense'].min()))

    # Composite Score
    df['Score'] = (
        METRIC_WEIGHTS['sharpe_composite'] * df['sharpe_composite'] +
        METRIC_WEIGHTS['sortino_composite'] * df['sortino_composite'] +
        METRIC_WEIGHTS['delta'] * df['Delta'] +
        METRIC_WEIGHTS['total_return'] * df['Total Return (5Y)'] +
        METRIC_WEIGHTS['aum'] * df['aum_score'] +
        METRIC_WEIGHTS['expense'] * df['expense_score']
    )

    # Integrity Penalty for high Sharpe
    df.loc[df['Sharpe (5Y)'] > INTEGRITY_PENALTY_THRESHOLD, 'Score'] -= INTEGRITY_PENALTY_AMOUNT

    # Std Dev Penalty (top quartile)
    if 'Std Dev (5Y)' in df:
        q3 = df['Std Dev (5Y)'].quantile(0.75)
        df.loc[df['Std Dev (5Y)'] >= q3, 'Score'] -= 0.1

    # 2022 Return Penalty (bottom quartile)
    if '2022 Return' in df:
        q1 = df['2022 Return'].quantile(0.25)
        df.loc[df['2022 Return'] <= q1, 'Score'] -= 0.1

    return df, None


# ─── 3A) TIER ASSIGNMENT ─────────────────────────────────────────────────
def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    def tier(score):
        if pd.isna(score): return "No Data"  # Fixed: was pd.isn(score)
        if score >= 8.5: return "Tier 1"
        if score >= 6.0: return "Tier 2"
        return "Tier 3"
    df['Tier'] = df['Score'].apply(tier)
    return df

# ─── STYLE TIERS ────────────────────────────────────────────────
def style_tiers(df, tier_column="Tier"):
    def color(val):
        if val == "Tier 1":
            return "background-color: #d4edda; color: #155724"
        elif val == "Tier 2":
            return "background-color: #fff3cd; color: #856404"
        elif val == "Tier 3":
            return "background-color: #f8d7da; color: #721c24"
        return ""
    return df.style.applymap(color, subset=[tier_column])

# ─── 4) MAIN DASHBOARD FUNCTION ────────────────────────────────────────────
def create_dashboard():
    st.title("🏦 Semi-Liquid Fund Selection Dashboard")
    st.markdown("---")

    df_1y = load_inception_group("1Y+ Inception Funds"); df_1y['Inception Group'] = '1Y+'
    df_3y = load_inception_group("3Y+ Inception Funds"); df_3y['Inception Group'] = '3Y+'
    df_5y = load_inception_group("5Y+ Inception Funds"); df_5y['Inception Group'] = '5Y+'

    scored_1y, _ = calculate_scores_1y(df_1y)
    scored_3y, _ = calculate_scores_3y(df_3y)
    scored_5y, _ = calculate_scores_5y(df_5y)

    df_all = pd.concat([scored_1y, scored_3y, scored_5y], ignore_index=True)
    df_tiered = assign_tiers(df_all)

    st.sidebar.header("🔍 Filters & Configuration")
    inception_opts = ["1Y+", "3Y+", "5Y+"]
    tiers_opts = ["Tier 1", "Tier 2", "Tier 3", "No Data"]
    cat_opts = sorted(df_tiered['Category'].dropna().unique())

    selected_inceptions = st.sidebar.multiselect("Select Inception Group(s):", inception_opts, default=inception_opts)
    selected_tiers = st.sidebar.multiselect("Select Tier(s):", tiers_opts, default=tiers_opts[:-1])
    selected_categories = st.sidebar.multiselect("Select Category(ies):", cat_opts, default=cat_opts)

    filtered_df = df_tiered[
        df_tiered['Inception Group'].isin(selected_inceptions) &
        df_tiered['Tier'].isin(selected_tiers) &
        df_tiered['Category'].isin(selected_categories)
    ]

    last_date = df_tiered["Inception Date"].max()
    st.caption(f"📅 Last refreshed: {last_date.strftime('%Y-%m-%d')}")

    # 1. Top 10 Overall
    st.subheader("🥇 Top 10 Funds Overall")
    top10 = filtered_df.sort_values("Score", ascending=False).head(10)
    st.dataframe(style_tiers(top10[["Ticker", "Fund", "Category", "Score", "Tier", "Inception Group"]]), use_container_width=True)

    # 2. Top 10 by Category (Sorted by Tier)
    st.subheader("🏅 Top 10 by Category (Sorted by Tier)")
    for cat in filtered_df['Category'].dropna().unique():
        st.markdown(f"**📂 {cat}**")
        group = filtered_df[filtered_df['Category'] == cat].sort_values(['Tier', 'Score'], ascending=[True, False]).head(10)
        st.dataframe(style_tiers(group[["Ticker", "Fund", "Score", "Tier", "Inception Group"]]), use_container_width=True)

    # 3. Top 5 by Inception
    st.subheader("🏆 Top 5 by Inception Group")
    for g in ["1Y+", "3Y+", "5Y+"]:
        st.markdown(f"**📈 {g} Funds**")
        top = filtered_df[filtered_df['Inception Group'] == g].sort_values("Score", ascending=False).head(5)
        st.dataframe(style_tiers(top[["Ticker", "Fund", "Score", "Tier"]]), use_container_width=True)

    # 4. Average Score by Group
    st.subheader("📊 Average Score by Inception Group")
    avg_score = filtered_df.groupby("Inception Group")["Score"].mean().reset_index()
    st.bar_chart(avg_score.set_index("Inception Group"))

    # 5. Heatmap (Category × Inception Group)
    st.subheader("🌡️ Score Heatmap by Category & Inception")
    heatmap_data = filtered_df.groupby(["Category", "Inception Group"])["Score"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Download
    @st.cache_data
    def _convert_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button("📥 Download Filtered Data", data=_convert_to_csv(filtered_df), file_name="filtered_funds.csv", mime="text/csv")

if __name__ == "__main__":
    create_dashboard()