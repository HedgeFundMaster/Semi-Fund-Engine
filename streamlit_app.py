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
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_inception_group(tab_keyword: str) -> pd.DataFrame:
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID)
    
    ws = next((w for w in sheet.worksheets() if tab_keyword in w.title), None)
    if ws is None:
        all_worksheets = [w.title for w in sheet.worksheets()]
        st.error(f"⚠️ No worksheet with '{tab_keyword}' in the title.")
        st.error(f"Available sheets: {all_worksheets}")
        st.stop()
    
    try:
        # Get all values as a list of lists
        raw_values = ws.get_all_values()
        if not raw_values:
            st.error(f"⚠️ Worksheet '{ws.title}' is empty!")
            st.stop()
        
        # Separate header and data
        header = raw_values[0]
        data_rows = raw_values[1:]
        
        if not data_rows:
            st.error(f"⚠️ No data rows found in '{ws.title}'!")
            st.stop()
        
        # Clean header (remove spaces)
        header = [str(col).strip() for col in header]
        header_len = len(header)
        
        # Process each data row to ensure consistent length
        processed_data = []
        for i, row in enumerate(data_rows):
            # Convert all values to strings first, then clean
            clean_row = [str(val).strip() if val else '' for val in row]
            
            # Ensure row matches header length
            if len(clean_row) < header_len:
                # Pad with empty strings
                clean_row.extend([''] * (header_len - len(clean_row)))
            elif len(clean_row) > header_len:
                # Truncate if too long
                clean_row = clean_row[:header_len]
            
            processed_data.append(clean_row)
        
        # Create DataFrame from processed data
        df = pd.DataFrame(processed_data, columns=header)
        
        # Replace empty strings with None for proper NaN handling
        df = df.replace('', None)
        
        # Convert numeric columns safely
        for col in df.columns:
            try:
                # Skip if column doesn't exist or is malformed
                if col not in df.columns:
                    continue
                    
                # Get the column as a pandas Series
                col_data = df[col]
                
                # Only proceed if it's actually a Series
                if not isinstance(col_data, pd.Series):
                    st.warning(f"Column '{col}' is not a proper Series, skipping numeric conversion")
                    continue
                
                # Try numeric conversion
                df[col] = pd.to_numeric(col_data, errors='ignore')
                
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to numeric: {str(e)}")
                continue
        
        # Handle date column if it exists
        if 'Inception Date' in df.columns:
            try:
                df['Inception Date'] = pd.to_datetime(df['Inception Date'], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert Inception Date: {str(e)}")
        
        # Check for required columns
        required_cols = ['Ticker', 'Fund']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"⚠️ Missing required columns in {tab_keyword}: {missing_cols}")
            st.stop()
        
        # Remove rows where both Ticker and Fund are missing
        df = df.dropna(subset=required_cols, how='all')
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        return df
        
    except Exception as e:
        st.error(f"🚨 Error loading data from '{ws.title}': {str(e)}")
        st.error("Please check your Google Sheet data format and try again.")
        st.stop()
# ─── 3) SCORING IMPLEMENTATIONS ──────────────────────────────────────────────────
def standardize_column_names(df):
    """Standardize column names across different sheets"""
    df = df.copy()
    
    # Create a mapping for inconsistent column names
    column_mapping = {
        'Sharpe (1y)': 'Sharpe (1Y)',
        'Sortino (1y)': 'Sortino (1Y)',
        'Sortino (3Y) ': 'Sortino (3Y)',  # Remove trailing space
        'Std Dev (1Y) ': 'Std Dev (1Y)',  # Remove trailing space
        'Std Dev (3Y) ': 'Std Dev (3Y)',  # Remove trailing space
        'Std Dev (5Y) ': 'Std Dev (5Y)',  # Remove trailing space
        'VaR ': 'VaR',  # Remove trailing space
    }
    
    # Apply the mapping
    df = df.rename(columns=column_mapping)
    
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def safe_numeric_conversion(df, columns):
    """Safely convert columns to numeric, handling any conversion issues"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN values with 0 for calculations
            df[col] = df[col].fillna(0)
    return df

def calculate_scores_1y(df: pd.DataFrame):
    df = df.copy()
    df = standardize_column_names(df)
    
    # Ensure numeric conversion for required columns
    numeric_cols = ['Total Return', 'Sharpe (1Y)', 'Sortino (1Y)', 'AUM', 'Net Expense', 'Std Dev (1Y)', 'VaR']
    df = safe_numeric_conversion(df, numeric_cols)
    
    df['Delta'] = df['Total Return'] - df.groupby("Category")['Total Return'].transform("mean")
    df["sharpe_composite"] = df['Sharpe (1Y)']
    df['sortino_composite'] = df['Sortino (1Y)']
    
    # Safe AUM score calculation with error handling
    try:
        aum_log = np.log1p(df['AUM'])
        aum_min, aum_max = aum_log.min(), aum_log.max()
        if aum_max != aum_min:
            df['aum_score'] = (aum_log - aum_min) / (aum_max - aum_min)
        else:
            df['aum_score'] = 0.5  # Default if all AUM values are the same
    except:
        df['aum_score'] = 0.5  # Default fallback
    
    # Safe expense score calculation
    try:
        exp_min, exp_max = df['Net Expense'].min(), df['Net Expense'].max()
        if exp_max != exp_min:
            df['expense_score'] = 1 - ((df['Net Expense'] - exp_min) / (exp_max - exp_min))
        else:
            df['expense_score'] = 0.5  # Default if all expense values are the same
    except:
        df['expense_score'] = 0.5  # Default fallback
    
    df['Score'] = (
        METRIC_WEIGHTS['sharpe_composite'] * df['sharpe_composite'] +
        METRIC_WEIGHTS['sortino_composite'] * df['sortino_composite'] +
        METRIC_WEIGHTS['delta'] * df['Delta'] +
        METRIC_WEIGHTS['total_return'] * df['Total Return'] +
        METRIC_WEIGHTS['aum'] * df['aum_score'] +
        METRIC_WEIGHTS['expense'] * df['expense_score']
    )
    
    if 'Sharpe (1Y)' in df.columns:
        df.loc[df['Sharpe (1Y)'] > INTEGRITY_PENALTY_THRESHOLD, 'Score'] -= INTEGRITY_PENALTY_AMOUNT 

    if 'Std Dev (1Y)' in df.columns:
        q3 = df['Std Dev (1Y)'].quantile(0.75)
        df.loc[df['Std Dev (1Y)'] > q3, 'Score'] -= 0.1

    if 'VaR' in df.columns:
        q3 = df['VaR'].quantile(0.75)
        df.loc[df['VaR'] > q3, 'Score'] -= 0.1 
    
    return df, None 

def calculate_scores_3y(df: pd.DataFrame):
    df = df.copy()
    df = standardize_column_names(df)
    
    numeric_cols = ['Total Return (3Y)', 'Sharpe (3Y)', 'Sortino (3Y)', 'Sharpe (1Y)', 'Sortino (1Y)', 
                   'AUM', 'Net Expense', 'Std Dev (3Y)', '2022 Return']
    df = safe_numeric_conversion(df, numeric_cols)
    
    df['Delta'] = df['Total Return (3Y)'] - df.groupby("Category")['Total Return (3Y)'].transform("mean")   
    df['sharpe_composite'] = 0.5 * df['Sharpe (3Y)'] + 0.5 * df['Sharpe (1Y)']
    df['sortino_composite'] = 0.5 * df['Sortino (3Y)'] + 0.5 * df['Sortino (1Y)']
    
    # Safe calculations
    try:
        aum_log = np.log1p(df['AUM'])
        aum_min, aum_max = aum_log.min(), aum_log.max()
        if aum_max != aum_min:
            df['aum_score'] = (aum_log - aum_min) / (aum_max - aum_min)
        else:
            df['aum_score'] = 0.5
    except:
        df['aum_score'] = 0.5
        
    try:
        exp_min, exp_max = df['Net Expense'].min(), df['Net Expense'].max()
        if exp_max != exp_min:
            df['expense_score'] = 1 - ((df['Net Expense'] - exp_min) / (exp_max - exp_min))
        else:
            df['expense_score'] = 0.5
    except:
        df['expense_score'] = 0.5

    df['Score'] = (
        METRIC_WEIGHTS['sharpe_composite'] * df['sharpe_composite'] +
        METRIC_WEIGHTS['sortino_composite'] * df['sortino_composite'] +
        METRIC_WEIGHTS['delta'] * df['Delta'] +
        METRIC_WEIGHTS['total_return'] * df['Total Return (3Y)'] +
        METRIC_WEIGHTS['aum'] * df['aum_score'] +
        METRIC_WEIGHTS['expense'] * df['expense_score']
    )

    # Penalties with safe checks
    if 'Sharpe (3Y)' in df.columns:
        df.loc[df['Sharpe (3Y)'] > INTEGRITY_PENALTY_THRESHOLD, 'Score'] -= INTEGRITY_PENALTY_AMOUNT

    if 'Std Dev (3Y)' in df.columns:
        q3 = df['Std Dev (3Y)'].quantile(0.75)
        df.loc[df['Std Dev (3Y)'] >= q3, 'Score'] -= 0.1

    if '2022 Return' in df.columns:
        q1 = df['2022 Return'].quantile(0.25)
        df.loc[df['2022 Return'] <= q1, 'Score'] -= 0.1

    return df, None

def calculate_scores_5y(df: pd.DataFrame):
    df = df.copy()
    df = standardize_column_names(df)
    
    numeric_cols = ['Total Return (5Y)', 'Sharpe (5Y)', 'Sortino (5Y)', 'Sharpe (3Y)', 'Sortino (3Y)',
                   'Sharpe (1Y)', 'Sortino (1Y)', 'AUM', 'Net Expense', 'Std Dev (5Y)', '2022 Return']
    df = safe_numeric_conversion(df, numeric_cols)
    
    df['Delta'] = df['Total Return (5Y)'] - df.groupby("Category")['Total Return (5Y)'].transform("mean")

    # Composite Sharpe + Sortino weighting with safe fallbacks
    df['sharpe_composite'] = (
        0.50 * df.get('Sharpe (5Y)', 0) +
        0.30 * df.get('Sharpe (3Y)', 0) +
        0.20 * df.get('Sharpe (1Y)', 0)
    )

    df['sortino_composite'] = (
        0.50 * df.get('Sortino (5Y)', 0) +
        0.30 * df.get('Sortino (3Y)', 0) +
        0.20 * df.get('Sortino (1Y)', 0)
    )

    # Safe calculations
    try:
        aum_log = np.log1p(df['AUM'])
        aum_min, aum_max = aum_log.min(), aum_log.max()
        if aum_max != aum_min:
            df['aum_score'] = (aum_log - aum_min) / (aum_max - aum_min)
        else:
            df['aum_score'] = 0.5
    except:
        df['aum_score'] = 0.5
        
    try:
        exp_min, exp_max = df['Net Expense'].min(), df['Net Expense'].max()
        if exp_max != exp_min:
            df['expense_score'] = 1 - ((df['Net Expense'] - exp_min) / (exp_max - exp_min))
        else:
            df['expense_score'] = 0.5
    except:
        df['expense_score'] = 0.5

    # Composite Score - need to add Total Return (5Y) column check
    if 'Total Return (5Y)' not in df.columns:
        df['Total Return (5Y)'] = 0  # Default fallback
        
    df['Score'] = (
        METRIC_WEIGHTS['sharpe_composite'] * df['sharpe_composite'] +
        METRIC_WEIGHTS['sortino_composite'] * df['sortino_composite'] +
        METRIC_WEIGHTS['delta'] * df['Delta'] +
        METRIC_WEIGHTS['total_return'] * df['Total Return (5Y)'] +
        METRIC_WEIGHTS['aum'] * df['aum_score'] +
        METRIC_WEIGHTS['expense'] * df['expense_score']
    )

    # Penalties with safe checks
    if 'Sharpe (5Y)' in df.columns:
        df.loc[df['Sharpe (5Y)'] > INTEGRITY_PENALTY_THRESHOLD, 'Score'] -= INTEGRITY_PENALTY_AMOUNT

    if 'Std Dev (5Y)' in df.columns:
        q3 = df['Std Dev (5Y)'].quantile(0.75)
        df.loc[df['Std Dev (5Y)'] >= q3, 'Score'] -= 0.1

    if '2022 Return' in df.columns:
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