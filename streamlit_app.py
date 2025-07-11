import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

import logging
logging.basicConfig(level=logging.DEBUG)
st.set_page_config(page_title="Fund Dashboard", layout="wide")

def score_funds_by_period(df_long, period_years=None):
    """
    Scores funds based on Total Return Level over a given period.
    - If period_years is None → uses all available data (since inception).
    - If period_years = 1 or 3 → uses trailing returns.
    """
    today = pd.Timestamp.today()
    
    if period_years:
        cutoff_date = today - pd.DateOffset(years=period_years)
        df_filtered = df_long[df_long["Date"] >= cutoff_date].copy()
    else:
        df_filtered = df_long.copy()

    # Convert to wide
    df_temp = df_filtered.pivot_table(index=["Symbol", "Name", "Date"], columns="Metric", values="Value").reset_index()

    if "Total Return Level" not in df_temp.columns:
        return pd.DataFrame()  # no score possible

    df_temp["Peer Avg"] = df_temp.groupby("Date")["Total Return Level"].transform("mean")
    df_temp["Delta"] = df_temp["Total Return Level"] - df_temp["Peer Avg"]

    latest_date = df_temp["Date"].max()

    # New guard clause
    if pd.isna(latest_date) or latest_date not in df_temp["Date"].values:
        return pd.DataFrame()
    df_score = df_temp[df_temp["Date"] == latest_date].copy()

    df_score["Score"] = df_score["Total Return Level"] * 3.0 + df_score["Delta"] * 2.0

    if "Daily Value at Risk (VaR) 5% (1Y Lookback)" in df_score.columns:
        df_score["Score"] -= df_score["Daily Value at Risk (VaR) 5% (1Y Lookback)"] * 1.5

    if "Annualized Standard Deviation of Monthly Returns (1Y Lookback)" in df_score.columns:
        df_score["Score"] -= df_score["Annualized Standard Deviation of Monthly Returns (1Y Lookback)"] * 1.2

    def tier(score):
        if pd.isna(score): return "No Data"
        if score >= 8.5: return "Tier 1"
        if score >= 6.0: return "Tier 2"
        return "Tier 3"

    df_score["Tier"] = df_score["Score"].apply(tier)
    df_score["Period"] = f"{period_years}Y" if period_years else "Since Inception"
    
    return df_score


# ─── 1) AUTH ───────────────────────────────────────────────────────────────
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
import json
# Build the service-account dict from individual secret fields
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

# ─── 2) PULL & CLEAN ────────────────────────────────────────────────────────
SHEET_ID = "1p7rZ4sX3uKcSpRy5hHfau0isszyHSDQfjcKgNBH29XI"
sheet    = client.open_by_key(SHEET_ID)

# Find Time Series worksheet
ts_ws = None
for ws in sheet.worksheets():
    if "Time Series" in ws.title:
        ts_ws = ws
        break

if ts_ws is None:
    st.error("No worksheet with 'Time Series' in the title found!")
    st.stop()

st.write(f"Using worksheet: {ts_ws.title}")

values = ts_ws.get_all_values()
st.write(f"Total rows retrieved: {len(values)}")

# Find header row
header_idx = None
for i, row in enumerate(values):
    if row and row[0].strip() == "Symbol":
        header_idx = i
        break

if header_idx is None:
    st.error("No header row with 'Symbol' found!")
    st.stop()

st.write(f"Header row found at index: {header_idx}")

raw_headers = values[header_idx]
raw_data = values[header_idx+1:]

st.write(f"Raw headers: {raw_headers}")
st.write(f"Data rows: {len(raw_data)}")

# FIXED: Better date detection and parsing function
def is_date_column(header_str):
    """Check if a header represents a date column"""
    header_str = str(header_str).strip()
    
    # Check for YYYY-MM-DD format
    if len(header_str) == 10 and header_str.count('-') == 2:
        try:
            pd.to_datetime(header_str, format="%Y-%m-%d")
            return True
        except:
            pass
    
    # Check for Excel serial number (numeric and in reasonable range)
    try:
        serial = float(header_str)
        if 1 <= serial <= 2958465:  # Valid Excel date range
            return True
    except:
        pass
    
    return False

def convert_to_standard_date(date_str):
    """Convert various date formats to YYYY-MM-DD string"""
    date_str = str(date_str).strip()
    
    # Already in YYYY-MM-DD format
    if len(date_str) == 10 and date_str.count('-') == 2:
        try:
            # Validate it's a real date
            pd.to_datetime(date_str, format="%Y-%m-%d")
            return date_str
        except:
            pass
    
    # Try Excel serial conversion
    try:
        serial = float(date_str)
        if 1 <= serial <= 2958465:
            epoch = datetime(1899, 12, 30)
            converted_date = epoch + timedelta(days=int(serial))
            return converted_date.strftime("%Y-%m-%d")
    except:
        pass
    
    # Try general parsing
    try:
        parsed_date = pd.to_datetime(date_str)
        return parsed_date.strftime("%Y-%m-%d")
    except:
        pass
    
    return None

# Process headers
clean_headers = []
valid_idx = []

for i, header in enumerate(raw_headers):
    header = str(header).strip()
    
    if not header:
        continue
    
    # Check if it's a date column
    if is_date_column(header):
        converted_date = convert_to_standard_date(header)
        if converted_date:
            clean_headers.append(converted_date)
            valid_idx.append(i)
        else:
            st.warning(f"Could not convert date header: {header}")
    else:
        # Non-date column (Symbol, Name, Metric, etc.)
        clean_headers.append(header)
        valid_idx.append(i)

# Debug output
st.write("=== HEADER PROCESSING DEBUG ===")
st.write(f"Original headers: {raw_headers}")
st.write(f"Clean headers: {clean_headers}")
st.write(f"Valid indices: {valid_idx}")

# Build DataFrame with clean headers
rows = [[row[i] for i in valid_idx] for row in raw_data]
df_initial = pd.DataFrame(rows, columns=clean_headers)

st.write(f"Initial DataFrame shape: {df_initial.shape}")
st.write("Initial DataFrame columns:", list(df_initial.columns))

# Check required columns
required_cols = ["Symbol", "Name", "Metric"]
missing_cols = [col for col in required_cols if col not in df_initial.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.write(f"Available columns: {list(df_initial.columns)}")
    st.stop()

# Melt to long form
df_long = df_initial.melt(
    id_vars=["Symbol", "Name", "Metric"],
    var_name="Date",
    value_name="Value"
)

st.write(f"After melting: {df_long.shape}")

# ─── 2.5) SIMPLIFIED DATE CONVERSION ────────────────────────────────────────

# Debug: Show what we're working with
st.write("=== DATE CONVERSION DEBUG ===")
st.write("Sample Date values from melted data:")
st.write(df_long["Date"].head(10).tolist())
st.write("Unique dates:", df_long["Date"].nunique())

# Clean and convert Date column
df_long["Date"] = df_long["Date"].apply(lambda d: convert_to_standard_date(d))
df_long = df_long[df_long["Date"].notna()]
df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")

# Check results
valid_dates = df_long["Date"].notna().sum()
invalid_dates = df_long["Date"].isna().sum()

st.write(f"✅ Valid dates: {valid_dates} | ❌ Invalid dates: {invalid_dates}")

# Only stop if ALL dates are invalid (very rare after fallback clean)
if valid_dates == 0:
    st.error("🛑 No valid dates found — check spreadsheet headers for typos!")
    st.stop()

# Drop rows with invalid dates
df_long = df_long[df_long["Date"].notna()]

# Convert Value column to numeric
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long = df_long.dropna(subset=["Value"])

st.write(f"Final df_long shape: {df_long.shape}")
st.write(f"Date range: {df_long['Date'].min()} to {df_long['Date'].max()}")

if len(df_long) == 0:
    st.error("No valid data remaining after cleaning!")
    st.stop()

# ─── 3A) CREATE PIVOT TABLE AND PROCEED WITH ANALYSIS ──────────────────────

# Create pivot table for wide format analysis
df_wide = df_long.pivot_table(
    index=["Symbol", "Name", "Date"], 
    columns="Metric", 
    values="Value"
).reset_index()

# Ensure Date column is datetime
df_wide["Date"] = pd.to_datetime(df_wide["Date"], errors="coerce")
latest_date = df_wide["Date"].max()

st.write(f"✅ Data successfully processed!")
st.write(f"📊 df_wide shape: {df_wide.shape}")
st.write(f"📅 Latest date: {latest_date}")
st.write(f"🏢 Number of funds: {df_wide['Symbol'].nunique()}")
st.write(f"📈 Available metrics: {[col for col in df_wide.columns if col not in ['Symbol', 'Name', 'Date']]}")

# Debug Fix: List above from lines 64-268

# Get latest date AFTER creating df_wide
latest_date = df_wide["Date"].max()

# Debug: Check if we have valid data
st.write(f"df_wide shape: {df_wide.shape}")
st.write(f"Latest date: {latest_date}")
st.write(f"Date range: {df_wide['Date'].min()} to {df_wide['Date'].max()}")

# Check if we have the required columns
required_metrics = ["Total Return Level"]
missing_metrics = [col for col in required_metrics if col not in df_wide.columns]
if missing_metrics:
    st.error(f"Missing required metrics in df_wide: {missing_metrics}")
    st.write(f"Available columns: {list(df_wide.columns)}")
    st.stop()

# Sort for rolling calculations
df_wide = df_wide.sort_values(["Symbol", "Date"])

# Calculate peer averages for Delta calculation
df_wide["Peer Avg"] = df_wide.groupby("Date")["Total Return Level"].transform("mean")
df_wide["Delta"] = df_wide["Total Return Level"] - df_wide["Peer Avg"]

# Rolling returns (simplified - using actual rolling windows based on data points)
df_wide["Return 1Y"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()  # Assuming monthly data
)
df_wide["Return 3Y"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
    lambda x: x.rolling(window=36, min_periods=1).mean()  # Assuming monthly data
)
df_wide["Return_Since_Inception"] = df_wide.groupby("Symbol")["Total Return Level"].transform("mean")

# ─── 3B) SCORE CALCULATION ─────────────────────────────────────────────────
# Weights and penalties
WEIGHT_RETURN = 3.0
WEIGHT_DELTA = 2.0
PENALTY_VAR = 1.5
PENALTY_STDDEV = 1.2

# Latest snapshot - ensure we have data for the latest date
df_latest = df_wide[df_wide["Date"] == latest_date].copy()

# Debug: Check if we have data for latest date
st.write(f"Records for latest date ({latest_date}): {len(df_latest)}")

if len(df_latest) == 0:
    st.error(f"No data found for latest date: {latest_date}")
    st.write("Available dates:")
    st.write(df_wide["Date"].value_counts().head(10))
    st.stop()

# Initialize score columns with the base calculation
df_latest["Score_1Y"] = df_latest["Return 1Y"] * WEIGHT_RETURN
df_latest["Score_3Y"] = df_latest["Return 3Y"] * WEIGHT_RETURN  
df_latest["Score_Since_Inception"] = df_latest["Return_Since_Inception"] * WEIGHT_RETURN

# Add Delta component if available
if "Delta" in df_latest.columns:
    df_latest["Score_1Y"] += df_latest["Delta"] * WEIGHT_DELTA
    df_latest["Score_3Y"] += df_latest["Delta"] * WEIGHT_DELTA
    df_latest["Score_Since_Inception"] += df_latest["Delta"] * WEIGHT_DELTA

# Apply risk penalties if columns exist
risk_columns = {
    "Daily Value at Risk (VaR) 5% (1Y Lookback)": PENALTY_VAR,
    "Annualized Standard Deviation of Monthly Returns (1Y Lookback)": PENALTY_STDDEV
}

for risk_col, penalty_weight in risk_columns.items():
    if risk_col in df_latest.columns:
        df_latest["Score_1Y"] -= df_latest[risk_col] * penalty_weight
        df_latest["Score_3Y"] -= df_latest[risk_col] * penalty_weight
        df_latest["Score_Since_Inception"] -= df_latest[risk_col] * penalty_weight
        st.write(f"Applied {risk_col} penalty")
    else:
        st.write(f"Risk metric not found: {risk_col}")

# Tier assignment function
def tier(s: float) -> str:
    if pd.isna(s): return "No Data"
    if s >= 8.5: return "Tier 1"
    if s >= 6.0: return "Tier 2"
    return "Tier 3"

# Apply tier assignments
df_latest["Tier_1Y"] = df_latest["Score_1Y"].apply(tier)
df_latest["Tier_3Y"] = df_latest["Score_3Y"].apply(tier)
df_latest["Tier_Since_Inception"] = df_latest["Score_Since_Inception"].apply(tier)

# ─── 3C) COMPOSITE SCORE CALCULATION ──────────────────────────────────────
# Weighted composite score
df_latest["Score"] = (
    0.4 * df_latest["Score_1Y"] +
    0.3 * df_latest["Score_3Y"] +
    0.3 * df_latest["Score_Since_Inception"]
)

# Assign composite tier
df_latest["Tier"] = df_latest["Score"].apply(tier)

# Create top 10 tables
top10_1Y = df_latest.sort_values("Score_1Y", ascending=False).head(10)
top10_3Y = df_latest.sort_values("Score_3Y", ascending=False).head(10)
top10_SI = df_latest.sort_values("Score_Since_Inception", ascending=False).head(10)
top10 = df_latest.sort_values("Score", ascending=False).head(10)

# Debug: Show score statistics
st.write("Score Statistics:")
st.write(f"Score_1Y: min={df_latest['Score_1Y'].min():.2f}, max={df_latest['Score_1Y'].max():.2f}")
st.write(f"Score_3Y: min={df_latest['Score_3Y'].min():.2f}, max={df_latest['Score_3Y'].max():.2f}")
st.write(f"Composite Score: min={df_latest['Score'].min():.2f}, max={df_latest['Score'].max():.2f}")



# ─── 4) STREAMLIT UI ──────────────────────────────────────────────────────
st.title("Semi-Liquid Fund Selection Dashboard")
st.markdown(f"**Data as of:** {latest_date.date()}")

st.subheader("Top 10 Funds by Composite Score")
st.dataframe(top10[["Symbol", "Name", "Score", "Tier"]], height=300)

st.subheader("🔄 Multi-Period Scoring")

tabs = st.tabs(["1-Year", "3-Year", "Since Inception"])

periods = [1, 3, None]  # corresponding to tab order

for i, tab in enumerate(tabs):
    with tab:
        period = periods[i]
        result_df = score_funds_by_period(df_long, period)
        if result_df.empty:
            st.warning(f"No scoring data available for {period or 'Since Inception'} period.")
        else:
            top_n = result_df.sort_values("Score", ascending=False).head(10)
            st.write(f"**Top 10 Funds – {result_df['Period'].iloc[0]}**")
            st.dataframe(top_n[["Symbol", "Name", "Score", "Tier"]], height=400)
            st.write(df_long.groupby("Symbol")["Date"].nunique().sort_values(ascending=False))

st.subheader("Full Fund Scores & Metrics")
st.dataframe(df_latest, height=600)

col1, col2 = st.columns(2)
col1.metric("Latest Date", latest_date.strftime("%Y-%m-%d"))
col2.metric("Funds Scored", len(df_latest))

def color_tier(val):
    if val == "Tier 1":
        return "background-color: #d4edda"  # green
    elif val == "Tier 2":
        return "background-color: #fff3cd"  # yellow
    elif val == "Tier 3":
        return "background-color: #f8d7da"  # red
    return ""

styled_df = df_latest.style.applymap(color_tier, subset=["Tier"])
st.subheader("Full Fund Scores & Metrics")
st.dataframe(styled_df, height=600)

import matplotlib.pyplot as plt

top_plot = top10.sort_values("Score", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_plot["Name"], top_plot["Score"])
ax.set_xlabel("Composite Score")
ax.set_title("Top 10 Funds by Composite Score")
st.pyplot(fig)

with st.sidebar:
    st.header("📘 Score Breakdown")
    st.markdown("""
    - **Total Return Level**: Weighted 3x
    - **Delta**: Weighted 2x  
    - **VaR (penalty)**: Weighted -1.5x  
    - **Std Dev (penalty)**: Weighted -1.2x

    Final score = rewards performance, penalizes risk.
    """)

selected_tiers = st.multiselect(
    "Select Tiers to Display",
    options=["Tier 1", "Tier 2", "Tier 3", "No Data"],
    default=["Tier 1", "Tier 2", "Tier 3"]
)
filtered = df_latest[df_latest["Tier"].isin(selected_tiers)]
st.dataframe(filtered, height=600)


