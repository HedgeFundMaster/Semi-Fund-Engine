import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

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
creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
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

# Function to convert Excel serial date to 'YYYY-MM-DD'

def convert_excel_date(date_str):
    """Convert Excel serial (e.g. "46022") to 'YYYY-MM-DD', else return None."""
    try:
        serial = int(date_str)
        epoch  = datetime(1899, 12, 30)
        return (epoch + timedelta(days=serial)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # If it already looks like YYYY-MM-DD, keep it
        if isinstance(date_str, str) and len(date_str)==10 and date_str.count('-')==2:
            return date_str
        return None

clean_headers = []  # final column names
valid_idx     = []  # which raw_headers columns to keep

# Process each raw header
for i, h in enumerate(raw_headers):
    h = h.strip()
    if not h:
        # skip blanks entirely
        continue

    # Try Excel serial → date
    parsed = convert_excel_date(h)
    if parsed:
        clean_headers.append(parsed)
        valid_idx.append(i)
    else:
        # keep non‐date metrics like "Symbol","Name","Metric"
        clean_headers.append(h)
        valid_idx.append(i)

# Debug print
st.write("Using columns:", clean_headers)

# Slice raw_data into rows of only those valid columns
rows = [[r[i] for i in valid_idx] for r in raw_data]

# Build the wide initial DataFrame
df_initial = pd.DataFrame(rows, columns=clean_headers)
st.write("Initial DF columns:", list(df_initial.columns))


# Check required columns
required_cols = ["Symbol", "Name", "Metric"]
missing_cols = [col for col in required_cols if col not in df_initial.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.write(f"Available columns: {list(df_initial.columns)}")
    st.stop()
# Melt to long form
df_long = df_initial.melt(
    id_vars=["Symbol","Name","Metric"],
    var_name="Date",
    value_name="Value"
)

# Coerce Value to numeric and drop NaNs
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long = df_long.dropna(subset=["Value"])

# ─── 2.5) DATE FILTERING ────────────────────────────────────────────────────

# Get date columns (exclude the first 3 columns)
date_cols = [col for col in df_initial.columns if col not in required_cols]
st.write(f"Date columns found: {date_cols}")

# Convert Date column to datetime (coerce invalids to NaT)
df_long["Date"] = pd.to_datetime(df_long["Date"], format="%Y-%m-%d", errors="coerce")

# Drop any rows where Date conversion failed
df_long = df_long[df_long["Date"].notna()]

st.write(f"After date filtering: {len(df_long)} rows")

if len(df_long) == 0:
    st.error("No rows with valid dates found!")
    st.stop()

# Convert Value column to numeric and drop NaNs
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long = df_long.dropna(subset=["Value"])

st.write("After numeric conversion:")
st.write(f"Non-null values: {df_long['Value'].notna().sum()}")
st.write(f"Null values: {df_long['Value'].isna().sum()}")

# Final check
if df_long["Value"].notna().sum() == 0:
    st.error("No numeric values found after conversion!")
    st.stop()
# Create pivot table AFTER cleaning
df_wide = df_long.pivot_table(
    index=["Symbol", "Name", "Date"], 
    columns="Metric", 
    values="Value"
).reset_index()

# ─── 3A) ROLLING RETURN METRICS ─────────────────────────────────────────────
df_wide["Date"] = pd.to_datetime(df_wide["Date"], errors="coerce")
latest_date = df_wide["Date"].max()

# Sort for rolling calculations
df_wide = df_wide.sort_values(["Symbol", "Date"])

# Rolling returns
df_wide["Return 1Y"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
    lambda x: x.rolling(window=2, min_periods=1).mean()
)
df_wide["Return 3Y"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)
df_wide["Return_Since_Inception"] = df_wide.groupby("Symbol")["Total Return Level"].transform("mean")


# ─── 3B) SCORE CALCULATION ─────────────────────────────────────────────────
# Weights and penalties
WEIGHT_RETURN = 3.0
WEIGHT_DELTA = 2.0
PENALTY_VAR = 1.5
PENALTY_STDDEV = 1.2

# Latest snapshot
df_latest = df_wide[df_wide["Date"] == latest_date].copy()

# Score formulas
df_latest["Score_1Y"] = df_latest["Return 1Y"] * WEIGHT_RETURN + df_latest["Delta"] * WEIGHT_DELTA
df_latest["Score_3Y"] = df_latest["Return 3Y"] * WEIGHT_RETURN + df_latest["Delta"] * WEIGHT_DELTA
df_latest["Score_Since_Inception"] = df_latest["Return_Since_Inception"] * WEIGHT_RETURN + df_latest["Delta"] * WEIGHT_DELTA

# Apply risk penalties
for score_col in ["Score_1Y", "Score_3Y", "Score_Since_Inception"]:
    if "Daily Value at Risk (VaR) 5% (1Y Lookback)" in df_latest.columns:
        df_latest[score_col] -= df_latest["Daily Value at Risk (VaR) 5% (1Y Lookback)"] * PENALTY_VAR
    if "Annualized Standard Deviation of Monthly Returns (1Y Lookback)" in df_latest.columns:
        df_latest[score_col] -= df_latest["Annualized Standard Deviation of Monthly Returns (1Y Lookback)"] * PENALTY_STDDEV

# Tier assignment
def tier(s: float) -> str:
    if pd.isna(s): return "No Data"
    if s >= 8.5: return "Tier 1"
    if s >= 6.0: return "Tier 2"
    return "Tier 3"

df_latest["Tier_1Y"] = df_latest["Score_1Y"].apply(tier)
df_latest["Tier_3Y"] = df_latest["Score_3Y"].apply(tier)
df_latest["Tier_Since_Inception"] = df_latest["Score_Since_Inception"].apply(tier)

# Top 10 tables
top10_1Y = df_latest.sort_values("Score_1Y", ascending=False).head(10)
top10_3Y = df_latest.sort_values("Score_3Y", ascending=False).head(10)
top10_SI = df_latest.sort_values("Score_Since_Inception", ascending=False).head(10)

# Composite score and tier
df_latest["Score"] = (
    0.4 * df_latest["Score_1Y"] +
    0.3 * df_latest["Score_3Y"] +
    0.3 * df_latest["Score_Since_Inception"]
)
df_latest["Tier"] = df_latest["Score"].apply(tier)

# Re-create top10 composite view
top10 = df_latest.sort_values("Score", ascending=False).head(10)

# ─── 3C) COMPOSITE SCORE CALCULATION ──────────────────────────────────────
# Weighted composite score
df_latest["Score"] = (
    0.4 * df_latest["Score_1Y"] +
    0.3 * df_latest["Score_3Y"] +
    0.3 * df_latest["Score_Since_Inception"]
)

# Assign composite tier
df_latest["Tier"] = df_latest["Score"].apply(tier)

# Top 10 composite funds
top10 = df_latest.sort_values("Score", ascending=False).head(10)



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


