import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

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

# Clean headers and convert Excel serial numbers to dates
def convert_excel_date(date_str):
    """Convert Excel serial number to date string"""
    try:
        # Try parsing as Excel serial number
        serial_num = int(date_str)
        # Excel epoch is 1900-01-01, but Excel incorrectly treats 1900 as a leap year, so we use 1899-12-30 as the epoch
    
        excel_epoch = datetime(1899, 12, 30)
        converted_date = excel_epoch + timedelta(days=serial_num)
        return converted_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # If it's already a date string, return as is
        if isinstance(date_str, str) and len(date_str) == 10 and date_str.count('-') == 2:
            return date_str
        return None

# Process headers
clean_headers = []
for h in raw_headers:
    h = h.strip()
    if not h:
        continue
    
    # Convert Excel serial numbers to dates
    if h.isdigit():
        converted = convert_excel_date(h)
        if converted:
            clean_headers.append(converted)
        else:
            clean_headers.append(h)
    else:
        clean_headers.append(h)

st.write(f"Clean headers: {clean_headers}")

from datetime import datetime, timedelta

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

# Pivot to wide form
df_wide = df_long.pivot_table(
    index=["Symbol","Name","Date"],
    columns="Metric",
    values="Value"
).reset_index()

# (and then your date-filter, parse, peer avg, scoring, etc.)

# Get date columns (exclude the first 3 columns)
date_cols = [col for col in df_initial.columns if col not in required_cols]
st.write(f"Date columns found: {date_cols}")

# Filter for valid dates only
valid_date_pattern = r'^\d{4}-\d{2}-\d{2}$'
df_long = df_long[df_long["Date"].str.match(valid_date_pattern, na=False)]

st.write(f"After date filtering: {len(df_long)} rows")

if len(df_long) == 0:
    st.error("No rows with valid dates found!")
    st.stop()

# Convert Date column to datetime
df_long["Date"] = pd.to_datetime(df_long["Date"], format="%Y-%m-%d")

# Convert Value column to numeric
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

st.write("After numeric conversion:")
st.write(f"Non-null values: {df_long['Value'].notna().sum()}")
st.write(f"Null values: {df_long['Value'].isna().sum()}")

# Check if we have any numeric data
if df_long["Value"].notna().sum() == 0:
    st.error("No numeric values found after conversion!")
    st.stop()

# Create pivot table
df_wide = df_long.pivot_table(
    index=["Symbol", "Name", "Date"], 
    columns="Metric", 
    values="Value"
).reset_index()

st.write("Pivot table result:")
st.dataframe(df_wide.head())

# Check required columns for calculations
required_metrics = ["Total Return Level"]
missing_metrics = [col for col in required_metrics if col not in df_wide.columns]
if missing_metrics:
    st.error(f"Missing required metrics: {missing_metrics}")
    st.write(f"Available metric columns: {[col for col in df_wide.columns if col not in ['Symbol', 'Name', 'Date']]}")
    st.stop()

# Calculate peer average & delta
df_wide["Peer Avg"] = df_wide.groupby("Date")["Total Return Level"].transform("mean")
df_wide["Delta"] = df_wide["Total Return Level"] - df_wide["Peer Avg"]

# ─── 3) SCORE & TIER ────────────────────────────────────────────────────────
latest_date = df_wide["Date"].max()
df_score = df_wide[df_wide["Date"] == latest_date].copy()

st.write(f"Latest date: {latest_date}")
st.write(f"Funds on latest date: {len(df_score)}")

if len(df_score) == 0:
    st.error("No data for the latest date!")
    st.stop()

# Calculate score with available columns
df_score["Score"] = df_score["Total Return Level"] * 3.0 + df_score["Delta"] * 2.0

# Subtract VaR and StdDev if available
if "Daily Value at Risk (VaR) 5% (1Y Lookback)" in df_score.columns:
    df_score["Score"] -= df_score["Daily Value at Risk (VaR) 5% (1Y Lookback)"] * 1.5

if "Annualized Standard Deviation of Monthly Returns (1Y Lookback)" in df_score.columns:
    df_score["Score"] -= df_score["Annualized Standard Deviation of Monthly Returns (1Y Lookback)"] * 1.2

def tier(s: float) -> str:
    if pd.isna(s):
        return "No Data"
    if s >= 8.5: return "Tier 1"
    if s >= 6.0: return "Tier 2"
    return "Tier 3"

df_score["Tier"] = df_score["Score"].apply(tier)

# Prepare Top 10
top10 = df_score.sort_values("Score", ascending=False).head(10)

# ─── 4) STREAMLIT UI ──────────────────────────────────────────────────────
st.title("Semi-Liquid Fund Selection Dashboard")
st.markdown(f"**Data as of:** {latest_date.date()}")

st.subheader("Top 10 Funds by Composite Score")
st.dataframe(top10[["Symbol", "Name", "Score", "Tier"]], height=300)

st.subheader("Full Fund Scores & Metrics")
st.dataframe(df_score, height=600)


