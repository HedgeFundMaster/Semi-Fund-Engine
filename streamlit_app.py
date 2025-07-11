# ─── 0) STREAMLIT CONFIG & LIBRARIES ───────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="Fund Dashboard", layout="wide")

st.write("✅ App code loaded")    # ← add this, redeploy, and look for it in the UI

import logging
logging.basicConfig(level=logging.DEBUG)

import os, json
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

# Scoring weights and penalties
WEIGHT_RETURN = 3.0
WEIGHT_DELTA = 2.0
PENALTY_VAR = 1.5
PENALTY_STDDEV = 1.2

# Composite score weights
COMPOSITE_WEIGHTS = {
    "1Y": 0.4,
    "3Y": 0.3,
    "SI": 0.3
}

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

# ─── 3) DATA LOADING & CLEANING FUNCTIONS ──────────────────────────────────
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

def detect_data_frequency(df):
    """Detect if data is monthly, quarterly, or other frequency"""
    if len(df) < 2:
        return "unknown", 12  # default to monthly
    
    # Sort by date to get consistent intervals
    df_sorted = df.sort_values('Date')
    dates = df_sorted['Date'].unique()
    
    if len(dates) < 2:
        return "unknown", 12
    
    # Calculate typical interval between dates
    intervals = []
    for i in range(1, min(len(dates), 10)):  # Check first 10 intervals
        diff = (dates[i] - dates[i-1]).days
        intervals.append(diff)
    
    avg_interval = np.mean(intervals)
    
    # Determine frequency
    if 25 <= avg_interval <= 35:  # Monthly (around 30 days)
        return "monthly", 12
    elif 85 <= avg_interval <= 95:  # Quarterly (around 90 days)
        return "quarterly", 4
    elif 360 <= avg_interval <= 370:  # Annual
        return "annual", 1
    else:
        return "unknown", 12  # default to monthly

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_clean_data():
    """Load and clean data from Google Sheets with enhanced error handling"""
    
    with st.spinner("🔄 Loading data from Google Sheets..."):
        try:
            client = get_gspread_client()
            sheet = client.open_by_key(SHEET_ID)
            
            # Find Time Series worksheet
            ts_ws = None
            for ws in sheet.worksheets():
                if "Time Series" in ws.title:
                    ts_ws = ws
                    break
            
            if ts_ws is None:
                st.error("⚠️ No worksheet with 'Time Series' in the title found!")
                st.info("💡 Please ensure your Google Sheet has a worksheet with 'Time Series' in the name.")
                st.stop()
            
            # Get all values
            values = ts_ws.get_all_values()
            
            if not values:
                st.error("⚠️ No data found in the Time Series worksheet!")
                st.stop()
            
            # Find header row
            header_idx = None
            for i, row in enumerate(values):
                if row and row[0].strip() == "Symbol":
                    header_idx = i
                    break
            
            if header_idx is None:
                st.error("⚠️ No header row with 'Symbol' found!")
                st.info("💡 Please ensure your first column header is exactly 'Symbol'.")
                st.stop()
            
            raw_headers = values[header_idx]
            raw_data = values[header_idx+1:]
            
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
                    # Non-date column (Symbol, Name, Metric, etc.)
                    clean_headers.append(header)
                    valid_idx.append(i)
            
            # Build DataFrame with clean headers
            rows = [[row[i] for i in valid_idx] for row in raw_data]
            df_initial = pd.DataFrame(rows, columns=clean_headers)
            
            # Check required columns
            required_cols = ["Symbol", "Name", "Metric"]
            missing_cols = [col for col in required_cols if col not in df_initial.columns]
            if missing_cols:
                st.error(f"⚠️ Missing required columns: {missing_cols}")
                st.info("💡 Please ensure your sheet has columns: Symbol, Name, Metric")
                st.stop()
            
            # Melt to long form
            df_long = df_initial.melt(
                id_vars=["Symbol", "Name", "Metric"],
                var_name="Date",
                value_name="Value"
            )
            
            # Clean and convert Date column
            df_long["Date"] = df_long["Date"].apply(lambda d: convert_to_standard_date(d))
            df_long = df_long[df_long["Date"].notna()]
            df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
            
            # Remove invalid dates and values
            df_long = df_long[df_long["Date"].notna()]
            df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
            df_long = df_long.dropna(subset=["Value"])
            
            # Data quality checks
            if len(df_long) == 0:
                st.error("⚠️ No valid data remaining after cleaning!")
                st.info("💡 Please check your data format and try again.")
                st.stop()
            
            # Remove obvious outliers (values beyond reasonable bounds)
            df_long = df_long[df_long["Value"].between(-1000, 10000)]
            
            return df_long
            
        except Exception as e:
            st.error(f"🚨 Error loading data: {str(e)}")
            st.info("💡 Please check your Google Sheets connection and data format.")
            st.stop()

def calculate_data_quality_metrics(df):
    """Calculate data quality indicators"""
    total_records = len(df)
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    
    # Check for duplicate records
    duplicates = df.duplicated().sum()
    
    # Date range completeness
    date_range = df['Date'].max() - df['Date'].min()
    unique_dates = df['Date'].nunique()
    
    # Data freshness (days since last update)
    last_update = df['Date'].max()
    days_since_update = (datetime.now() - last_update).days
    
    quality_score = max(0, 100 - (missing_values/total_records * 50) - (duplicates/total_records * 30))
    
    return {
        'total_records': total_records,
        'missing_values': missing_values,
        'duplicates': duplicates,
        'date_range_days': date_range.days,
        'unique_dates': unique_dates,
        'days_since_update': days_since_update,
        'quality_score': quality_score
    }

# ─── 4) SCORING FUNCTIONS ──────────────────────────────────────────────────
def calculate_rolling_returns(df_wide):
    """Calculate rolling returns based on detected data frequency"""
    
    # Detect data frequency
    frequency, periods_per_year = detect_data_frequency(df_wide)
    
    # Calculate rolling windows based on frequency
    window_1y = periods_per_year
    window_3y = periods_per_year * 3
    
    # Sort for rolling calculations
    df_wide = df_wide.sort_values(["Symbol", "Date"])
    
    # Calculate peer averages for Delta calculation
    df_wide["Peer Avg"] = df_wide.groupby("Date")["Total Return Level"].transform("mean")
    df_wide["Delta"] = df_wide["Total Return Level"] - df_wide["Peer Avg"]
    
    # Rolling returns with proper annualization
    df_wide["Return 1Y"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
        lambda x: x.rolling(window=window_1y, min_periods=max(1, window_1y//2)).apply(
            lambda vals: (vals.iloc[-1] / vals.iloc[0]) ** (periods_per_year / len(vals)) - 1 if len(vals) > 1 else 0
        )
    )
    
    df_wide["Return 3Y"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
        lambda x: x.rolling(window=window_3y, min_periods=max(1, window_3y//2)).apply(
            lambda vals: (vals.iloc[-1] / vals.iloc[0]) ** (periods_per_year / len(vals)) - 1 if len(vals) > 1 else 0
        )
    )
    
    # Since inception return (annualized)
    df_wide["Return_Since_Inception"] = df_wide.groupby("Symbol")["Total Return Level"].transform(
        lambda x: (x.iloc[-1] / x.iloc[0]) ** (periods_per_year / len(x)) - 1 if len(x) > 1 else 0
    )
    
    return df_wide, frequency

def calculate_scores(df_latest):
    """Calculate fund scores with enhanced risk adjustments"""
    
    # Initialize score columns
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
    
    applied_penalties = []
    for risk_col, penalty_weight in risk_columns.items():
        if risk_col in df_latest.columns:
            # Apply penalty (ensure risk metrics are positive)
            risk_values = df_latest[risk_col].abs()
            df_latest["Score_1Y"] -= risk_values * penalty_weight
            df_latest["Score_3Y"] -= risk_values * penalty_weight
            df_latest["Score_Since_Inception"] -= risk_values * penalty_weight
            applied_penalties.append(risk_col)
    
    # Calculate composite score
    df_latest["Score"] = (
        COMPOSITE_WEIGHTS["1Y"] * df_latest["Score_1Y"] +
        COMPOSITE_WEIGHTS["3Y"] * df_latest["Score_3Y"] +
        COMPOSITE_WEIGHTS["SI"] * df_latest["Score_Since_Inception"]
    )
    
    return df_latest, applied_penalties

def assign_tiers(df_latest):
    """Assign tiers based on scores with dynamic thresholds"""
    
    def tier(s: float) -> str:
        if pd.isna(s): return "No Data"
        if s >= 8.5: return "Tier 1"
        if s >= 6.0: return "Tier 2"
        return "Tier 3"
    
    # Apply tier assignments
    df_latest["Tier_1Y"] = df_latest["Score_1Y"].apply(tier)
    df_latest["Tier_3Y"] = df_latest["Score_3Y"].apply(tier)
    df_latest["Tier_Since_Inception"] = df_latest["Score_Since_Inception"].apply(tier)
    df_latest["Tier"] = df_latest["Score"].apply(tier)
    
    return df_latest

# ─── 5) VISUALIZATION FUNCTIONS ────────────────────────────────────────────
def create_score_distribution_chart(df_latest):
    """Create interactive score distribution chart"""
    fig = px.histogram(
        df_latest, 
        x="Score", 
        nbins=20,
        title="Score Distribution",
        labels={"Score": "Composite Score", "count": "Number of Funds"}
    )
    fig.update_layout(height=400)
    return fig

def create_top_funds_chart(df_latest):
    """Create interactive top funds chart"""
    top_funds = df_latest.sort_values("Score", ascending=False).head(10)
    
    fig = px.bar(
        top_funds,
        x="Score",
        y="Name",
        orientation="h",
        title="Top 10 Funds by Composite Score",
        labels={"Score": "Composite Score", "Name": "Fund Name"},
        color="Score",
        color_continuous_scale="RdYlGn"
    )
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    return fig

def create_risk_return_scatter(df_latest):
    """Create risk-return scatter plot"""
    if "Annualized Standard Deviation of Monthly Returns (1Y Lookback)" in df_latest.columns:
        fig = px.scatter(
            df_latest,
            x="Annualized Standard Deviation of Monthly Returns (1Y Lookback)",
            y="Return 1Y",
            color="Tier",
            hover_data=["Symbol", "Name", "Score"],
            title="Risk-Return Profile",
            labels={
                "Annualized Standard Deviation of Monthly Returns (1Y Lookback)": "Risk (Std Dev)",
                "Return 1Y": "1-Year Return"
            }
        )
        fig.update_layout(height=500)
        return fig
    return None

# ─── 6) MAIN DASHBOARD FUNCTION ────────────────────────────────────────────
def create_dashboard():
    """Create the main dashboard interface"""
    
    st.title("🏦 Semi-Liquid Fund Selection Dashboard")
    st.markdown("---")
    
    # Load data
    df_long = load_and_clean_data()
    
    # Calculate data quality metrics
    quality_metrics = calculate_data_quality_metrics(df_long)
    
    # Create pivot table
    df_wide = df_long.pivot_table(
        index=["Symbol", "Name", "Date"], 
        columns="Metric", 
        values="Value"
    ).reset_index()
    
    # Ensure Date column is datetime
    df_wide["Date"] = pd.to_datetime(df_wide["Date"], errors="coerce")
    latest_date = df_wide["Date"].max()
    
    # Check if we have the required columns
    required_metrics = ["Total Return Level"]
    missing_metrics = [col for col in required_metrics if col not in df_wide.columns]
    if missing_metrics:
        st.error(f"⚠️ Missing required metrics: {missing_metrics}")
        st.info("💡 Please ensure your data includes 'Total Return Level' metric.")
        st.stop()
    
    # Calculate rolling returns and scores
    df_wide, frequency = calculate_rolling_returns(df_wide)
    
    # Get latest snapshot
    df_latest = df_wide[df_wide["Date"] == latest_date].copy()
    
    if len(df_latest) == 0:
        st.error(f"⚠️ No data available for the latest date: {latest_date}")
        st.info("💡 Please check your data source or try refreshing the page.")
        st.stop()
    
    # Calculate scores and assign tiers
    df_latest, applied_penalties = calculate_scores(df_latest)
    df_latest = assign_tiers(df_latest)
    
    # ─── HEADER METRICS ───────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📅 Latest Date", latest_date.strftime("%Y-%m-%d"))
    
    with col2:
        st.metric("🏢 Funds Scored", len(df_latest))
    
    with col3:
        st.metric("📊 Data Quality", f"{quality_metrics['quality_score']:.1f}%")
    
    with col4:
        st.metric("🔄 Data Frequency", frequency.title())
    
    with col5:
        freshness_color = "normal" if quality_metrics['days_since_update'] <= 7 else "inverse"
        st.metric("⏱️ Data Freshness", f"{quality_metrics['days_since_update']} days", 
                 delta=None, delta_color=freshness_color)
    
    # ─── MAIN CONTENT ─────────────────────────────────────────────────────
    
    # Top 10 Summary
    st.subheader("🏆 Top 10 Funds by Composite Score")
    top10 = df_latest.sort_values("Score", ascending=False).head(10)
    
    # Display top 10 with enhanced formatting
    display_cols = ["Symbol", "Name", "Score", "Tier", "Return 1Y", "Return 3Y"]
    display_df = top10[display_cols].copy()
    
    # Format percentages
    for col in ["Return 1Y", "Return 3Y"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    
    # Format scores
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Color-code tiers
    def color_tier(val):
        if val == "Tier 1":
            return "background-color: #d4edda; color: #155724"  # green
        elif val == "Tier 2":
            return "background-color: #fff3cd; color: #856404"  # yellow
        elif val == "Tier 3":
            return "background-color: #f8d7da; color: #721c24"  # red
        return ""
    
    styled_df = display_df.style.applymap(color_tier, subset=["Tier"])
    st.dataframe(styled_df, height=400, use_container_width=True)
    
    # ─── INTERACTIVE CHARTS ──────────────────────────────────────────────
    st.subheader("📈 Interactive Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        score_dist_fig = create_score_distribution_chart(df_latest)
        st.plotly_chart(score_dist_fig, use_container_width=True)
    
    with chart_col2:
        risk_return_fig = create_risk_return_scatter(df_latest)
        if risk_return_fig:
            st.plotly_chart(risk_return_fig, use_container_width=True)
        else:
            st.info("💡 Risk-return chart not available (missing risk metrics)")
    
    # Top funds chart
    top_funds_fig = create_top_funds_chart(df_latest)
    st.plotly_chart(top_funds_fig, use_container_width=True)
    
    # ─── MULTI-PERIOD ANALYSIS ──────────────────────────────────────────
    st.subheader("🔄 Multi-Period Analysis")
    
    tabs = st.tabs(["1-Year Performance", "3-Year Performance", "Since Inception"])
    
    with tabs[0]:
        st.write("**Top 10 Funds – 1-Year Performance**")
        top10_1Y = df_latest.sort_values("Score_1Y", ascending=False).head(10)
        display_1Y = top10_1Y[["Symbol", "Name", "Score_1Y", "Tier_1Y", "Return 1Y"]].copy()
        display_1Y["Score_1Y"] = display_1Y["Score_1Y"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_1Y["Return 1Y"] = display_1Y["Return 1Y"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        styled_1Y = display_1Y.style.applymap(color_tier, subset=["Tier_1Y"])
        st.dataframe(styled_1Y, height=400, use_container_width=True)
    
    with tabs[1]:
        st.write("**Top 10 Funds – 3-Year Performance**")
        top10_3Y = df_latest.sort_values("Score_3Y", ascending=False).head(10)
        display_3Y = top10_3Y[["Symbol", "Name", "Score_3Y", "Tier_3Y", "Return 3Y"]].copy()
        display_3Y["Score_3Y"] = display_3Y["Score_3Y"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_3Y["Return 3Y"] = display_3Y["Return 3Y"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        styled_3Y = display_3Y.style.applymap(color_tier, subset=["Tier_3Y"])
        st.dataframe(styled_3Y, height=400, use_container_width=True)
    
    with tabs[2]:
        st.write("**Top 10 Funds – Since Inception**")
        top10_SI = df_latest.sort_values("Score_Since_Inception", ascending=False).head(10)
        display_SI = top10_SI[["Symbol", "Name", "Score_Since_Inception", "Tier_Since_Inception", "Return_Since_Inception"]].copy()
        display_SI["Score_Since_Inception"] = display_SI["Score_Since_Inception"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_SI["Return_Since_Inception"] = display_SI["Return_Since_Inception"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        styled_SI = display_SI.style.applymap(color_tier, subset=["Tier_Since_Inception"])
        st.dataframe(styled_SI, height=400, use_container_width=True)
    
    # ─── DETAILED ANALYSIS ──────────────────────────────────────────────
    st.subheader("🔍 Detailed Fund Analysis")
    
    # Tier filtering
    selected_tiers = st.multiselect(
        "Select Tiers to Display:",
        options=["Tier 1", "Tier 2", "Tier 3", "No Data"],
        default=["Tier 1", "Tier 2", "Tier 3"]
    )
    
    filtered_df = df_latest[df_latest["Tier"].isin(selected_tiers)]
    
    # Display all fund scores
    all_display_cols = ["Symbol", "Name", "Score", "Tier", "Return 1Y", "Return 3Y", "Return_Since_Inception"]
    if "Delta" in filtered_df.columns:
        all_display_cols.append("Delta")
    
    all_display_df = filtered_df[all_display_cols].copy()
    
    # Format the display
    for col in ["Return 1Y", "Return 3Y", "Return_Since_Inception"]:
        if col in all_display_df.columns:
            all_display_df[col] = all_display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    
    if "Delta" in all_display_df.columns:
        all_display_df["Delta"] = all_display_df["Delta"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    all_display_df["Score"] = all_display_df["Score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    styled_all = all_display_df.style.applymap(color_tier, subset=["Tier"])
    st.dataframe(styled_all, height=600, use_container_width=True)
    
    # ─── SIDEBAR CONFIGURATION ──────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Dashboard Configuration")
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Score breakdown
        st.header("📘 Score Methodology")
        st.markdown(f"""
        **Performance Rewards:**
        - Total Return Level: {WEIGHT_RETURN}x weight
        - Delta (vs Peer Avg): {WEIGHT_DELTA}x weight
        
        **Risk Penalties:**
        - VaR (Value at Risk): -{PENALTY_VAR}x weight
        - Standard Deviation: -{PENALTY_STDDEV}x weight
        
        **Composite Score Weights:**
        - 1-Year: {COMPOSITE_WEIGHTS['1Y']*100:.0f}%
        - 3-Year: {COMPOSITE_WEIGHTS['3Y']*100:.0f}%
        - Since Inception: {COMPOSITE_WEIGHTS['SI']*100:.0f}%
        """)
        
        if applied_penalties:
            st.success(f"✅ Applied {len(applied_penalties)} risk penalties")
        else:
            st.warning("⚠️ No risk penalties applied (missing risk metrics)")
        
        st.markdown("---")
        
        # Data quality information
        st.header("🔍 Data Quality")
        st.metric("Total Records", quality_metrics['total_records'])
        st.metric("Date Range", f"{quality_metrics['date_range_days']} days")
        st.metric("Unique Dates", quality_metrics['unique_dates'])
        
        if quality_metrics['missing_values'] > 0:
            st.warning(f"⚠️ {quality_metrics['missing_values']} missing values detected")
        if quality_metrics['duplicates'] > 0:
            st.warning(f"⚠️ {quality_metrics['duplicates']} duplicate records detected")

# ─── RUN THE DASHBOARD ────────────────────────────────────────────────────
if __name__ == "__main__":
    create_dashboard()
