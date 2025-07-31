# ─── 0) STREAMLIT CONFIG & LIBRARIES ───────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="Semi Liquid Alternatives Fund Dashboard", layout="wide", page_icon="🏦")

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .tier-1 { background: #d4edda; color: #155724; }
    .tier-2 { background: #fff3cd; color: #856404; }
    .tier-3 { background: #f8d7da; color: #721c24; }
    
    .fund-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


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

# ─── 1A) COMPOSITE SCORE CONFIGURATION (Enhanced Framework) ─────────────────────────────────────

# ENHANCED METRIC WEIGHTS - Total Return as Primary Driver
METRIC_WEIGHTS = {
    "total_return": 0.30,        # PRIMARY: Actual fund performance (was 0.05)
    "sharpe_composite": 0.25,    # Risk-adjusted return (reduced from 0.35)
    "sortino_composite": 0.20,   # Downside risk adjustment (reduced from 0.25)
    "delta": 0.10,               # Category-relative performance
    "aum": 0.075,                # Assets under management scale (was 0.05)
    "expense": 0.075             # Cost efficiency (was 0.05)
}

# DATA QUALITY & SCORING PARAMETERS
INTEGRITY_PENALTY_THRESHOLD = 4.5   # Sharpe > 4.5 flagged as suspicious
INTEGRITY_PENALTY_AMOUNT = 0.25     # Score penalty for suspicious metrics

# TIER ASSIGNMENT PARAMETERS (Dynamic - Recalculated Based on Distribution)
DEFAULT_TIER_THRESHOLDS = {
    "tier1_percentile": 0.85,        # Top 15% of funds -> Tier 1
    "tier2_percentile": 0.50,        # 50th-85th percentile -> Tier 2
    "min_tier1_score": 0.5,          # Minimum absolute score for Tier 1
    "min_tier2_score": -0.5          # Minimum absolute score for Tier 2
}

# DATA COMPLETENESS REQUIREMENTS - VERY RELAXED FOR MORE FUNDS
MIN_DATA_REQUIREMENTS = {
    "1Y+": {"required": ["Total Return", "Sharpe (1Y)", "Sortino (1Y)"], "max_missing": 3},
    "3Y+": {"required": ["Total Return (3Y)", "Sharpe (3Y)", "Sortino (3Y)"], "max_missing": 3},
    "5Y+": {"required": ["Total Return (5Y)", "Sharpe (5Y)", "Sortino (5Y)"], "max_missing": 3}
}

# SCORING METHODOLOGY FLAGS
ENHANCED_NAN_HANDLING = True         # Use proper NaN handling (no 0-filling)
DYNAMIC_TIER_THRESHOLDS = True       # Use percentile-based tier assignment
VALIDATE_SCORE_CALCULATIONS = True   # Enable mathematical error detection

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
        header = [str(col).strip() for col in header if str(col).strip()]
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
                # Skip if column doesn't exist, is malformed, or is empty string
                if col not in df.columns or col == '' or not col:
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
    """Safely convert columns to numeric, preserving NaN values for proper data quality handling"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Keep NaN values - do NOT fill with 0 to prevent artificial score inflation
    return df

def check_data_completeness(row, inception_group):
    """Check if a fund has sufficient data for reliable scoring using configured requirements"""
    if inception_group not in MIN_DATA_REQUIREMENTS:
        return False
    
    requirements = MIN_DATA_REQUIREMENTS[inception_group]
    required_cols = requirements['required'] 
    max_missing = requirements['max_missing']
    
    missing_count = sum(1 for col in required_cols if pd.isna(row.get(col, np.nan)))
    return missing_count <= max_missing

def validate_fund_data_quality(df, inception_group):
    """Mark funds with insufficient data for scoring"""
    df = df.copy()
    df['has_sufficient_data'] = df.apply(lambda row: check_data_completeness(row, inception_group), axis=1)
    return df

def safe_score_calculation(df, components):
    """Calculate scores for ALL funds - use data quality for tier assignment only"""
    scores = pd.Series(0.0, index=df.index)
    
    # Calculate weighted sum for ALL funds - let tier assignment handle data quality
    for component, weight in components.items():
        if component in df.columns:
            # Use 0 for NaN in individual components for all funds
            component_scores = df[component].fillna(0) * weight
            scores = scores + component_scores
    
    return scores

# ─── DATA QUALITY ANALYSIS FUNCTIONS ─────────────────────────────────────────

def calculate_data_completeness_score(df, inception_group):
    """Calculate data completeness score for each fund (0-100%)"""
    if inception_group not in MIN_DATA_REQUIREMENTS:
        return pd.Series(0, index=df.index)
    
    required_cols = MIN_DATA_REQUIREMENTS[inception_group]['required']
    completeness_scores = []
    
    for _, row in df.iterrows():
        non_missing = sum(1 for col in required_cols if pd.notna(row.get(col, np.nan)))
        score = (non_missing / len(required_cols)) * 100
        completeness_scores.append(score)
    
    return pd.Series(completeness_scores, index=df.index)

def analyze_missing_data_by_column(df):
    """Analyze missing data patterns by column"""
    analysis = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df.columns:
            total_count = len(df)
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_count) * 100
            
            analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'non_missing_count': total_count - missing_count
            }
    
    return analysis

def identify_integrity_penalty_funds(df):
    """Identify funds flagged for integrity penalties"""
    flagged_funds = []
    
    # Check for high Sharpe ratios (suspicious)
    for col in ['Sharpe (1Y)', 'Sharpe (3Y)', 'Sharpe (5Y)']:
        if col in df.columns:
            high_sharpe = df[df[col] > INTEGRITY_PENALTY_THRESHOLD]
            for _, fund in high_sharpe.iterrows():
                flagged_funds.append({
                    'Ticker': fund.get('Ticker', 'N/A'),
                    'Fund': fund.get('Fund', 'N/A'),
                    'Reason': f'High {col}',
                    'Value': fund.get(col, np.nan),
                    'Threshold': INTEGRITY_PENALTY_THRESHOLD
                })
    
    return pd.DataFrame(flagged_funds)

def calculate_category_data_quality(df):
    """Calculate data quality metrics by category"""
    if 'Category' not in df.columns:
        return pd.DataFrame()
    
    category_metrics = []
    
    for category in df['Category'].dropna().unique():
        cat_df = df[df['Category'] == category]
        
        total_funds = len(cat_df)
        funds_with_data = cat_df.get('has_sufficient_data', True).sum() if 'has_sufficient_data' in cat_df.columns else total_funds
        valid_scores = (~pd.isna(cat_df['Score'])).sum()
        
        # Calculate average completeness score
        if 'data_completeness_score' in cat_df.columns:
            avg_completeness = cat_df['data_completeness_score'].mean()
        else:
            avg_completeness = 100  # Default if not calculated
        
        category_metrics.append({
            'Category': category,
            'Total_Funds': total_funds,
            'Funds_With_Sufficient_Data': funds_with_data if isinstance(funds_with_data, int) else total_funds,
            'Valid_Scores': valid_scores,
            'Data_Completeness_Pct': avg_completeness,
            'Data_Quality_Score': (funds_with_data / total_funds * 100) if total_funds > 0 else 0
        })
    
    return pd.DataFrame(category_metrics)

def track_tier_distribution(df):
    """Track tier distribution for monitoring changes"""
    tier_dist = df['Tier'].value_counts(normalize=True) * 100
    
    return {
        'Tier 1': tier_dist.get('Tier 1', 0),
        'Tier 2': tier_dist.get('Tier 2', 0), 
        'Tier 3': tier_dist.get('Tier 3', 0),
        'No Data': tier_dist.get('No Data', 0),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def generate_data_quality_alerts(df, previous_tier_dist=None):
    """Generate data quality monitoring alerts"""
    alerts = []
    
    # Current tier distribution
    current_tier_dist = track_tier_distribution(df)
    
    # Alert for no Tier 2 funds
    if current_tier_dist['Tier 2'] == 0:
        alerts.append({
            'level': 'error',
            'message': 'No Tier 2 funds detected - check scoring methodology',
            'details': f"Tier distribution: T1={current_tier_dist['Tier 1']:.1f}%, T2={current_tier_dist['Tier 2']:.1f}%, T3={current_tier_dist['Tier 3']:.1f}%"
        })
    
    # Alert for high missing data
    no_data_pct = current_tier_dist['No Data']
    if no_data_pct > 30:
        alerts.append({
            'level': 'warning',
            'message': f'High missing data: {no_data_pct:.1f}% of funds lack critical metrics',
            'details': 'Consider reviewing data collection processes'
        })
    
    # Alert for suspicious metrics
    integrity_flags = identify_integrity_penalty_funds(df)
    if not integrity_flags.empty:
        alerts.append({
            'level': 'warning',
            'message': f'{len(integrity_flags)} funds flagged for suspicious metrics',
            'details': f"Funds: {', '.join(integrity_flags['Ticker'].head(5).tolist())}"
        })
    
    # Tier distribution change alert (if previous data available)
    if previous_tier_dist:
        tier2_change = abs(current_tier_dist['Tier 2'] - previous_tier_dist.get('Tier 2', 0))
        if tier2_change > 10:  # More than 10% change
            alerts.append({
                'level': 'info',
                'message': f'Significant Tier 2 distribution change: {tier2_change:.1f}%',
                'details': f"Previous: {previous_tier_dist.get('Tier 2', 0):.1f}%, Current: {current_tier_dist['Tier 2']:.1f}%"
            })
    
    return alerts

def calculate_scores_1y(df: pd.DataFrame):
    df = df.copy()
    df = standardize_column_names(df)
    
    # Ensure numeric conversion for required columns
    numeric_cols = ['Total Return', 'Sharpe (1Y)', 'Sortino (1Y)', 'AUM', 'Net Expense', 'Std Dev (1Y)', 'VaR']
    df = safe_numeric_conversion(df, numeric_cols)
    
    # Validate data completeness BEFORE scoring
    df = validate_fund_data_quality(df, '1Y+')
    df['data_completeness_score'] = calculate_data_completeness_score(df, '1Y+')
    
    
    # Calculate Delta only for funds with valid Total Return data
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
    
    # Use safe score calculation that only scores funds with sufficient data
    score_components = {
        'sharpe_composite': METRIC_WEIGHTS['sharpe_composite'],
        'sortino_composite': METRIC_WEIGHTS['sortino_composite'], 
        'Delta': METRIC_WEIGHTS['delta'],
        'Total Return': METRIC_WEIGHTS['total_return'],
        'aum_score': METRIC_WEIGHTS['aum'],
        'expense_score': METRIC_WEIGHTS['expense']
    }
    df['Score'] = safe_score_calculation(df, score_components)
    
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
    
    # Validate data completeness BEFORE scoring
    df = validate_fund_data_quality(df, '3Y+')
    df['data_completeness_score'] = calculate_data_completeness_score(df, '3Y+')
    
    
    df['Delta'] = df['Total Return (3Y)'] - df.groupby("Category")['Total Return (3Y)'].transform("mean")   
    # Calculate composite scores only using available data - avoid NaN propagation
    df['sharpe_composite'] = df.apply(lambda row: 
        np.nanmean([row.get('Sharpe (3Y)', np.nan), row.get('Sharpe (1Y)', np.nan)]) 
        if not (pd.isna(row.get('Sharpe (3Y)', np.nan)) and pd.isna(row.get('Sharpe (1Y)', np.nan))) else np.nan, axis=1)
    df['sortino_composite'] = df.apply(lambda row: 
        np.nanmean([row.get('Sortino (3Y)', np.nan), row.get('Sortino (1Y)', np.nan)]) 
        if not (pd.isna(row.get('Sortino (3Y)', np.nan)) and pd.isna(row.get('Sortino (1Y)', np.nan))) else np.nan, axis=1)
    
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

    # Use safe score calculation that only scores funds with sufficient data
    score_components = {
        'sharpe_composite': METRIC_WEIGHTS['sharpe_composite'],
        'sortino_composite': METRIC_WEIGHTS['sortino_composite'], 
        'Delta': METRIC_WEIGHTS['delta'],
        'Total Return (3Y)': METRIC_WEIGHTS['total_return'],
        'aum_score': METRIC_WEIGHTS['aum'],
        'expense_score': METRIC_WEIGHTS['expense']
    }
    df['Score'] = safe_score_calculation(df, score_components)

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
    
    # Validate data completeness BEFORE scoring
    df = validate_fund_data_quality(df, '5Y+')
    df['data_completeness_score'] = calculate_data_completeness_score(df, '5Y+')
    
    
    df['Delta'] = df['Total Return (5Y)'] - df.groupby("Category")['Total Return (5Y)'].transform("mean")

    # Composite Sharpe + Sortino weighting with proper NaN handling
    def calc_weighted_composite(row, cols_weights):
        values = []
        weights = []
        for col, weight in cols_weights:
            if col in row.index and pd.notna(row[col]):
                values.append(row[col] * weight)
                weights.append(weight)
        if len(values) > 0:
            return sum(values) / sum(weights) * sum([w for c, w in cols_weights])  # Normalize back to original scale
        return np.nan
    
    sharpe_components = [('Sharpe (5Y)', 0.50), ('Sharpe (3Y)', 0.30), ('Sharpe (1Y)', 0.20)]
    sortino_components = [('Sortino (5Y)', 0.50), ('Sortino (3Y)', 0.30), ('Sortino (1Y)', 0.20)]
    
    df['sharpe_composite'] = df.apply(lambda row: calc_weighted_composite(row, sharpe_components), axis=1)
    df['sortino_composite'] = df.apply(lambda row: calc_weighted_composite(row, sortino_components), axis=1)

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
        
    # Use safe score calculation that only scores funds with sufficient data
    score_components = {
        'sharpe_composite': METRIC_WEIGHTS['sharpe_composite'],
        'sortino_composite': METRIC_WEIGHTS['sortino_composite'], 
        'Delta': METRIC_WEIGHTS['delta'],
        'Total Return (5Y)': METRIC_WEIGHTS['total_return'],
        'aum_score': METRIC_WEIGHTS['aum'],
        'expense_score': METRIC_WEIGHTS['expense']
    }
    df['Score'] = safe_score_calculation(df, score_components)

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
    """Assign tiers with data completeness validation and dynamic thresholds"""
    
    def get_dynamic_thresholds(valid_scores):
        """Calculate dynamic tier thresholds based on score distribution"""
        if DYNAMIC_TIER_THRESHOLDS and len(valid_scores) > 0:
            tier1_threshold = valid_scores.quantile(DEFAULT_TIER_THRESHOLDS['tier1_percentile'])
            tier2_threshold = valid_scores.quantile(DEFAULT_TIER_THRESHOLDS['tier2_percentile'])
            
            # Apply minimum thresholds
            tier1_threshold = max(tier1_threshold, DEFAULT_TIER_THRESHOLDS['min_tier1_score'])
            tier2_threshold = max(tier2_threshold, DEFAULT_TIER_THRESHOLDS['min_tier2_score'])
            
            # Ensure separation
            if tier1_threshold <= tier2_threshold:
                tier1_threshold = tier2_threshold + 0.5
                
            return tier1_threshold, tier2_threshold
        else:
            return 8.5, 6.0  # Static fallback thresholds
    
    # Calculate dynamic thresholds from all valid scores (ignore data sufficiency for scoring)
    valid_scores = df[~pd.isna(df['Score'])]['Score']
    tier1_thresh, tier2_thresh = get_dynamic_thresholds(valid_scores)
    
    def tier(row):
        score = row['Score']
        
        # Only mark as 'No Data' if score is truly missing (NaN)
        # Allow funds with any valid score to be tiered
        if pd.isna(score):
            return "No Data"
            
        # Use dynamic thresholds
        if score >= tier1_thresh: return "Tier 1"
        if score >= tier2_thresh: return "Tier 2"
        return "Tier 3"
    
    df['Tier'] = df.apply(tier, axis=1)
    df['tier1_threshold'] = tier1_thresh
    df['tier2_threshold'] = tier2_thresh
    
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

# ─── ENHANCED EXPORT FUNCTIONS ────────────────────────────────────────────

def clean_dataframe_for_export(df: pd.DataFrame, inception_group: str = None) -> pd.DataFrame:
    """
    Clean dataframe for export by removing unnecessary columns and filtering based on inception group
    
    Args:
        df: The dataframe to clean
        inception_group: Specific inception group to filter for ('1Y+', '3Y+', '5Y+', or None for all)
    
    Returns:
        Cleaned dataframe ready for export
    """
    df_clean = df.copy()
    
    # Remove unnamed columns
    unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in str(col)]
    df_clean = df_clean.drop(columns=unnamed_cols)
    
    # Remove columns with >90% missing values
    threshold = len(df_clean) * 0.1  # At least 10% non-null values required
    df_clean = df_clean.dropna(axis=1, thresh=threshold)
    
    # Define column sets for each inception group
    base_columns = ['Ticker', 'Fund', 'Category', 'Score', 'Tier', 'Inception Group', 'Inception Date', 'AUM', 'Net Expense']
    
    column_sets = {
        '1Y+': base_columns + [
            'Total Return', 'Sharpe (1Y)', 'Sortino (1Y)', 'Std Dev (1Y)', 'VaR', 'Delta',
            'sharpe_composite', 'sortino_composite', 'aum_score', 'expense_score'
        ],
        '3Y+': base_columns + [
            'Total Return (3Y)', 'Sharpe (3Y)', 'Sortino (3Y)', 'Sharpe (1Y)', 'Sortino (1Y)',
            'Std Dev (3Y)', '2022 Return', 'Delta', 'sharpe_composite', 'sortino_composite',
            'aum_score', 'expense_score'
        ],
        '5Y+': base_columns + [
            'Total Return (5Y)', 'Sharpe (5Y)', 'Sortino (5Y)', 'Sharpe (3Y)', 'Sortino (3Y)',
            'Sharpe (1Y)', 'Sortino (1Y)', 'Std Dev (5Y)', '2022 Return', 'Delta',
            'sharpe_composite', 'sortino_composite', 'aum_score', 'expense_score'
        ]
    }
    
    # Filter columns based on inception group
    if inception_group and inception_group in column_sets:
        # Keep only columns that exist in the dataframe and are relevant to the inception group
        relevant_cols = [col for col in column_sets[inception_group] if col in df_clean.columns]
        df_clean = df_clean[relevant_cols]
    
    # Reorder columns for better readability
    priority_cols = ['Ticker', 'Fund', 'Category', 'Score', 'Tier', 'Inception Group']
    other_cols = [col for col in df_clean.columns if col not in priority_cols]
    
    # Arrange columns with priority columns first
    final_column_order = []
    for col in priority_cols:
        if col in df_clean.columns:
            final_column_order.append(col)
    
    final_column_order.extend(other_cols)
    df_clean = df_clean[final_column_order]
    
    # Round numeric columns to reasonable precision
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_columns] = df_clean[numeric_columns].round(4)
    
    return df_clean

def get_export_filename(inception_group: str = None, include_timestamp: bool = True, prefix: str = None) -> str:
    """Generate appropriate filename for export"""
    if prefix:
        base_name = prefix
    elif inception_group:
        base_name = f"funds_{inception_group.replace('+', 'plus')}"
    else:
        base_name = "all_funds"
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{base_name}_{timestamp}.csv"
    
    return f"{base_name}.csv"

@st.cache_data
def convert_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes for download"""
    return df.to_csv(index=False).encode("utf-8")

def create_enhanced_download_section(df_quality_filtered: pd.DataFrame, filtered_df: pd.DataFrame, selected_inceptions: list):
    """Create enhanced download section with data quality export options"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Options")
    
    # Add caption showing filtered data info
    fund_count = len(filtered_df)
    quality_count = len(df_quality_filtered)
    if not filtered_df.empty and 'Inception Date' in filtered_df.columns:
        latest_inception = filtered_df['Inception Date'].max()
        if pd.notna(latest_inception):
            st.sidebar.caption(f"📊 {fund_count} funds (from {quality_count} quality-filtered) | Latest: {latest_inception.strftime('%Y-%m-%d')}")
        else:
            st.sidebar.caption(f"📊 {fund_count} funds (from {quality_count} quality-filtered)")
    else:
        st.sidebar.caption(f"📊 {fund_count} funds (from {quality_count} quality-filtered)")
    
    # Option 1: Download filtered fund data
    if not filtered_df.empty:
        cleaned_filtered = clean_dataframe_for_export(filtered_df)
        st.sidebar.download_button(
            "📋 Download Filtered Funds",
            data=convert_to_csv(cleaned_filtered),
            file_name=get_export_filename(),
            mime="text/csv",
            help="Download currently filtered fund data with all metrics"
        )
    
    # Option 2: Download data quality report
    if not df_quality_filtered.empty:
        quality_report = create_data_quality_export(df_quality_filtered)
        st.sidebar.download_button(
            "🛡️ Download Data Quality Report",
            data=convert_to_csv(quality_report),
            file_name=get_export_filename(prefix="data_quality_report"),
            mime="text/csv",
            help="Download comprehensive data quality analysis with fund-level completeness details"
        )
    
    # Option 3: Download by individual inception group
    st.sidebar.markdown("**📂 Download by Inception Group:**")
    
    for inception in ['1Y+', '3Y+', '5Y+']:
        group_data = df_quality_filtered[df_quality_filtered['Inception Group'] == inception]
        
        if not group_data.empty:
            cleaned_group = clean_dataframe_for_export(group_data, inception_group=inception)
            fund_count_group = len(cleaned_group)
            
            st.sidebar.download_button(
                f"📈 {inception} Funds ({fund_count_group})",
                data=convert_to_csv(cleaned_group),
                file_name=get_export_filename(inception_group=inception),
                mime="text/csv",
                help=f"Download all {inception} inception funds with relevant metrics only"
            )

def create_download_section(df_tiered: pd.DataFrame, filtered_df: pd.DataFrame, selected_inceptions: list):
    """Legacy download section - keeping for compatibility"""
    return create_enhanced_download_section(df_tiered, filtered_df, selected_inceptions)

# ─── DATA QUALITY DASHBOARD FUNCTIONS ────────────────────────────────────────────

def apply_data_quality_filters(df, min_completeness=0, exclude_integrity=False, exclude_missing_critical=False, complete_aum_expense_only=False):
    """Apply data quality filters to dataframe"""
    filtered_df = df.copy()
    
    # Filter by minimum data completeness
    if 'data_completeness_score' in filtered_df.columns and min_completeness > 0:
        filtered_df = filtered_df[filtered_df['data_completeness_score'] >= min_completeness]
    
    # Exclude funds with integrity penalties
    if exclude_integrity:
        integrity_flags = identify_integrity_penalty_funds(filtered_df)
        if not integrity_flags.empty:
            flagged_tickers = integrity_flags['Ticker'].tolist()
            filtered_df = filtered_df[~filtered_df['Ticker'].isin(flagged_tickers)]
    
    # Exclude funds with missing critical metrics
    if exclude_missing_critical and 'has_sufficient_data' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['has_sufficient_data'] == True]
    
    # Show only funds with complete AUM/expense data
    if complete_aum_expense_only:
        filtered_df = filtered_df[
            (~pd.isna(filtered_df.get('AUM', np.nan))) & 
            (~pd.isna(filtered_df.get('Net Expense', np.nan)))
        ]
    
    return filtered_df

def create_data_quality_export(df):
    """Create comprehensive data quality report for export"""
    export_data = []
    
    for _, fund in df.iterrows():
        completeness_score = fund.get('data_completeness_score', 100)
        has_sufficient_data = fund.get('has_sufficient_data', True)
        
        export_data.append({
            'Ticker': fund.get('Ticker', 'N/A'),
            'Fund': fund.get('Fund', 'N/A'),
            'Category': fund.get('Category', 'N/A'),
            'Inception_Group': fund.get('Inception Group', 'N/A'),
            'Data_Completeness_Score': completeness_score,
            'Has_Sufficient_Data': has_sufficient_data,
            'Tier': fund.get('Tier', 'N/A'),
            'Total_Return_Missing': pd.isna(fund.get('Total Return', np.nan)),
            'Sharpe_1Y_Missing': pd.isna(fund.get('Sharpe (1Y)', np.nan)),
            'Sortino_1Y_Missing': pd.isna(fund.get('Sortino (1Y)', np.nan)),
            'AUM_Missing': pd.isna(fund.get('AUM', np.nan)),
            'Expense_Missing': pd.isna(fund.get('Net Expense', np.nan)),
            'Final_Score': fund.get('Score', np.nan)
        })
    
    return pd.DataFrame(export_data)

# ─── NEW TAB FUNCTIONS ────────────────────────────────────────────

def create_main_rankings_tab(df_tiered):
    """Main Rankings tab with enhanced filters and fund listings"""
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Semi-Liquid Alternatives Fund Selection Dashboard</h1>
        <p>Advanced quantitative analysis and performance scoring for alternative investments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar filters
    with st.sidebar:
        # Version indicator
        st.markdown("---")
        st.markdown("**🚀 Phase 2 - Enhanced Analytics**")
        st.caption("✨ Advanced scoring, comparisons, and interactive features")
        st.markdown("---")
        
        st.markdown("### 🔍 **Smart Filters**")
        st.caption("🎯 Use filters to find funds matching your criteria")
        
        
        # Score range filter with enhanced help
        if not df_tiered[~pd.isna(df_tiered['Score'])].empty:
            score_min = float(df_tiered['Score'].min())
            score_max = float(df_tiered['Score'].max())
            score_range = st.slider(
                "📊 Composite Score Range",
                min_value=score_min,
                max_value=score_max,
                value=(score_min, score_max),
                step=0.1,
                help="Filter funds by their composite performance score. Higher scores indicate better overall performance across all metrics."
            )
        else:
            score_range = None
        
        # AUM range filter with enhanced help
        if 'AUM' in df_tiered.columns and not df_tiered['AUM'].isna().all():
            aum_values = df_tiered['AUM'].dropna()
            if not aum_values.empty:
                aum_min = float(aum_values.min())
                aum_max = float(aum_values.max())
                aum_range = st.slider(
                    "💰 AUM Range (Millions)",
                    min_value=aum_min,
                    max_value=aum_max,
                    value=(aum_min, aum_max),
                    step=10.0,
                    help="Filter by Assets Under Management. Larger funds may offer more stability but smaller funds might be more agile."
                )
            else:
                aum_range = None
        else:
            aum_range = None
        
        # Expense ratio filter with enhanced help
        if 'Net Expense' in df_tiered.columns and not df_tiered['Net Expense'].isna().all():
            expense_values = df_tiered['Net Expense'].dropna()
            if not expense_values.empty:
                expense_min = float(expense_values.min())
                expense_max = float(expense_values.max())
                expense_range = st.slider(
                    "💸 Expense Ratio Range (%)",
                    min_value=expense_min,
                    max_value=expense_max,
                    value=(expense_min, expense_max),
                    step=0.01,
                    help="Filter by annual expense ratio. Lower expenses mean more returns stay in your pocket."
                )
            else:
                expense_range = None
        else:
            expense_range = None
        
        # Fund search with enhanced help
        fund_options = ["All Funds"] + sorted(df_tiered['Ticker'].dropna().unique().tolist())
        selected_fund = st.selectbox(
            "🔍 Search Specific Fund:", 
            fund_options,
            help="Search for a specific fund by ticker symbol to view detailed analysis"
        )
        
        # Standard filters with enhanced help
        st.markdown("---")
        inception_opts = st.multiselect(
            "📅 Inception Groups:", 
            ["1Y+", "3Y+", "5Y+"], 
            default=["1Y+", "3Y+", "5Y+"],
            help="Filter by fund age. Older funds have longer track records but younger funds may be more innovative."
        )
        tier_opts = st.multiselect(
            "🏆 Performance Tiers:", 
            ["Tier 1", "Tier 2", "Tier 3", "No Data"], 
            default=["Tier 1", "Tier 2", "Tier 3", "No Data"],
            help="Filter by performance tier. Tier 1 = Best, Tier 2 = Good, Tier 3 = Below Average"
        )
        # Handle categories including potential NaN values
        all_categories = df_tiered['Category'].unique()
        # Remove NaN from the list but keep track of it
        categories_for_display = [cat for cat in all_categories if pd.notna(cat)]
        categories_for_display = sorted(categories_for_display)
        
        
        category_opts = st.multiselect(
            "📂 Fund Categories:", 
            categories_for_display, 
            default=categories_for_display,
            help="Filter by investment category/strategy. Select specific categories to focus your analysis."
        )
        
        # Display options
        st.markdown("---")
        st.markdown("**📊 Display Options:**")
        show_all_rows = st.checkbox(
            "Show all funds in Complete Rankings", 
            value=True, 
            help="Uncheck to limit to top 50 funds for faster loading on slower connections"
        )
        
        # Scoring Methodology Explanation
        st.markdown("---")
        with st.expander("📊 **Scoring Methodology**", expanded=False):
            st.markdown("### 📈 **Metric Weights**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Core Performance:**
                - 📊 Total Return: **{METRIC_WEIGHTS['total_return']:.0%}**
                - ⚡ Sharpe Ratio: **{METRIC_WEIGHTS['sharpe_composite']:.0%}**
                - 🛡️ Sortino Ratio: **{METRIC_WEIGHTS['sortino_composite']:.0%}**
                """)
            
            with col2:
                st.markdown(f"""
                **Additional Factors:**
                - 📈 Category Delta: **{METRIC_WEIGHTS['delta']:.0%}**
                - 💰 AUM Scale: **{METRIC_WEIGHTS['aum']:.1%}**
                - 💸 Expense Efficiency: **{METRIC_WEIGHTS['expense']:.1%}**
                """)
            
            st.markdown("### ⚠️ **Penalty System**")
            st.markdown(f"""
            - 🚩 **Integrity Penalty:** -{INTEGRITY_PENALTY_AMOUNT} for Sharpe > {INTEGRITY_PENALTY_THRESHOLD}
            - 📉 **High Volatility:** -0.1 for top quartile standard deviation
            - 📉 **Poor 2022 Performance:** -0.1 for bottom quartile 2022 returns
            """)
            
            st.markdown("### 🎯 **Dynamic Tier Thresholds**")
            if 'tier1_threshold' in df_tiered.columns and 'tier2_threshold' in df_tiered.columns:
                tier1_thresh = df_tiered['tier1_threshold'].iloc[0]
                tier2_thresh = df_tiered['tier2_threshold'].iloc[0]
                st.markdown(f"""
                - 🥇 **Tier 1:** Score ≥ {tier1_thresh:.2f} (Top {DEFAULT_TIER_THRESHOLDS['tier1_percentile']:.0%})
                - 🥈 **Tier 2:** Score ≥ {tier2_thresh:.2f} (Top {DEFAULT_TIER_THRESHOLDS['tier2_percentile']:.0%})
                - 🥉 **Tier 3:** Score < {tier2_thresh:.2f}
                """)
            
            st.markdown("### 🧮 **Composite Score Formula**")
            st.code("""
            Score = (Total_Return × 0.30) + 
                   (Sharpe_Composite × 0.25) + 
                   (Sortino_Composite × 0.20) + 
                   (Category_Delta × 0.10) + 
                   (AUM_Score × 0.075) + 
                   (Expense_Score × 0.075) 
                   - Penalties
            """)
        
        # Fund Comparison Tool
        st.markdown("---")
        with st.expander("🔍 **Fund Comparison Tool**", expanded=False):
            st.markdown("### Compare Funds Side-by-Side")
            
            # Get available funds for comparison
            available_funds = df_tiered['Ticker'].dropna().unique().tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                fund1 = st.selectbox("Select Fund 1:", ["None"] + available_funds, key="fund1")
            with col2:
                fund2 = st.selectbox("Select Fund 2:", ["None"] + available_funds, key="fund2")
            
            # Optional third fund
            fund3 = st.selectbox("Select Fund 3 (Optional):", ["None"] + available_funds, key="fund3")
            
            # Perform comparison if funds are selected
            selected_funds = [f for f in [fund1, fund2, fund3] if f != "None"]
            
            if len(selected_funds) >= 2:
                comparison_data = []
                
                for ticker in selected_funds:
                    fund_data = df_tiered[df_tiered['Ticker'] == ticker].iloc[0]
                    comparison_data.append({
                        'Ticker': ticker,
                        'Fund Name': fund_data.get('Fund', 'N/A'),
                        'Score': f"{fund_data.get('Score', 0):.2f}",
                        'Tier': fund_data.get('Tier', 'N/A'),
                        'Category': fund_data.get('Category', 'N/A'),
                        'Total Return': f"{fund_data.get('Total Return', 0):.2f}%" if pd.notna(fund_data.get('Total Return')) else 'N/A',
                        'Sharpe (1Y)': f"{fund_data.get('Sharpe (1Y)', 0):.2f}" if pd.notna(fund_data.get('Sharpe (1Y)')) else 'N/A',
                        'Sortino (1Y)': f"{fund_data.get('Sortino (1Y)', 0):.2f}" if pd.notna(fund_data.get('Sortino (1Y)')) else 'N/A',
                        'AUM ($M)': f"{fund_data.get('AUM', 0):.0f}" if pd.notna(fund_data.get('AUM')) else 'N/A',
                        'Expense Ratio': f"{fund_data.get('Net Expense', 0):.2f}%" if pd.notna(fund_data.get('Net Expense')) else 'N/A',
                        'Inception Group': fund_data.get('Inception Group', 'N/A')
                    })
                
                comparison_df = pd.DataFrame(comparison_data).T
                comparison_df.columns = [f"Fund {i+1}" for i in range(len(selected_funds))]
                
                st.markdown("#### 📊 **Comparison Table**")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Score comparison chart
                if len(selected_funds) >= 2:
                    st.markdown("#### 📈 **Score Comparison**")
                    scores = [df_tiered[df_tiered['Ticker'] == ticker]['Score'].iloc[0] for ticker in selected_funds]
                    
                    fig_comparison = px.bar(
                        x=selected_funds,
                        y=scores,
                        title="Fund Score Comparison",
                        labels={'x': 'Fund', 'y': 'Composite Score'},
                        color=scores,
                        color_continuous_scale='RdYlGn'
                    )
                    fig_comparison.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Phase 3 Roadmap Preview
        st.markdown("---")
        with st.expander("🚀 **Phase 3 Roadmap - Coming Soon**", expanded=False):
            st.markdown("### 🎯 **Advanced Features in Development**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Performance Attribution:**
                • Factor-based return decomposition
                • Style drift analysis
                • Benchmark relative performance
                • Risk-adjusted attribution
                
                **🎯 Risk-Return Optimization:**
                • Modern Portfolio Theory integration
                • Efficient frontier visualization
                • Monte Carlo simulations
                • VaR/CVaR calculations
                """)
            
            with col2:
                st.markdown("""
                **🔗 Correlation Analysis:**
                • Cross-fund correlation matrices
                • Market regime analysis
                • Diversification benefit scoring
                • Dynamic correlation tracking
                
                **🏗️ Custom Portfolio Builder:**
                • Drag-and-drop portfolio construction
                • Weight optimization tools
                • Constraint-based allocation
                • Real-time portfolio analytics
                """)
            
            st.info("💡 **Next Update:** Phase 3 will transform this into a comprehensive portfolio construction and risk management platform!")
        
    
    # Apply filters
    filtered_df = df_tiered.copy()
    
    # Apply score range filter
    if score_range and not df_tiered[~pd.isna(df_tiered['Score'])].empty:
        valid_scores = ~pd.isna(filtered_df['Score'])
        score_filtered = filtered_df[valid_scores & (filtered_df['Score'] >= score_range[0]) & (filtered_df['Score'] <= score_range[1])]
        
        # Include funds with NaN scores if the range covers the full spectrum
        score_values = filtered_df['Score'].dropna()
        if not score_values.empty and score_range[0] <= score_values.min() and score_range[1] >= score_values.max():
            nan_score_funds = filtered_df[filtered_df['Score'].isna()]
            filtered_df = pd.concat([score_filtered, nan_score_funds])
        else:
            filtered_df = score_filtered
    
    # Apply AUM range filter
    if aum_range and 'AUM' in filtered_df.columns:
        valid_aum = ~pd.isna(filtered_df['AUM'])
        aum_filtered = filtered_df[valid_aum & (filtered_df['AUM'] >= aum_range[0]) & (filtered_df['AUM'] <= aum_range[1])]
        
        # Include funds with NaN AUM if the range covers the full spectrum
        aum_values = filtered_df['AUM'].dropna()
        if not aum_values.empty and aum_range[0] <= aum_values.min() and aum_range[1] >= aum_values.max():
            nan_aum_funds = filtered_df[filtered_df['AUM'].isna()]
            filtered_df = pd.concat([aum_filtered, nan_aum_funds])
        else:
            filtered_df = aum_filtered
    
    # Apply expense range filter
    if expense_range and 'Net Expense' in filtered_df.columns:
        valid_expense = ~pd.isna(filtered_df['Net Expense'])
        expense_filtered = filtered_df[valid_expense & (filtered_df['Net Expense'] >= expense_range[0]) & (filtered_df['Net Expense'] <= expense_range[1])]
        
        # Include funds with NaN expense if the range covers the full spectrum
        expense_values = filtered_df['Net Expense'].dropna()
        if not expense_values.empty and expense_range[0] <= expense_values.min() and expense_range[1] >= expense_values.max():
            nan_expense_funds = filtered_df[filtered_df['Net Expense'].isna()]
            filtered_df = pd.concat([expense_filtered, nan_expense_funds])
        else:
            filtered_df = expense_filtered
    
    # Apply fund search filter
    if selected_fund != "All Funds":
        filtered_df = filtered_df[filtered_df['Ticker'] == selected_fund]
    
    # Apply standard filters
    filtered_df = filtered_df[filtered_df['Inception Group'].isin(inception_opts)]
    filtered_df = filtered_df[filtered_df['Tier'].isin(tier_opts)]
    
    # Apply category filter - include NaN categories when all categories are selected
    if len(category_opts) == len(categories_for_display):
        filtered_df = filtered_df[(filtered_df['Category'].isin(category_opts)) | (filtered_df['Category'].isna())]
    else:
        filtered_df = filtered_df[filtered_df['Category'].isin(category_opts)]
    
    # Enhanced Key metrics cards with trend indicators
    col1, col2, col3, col4 = st.columns(4)
    
    total_funds = len(filtered_df)
    tier1_count = len(filtered_df[filtered_df['Tier'] == 'Tier 1'])
    tier2_count = len(filtered_df[filtered_df['Tier'] == 'Tier 2'])
    tier3_count = len(filtered_df[filtered_df['Tier'] == 'Tier 3'])
    avg_score = filtered_df['Score'].mean() if not filtered_df['Score'].isna().all() else 0
    
    # Calculate quality metrics for delta indicators
    tier1_pct = (tier1_count / total_funds * 100) if total_funds > 0 else 0
    tier2_pct = (tier2_count / total_funds * 100) if total_funds > 0 else 0
    
    with col1:
        st.metric(
            "📊 Total Funds", 
            value=f"{total_funds}",
            help="Total number of funds matching current filters"
        )
    with col2:
        st.metric(
            "🥇 Tier 1 Funds", 
            value=f"{tier1_count}",
            delta=f"{tier1_pct:.1f}% of total",
            help="Top-tier funds with exceptional performance"
        )
    with col3:
        st.metric(
            "🥈 Tier 2 Funds", 
            value=f"{tier2_count}",
            delta=f"{tier2_pct:.1f}% of total",
            help="High-quality funds with strong performance"
        )
    with col4:
        # Add performance indicator for average score
        score_quality = "Excellent" if avg_score > 8 else "Good" if avg_score > 6 else "Moderate" if avg_score > 4 else "Below Average"
        st.metric(
            "🎯 Average Score", 
            value=f"{avg_score:.2f}",
            delta=f"{score_quality}",
            help="Mean composite score of filtered funds"
        )
    
    st.markdown("---")
    
    # Top funds display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥇 Top 10 Overall")
        if not filtered_df.empty:
            top10 = filtered_df.sort_values("Score", ascending=False).head(10)
            display_cols = ["Ticker", "Fund", "Category", "Score", "Tier", "Inception Group"]
            st.dataframe(style_tiers(top10[display_cols]), use_container_width=True)
        else:
            st.info("No funds match the current filters")
    
    with col2:
        st.subheader("🏅 Top 5 by Inception Group")
        for group in ["1Y+", "3Y+", "5Y+"]:
            if group in filtered_df['Inception Group'].values:
                st.markdown(f"**📈 {group} Funds**")
                group_data = filtered_df[filtered_df['Inception Group'] == group].sort_values("Score", ascending=False).head(5)
                if not group_data.empty:
                    display_cols = ["Ticker", "Fund", "Score", "Tier"]
                    st.dataframe(style_tiers(group_data[display_cols]), use_container_width=True)
    
    # Score Breakdown for Top Performers
    if not filtered_df.empty:
        st.markdown("---")
        st.subheader("🧮 Score Breakdown Analysis")
        
        # Get top 5 funds for breakdown
        top_funds = filtered_df.sort_values("Score", ascending=False).head(5)
        
        # Create score breakdown data
        breakdown_data = []
        for _, fund in top_funds.iterrows():
            # Calculate component contributions
            total_return_contrib = fund.get('Total Return', 0) * METRIC_WEIGHTS['total_return'] if pd.notna(fund.get('Total Return')) else 0
            sharpe_contrib = fund.get('sharpe_composite', 0) * METRIC_WEIGHTS['sharpe_composite'] if pd.notna(fund.get('sharpe_composite')) else 0
            sortino_contrib = fund.get('sortino_composite', 0) * METRIC_WEIGHTS['sortino_composite'] if pd.notna(fund.get('sortino_composite')) else 0
            delta_contrib = fund.get('Delta', 0) * METRIC_WEIGHTS['delta'] if pd.notna(fund.get('Delta')) else 0
            aum_contrib = fund.get('aum_score', 0) * METRIC_WEIGHTS['aum'] if pd.notna(fund.get('aum_score')) else 0
            expense_contrib = fund.get('expense_score', 0) * METRIC_WEIGHTS['expense'] if pd.notna(fund.get('expense_score')) else 0
            
            breakdown_data.append({
                'Fund': f"{fund['Ticker']} ({fund['Tier']})",
                'Total Return': total_return_contrib,
                'Sharpe Ratio': sharpe_contrib,
                'Sortino Ratio': sortino_contrib,
                'Category Delta': delta_contrib,
                'AUM Score': aum_contrib,
                'Expense Score': expense_contrib,
                'Total Score': fund['Score']
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Component Contributions")
            
            # Create stacked bar chart
            components = ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 'Category Delta', 'AUM Score', 'Expense Score']
            
            fig_breakdown = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            for i, component in enumerate(components):
                fig_breakdown.add_trace(go.Bar(
                    name=component,
                    x=breakdown_df['Fund'],
                    y=breakdown_df[component],
                    marker_color=colors[i % len(colors)]
                ))
            
            fig_breakdown.update_layout(
                barmode='stack',
                title="Score Component Breakdown - Top 5 Funds",
                xaxis_title="Fund (Tier)",
                yaxis_title="Score Contribution",
                height=400,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Key Insights")
            
            # Find the fund with highest total return contribution
            best_return = breakdown_df.loc[breakdown_df['Total Return'].idxmax()]
            best_risk_adj = breakdown_df.loc[breakdown_df['Sharpe Ratio'].idxmax()]
            
            st.markdown(f"""
            **🏆 Best Total Return:**  
            {best_return['Fund']}  
            Contribution: {best_return['Total Return']:.3f}
            
            **⚡ Best Risk-Adjusted:**  
            {best_risk_adj['Fund']}  
            Sharpe Contribution: {best_risk_adj['Sharpe Ratio']:.3f}
            
            **💡 Component Analysis:**
            • Total Return drives {METRIC_WEIGHTS['total_return']:.0%} of score
            • Risk metrics contribute {METRIC_WEIGHTS['sharpe_composite'] + METRIC_WEIGHTS['sortino_composite']:.0%}
            • Efficiency factors add {METRIC_WEIGHTS['aum'] + METRIC_WEIGHTS['expense']:.0%}
            """)
            
            # Score distribution info
            avg_score = top_funds['Score'].mean()
            score_range = top_funds['Score'].max() - top_funds['Score'].min()
            
            st.markdown(f"""
            **📈 Top 5 Statistics:**
            • Average Score: {avg_score:.2f}
            • Score Range: {score_range:.2f}
            """)
            
            # Performance badges
            top_fund = top_funds.iloc[0]
            badges = []
            if pd.notna(top_fund.get('Total Return')) and top_fund['Total Return'] > 10:
                badges.append("🔥 High Return")
            if pd.notna(top_fund.get('Sharpe (1Y)')) and top_fund['Sharpe (1Y)'] > 1.5:
                badges.append("⚡ Excellent Sharpe")
            if pd.notna(top_fund.get('Net Expense')) and top_fund['Net Expense'] < 0.5:
                badges.append("💎 Low Cost")
            
            if badges:
                st.markdown("**🏅 Top Fund Badges:**")
                for badge in badges:
                    st.markdown(f"• {badge}")
    
    # Full rankings table
    st.subheader("📊 Complete Rankings")
    if not filtered_df.empty:
        # Sort the data first to ensure we're not losing anything in the sort operation
        sorted_df = filtered_df.sort_values("Score", ascending=False)
        display_columns = ["Ticker", "Fund", "Category", "Score", "Tier", "Inception Group", "AUM", "Net Expense"]
        
        # Apply display limit if requested
        if show_all_rows:
            display_df = sorted_df
        else:
            display_df = sorted_df.head(50)
        
        # Display with sufficient height to show all rows
        table_height = min(600, len(display_df) * 35 + 50) if show_all_rows else 400
        st.dataframe(
            style_tiers(display_df[display_columns]), 
            use_container_width=True,
            height=table_height
        )
        
        if not show_all_rows and len(sorted_df) > 50:
            st.info(f"💡 Showing top 50 funds. Check 'Show all funds' in sidebar to see all {len(sorted_df)} funds.")
    else:
        st.info("No funds match the current filters")
    
    # Add download functionality
    create_enhanced_download_section(df_tiered, filtered_df, inception_opts)

def create_analytics_deep_dive_tab(df_tiered):
    """Analytics Deep Dive tab with interactive charts and analysis"""
    st.title("📈 Analytics Deep Dive")
    
    # Interactive charts section
    st.subheader("📊 Interactive Analysis")
    
    # Create two columns for chart layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive heatmap
        st.markdown("**Score Heatmap: Category vs Inception Group**")
        if not df_tiered.empty and 'Category' in df_tiered.columns and 'Inception Group' in df_tiered.columns:
            heatmap_data = df_tiered.groupby(["Category", "Inception Group"])["Score"].mean().unstack(fill_value=0)
            
            if not heatmap_data.empty:
                fig_heatmap = px.imshow(
                    heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='RdYlBu_r',
                    title="Average Score Heatmap",
                    labels={'x': 'Inception Group', 'y': 'Category', 'color': 'Avg Score'},
                    text_auto='.2f'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No data available for heatmap")
        else:
            st.info("Insufficient data for heatmap")
    
    with col2:
        # Interactive bar chart for average scores
        st.markdown("**Average Scores by Inception Group**")
        if not df_tiered.empty and 'Inception Group' in df_tiered.columns:
            avg_scores = df_tiered.groupby("Inception Group")["Score"].agg(['mean', 'count', 'std']).reset_index()
            avg_scores.columns = ['Inception Group', 'Average Score', 'Fund Count', 'Std Dev']
            
            fig_bar = px.bar(
                avg_scores,
                x='Inception Group',
                y='Average Score',
                title="Average Scores by Group",
                text='Average Score',
                hover_data=['Fund Count', 'Std Dev']
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data available for bar chart")
    
    # Risk-Return scatter plot
    st.subheader("📈 Risk-Return Analysis")
    
    if not df_tiered.empty and 'Total Return' in df_tiered.columns:
        # Determine which Sharpe column to use
        sharpe_col = None
        for col in ['Sharpe (5Y)', 'Sharpe (3Y)', 'Sharpe (1Y)']:
            if col in df_tiered.columns and not df_tiered[col].isna().all():
                sharpe_col = col
                break
        
        if sharpe_col:
            scatter_df = df_tiered[~pd.isna(df_tiered['Total Return']) & ~pd.isna(df_tiered[sharpe_col])].copy()
            
            if not scatter_df.empty:
                fig_scatter = px.scatter(
                    scatter_df,
                    x=sharpe_col,
                    y='Total Return',
                    color='Tier',
                    size='AUM',
                    hover_name='Ticker',
                    hover_data=['Fund', 'Category', 'Score'],
                    title=f"Risk-Return Analysis: Total Return vs {sharpe_col}",
                    labels={sharpe_col: f'{sharpe_col} (Risk-Adjusted Return)', 'Total Return': 'Total Return (%)'}
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Insufficient data for risk-return analysis")
        else:
            st.info("No Sharpe ratio data available for risk-return analysis")
    else:
        st.info("No Total Return data available")
    
    # Category analysis
    st.subheader("📂 Category Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Category' in df_tiered.columns:
            category_stats = df_tiered.groupby('Category').agg({
                'Score': ['count', 'mean', 'std'],
                'Total Return': 'mean',
                'AUM': 'mean'
            }).round(2)
            category_stats.columns = ['Fund Count', 'Avg Score', 'Score Std', 'Avg Return', 'Avg AUM']
            st.dataframe(category_stats, use_container_width=True)
    
    with col2:
        st.markdown("**Tier Distribution by Category (%)**")
    
    # Build the crosstab (wide %)
    tier_by_category = (
        pd.crosstab(df_tiered['Category'], df_tiered['Tier'], normalize='index')
          .mul(100)
    )
    
    # 2) Melt into long form
    df_long = (
        tier_by_category
          .reset_index()
          .melt(
             id_vars='Category',
             var_name='Tier',
             value_name='Pct'
          )
    )
    
    # 3) Plot with error handling
    try:
        fig_stacked = px.bar(
            df_long,
            x='Category',
            y='Pct',
            color='Tier',
            barmode='stack',
            title="Tier Distribution by Category (%)"
        )
        
        # 4) Render in bottom‑right (inside col2)
        st.plotly_chart(fig_stacked, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating tier distribution chart: {str(e)}")
        st.info("Chart creation failed - check data availability")

def create_basic_data_quality_tab(df_tiered):
    """Basic Data Quality tab with essential metrics only"""
    st.title("🛡️ Data Quality Overview")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_funds = len(df_tiered)
    funds_with_data = df_tiered.get('has_sufficient_data', True).sum() if 'has_sufficient_data' in df_tiered.columns else total_funds
    valid_scores = (~pd.isna(df_tiered['Score'])).sum()
    avg_completeness = df_tiered.get('data_completeness_score', pd.Series([100]*len(df_tiered))).mean()
    
    with col1:
        st.metric("Total Funds", total_funds)
    with col2:
        st.metric("Funds with Data", f"{funds_with_data if isinstance(funds_with_data, int) else total_funds}")
    with col3:
        st.metric("Valid Scores", valid_scores)
    with col4:
        st.metric("Avg Completeness", f"{avg_completeness:.1f}%")
    
    st.markdown("---")
    
    # Simple tier distribution chart
    st.subheader("📊 Tier Distribution")
    try:
        tier_counts = df_tiered['Tier'].value_counts()
        if not tier_counts.empty:
            fig_tier = px.bar(
                x=tier_counts.index,
                y=tier_counts.values,
                title="Fund Distribution by Tier",
                labels={'x': 'Tier', 'y': 'Number of Funds'}
            )
            st.plotly_chart(fig_tier, use_container_width=True)
        else:
            st.info("No tier data available")
    except Exception as e:
        st.error(f"Error creating tier chart: {str(e)}")
    
    # Basic data table
    st.subheader("📋 Fund Data Summary")
    if not df_tiered.empty:
        display_cols = ['Ticker', 'Fund', 'Category', 'Score', 'Tier', 'Inception Group']
        available_cols = [col for col in display_cols if col in df_tiered.columns]
        st.dataframe(df_tiered[available_cols].head(20), use_container_width=True)
    else:
        st.info("No data available")

# ─── PHASE 3 PLACEHOLDER FUNCTIONS ────────────────────────────────────────────

def create_performance_attribution_placeholder():
    """Placeholder for Phase 3 Performance Attribution features"""
    st.title("📊 Performance Attribution Analysis")
    
    # Coming soon notice
    st.info("🚀 **Phase 3 Feature - Coming Soon!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🎯 **Planned Features:**
        
        ### 📈 **Factor-Based Analysis**
        - Return decomposition by risk factors
        - Style drift detection and analysis
        - Market timing vs. security selection
        - Alpha and beta attribution
        
        ### 📊 **Benchmark Comparison**
        - Relative performance tracking
        - Tracking error analysis
        - Information ratio calculations
        - Rolling attribution windows
        """)
    
    with col2:
        st.markdown("""
        ## 🔧 **Advanced Analytics:**
        
        ### 🎨 **Interactive Visualizations**
        - Attribution waterfall charts
        - Factor exposure heatmaps
        - Time-series attribution plots
        - Risk-return scatter analysis
        
        ### 📋 **Detailed Reports**
        - Monthly attribution summaries
        - Sector/style attribution
        - Currency impact analysis
        - Custom benchmark creation
        """)
    
    # Mockup visualization placeholder
    st.markdown("---")
    st.markdown("### 🎨 **Preview: Attribution Waterfall Chart**")
    st.image("https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=Attribution+Waterfall+Chart+%28Phase+3%29", 
             caption="Interactive attribution analysis coming in Phase 3")

def create_portfolio_builder_placeholder():
    """Placeholder for Phase 3 Portfolio Builder features"""
    st.title("🎯 Custom Portfolio Builder")
    
    # Coming soon notice
    st.info("🚀 **Phase 3 Feature - Coming Soon!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🏗️ **Portfolio Construction:**
        
        ### 🎛️ **Interactive Builder**
        - Drag-and-drop fund selection
        - Real-time weight adjustment
        - Constraint-based optimization
        - Target allocation modeling
        
        ### 📊 **Optimization Tools**
        - Mean-variance optimization
        - Risk parity allocation
        - Black-Litterman integration
        - Custom objective functions
        """)
    
    with col2:
        st.markdown("""
        ## 📈 **Portfolio Analytics:**
        
        ### 🔍 **Real-Time Metrics**
        - Expected return calculations
        - Risk (volatility) projections
        - Sharpe ratio optimization
        - Diversification ratios
        
        ### 🎨 **Visualization Suite**
        - Efficient frontier plots
        - Asset allocation pie charts
        - Risk contribution analysis
        - Correlation matrix heatmaps
        """)
    
    # Mockup visualization placeholder
    st.markdown("---")
    st.markdown("### 🎨 **Preview: Portfolio Construction Interface**")
    st.image("https://via.placeholder.com/800x400/2196F3/FFFFFF?text=Interactive+Portfolio+Builder+%28Phase+3%29", 
             caption="Drag-and-drop portfolio construction coming in Phase 3")

def create_risk_analytics_placeholder():
    """Placeholder for Phase 3 Risk Analytics features"""
    st.title("🔗 Advanced Risk Analytics")
    
    # Coming soon notice
    st.info("🚀 **Phase 3 Feature - Coming Soon!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🔗 **Correlation Analysis:**
        
        ### 📊 **Cross-Fund Analysis**
        - Dynamic correlation matrices
        - Rolling correlation windows
        - Regime-based correlations
        - Tail dependency measures
        
        ### 🌐 **Market Regime Detection**
        - Bull/bear market identification
        - Volatility regime analysis
        - Crisis period detection
        - Structural break analysis
        """)
    
    with col2:
        st.markdown("""
        ## ⚠️ **Risk Management:**
        
        ### 📉 **VaR & Risk Metrics**
        - Value at Risk (VaR) calculations
        - Conditional VaR (Expected Shortfall)
        - Maximum Drawdown analysis
        - Stress testing scenarios
        
        ### 🎯 **Monte Carlo Simulations**
        - Return distribution modeling
        - Scenario generation
        - Path-dependent analytics
        - Confidence interval estimation
        """)
    
    # Mockup visualization placeholder
    st.markdown("---")
    st.markdown("### 🎨 **Preview: Risk Analytics Dashboard**")
    st.image("https://via.placeholder.com/800x400/FF9800/FFFFFF?text=Advanced+Risk+Analytics+%28Phase+3%29", 
             caption="Comprehensive risk analysis coming in Phase 3")

# ─── SIMPLIFIED DASHBOARD FUNCTION (STABLE VERSION) ────────────────────────────────────────────
def create_dashboard():
    # Add cache control button
    if st.sidebar.button('🔄 Refresh Data (Clear Cache)'):
        st.cache_data.clear()
        st.experimental_rerun()
    
    st.sidebar.markdown(f"**📅 Cache Status:** Active (5min TTL)")
    st.sidebar.markdown(f"**🕒 Last Refresh:** {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.markdown("---")
    
    # Load and process data (shared across tabs)
    with st.spinner("Loading data from Google Sheets..."):
        df_1y = load_inception_group("1Y+ Inception Funds"); df_1y['Inception Group'] = '1Y+'
        df_3y = load_inception_group("3Y+ Inception Funds"); df_3y['Inception Group'] = '3Y+'
        df_5y = load_inception_group("5Y+ Inception Funds"); df_5y['Inception Group'] = '5Y+'

        scored_1y, _ = calculate_scores_1y(df_1y)
        scored_3y, _ = calculate_scores_3y(df_3y)
        scored_5y, _ = calculate_scores_5y(df_5y)

        df_all = pd.concat([scored_1y, scored_3y, scored_5y], ignore_index=True)
        df_tiered = assign_tiers(df_all)
    
    
    st.markdown("---")
    
    # Fund count validation summary
    total_loaded = len(df_tiered)
    funds_with_scores = (~pd.isna(df_tiered['Score'])).sum()
    tier_counts = df_tiered['Tier'].value_counts()
    
    # Display tier thresholds used
    tier1_thresh = df_tiered['tier1_threshold'].iloc[0] if 'tier1_threshold' in df_tiered.columns else 'N/A'
    tier2_thresh = df_tiered['tier2_threshold'].iloc[0] if 'tier2_threshold' in df_tiered.columns else 'N/A'
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Fund Count Summary")
    st.sidebar.metric("Total Loaded", total_loaded)
    st.sidebar.metric("With Scores", funds_with_scores)
    for tier, count in tier_counts.items():
        st.sidebar.metric(f"{tier}", count)
    
    st.sidebar.markdown("**🎯 Tier Thresholds**")
    st.sidebar.text(f"Tier 1: ≥ {tier1_thresh:.2f}" if isinstance(tier1_thresh, (int, float)) else f"Tier 1: {tier1_thresh}")
    st.sidebar.text(f"Tier 2: ≥ {tier2_thresh:.2f}" if isinstance(tier2_thresh, (int, float)) else f"Tier 2: {tier2_thresh}")
    
    # Enhanced tabbed interface with Phase 3 preparation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Fund Rankings", 
        "🛡️ Data Quality",
        "📊 Performance Attribution*",
        "🎯 Portfolio Builder*",
        "🔗 Risk Analytics*"
    ])
    
    with tab1:
        create_main_rankings_tab(df_tiered)
    
    with tab2:
        create_basic_data_quality_tab(df_tiered)
    
    with tab3:
        create_performance_attribution_placeholder()
    
    with tab4:
        create_portfolio_builder_placeholder()
    
    with tab5:
        create_risk_analytics_placeholder()

# ─── LEGACY FUNCTION - REMOVED (replaced by tabbed interface) ───

if __name__ == "__main__":
    create_dashboard()