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

# DATA COMPLETENESS REQUIREMENTS
MIN_DATA_REQUIREMENTS = {
    "1Y+": {"required": ["Total Return", "Sharpe (1Y)", "Sortino (1Y)"], "max_missing": 1},
    "3Y+": {"required": ["Total Return (3Y)", "Sharpe (3Y)", "Sortino (3Y)"], "max_missing": 1},
    "5Y+": {"required": ["Total Return (5Y)", "Sharpe (5Y)", "Sortino (5Y)"], "max_missing": 1}
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
    """Calculate scores only for funds with sufficient data, NaN for others"""
    scores = pd.Series(np.nan, index=df.index)
    
    # Only calculate scores for funds with sufficient data
    valid_funds = df['has_sufficient_data'] == True
    
    if valid_funds.any():
        # Calculate weighted sum for valid funds only
        for component, weight in components.items():
            if component in df.columns:
                # Use 0 for NaN in individual components, but only for valid funds
                component_scores = df[component].fillna(0) * weight
                scores[valid_funds] = scores[valid_funds].fillna(0) + component_scores[valid_funds]
    
    return scores

def calculate_scores_1y(df: pd.DataFrame):
    df = df.copy()
    df = standardize_column_names(df)
    
    # Ensure numeric conversion for required columns
    numeric_cols = ['Total Return', 'Sharpe (1Y)', 'Sortino (1Y)', 'AUM', 'Net Expense', 'Std Dev (1Y)', 'VaR']
    df = safe_numeric_conversion(df, numeric_cols)
    
    # Validate data completeness BEFORE scoring
    df = validate_fund_data_quality(df, '1Y+')
    
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
    
    # Calculate dynamic thresholds from all valid scores
    valid_scores = df[~pd.isna(df['Score']) & df.get('has_sufficient_data', True)]['Score']
    tier1_thresh, tier2_thresh = get_dynamic_thresholds(valid_scores)
    
    def tier(row):
        score = row['Score']
        has_data = row.get('has_sufficient_data', True)
        
        # Mark funds with insufficient data as 'No Data'
        if not has_data or pd.isna(score):
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

def get_export_filename(inception_group: str = None, include_timestamp: bool = True) -> str:
    """Generate appropriate filename for export"""
    base_name = f"funds_{inception_group.replace('+', 'plus')}" if inception_group else "all_funds"
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{base_name}_{timestamp}.csv"
    
    return f"{base_name}.csv"

@st.cache_data
def convert_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes for download"""
    return df.to_csv(index=False).encode("utf-8")

def create_download_section(df_tiered: pd.DataFrame, filtered_df: pd.DataFrame, selected_inceptions: list):
    """Create enhanced download section with multiple options"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Options")
    
    # Add caption showing filtered data info
    fund_count = len(filtered_df)
    if not filtered_df.empty and 'Inception Date' in filtered_df.columns:
        latest_inception = filtered_df['Inception Date'].max()
        if pd.notna(latest_inception):
            st.sidebar.caption(f"📊 {fund_count} funds | Latest inception: {latest_inception.strftime('%Y-%m-%d')}")
        else:
            st.sidebar.caption(f"📊 {fund_count} funds")
    else:
        st.sidebar.caption(f"📊 {fund_count} funds")
    
    # Option 1: Download filtered data (all selected inception groups combined)
    if not filtered_df.empty:
        cleaned_filtered = clean_dataframe_for_export(filtered_df)
        st.sidebar.download_button(
            "📋 Download Filtered Data",
            data=convert_to_csv(cleaned_filtered),
            file_name=get_export_filename(),
            mime="text/csv",
            help="Download currently filtered data with all selected inception groups"
        )
    
    # Option 2: Download by individual inception group
    st.sidebar.markdown("**📂 Download by Inception Group:**")
    
    for inception in ['1Y+', '3Y+', '5Y+']:
        group_data = df_tiered[df_tiered['Inception Group'] == inception]
        
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

# ─── UPDATED MAIN DASHBOARD FUNCTION ────────────────────────────────────────────
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
    
    # Score Distribution Analysis & Data Quality Monitor
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Quality & Distribution")
    
    # Calculate distribution statistics
    tier_counts = df_tiered['Tier'].value_counts()
    total_funds = len(df_tiered)
    valid_scores = df_tiered[~pd.isna(df_tiered['Score'])]['Score']
    
    # Show tier distribution
    for tier_name in ["Tier 1", "Tier 2", "Tier 3", "No Data"]:
        count = tier_counts.get(tier_name, 0)
        pct = (count / total_funds * 100) if total_funds > 0 else 0
        st.sidebar.metric(f"{tier_name}", f"{count} ({pct:.1f}%)")
    
    # Show dynamic thresholds if available
    if 'tier1_threshold' in df_tiered.columns and 'tier2_threshold' in df_tiered.columns:
        t1_thresh = df_tiered['tier1_threshold'].iloc[0] if not df_tiered.empty else 8.5
        t2_thresh = df_tiered['tier2_threshold'].iloc[0] if not df_tiered.empty else 6.0
        st.sidebar.info(f"🎯 Dynamic Thresholds:\\nTier 1: ≥{t1_thresh:.2f}\\nTier 2: ≥{t2_thresh:.2f}")
    
    # Data quality alerts
    no_data_count = tier_counts.get("No Data", 0)
    tier2_count = tier_counts.get("Tier 2", 0)
    
    if tier2_count == 0 and total_funds > 20:
        st.sidebar.error("🚨 No Tier 2 funds detected!")
        if len(valid_scores) > 0:
            st.sidebar.info(f"Score range: {valid_scores.min():.2f} to {valid_scores.max():.2f}")
    
    if no_data_count > total_funds * 0.2:
        st.sidebar.warning(f"⚠️ {no_data_count} funds ({no_data_count/total_funds*100:.1f}%) lack critical data")
    
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

    # Enhanced data summary with fund count and latest inception date
    fund_count = len(filtered_df)
    if not filtered_df.empty and 'Inception Date' in filtered_df.columns:
        latest_inception = filtered_df['Inception Date'].max()
        if pd.notna(latest_inception):
            st.caption(f"📊 Showing {fund_count} funds | Latest inception: {latest_inception.strftime('%Y-%m-%d')}")
        else:
            st.caption(f"📊 Showing {fund_count} funds")
    else:
        st.caption(f"📊 Showing {fund_count} funds")

    # Rest of the dashboard code remains the same...
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

    # Score Distribution Analysis Section
    st.subheader("📈 Score Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Score Distribution by Inception Group**")
        fig_hist = px.histogram(
            df_tiered[~pd.isna(df_tiered['Score'])], 
            x='Score', 
            color='Inception Group',
            nbins=30,
            title="Score Distribution Histogram"
        )
        fig_hist.add_vline(x=df_tiered['tier1_threshold'].iloc[0] if 'tier1_threshold' in df_tiered.columns else 8.5, 
                          line_dash="dash", line_color="green", annotation_text="Tier 1")
        fig_hist.add_vline(x=df_tiered['tier2_threshold'].iloc[0] if 'tier2_threshold' in df_tiered.columns else 6.0, 
                          line_dash="dash", line_color="orange", annotation_text="Tier 2")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("**Data Completeness Analysis**")
        # Show data completeness by inception group
        completeness_data = []
        for group in ['1Y+', '3Y+', '5Y+']:
            group_df = df_tiered[df_tiered['Inception Group'] == group]
            if not group_df.empty:
                total = len(group_df)
                has_data = group_df.get('has_sufficient_data', True).sum() if 'has_sufficient_data' in group_df.columns else total
                valid_scores = (~pd.isna(group_df['Score'])).sum()
                
                completeness_data.append({
                    'Group': group,
                    'Total Funds': total,
                    'Sufficient Data': has_data if isinstance(has_data, int) else total,
                    'Valid Scores': valid_scores,
                    'Completeness %': (has_data/total*100) if isinstance(has_data, int) and total > 0 else 100
                })
        
        if completeness_data:
            completeness_df = pd.DataFrame(completeness_data)
            st.dataframe(completeness_df, use_container_width=True)
        
        # Score statistics table
        st.markdown("**Score Statistics by Group**")
        stats_data = []
        for group in ['1Y+', '3Y+', '5Y+']:
            group_scores = df_tiered[(df_tiered['Inception Group'] == group) & (~pd.isna(df_tiered['Score']))]['Score']
            if len(group_scores) > 0:
                stats_data.append({
                    'Group': group,
                    'Count': len(group_scores),
                    'Min': group_scores.min(),
                    'Max': group_scores.max(),
                    'Mean': group_scores.mean(),
                    'Std': group_scores.std()
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df.round(3), use_container_width=True)
    
    # Debugging section for problematic funds
    if st.expander("🔍 Debug Problematic Funds"):
        problematic_tickers = ['CREMX', 'XHLDX', 'REFLX', 'PNDRX', 'XHFAX', 'FORFX']
        problematic_funds = df_tiered[df_tiered['Ticker'].isin(problematic_tickers)]
        
        if not problematic_funds.empty:
            st.markdown("**Known problematic funds analysis:**")
            debug_cols = ['Ticker', 'Fund', 'Tier', 'Score', 'Total Return', 'Sharpe (1Y)', 'Sortino (1Y)', 'has_sufficient_data']
            available_debug_cols = [col for col in debug_cols if col in problematic_funds.columns]
            st.dataframe(problematic_funds[available_debug_cols], use_container_width=True)
        else:
            st.info("No problematic funds found in current dataset.")
    
    # Enhanced Download Section
    create_download_section(df_tiered, filtered_df, selected_inceptions)

if __name__ == "__main__":
    create_dashboard()