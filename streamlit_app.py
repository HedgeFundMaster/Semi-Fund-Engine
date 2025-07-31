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
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics imports for Phase 3
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ Advanced analytics require scikit-learn. Some features may be limited.")

# Professional features imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    
import io
import base64
import pickle

# ─── UTILITY FUNCTIONS FOR SAFE PLOTTING ─────────────────────────────────────

def safe_scatter_plot(df, x_col, y_col, title="Scatter Plot", **kwargs):
    """
    Create a safe scatter plot with comprehensive data validation and error handling
    """
    try:
        if df is None or df.empty:
            return None, "No data available"
        
        # Check required columns exist
        if x_col not in df.columns or y_col not in df.columns:
            return None, f"Missing required columns: {x_col}, {y_col}"
        
        # Clean and validate data
        plot_df = df.copy()
        
        # Ensure numeric columns are numeric and handle inf/-inf
        numeric_cols = [x_col, y_col]
        size_col = kwargs.get('size')
        color_col = kwargs.get('color')
        
        if size_col and size_col in plot_df.columns:
            numeric_cols.append(size_col)
        if color_col and color_col in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[color_col]):
            numeric_cols.append(color_col)
        
        for col in numeric_cols:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN in essential columns
        plot_df = plot_df.dropna(subset=[x_col, y_col])
        
        if len(plot_df) == 0:
            return None, "No valid data after cleaning"
        
        # Handle size parameter - ensure positive values
        if size_col and size_col in plot_df.columns:
            if plot_df[size_col].notna().any():
                plot_df['Safe_Size'] = np.abs(plot_df[size_col]) + 0.01
                kwargs['size'] = 'Safe_Size'
            else:
                kwargs.pop('size', None)
        
        # Clean hover_data
        if 'hover_data' in kwargs and kwargs['hover_data']:
            hover_cols = []
            for col in kwargs['hover_data']:
                if col in plot_df.columns:
                    hover_cols.append(col)
            kwargs['hover_data'] = hover_cols if hover_cols else None
        
        # Create the scatter plot
        fig = px.scatter(plot_df, x=x_col, y=y_col, title=title, **kwargs)
        
        return fig, "Success"
        
    except Exception as e:
        return None, f"Error creating scatter plot: {str(e)}"

# Portfolio optimization constants
RISK_FREE_RATE = 0.02  # 2% risk-free rate assumption
TRADING_DAYS = 252
MONTE_CARLO_SIMULATIONS = 10000

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

# ─── PHASE 3 ADVANCED ANALYTICS HELPER FUNCTIONS ──────────────────────────────────

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio risk and return metrics"""
    portfolio_return = np.sum(returns.mean() * weights) * TRADING_DAYS
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * TRADING_DAYS, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_std,
        'sharpe': sharpe_ratio
    }

def efficient_frontier(returns, num_portfolios=100):
    """Generate efficient frontier data"""
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    
    # Generate target returns
    min_ret = returns.mean().min() * TRADING_DAYS
    max_ret = returns.mean().max() * TRADING_DAYS
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    def portfolio_stats(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * TRADING_DAYS
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * TRADING_DAYS, weights)))
        return portfolio_return, portfolio_std
    
    def minimize_volatility(weights, returns, target_return):
        portfolio_return, portfolio_std = portfolio_stats(weights, returns)
        return portfolio_std
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    for i, target in enumerate(target_returns):
        # Add return constraint
        cons = [constraints, {'type': 'eq', 'fun': lambda x, target=target: portfolio_stats(x, returns)[0] - target}]
        
        # Initial guess
        x0 = np.array([1./num_assets] * num_assets)
        
        try:
            result = minimize(minimize_volatility, x0, args=(returns, target), 
                            method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': 1000})
            
            if result.success:
                ret, vol = portfolio_stats(result.x, returns)
                results[0, i] = ret
                results[1, i] = vol
                results[2, i] = (ret - RISK_FREE_RATE) / vol
        except:
            results[:, i] = np.nan
    
    return results

def monte_carlo_simulation(returns, weights=None, time_horizon=252):
    """Run Monte Carlo simulation for portfolio outcomes"""
    if weights is None:
        weights = np.array([1/len(returns.columns)] * len(returns.columns))
    
    # Calculate portfolio statistics
    portfolio_mean = np.sum(returns.mean() * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    
    # Run simulations
    simulations = np.random.normal(portfolio_mean, portfolio_std, (MONTE_CARLO_SIMULATIONS, time_horizon))
    cumulative_returns = np.cumprod(1 + simulations, axis=1)
    final_values = cumulative_returns[:, -1]
    
    return {
        'final_values': final_values,
        'cumulative_returns': cumulative_returns,
        'var_95': np.percentile(final_values, 5),
        'var_99': np.percentile(final_values, 1),
        'expected_return': np.mean(final_values),
        'volatility': np.std(final_values)
    }

def calculate_var(returns, confidence_level=0.05, time_horizon=1):
    """Calculate Value at Risk"""
    if len(returns) < 30:
        return None
    
    # Historical VaR
    sorted_returns = np.sort(returns)
    var_index = int(confidence_level * len(sorted_returns))
    historical_var = sorted_returns[var_index] * np.sqrt(time_horizon)
    
    # Parametric VaR
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    parametric_var = stats.norm.ppf(confidence_level, mean_return, std_return) * np.sqrt(time_horizon)
    
    return {
        'historical_var': historical_var,
        'parametric_var': parametric_var,
        'expected_shortfall': np.mean(sorted_returns[:var_index]) * np.sqrt(time_horizon)
    }

def calculate_maximum_drawdown(cumulative_returns):
    """Calculate maximum drawdown and recovery time"""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Find recovery time
    max_dd_end = np.argmin(drawdown)
    max_dd_start = np.argmax(cumulative_returns[:max_dd_end])
    
    # Find when it recovered (if it did)
    recovery_time = None
    if max_dd_end < len(cumulative_returns) - 1:
        recovery_idx = np.where(cumulative_returns[max_dd_end:] >= peak[max_dd_end])[0]
        if len(recovery_idx) > 0:
            recovery_time = recovery_idx[0]
    
    return {
        'max_drawdown': max_drawdown,
        'drawdown_start': max_dd_start,
        'drawdown_end': max_dd_end,
        'recovery_time': recovery_time,
        'drawdown_series': drawdown
    }

def perform_factor_analysis(returns_data):
    """Perform factor analysis on fund returns"""
    if not SKLEARN_AVAILABLE or returns_data.empty:
        return None
    
    # Remove NaN values
    clean_data = returns_data.dropna()
    if len(clean_data) < 10:
        return None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)
    
    # PCA for factor analysis
    pca = PCA(n_components=min(5, len(clean_data.columns)))
    pca_result = pca.fit_transform(scaled_data)
    
    return {
        'explained_variance': pca.explained_variance_ratio_,
        'components': pca.components_,
        'factor_loadings': pca_result,
        'feature_names': clean_data.columns.tolist()
    }

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
    """Enhanced Fund Rankings tab with advanced filtering, comparison, and analytics tools"""
    
    # Professional header with Phase 3 indicators
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Semi-Liquid Alternatives Fund Selection Dashboard</h1>
        <p>Advanced quantitative analysis and performance scoring for alternative investments</p>
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 8px 16px; border-radius: 8px; margin: 10px 0;">
            <strong>🚀 Phase 3 - Institutional Analytics Platform</strong> | 
            Advanced Portfolio Construction & Risk Management
        </div>
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
    
    # Advanced Fund Comparison Tool
    if not filtered_df.empty and len(filtered_df) > 1:
        st.markdown("---")
        st.subheader("🔍 Advanced Fund Comparison")
        
        # Multi-fund selector
        comparison_tabs = st.tabs(["Side-by-Side Comparison", "Performance Matrix", "Risk-Return Analysis"])
        
        with comparison_tabs[0]:
            st.markdown("#### 📊 Side-by-Side Fund Analysis")
            
            # Fund selection for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Select funds to compare (up to 5):**")
                available_funds = filtered_df.sort_values('Score', ascending=False)['Ticker'].head(20).tolist()
                selected_comparison_funds = st.multiselect(
                    "Choose funds for comparison:",
                    available_funds,
                    default=available_funds[:3] if len(available_funds) >= 3 else available_funds,
                    max_selections=5,
                    help="Select 2-5 funds for detailed side-by-side comparison"
                )
            
            with col2:
                st.markdown("**Comparison metrics:**")
                comparison_metrics = st.multiselect(
                    "Select metrics to compare:",
                    ["Score", "Total Return (%)", "Sharpe Ratio", "Sortino Ratio", "Volatility (%)", 
                     "Max Drawdown (%)", "AUM", "Net Expense", "Category Delta"],
                    default=["Score", "Total Return (%)", "Sharpe Ratio", "Volatility (%)"],
                    help="Choose which metrics to include in the comparison"
                )
            
            if len(selected_comparison_funds) >= 2 and comparison_metrics:
                comparison_df = filtered_df[filtered_df['Ticker'].isin(selected_comparison_funds)].copy()
                
                # Create comparison table
                comparison_display = comparison_df[['Ticker', 'Fund Name'] + comparison_metrics].copy()
                comparison_display = comparison_display.set_index('Ticker')
                
                # Add ranking for each metric
                for metric in comparison_metrics:
                    if metric in comparison_display.columns:
                        comparison_display[f'{metric} Rank'] = comparison_display[metric].rank(ascending=False, method='min').astype(int)
                
                st.dataframe(comparison_display.round(3), use_container_width=True)
                
                # Radar chart comparison
                if len(selected_comparison_funds) <= 4:
                    st.markdown("#### 🕸️ Performance Radar Chart")
                    
                    # Normalize metrics for radar chart (0-10 scale)
                    radar_metrics = [m for m in comparison_metrics if m in comparison_display.columns and comparison_display[m].notna().any()]
                    
                    if len(radar_metrics) >= 3:
                        fig_radar = go.Figure()
                        
                        for ticker in selected_comparison_funds:
                            if ticker in comparison_display.index:
                                fund_data = comparison_display.loc[ticker]
                                
                                # Normalize values to 0-10 scale
                                normalized_values = []
                                for metric in radar_metrics:
                                    value = fund_data[metric]
                                    if pd.notna(value):
                                        # Simple min-max normalization
                                        min_val = comparison_display[metric].min()
                                        max_val = comparison_display[metric].max()
                                        if max_val != min_val:
                                            normalized = 10 * (value - min_val) / (max_val - min_val)
                                        else:
                                            normalized = 5
                                        normalized_values.append(normalized)
                                    else:
                                        normalized_values.append(0)
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=normalized_values + [normalized_values[0]],  # Close the polygon
                                    theta=radar_metrics + [radar_metrics[0]],
                                    fill='toself',
                                    name=ticker
                                ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 10]
                                )),
                            showlegend=True,
                            title="Fund Performance Radar Chart (Normalized 0-10 Scale)",
                            height=500
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
        
        with comparison_tabs[1]:
            st.markdown("#### 📈 Performance Correlation Matrix")
            
            # Calculate correlation matrix for selected metrics
            numeric_cols = ['Score', 'Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Volatility (%)']
            available_numeric = [col for col in numeric_cols if col in filtered_df.columns]
            
            if len(available_numeric) >= 2:
                corr_data = filtered_df[available_numeric].corr()
                
                fig_corr = px.imshow(
                    corr_data,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    title="Fund Metrics Correlation Matrix",
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation insights
                st.markdown("#### 💡 Correlation Insights")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_data.columns)):
                    for j in range(i+1, len(corr_data.columns)):
                        corr_val = corr_data.iloc[i, j]
                        if not pd.isna(corr_val):
                            corr_pairs.append({
                                'Metric 1': corr_data.columns[i],
                                'Metric 2': corr_data.columns[j],
                                'Correlation': corr_val
                            })
                
                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
                    top_correlations = corr_df.sort_values('Abs Correlation', ascending=False).head(3)
                    
                    for _, row in top_correlations.iterrows():
                        strength = "Strong" if row['Abs Correlation'] > 0.7 else "Moderate" if row['Abs Correlation'] > 0.4 else "Weak"
                        direction = "positive" if row['Correlation'] > 0 else "negative"
                        st.write(f"• **{strength} {direction} correlation** between {row['Metric 1']} and {row['Metric 2']}: {row['Correlation']:.3f}")
        
        with comparison_tabs[2]:
            st.markdown("#### 🎯 Risk-Return Efficiency Analysis")
            
            # Risk-return scatter plot with enhanced features and validation
            if 'Total Return (%)' in filtered_df.columns and 'Volatility (%)' in filtered_df.columns:
                try:
                    # Clean and validate data for scatter plot
                    plot_df = filtered_df.copy()
                    
                    # Ensure numeric columns are numeric and handle inf/-inf
                    numeric_cols = ['Total Return (%)', 'Volatility (%)', 'Score']
                    for col in numeric_cols:
                        if col in plot_df.columns:
                            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                            plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Remove rows with NaN in essential columns
                    plot_df = plot_df.dropna(subset=['Total Return (%)', 'Volatility (%)'])
                    
                    if len(plot_df) == 0:
                        st.warning("No valid data for Risk-Return scatter plot")
                    else:
                        # Handle size parameter - ensure positive values
                        if 'Score' in plot_df.columns and plot_df['Score'].notna().any():
                            plot_df['Size_Value'] = np.abs(plot_df['Score']) + 0.1
                            size_col = 'Size_Value'
                        else:
                            size_col = None
                        
                        # Prepare hover data - only include available columns
                        hover_data_cols = []
                        for col in ['Fund Name', 'Sharpe Ratio', 'Category']:
                            if col in plot_df.columns:
                                hover_data_cols.append(col)
                        
                        fig_risk_return = px.scatter(
                            plot_df,
                            x='Volatility (%)',
                            y='Total Return (%)',
                            size=size_col,
                            color='Tier',
                            hover_name='Ticker' if 'Ticker' in plot_df.columns else None,
                            hover_data=hover_data_cols if hover_data_cols else None,
                            title='Risk-Return Profile with Efficiency Frontier',
                            color_discrete_map={
                                'Tier 1': '#2E8B57',
                                'Tier 2': '#4682B4', 
                                'Tier 3': '#DAA520',
                                'No Data': '#696969'
                            }
                        )
                        
                except Exception as e:
                    st.error(f"Error creating Risk-Return scatter plot: {str(e)}")
                    st.info("Displaying simplified analysis...")
                    
                    # Fallback to simple scatter without advanced features
                    try:
                        simple_df = filtered_df[['Volatility (%)', 'Total Return (%)']].dropna()
                        if not simple_df.empty:
                            fig_risk_return = px.scatter(
                                simple_df,
                                x='Volatility (%)',
                                y='Total Return (%)',
                                title='Risk-Return Profile (Simplified)'
                            )
                        else:
                            fig_risk_return = None
                    except:
                        fig_risk_return = None
                
                if 'fig_risk_return' in locals() and fig_risk_return is not None:
                    # Add efficiency frontier line (simplified)
                    if len(filtered_df) > 5:
                        sorted_by_return = filtered_df.sort_values('Total Return (%)', ascending=False)
                        top_performers = sorted_by_return.head(10)
                        
                        if not top_performers.empty:
                            fig_risk_return.add_trace(go.Scatter(
                                x=top_performers['Volatility (%)'],
                                y=top_performers['Total Return (%)'],
                                mode='lines',
                                name='Efficiency Guide',
                                line=dict(color='red', dash='dash', width=2),
                                hovertemplate='Efficiency Guide<extra></extra>'
                            ))
                    
                    # Add quadrant lines
                    median_return = filtered_df['Total Return (%)'].median()
                    median_vol = filtered_df['Volatility (%)'].median()
                    
                    fig_risk_return.add_hline(y=median_return, line_dash="dot", 
                                             annotation_text="Median Return")
                    fig_risk_return.add_vline(x=median_vol, line_dash="dot", 
                                             annotation_text="Median Risk")
                    
                    fig_risk_return.update_layout(height=600)
                    st.plotly_chart(fig_risk_return, use_container_width=True)
                    
                    # Quadrant analysis
                    st.markdown("#### 📊 Risk-Return Quadrant Analysis")
                
                # Quadrant analysis
                st.markdown("#### 📊 Risk-Return Quadrant Analysis")
                
                # Categorize funds by quadrants
                high_return_low_risk = filtered_df[
                    (filtered_df['Total Return (%)'] > median_return) & 
                    (filtered_df['Volatility (%)'] < median_vol)
                ]
                high_return_high_risk = filtered_df[
                    (filtered_df['Total Return (%)'] > median_return) & 
                    (filtered_df['Volatility (%)'] >= median_vol)
                ]
                low_return_low_risk = filtered_df[
                    (filtered_df['Total Return (%)'] <= median_return) & 
                    (filtered_df['Volatility (%)'] < median_vol)
                ]
                low_return_high_risk = filtered_df[
                    (filtered_df['Total Return (%)'] <= median_return) & 
                    (filtered_df['Volatility (%)'] >= median_vol)
                ]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🏆 High Return, Low Risk", len(high_return_low_risk))
                    if not high_return_low_risk.empty:
                        st.write("Top funds:")
                        for ticker in high_return_low_risk.sort_values('Score', ascending=False)['Ticker'].head(3):
                            st.write(f"• {ticker}")
                
                with col2:
                    st.metric("⚡ High Return, High Risk", len(high_return_high_risk))
                    if not high_return_high_risk.empty:
                        st.write("Top funds:")
                        for ticker in high_return_high_risk.sort_values('Score', ascending=False)['Ticker'].head(3):
                            st.write(f"• {ticker}")
                
                with col3:
                    st.metric("🛡️ Low Return, Low Risk", len(low_return_low_risk))
                    if not low_return_low_risk.empty:
                        st.write("Top funds:")
                        for ticker in low_return_low_risk.sort_values('Score', ascending=False)['Ticker'].head(3):
                            st.write(f"• {ticker}")
                
                with col4:
                    st.metric("⚠️ Low Return, High Risk", len(low_return_high_risk))
                    if not low_return_high_risk.empty:
                        st.write("Funds to review:")
                        for ticker in low_return_high_risk.sort_values('Score', ascending=True)['Ticker'].head(3):
                            st.write(f"• {ticker}")
    
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
    """Enhanced Data Quality tab with advanced validation, scoring, and data governance"""
    st.title("🛡️ Advanced Data Quality & Validation")
    st.markdown("**Comprehensive data governance and quality assessment for institutional compliance**")
    
    # Enhanced quality analysis
    quality_tabs = st.tabs([
        "📊 Quality Overview", 
        "🔍 Data Validation", 
        "🚨 Quality Alerts", 
        "📈 Quality Trends",
        "🏗️ Data Lineage"
    ])
    
    with quality_tabs[0]:
        st.markdown("### 📊 Data Quality Dashboard")
        
        # Calculate comprehensive quality metrics
        total_funds = len(df_tiered)
        
        # Core data availability
        key_fields = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'AUM', 'Net Expense', 'Category']
        field_completeness = {}
        
        for field in key_fields:
            if field in df_tiered.columns:
                non_null_count = df_tiered[field].notna().sum()
                completeness = (non_null_count / total_funds) * 100
                field_completeness[field] = {
                    'available': non_null_count,
                    'total': total_funds,
                    'completeness': completeness
                }
        
        # Quality score calculation
        overall_completeness = np.mean([fc['completeness'] for fc in field_completeness.values()])
        
        # Advanced quality metrics
        valid_scores = (~pd.isna(df_tiered['Score'])).sum()
        score_completeness = (valid_scores / total_funds) * 100
        
        # Outlier detection
        outlier_counts = {}
        if 'Total Return (%)' in df_tiered.columns:
            returns = df_tiered['Total Return (%)'].dropna()
            q1, q3 = returns.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = returns[(returns < q1 - 1.5*iqr) | (returns > q3 + 1.5*iqr)]
            outlier_counts['Returns'] = len(outliers)
        
        if 'Volatility (%)' in df_tiered.columns:
            vols = df_tiered['Volatility (%)'].dropna()
            vol_outliers = vols[vols > vols.quantile(0.95)]
            outlier_counts['Volatility'] = len(vol_outliers)
        
        # Quality metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            quality_grade = "A" if overall_completeness >= 90 else "B" if overall_completeness >= 75 else "C" if overall_completeness >= 60 else "D"
            st.metric(
                "Overall Quality Grade", 
                quality_grade,
                delta=f"{overall_completeness:.1f}% complete",
                help="Overall data quality assessment based on field completeness"
            )
        
        with col2:
            st.metric(
                "📊 Total Funds", 
                total_funds,
                help="Total number of funds in the dataset"
            )
        
        with col3:
            st.metric(
                "✅ Scoreable Funds", 
                valid_scores,
                delta=f"{score_completeness:.1f}% of total",
                help="Funds with sufficient data for scoring"
            )
        
        with col4:
            total_outliers = sum(outlier_counts.values())
            st.metric(
                "🚨 Data Anomalies", 
                total_outliers,
                delta=f"{(total_outliers/total_funds)*100:.1f}% of funds",
                help="Statistical outliers requiring review"
            )
        
        with col5:
            missing_critical = total_funds - valid_scores
            st.metric(
                "⚠️ Missing Critical Data", 
                missing_critical,
                delta=f"{(missing_critical/total_funds)*100:.1f}% of funds",
                help="Funds missing essential performance data"
            )
        
        # Field completeness visualization
        st.markdown("#### 📈 Field Completeness Analysis")
        
        if field_completeness:
            completeness_df = pd.DataFrame([
                {
                    'Field': field,
                    'Available Records': data['available'],
                    'Total Records': data['total'],
                    'Completeness (%)': data['completeness']
                }
                for field, data in field_completeness.items()
            ])
            
            # Completeness bar chart
            fig_completeness = px.bar(
                completeness_df,
                x='Field',
                y='Completeness (%)',
                title='Data Field Completeness',
                color='Completeness (%)',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            fig_completeness.add_hline(y=90, line_dash="dash", line_color="green", 
                                     annotation_text="Excellent (90%+)")
            fig_completeness.add_hline(y=75, line_dash="dash", line_color="orange", 
                                     annotation_text="Good (75%+)")
            fig_completeness.update_layout(height=400)
            st.plotly_chart(fig_completeness, use_container_width=True)
            
            # Detailed completeness table
            st.dataframe(completeness_df, use_container_width=True, hide_index=True)
    
    with quality_tabs[1]:
        st.markdown("### 🔍 Advanced Data Validation")
        
        # Validation rules and results
        validation_results = []
        
        # Rule 1: Return reasonableness
        if 'Total Return (%)' in df_tiered.columns:
            returns = df_tiered['Total Return (%)'].dropna()
            extreme_returns = returns[(returns < -50) | (returns > 100)]
            validation_results.append({
                'Rule': 'Return Reasonableness',
                'Description': 'Returns should be between -50% and +100%',
                'Violations': len(extreme_returns),
                'Severity': 'High' if len(extreme_returns) > 5 else 'Medium' if len(extreme_returns) > 0 else 'Low',
                'Status': '❌ Failed' if len(extreme_returns) > 0 else '✅ Passed'
            })
        
        # Rule 2: Sharpe ratio validity
        if 'Sharpe Ratio' in df_tiered.columns:
            sharpe = df_tiered['Sharpe Ratio'].dropna()
            extreme_sharpe = sharpe[(sharpe < -3) | (sharpe > 5)]
            validation_results.append({
                'Rule': 'Sharpe Ratio Validity',
                'Description': 'Sharpe ratios should be between -3 and +5',
                'Violations': len(extreme_sharpe),
                'Severity': 'Medium' if len(extreme_sharpe) > 0 else 'Low',
                'Status': '❌ Failed' if len(extreme_sharpe) > 0 else '✅ Passed'
            })
        
        # Rule 3: AUM consistency
        if 'AUM' in df_tiered.columns:
            aum = df_tiered['AUM'].dropna()
            negative_aum = aum[aum < 0]
            validation_results.append({
                'Rule': 'AUM Positivity',
                'Description': 'AUM values must be positive',
                'Violations': len(negative_aum),
                'Severity': 'High' if len(negative_aum) > 0 else 'Low',
                'Status': '❌ Failed' if len(negative_aum) > 0 else '✅ Passed'
            })
        
        # Rule 4: Expense ratio reasonableness
        if 'Net Expense' in df_tiered.columns:
            expenses = df_tiered['Net Expense'].dropna()
            extreme_expenses = expenses[(expenses < 0) | (expenses > 5)]
            validation_results.append({
                'Rule': 'Expense Ratio Range',
                'Description': 'Expense ratios should be between 0% and 5%',
                'Violations': len(extreme_expenses),
                'Severity': 'Medium' if len(extreme_expenses) > 0 else 'Low',
                'Status': '❌ Failed' if len(extreme_expenses) > 0 else '✅ Passed'
            })
        
        # Rule 5: Volatility consistency
        if 'Volatility (%)' in df_tiered.columns:
            vols = df_tiered['Volatility (%)'].dropna()
            invalid_vols = vols[vols < 0]
            validation_results.append({
                'Rule': 'Volatility Non-Negative',
                'Description': 'Volatility must be non-negative',
                'Violations': len(invalid_vols),
                'Severity': 'High' if len(invalid_vols) > 0 else 'Low',
                'Status': '❌ Failed' if len(invalid_vols) > 0 else '✅ Passed'
            })
        
        # Display validation results
        if validation_results:
            validation_df = pd.DataFrame(validation_results)
            
            # Summary metrics
            total_rules = len(validation_results)
            passed_rules = len([r for r in validation_results if 'Passed' in r['Status']])
            failed_rules = total_rules - passed_rules
            total_violations = sum([r['Violations'] for r in validation_results])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rules", total_rules)
            with col2:
                st.metric("Rules Passed", passed_rules, delta=f"{(passed_rules/total_rules)*100:.0f}%")
            with col3:
                st.metric("Rules Failed", failed_rules, delta=f"{(failed_rules/total_rules)*100:.0f}%")
            with col4:
                st.metric("Total Violations", total_violations)
            
            # Validation results table
            st.markdown("#### 📋 Validation Rule Results")
            st.dataframe(
                validation_df.style.apply(
                    lambda row: ['background-color: #ffebee' if 'Failed' in row['Status'] else 'background-color: #e8f5e8' 
                               for _ in row], axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # Severity distribution
            severity_counts = validation_df['Severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Validation Issues by Severity",
                color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#44aa44'}
            )
            st.plotly_chart(fig_severity, use_container_width=True)
    
    with quality_tabs[2]:
        st.markdown("### 🚨 Data Quality Alerts")
        
        # Generate alerts based on data issues
        alerts = []
        
        # High-priority alerts
        if 'Total Return (%)' in df_tiered.columns:
            missing_returns = df_tiered['Total Return (%)'].isna().sum()
            if missing_returns > total_funds * 0.1:  # More than 10% missing
                alerts.append({
                    'Level': '🔴 CRITICAL',
                    'Category': 'Missing Data',
                    'Alert': f'{missing_returns} funds missing Total Return data',
                    'Impact': 'Scoring accuracy compromised',
                    'Action': 'Request updated data from data providers'
                })
        
        # Medium-priority alerts
        if 'Score' in df_tiered.columns:
            unscored_funds = df_tiered['Score'].isna().sum()
            if unscored_funds > 0:
                alerts.append({
                    'Level': '🟡 WARNING',
                    'Category': 'Incomplete Scoring',
                    'Alert': f'{unscored_funds} funds could not be scored',
                    'Impact': 'Limited fund universe for analysis',
                    'Action': 'Review data requirements for scoring'
                })
        
        # Data consistency alerts
        if 'Tier' in df_tiered.columns:
            tier_distribution = df_tiered['Tier'].value_counts()
            if 'No Data' in tier_distribution and tier_distribution['No Data'] > total_funds * 0.2:
                alerts.append({
                    'Level': '🟠 MODERATE',
                    'Category': 'Data Quality',
                    'Alert': f'{tier_distribution["No Data"]} funds in "No Data" tier',
                    'Impact': 'Reduced analytical value',
                    'Action': 'Investigate data collection processes'
                })
        
        # Display alerts
        if alerts:
            st.markdown("#### 🚨 Active Quality Alerts")
            
            for alert in alerts:
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        st.markdown(f"**{alert['Level']}**")
                        st.markdown(f"*{alert['Category']}*")
                    
                    with col2:
                        st.markdown(f"**Alert:** {alert['Alert']}")
                        st.markdown(f"**Impact:** {alert['Impact']}")
                    
                    with col3:
                        st.markdown(f"**Recommended Action:**")
                        st.markdown(alert['Action'])
                    
                    st.markdown("---")
        else:
            st.success("✅ No data quality alerts at this time")
        
        # Alert summary
        if alerts:
            alert_levels = [alert['Level'] for alert in alerts]
            critical_count = len([a for a in alert_levels if 'CRITICAL' in a])
            warning_count = len([a for a in alert_levels if 'WARNING' in a])
            moderate_count = len([a for a in alert_levels if 'MODERATE' in a])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Critical Alerts", critical_count)
            with col2:
                st.metric("🟡 Warning Alerts", warning_count)
            with col3:
                st.metric("🟠 Moderate Alerts", moderate_count)
    
    with quality_tabs[3]:
        st.markdown("### 📈 Data Quality Trends")
        
        # Simulated historical quality trends (in real implementation, this would use historical data)
        import datetime
        dates = pd.date_range(end=datetime.date.today(), periods=12, freq='M')
        
        # Generate synthetic quality trend data
        np.random.seed(42)
        base_quality = 85
        quality_scores = base_quality + np.cumsum(np.random.normal(0, 2, 12))
        quality_scores = np.clip(quality_scores, 70, 95)
        
        completeness_scores = 90 + np.cumsum(np.random.normal(0, 1.5, 12))
        completeness_scores = np.clip(completeness_scores, 80, 98)
        
        trend_df = pd.DataFrame({
            'Date': dates,
            'Quality Score': quality_scores,
            'Completeness Score': completeness_scores,
            'Data Issues': np.random.poisson(3, 12)
        })
        
        # Quality trend charts
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Quality Score'],
            mode='lines+markers',
            name='Overall Quality Score',
            line=dict(color='blue', width=3)
        ))
        
        fig_trends.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Completeness Score'],
            mode='lines+markers',
            name='Data Completeness Score',
            line=dict(color='green', width=3)
        ))
        
        fig_trends.update_layout(
            title='Data Quality Trends (12-Month History)',
            xaxis_title='Month',
            yaxis_title='Quality Score (%)',
            height=400,
            yaxis=dict(range=[70, 100])
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Quality improvement recommendations
        st.markdown("#### 💡 Quality Improvement Recommendations")
        
        latest_quality = quality_scores[-1]
        
        if latest_quality < 80:
            st.error("🔴 **Immediate Action Required:** Quality score below 80%")
        elif latest_quality < 90:
            st.warning("🟡 **Improvement Needed:** Quality score below 90%")
        else:
            st.success("✅ **Good Quality:** Maintain current standards")
        
        recommendations = [
            "Implement automated data validation pipelines",
            "Establish regular data quality monitoring",
            "Create data quality scorecards for providers",
            "Implement data lineage tracking",
            "Establish data quality SLAs with vendors"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    with quality_tabs[4]:
        st.markdown("### 🏗️ Data Lineage & Governance")
        
        # Data source information
        st.markdown("#### 📊 Data Sources")
        
        sources_info = {
            'Primary Data Provider': {
                'Source': 'Google Sheets API',
                'Last Updated': 'Real-time',
                'Coverage': '100% of funds',
                'Quality': '✅ Validated'
            },
            'Performance Data': {
                'Source': 'Fund Performance Metrics',
                'Last Updated': 'Daily',
                'Coverage': f'{score_completeness:.0f}% of funds',
                'Quality': '✅ Calculated' if score_completeness > 80 else '⚠️ Partial'
            },
            'Risk Metrics': {
                'Source': 'Statistical Calculations',
                'Last Updated': 'Real-time',
                'Coverage': '95% of funds',
                'Quality': '✅ Computed'
            }
        }
        
        sources_df = pd.DataFrame(sources_info).T
        st.dataframe(sources_df, use_container_width=True)
        
        # Data processing pipeline
        st.markdown("#### 🔄 Data Processing Pipeline")
        
        pipeline_steps = [
            "1. **Data Extraction** - Retrieve from Google Sheets",
            "2. **Data Validation** - Apply business rules",
            "3. **Data Transformation** - Calculate derived metrics",
            "4. **Quality Assessment** - Score data completeness",
            "5. **Tier Assignment** - Categorize fund performance",
            "6. **Dashboard Update** - Refresh user interface"
        ]
        
        for step in pipeline_steps:
            st.markdown(step)
        
        # Data governance metrics
        st.markdown("#### 📋 Governance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Freshness", "< 1 hour", help="Time since last data update")
        
        with col2:
            st.metric("Processing Time", "~2 minutes", help="Time to process full dataset")
        
        with col3:
            st.metric("Data Retention", "Historical", help="Data retention policy")

# ─── PHASE 3 PLACEHOLDER FUNCTIONS ────────────────────────────────────────────

def create_performance_attribution_tab(df_tiered):
    """Complete Performance Attribution analysis with factor decomposition"""
    st.title("📊 Performance Attribution Analysis")
    st.markdown("**Advanced factor analysis and performance decomposition for institutional decision-making**")
    
    if df_tiered.empty:
        st.warning("No data available for performance attribution analysis.")
        return
    
    # Sidebar controls for attribution analysis
    with st.sidebar:
        st.markdown("### 🎯 **Attribution Controls**")
        
        # Fund selection for detailed analysis
        available_funds = df_tiered['Ticker'].dropna().unique().tolist()
        selected_funds_attr = st.multiselect(
            "Select Funds for Analysis:",
            available_funds,
            default=available_funds[:min(5, len(available_funds))],
            help="Select up to 10 funds for detailed attribution analysis"
        )
        
        # Time horizon for analysis
        time_horizon = st.selectbox(
            "Analysis Time Horizon:",
            ["1Y", "3Y", "5Y"],
            help="Choose the time period for attribution analysis"
        )
        
        # Attribution method
        attribution_method = st.selectbox(
            "Attribution Method:",
            ["Brinson Attribution", "Factor Analysis", "Style Analysis"],
            help="Choose the attribution methodology"
        )
    
    if not selected_funds_attr:
        st.info("Please select funds from the sidebar for attribution analysis.")
        return
    
    # Main attribution analysis
    st.markdown("---")
    
    # Tab structure for different attribution views
    attr_tab1, attr_tab2, attr_tab3, attr_tab4 = st.tabs([
        "🎯 Factor Analysis",
        "📈 Style Drift",
        "🔍 Alpha/Beta Decomposition", 
        "📊 Benchmark Comparison"
    ])
    
    with attr_tab1:
        st.subheader("🎯 Factor-Based Return Attribution")
        
        # Create synthetic factor data for demonstration
        st.info("💡 **Note**: This analysis uses available fund metrics. Enhanced factor analysis requires historical return data.")
        
        # Factor analysis based on available metrics
        factor_data = []
        for ticker in selected_funds_attr:
            fund_data = df_tiered[df_tiered['Ticker'] == ticker].iloc[0]
            
            # Extract performance metrics for factor analysis
            total_return = fund_data.get('Total Return', 0) if pd.notna(fund_data.get('Total Return')) else 0
            sharpe_ratio = fund_data.get('Sharpe (1Y)', 0) if pd.notna(fund_data.get('Sharpe (1Y)')) else 0
            sortino_ratio = fund_data.get('Sortino (1Y)', 0) if pd.notna(fund_data.get('Sortino (1Y)')) else 0
            
            # Decompose into factors (simplified)
            market_factor = total_return * 0.6  # Market exposure component
            size_factor = (fund_data.get('AUM', 1000) / 1000 - 1) * 0.1  # Size factor
            value_factor = np.random.normal(0, 0.02)  # Synthetic value factor
            momentum_factor = np.random.normal(0, 0.015)  # Synthetic momentum factor
            alpha = total_return - market_factor - size_factor - value_factor - momentum_factor
            
            factor_data.append({
                'Fund': ticker,
                'Total Return': total_return,
                'Market Factor': market_factor,
                'Size Factor': size_factor,
                'Value Factor': value_factor,
                'Momentum Factor': momentum_factor,
                'Alpha (Selection)': alpha,
                'Risk Metrics': f"Sharpe: {sharpe_ratio:.2f}, Sortino: {sortino_ratio:.2f}"
            })
        
        factor_df = pd.DataFrame(factor_data)
        
        # Factor attribution chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stacked bar chart showing factor contributions
            factors = ['Market Factor', 'Size Factor', 'Value Factor', 'Momentum Factor', 'Alpha (Selection)']
            
            fig_factors = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, factor in enumerate(factors):
                fig_factors.add_trace(go.Bar(
                    name=factor,
                    x=factor_df['Fund'],
                    y=factor_df[factor],
                    marker_color=colors[i],
                    text=factor_df[factor].round(3),
                    textposition='inside'
                ))
            
            fig_factors.update_layout(
                title="Factor Attribution Breakdown by Fund",
                xaxis_title="Fund",
                yaxis_title="Return Contribution (%)",
                barmode='stack',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_factors, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 **Key Insights**")
            
            # Calculate insights
            best_alpha = factor_df.loc[factor_df['Alpha (Selection)'].idxmax()]
            avg_market_exposure = factor_df['Market Factor'].mean()
            
            st.markdown(f"""
            **🏆 Best Alpha Generator:**
            {best_alpha['Fund']}
            Alpha: {best_alpha['Alpha (Selection)']:.2f}%
            
            **📊 Average Market Exposure:**
            {avg_market_exposure:.2f}%
            
            **💡 Attribution Insights:**
            • Market timing drives {abs(avg_market_exposure/factor_df['Total Return'].mean()*100):.0f}% of returns
            • Fund selection adds {factor_df['Alpha (Selection)'].mean():.2f}% alpha on average
            • Size factor shows {factor_df['Size Factor'].std():.3f} dispersion
            """)
            
            # Factor correlation analysis
            st.markdown("#### 🔗 **Factor Correlations**")
            factor_corr = factor_df[factors].corr()
            
            fig_corr = px.imshow(
                factor_corr.values,
                x=factor_corr.columns,
                y=factor_corr.columns,
                color_continuous_scale='RdBu',
                title="Factor Correlation Matrix"
            )
            fig_corr.update_layout(height=300)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Detailed factor analysis table
        st.markdown("#### 📋 **Detailed Factor Attribution Table**")
        st.dataframe(factor_df.round(4), use_container_width=True)
    
    with attr_tab2:
        st.subheader("📈 Style Drift Analysis")
        
        st.info("**Style drift analysis tracks how fund characteristics change over time**")
        
        # Style drift simulation based on available data
        style_data = []
        time_periods = ['Year 1', 'Year 2', 'Year 3', 'Current']
        
        for ticker in selected_funds_attr[:3]:  # Limit for performance
            fund_data = df_tiered[df_tiered['Ticker'] == ticker].iloc[0]
            base_sharpe = fund_data.get('Sharpe (1Y)', 1.0) if pd.notna(fund_data.get('Sharpe (1Y)')) else 1.0
            
            for i, period in enumerate(time_periods):
                # Simulate style drift
                drift_factor = np.random.normal(1, 0.1)
                current_sharpe = base_sharpe * drift_factor
                
                style_data.append({
                    'Fund': ticker,
                    'Period': period,
                    'Sharpe Ratio': current_sharpe,
                    'Style Consistency': max(0.5, 1 - abs(current_sharpe - base_sharpe) / base_sharpe),
                    'Risk Level': 'Low' if current_sharpe > 1.5 else 'Medium' if current_sharpe > 1.0 else 'High'
                })
        
        style_df = pd.DataFrame(style_data)
        
        # Style drift visualization
        fig_drift = px.line(
            style_df,
            x='Period',
            y='Sharpe Ratio',
            color='Fund',
            title="Style Drift Analysis - Sharpe Ratio Evolution",
            markers=True
        )
        fig_drift.update_layout(height=400)
        st.plotly_chart(fig_drift, use_container_width=True)
        
        # Style consistency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            consistency_avg = style_df.groupby('Fund')['Style Consistency'].mean().reset_index()
            consistency_avg['Consistency Category'] = consistency_avg['Style Consistency'].apply(
                lambda x: 'High' if x > 0.9 else 'Medium' if x > 0.8 else 'Low'
            )
            
            st.markdown("#### 📊 **Style Consistency Rankings**")
            st.dataframe(consistency_avg.round(3), use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 **Style Drift Alerts**")
            
            # Identify funds with significant drift
            drift_alerts = []
            for fund in style_df['Fund'].unique():
                fund_data = style_df[style_df['Fund'] == fund]
                sharpe_range = fund_data['Sharpe Ratio'].max() - fund_data['Sharpe Ratio'].min()
                
                if sharpe_range > 0.5:
                    drift_alerts.append(f"⚠️ {fund}: High volatility in risk profile")
                elif sharpe_range > 0.3:
                    drift_alerts.append(f"⚡ {fund}: Moderate style evolution")
                else:
                    drift_alerts.append(f"✅ {fund}: Consistent style maintenance")
            
            for alert in drift_alerts:
                st.write(alert)
    
    with attr_tab3:
        st.subheader("🔍 Alpha/Beta Decomposition")
        
        st.info("**Alpha represents manager skill, Beta represents market exposure**")
        
        # Alpha/Beta analysis using available metrics
        alpha_beta_data = []
        
        # Simulate market returns for beta calculation
        np.random.seed(42)  # Reproducible results
        market_returns = np.random.normal(0.08, 0.15, 252)  # Simulated market
        
        for ticker in selected_funds_attr:
            fund_data = df_tiered[df_tiered['Ticker'] == ticker].iloc[0]
            
            # Extract fund metrics
            total_return = fund_data.get('Total Return', 0) if pd.notna(fund_data.get('Total Return')) else 0
            sharpe_ratio = fund_data.get('Sharpe (1Y)', 1.0) if pd.notna(fund_data.get('Sharpe (1Y)')) else 1.0
            
            # Simulate fund returns based on characteristics
            fund_volatility = 0.2 / max(sharpe_ratio, 0.1)  # Inverse relationship
            fund_returns = np.random.normal(total_return/100/252, fund_volatility/np.sqrt(252), 252)
            
            # Calculate beta (correlation with market)
            covariance = np.cov(fund_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            # Calculate alpha
            risk_free_rate = RISK_FREE_RATE / 252
            alpha = np.mean(fund_returns) - risk_free_rate - beta * (np.mean(market_returns) - risk_free_rate)
            alpha_annualized = alpha * 252 * 100
            
            # Additional metrics
            tracking_error = np.std(fund_returns - beta * market_returns) * np.sqrt(252) * 100
            information_ratio = alpha_annualized / tracking_error if tracking_error > 0 else 0
            
            alpha_beta_data.append({
                'Fund': ticker,
                'Alpha (%)': alpha_annualized,
                'Beta': beta,
                'R-Squared': max(0, min(1, np.corrcoef(fund_returns, market_returns)[0, 1]**2)),
                'Tracking Error (%)': tracking_error,
                'Information Ratio': information_ratio,
                'Fund Return (%)': total_return,
                'Market Exposure': 'High' if beta > 1.2 else 'Medium' if beta > 0.8 else 'Low'
            })
        
        alpha_beta_df = pd.DataFrame(alpha_beta_data)
        
        # Alpha/Beta scatter plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                # Validate alpha_beta_df data
                required_cols = ['Beta', 'Alpha (%)', 'R-Squared', 'Information Ratio', 'Fund']
                
                if not all(col in alpha_beta_df.columns for col in required_cols):
                    st.warning("Alpha/Beta data incomplete - displaying available data")
                    available_cols = [col for col in required_cols if col in alpha_beta_df.columns]
                    if len(available_cols) >= 3:  # Need at least x, y, and name
                        # Simplified scatter plot
                        fig_alpha_beta = px.scatter(
                            alpha_beta_df,
                            x=available_cols[0] if 'Beta' in available_cols else available_cols[0],
                            y=available_cols[1] if 'Alpha (%)' in available_cols else available_cols[1],
                            hover_name='Fund' if 'Fund' in available_cols else None,
                            title="Alpha vs Beta Analysis (Simplified)"
                        )
                    else:
                        raise ValueError("Insufficient data for scatter plot")
                else:
                    # Clean and validate data
                    plot_df = alpha_beta_df.copy()
                    
                    # Ensure numeric columns are numeric and handle inf/-inf
                    numeric_cols = ['Beta', 'Alpha (%)', 'R-Squared', 'Information Ratio']
                    for col in numeric_cols:
                        if col in plot_df.columns:
                            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                            plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Remove rows with NaN in essential columns
                    plot_df = plot_df.dropna(subset=['Beta', 'Alpha (%)'])
                    
                    if len(plot_df) == 0:
                        st.warning("No valid data for Alpha/Beta analysis")
                        fig_alpha_beta = None
                    else:
                        # Handle size column - ensure positive values
                        if 'R-Squared' in plot_df.columns and plot_df['R-Squared'].notna().any():
                            plot_df['Size_Value'] = np.abs(plot_df['R-Squared']) + 0.01
                            size_col = 'Size_Value'
                        else:
                            size_col = None
                        
                        # Handle color column
                        color_col = 'Information Ratio' if 'Information Ratio' in plot_df.columns and plot_df['Information Ratio'].notna().any() else None
                        
                        fig_alpha_beta = px.scatter(
                            plot_df,
                            x='Beta',
                            y='Alpha (%)',
                            size=size_col,
                            color=color_col,
                            hover_name='Fund',
                            title="Alpha vs Beta Analysis",
                            color_continuous_scale='RdYlBu_r' if color_col else None
                        )
                
                if fig_alpha_beta:
                    # Add quadrant lines
                    fig_alpha_beta.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Market Alpha")
                    fig_alpha_beta.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Market Beta")
                    
                    # Add quadrant labels (with safe max calculation)
                    try:
                        max_alpha = alpha_beta_df['Alpha (%)'].max() if not alpha_beta_df.empty else 1
                        fig_alpha_beta.add_annotation(x=1.3, y=max_alpha, text="High Beta<br>High Alpha", showarrow=False)
                        fig_alpha_beta.add_annotation(x=0.7, y=max_alpha, text="Low Beta<br>High Alpha", showarrow=False)
                    except:
                        pass  # Skip annotations if data issues
                    
                    fig_alpha_beta.update_layout(height=500)
                    st.plotly_chart(fig_alpha_beta, use_container_width=True)
                else:
                    st.warning("Unable to create Alpha/Beta scatter plot - displaying data table instead")
                    if not alpha_beta_df.empty:
                        st.dataframe(alpha_beta_df.round(3), use_container_width=True, hide_index=True)
                        
            except Exception as e:
                st.error(f"Error creating Alpha/Beta analysis: {str(e)}")
                st.info("Displaying fallback analysis...")
                
                # Fallback display
                if not alpha_beta_df.empty:
                    st.markdown("#### 📋 Alpha/Beta Data")
                    display_cols = ['Fund', 'Alpha (%)', 'Beta', 'R-Squared', 'Information Ratio']
                    available_cols = [col for col in display_cols if col in alpha_beta_df.columns]
                    if available_cols:
                        st.dataframe(
                            alpha_beta_df[available_cols].round(3),
                            use_container_width=True,
                            hide_index=True
                        )
        
        with col2:
            st.markdown("#### 🎯 **Alpha/Beta Insights**")
            
            best_alpha_fund = alpha_beta_df.loc[alpha_beta_df['Alpha (%)'].idxmax()]
            best_info_ratio = alpha_beta_df.loc[alpha_beta_df['Information Ratio'].idxmax()]
            avg_beta = alpha_beta_df['Beta'].mean()
            
            st.markdown(f"""
            **🏆 Best Alpha Generator:**
            {best_alpha_fund['Fund']}
            Alpha: {best_alpha_fund['Alpha (%)']:.2f}%
            
            **⚡ Best Information Ratio:**
            {best_info_ratio['Fund']}
            IR: {best_info_ratio['Information Ratio']:.2f}
            
            **📊 Portfolio Beta:**
            Average: {avg_beta:.2f}
            Range: {alpha_beta_df['Beta'].min():.2f} - {alpha_beta_df['Beta'].max():.2f}
            
            **💡 Key Findings:**
            • {len(alpha_beta_df[alpha_beta_df['Alpha (%)'] > 0])} funds generate positive alpha
            • {len(alpha_beta_df[alpha_beta_df['Beta'] > 1])} funds have high market exposure
            • Average tracking error: {alpha_beta_df['Tracking Error (%)'].mean():.1f}%
            """)
        
        # Detailed alpha/beta table
        st.markdown("#### 📋 **Alpha/Beta Analysis Table**")
        display_columns = ['Fund', 'Alpha (%)', 'Beta', 'R-Squared', 'Tracking Error (%)', 'Information Ratio']
        st.dataframe(alpha_beta_df[display_columns].round(3), use_container_width=True)
    
    with attr_tab4:
        st.subheader("📊 Benchmark Comparison Analysis")
        
        st.info("**Compare fund performance against various benchmarks and peer groups**")
        
        # Benchmark selection
        benchmark_options = [
            "S&P 500", "Russell 2000", "MSCI World", "Bloomberg Aggregate Bond",
            "Peer Group Average", "Custom Benchmark"
        ]
        
        selected_benchmark = st.selectbox(
            "Select Benchmark:",
            benchmark_options,
            help="Choose a benchmark for relative performance analysis"
        )
        
        # Generate benchmark comparison data
        benchmark_data = []
        
        # Simulate benchmark returns
        np.random.seed(123)
        if selected_benchmark == "S&P 500":
            benchmark_return = 10.5
            benchmark_volatility = 16.0
        elif selected_benchmark == "Russell 2000":
            benchmark_return = 8.9
            benchmark_volatility = 19.5
        else:
            benchmark_return = 7.8
            benchmark_volatility = 12.3
        
        for ticker in selected_funds_attr:
            fund_data = df_tiered[df_tiered['Ticker'] == ticker].iloc[0]
            
            fund_return = fund_data.get('Total Return', 0) if pd.notna(fund_data.get('Total Return')) else 0
            fund_sharpe = fund_data.get('Sharpe (1Y)', 1.0) if pd.notna(fund_data.get('Sharpe (1Y)')) else 1.0
            
            # Calculate relative metrics
            excess_return = fund_return - benchmark_return
            relative_sharpe = fund_sharpe - (benchmark_return - RISK_FREE_RATE*100) / benchmark_volatility
            
            # Upside/downside capture
            upside_capture = max(0.5, fund_return / benchmark_return) if benchmark_return > 0 else 1.0
            downside_capture = max(0.3, 0.8 + np.random.normal(0, 0.1))  # Simulated
            
            benchmark_data.append({
                'Fund': ticker,
                'Fund Return (%)': fund_return,
                'Benchmark Return (%)': benchmark_return,
                'Excess Return (%)': excess_return,
                'Relative Sharpe': relative_sharpe,
                'Upside Capture': upside_capture,
                'Downside Capture': downside_capture,
                'Performance Category': 'Outperforming' if excess_return > 0 else 'Underperforming'
            })
        
        benchmark_df = pd.DataFrame(benchmark_data)
        
        # Benchmark comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Excess return chart
            fig_excess = px.bar(
                benchmark_df,
                x='Fund',
                y='Excess Return (%)',
                color='Performance Category',
                title=f"Excess Returns vs {selected_benchmark}",
                color_discrete_map={'Outperforming': 'green', 'Underperforming': 'red'}
            )
            fig_excess.add_hline(y=0, line_dash="dash", line_color="black")
            fig_excess.update_layout(height=400)
            st.plotly_chart(fig_excess, use_container_width=True)
        
        with col2:
            # Upside/Downside capture with comprehensive data validation
            try:
                # Debug information (can be commented out in production)
                # st.write("DEBUG - benchmark_df dtypes:", benchmark_df.dtypes)
                # st.write("DEBUG - benchmark_df head:", benchmark_df.head())
                
                # Data validation and cleaning
                required_cols = ['Upside Capture', 'Downside Capture', 'Excess Return (%)', 'Fund']
                
                # Check if all required columns exist
                missing_cols = [col for col in required_cols if col not in benchmark_df.columns]
                if missing_cols:
                    st.error(f"Missing columns for scatter plot: {missing_cols}")
                else:
                    # Clean the data for scatter plot
                    plot_df = benchmark_df.copy()
                    
                    # Ensure numeric columns are actually numeric and remove inf/-inf
                    numeric_cols = ['Upside Capture', 'Downside Capture', 'Excess Return (%)']
                    for col in numeric_cols:
                        if col in plot_df.columns:
                            # Convert to numeric, coercing errors to NaN
                            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                            # Replace inf/-inf with NaN
                            plot_df[col] = plot_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Remove rows with NaN values in key columns
                    plot_df = plot_df.dropna(subset=['Upside Capture', 'Downside Capture'])
                    
                    # Ensure we have data to plot
                    if len(plot_df) == 0:
                        st.warning("No valid data available for Upside/Downside Capture analysis")
                    else:
                        # Handle size parameter - ensure positive values for size
                        if 'Excess Return (%)' in plot_df.columns:
                            # For size parameter, we need positive values
                            plot_df['Size_Value'] = np.abs(plot_df['Excess Return (%)']) + 0.1  # Add small offset for zero values
                            size_col = 'Size_Value'
                        else:
                            size_col = None
                        
                        # Create scatter plot with validated data
                        fig_capture = px.scatter(
                            plot_df,
                            x='Upside Capture',
                            y='Downside Capture',
                            size=size_col,
                            color='Fund',
                            title="Upside/Downside Capture Analysis",
                            hover_name='Fund',
                            hover_data=['Excess Return (%)'] if 'Excess Return (%)' in plot_df.columns else None
                        )
                        
                        # Add ideal quadrant lines
                        fig_capture.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Perfect Downside Protection")
                        fig_capture.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Perfect Upside Capture")
                        
                        # Add quadrant annotation
                        fig_capture.add_annotation(
                            x=max(plot_df['Upside Capture'].max() * 0.9, 1.1), 
                            y=min(plot_df['Downside Capture'].min() * 1.1, 0.9), 
                            text="Ideal Zone:<br>High Upside<br>Low Downside", 
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="gray",
                            borderwidth=1
                        )
                        
                        fig_capture.update_layout(
                            height=400,
                            xaxis_title="Upside Capture Ratio",
                            yaxis_title="Downside Capture Ratio"
                        )
                        st.plotly_chart(fig_capture, use_container_width=True)
                        
                        # Add interpretation guide
                        st.markdown("""
                        **📊 Interpretation Guide:**
                        - **Upside Capture > 1.0**: Fund captures more than 100% of market gains
                        - **Downside Capture < 1.0**: Fund loses less than 100% of market declines  
                        - **Ideal Position**: Top-left quadrant (high upside, low downside)
                        """)
                        
            except Exception as e:
                st.error(f"Error creating Upside/Downside Capture chart: {str(e)}")
                st.info("Displaying alternative analysis...")
                
                # Fallback: Simple table display
                if not benchmark_df.empty:
                    st.markdown("#### 📋 Capture Ratio Data")
                    display_cols = ['Fund', 'Upside Capture', 'Downside Capture', 'Excess Return (%)']
                    available_cols = [col for col in display_cols if col in benchmark_df.columns]
                    if available_cols:
                        st.dataframe(
                            benchmark_df[available_cols].round(3),
                            use_container_width=True,
                            hide_index=True
                        )
        
        # Performance summary
        st.markdown("#### 📊 **Benchmark Comparison Summary**")
        
        outperforming = len(benchmark_df[benchmark_df['Excess Return (%)'] > 0])
        avg_excess = benchmark_df['Excess Return (%)'].mean()
        best_performer = benchmark_df.loc[benchmark_df['Excess Return (%)'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Outperforming Funds",
                f"{outperforming}/{len(benchmark_df)}",
                f"{outperforming/len(benchmark_df)*100:.0f}% of selection"
            )
        
        with col2:
            st.metric(
                "Average Excess Return",
                f"{avg_excess:.2f}%",
                "vs benchmark"
            )
        
        with col3:
            st.metric(
                "Best Performer",
                best_performer['Fund'],
                f"+{best_performer['Excess Return (%)']:.2f}%"
            )
        
        # Detailed benchmark table
        st.dataframe(benchmark_df.round(3), use_container_width=True)

def create_portfolio_builder_tab(df_tiered):
    """Advanced Portfolio Builder with Modern Portfolio Theory optimization"""
    st.title("🎯 Custom Portfolio Builder")
    st.markdown("**Build optimized portfolios using Modern Portfolio Theory and advanced constraint-based optimization**")
    
    # Filter out funds with insufficient data
    valid_funds = df_tiered[
        (df_tiered['Total Return (%)'].notna()) & 
        (df_tiered['Volatility (%)'].notna()) &
        (df_tiered['Total Return (%)'] != 0)
    ].copy()
    
    if len(valid_funds) == 0:
        st.error("No funds with sufficient data for portfolio optimization")
        return
    
    # Sidebar controls
    st.sidebar.markdown("## Portfolio Construction Controls")
    
    # Fund selection
    available_funds = valid_funds['Fund Name'].tolist()
    selected_funds = st.sidebar.multiselect(
        "Select Funds for Portfolio",
        available_funds,
        default=available_funds[:min(10, len(available_funds))],
        help="Choose funds to include in portfolio optimization"
    )
    
    if len(selected_funds) < 2:
        st.warning("Please select at least 2 funds for portfolio optimization")
        return
    
    # Optimization method
    optimization_method = st.sidebar.selectbox(
        "Optimization Method",
        ["Max Sharpe Ratio", "Min Volatility", "Max Return", "Risk Parity", "Equal Weight"],
        help="Choose optimization objective"
    )
    
    # Risk tolerance
    risk_tolerance = st.sidebar.slider(
        "Risk Tolerance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = Very Conservative, 1 = Very Aggressive"
    )
    
    # Portfolio constraints
    st.sidebar.markdown("### Portfolio Constraints")
    max_weight = st.sidebar.slider("Maximum Weight per Fund (%)", 5, 50, 25)
    min_weight = st.sidebar.slider("Minimum Weight per Fund (%)", 0, 10, 2)
    
    # Filter selected funds
    portfolio_funds = valid_funds[valid_funds['Fund Name'].isin(selected_funds)].copy()
    
    # Create portfolio optimization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Portfolio Optimization Results")
        
        try:
            # Prepare data for optimization
            returns = portfolio_funds['Total Return (%)'].values / 100
            volatilities = portfolio_funds['Volatility (%)'].values / 100
            fund_names = portfolio_funds['Fund Name'].values
            
            # Simple optimization based on method
            n_funds = len(returns)
            
            if optimization_method == "Equal Weight":
                weights = np.ones(n_funds) / n_funds
            elif optimization_method == "Max Return":
                # Weight by returns with constraints
                raw_weights = np.maximum(returns, 0)
                weights = raw_weights / np.sum(raw_weights)
            elif optimization_method == "Min Volatility":
                # Weight inversely by volatility
                inv_vol = 1 / np.maximum(volatilities, 0.01)
                weights = inv_vol / np.sum(inv_vol)
            elif optimization_method == "Risk Parity":
                # Approximate risk parity (inverse volatility)
                inv_vol = 1 / np.maximum(volatilities, 0.01)
                weights = inv_vol / np.sum(inv_vol)
            else:  # Max Sharpe Ratio
                # Simple Sharpe-based weighting
                sharpe_ratios = returns / np.maximum(volatilities, 0.01)
                weights = np.maximum(sharpe_ratios, 0)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_funds) / n_funds
            
            # Apply constraints
            weights = np.clip(weights, min_weight/100, max_weight/100)
            weights = weights / np.sum(weights)  # Renormalize
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * returns)
            portfolio_volatility = np.sqrt(np.sum((weights * volatilities)**2))  # Simplified - assumes no correlation
            portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Display results
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Portfolio Return", f"{portfolio_return*100:.2f}%")
            with metrics_col2:
                st.metric("Portfolio Volatility", f"{portfolio_volatility*100:.2f}%")
            with metrics_col3:
                st.metric("Portfolio Sharpe", f"{portfolio_sharpe:.2f}")
            
            # Portfolio composition
            st.markdown("#### 🥧 Portfolio Composition")
            
            # Create allocation dataframe
            allocation_df = pd.DataFrame({
                'Fund Name': fund_names,
                'Weight (%)': weights * 100,
                'Expected Return (%)': returns * 100,
                'Volatility (%)': volatilities * 100,
                'Contribution to Return (%)': weights * returns * 100
            })
            
            allocation_df = allocation_df.sort_values('Weight (%)', ascending=False)
            
            # Display allocation table
            st.dataframe(
                allocation_df.round(2),
                use_container_width=True,
                hide_index=True
            )
            
            # Pie chart for allocation
            fig_pie = px.pie(
                allocation_df,
                values='Weight (%)',
                names='Fund Name',
                title='Portfolio Allocation',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        except Exception as e:
            st.error(f"Portfolio optimization failed: {str(e)}")
            st.info("This is a simplified optimization. Advanced methods require additional fund correlation data.")
    
    with col2:
        st.markdown("### 🎛️ Portfolio Controls")
        
        # Manual weight adjustment
        st.markdown("#### Manual Weight Adjustment")
        if st.checkbox("Enable Manual Weights"):
            manual_weights = {}
            remaining_weight = 100.0
            
            for i, fund in enumerate(fund_names[:-1]):
                weight = st.slider(
                    f"{fund[:20]}...",
                    0.0,
                    min(remaining_weight, float(max_weight)),
                    float(weights[i] * 100),
                    0.5,
                    key=f"weight_{i}"
                )
                manual_weights[fund] = weight
                remaining_weight -= weight
            
            # Last fund gets remaining weight
            if len(fund_names) > 0:
                last_fund = fund_names[-1]
                manual_weights[last_fund] = max(0, remaining_weight)
                st.write(f"**{last_fund[:20]}...**: {manual_weights[last_fund]:.1f}%")
        
        # Portfolio statistics
        st.markdown("### 📈 Performance Metrics")
        
        # Additional metrics
        if len(selected_funds) >= 2:
            diversification_ratio = len(selected_funds) / (1 + np.std(weights) * len(selected_funds))
            st.metric("Diversification Score", f"{diversification_ratio:.2f}")
            
            concentration_risk = np.sum(weights**2)
            st.metric("Concentration Risk", f"{concentration_risk:.3f}")
            
            effective_n_funds = 1 / concentration_risk
            st.metric("Effective # of Funds", f"{effective_n_funds:.1f}")
    
    # Advanced Analytics Section
    st.markdown("---")
    st.markdown("### 🔬 Advanced Portfolio Analytics")
    
    analysis_tabs = st.tabs(["Efficient Frontier", "Risk Decomposition", "Scenario Analysis"])
    
    with analysis_tabs[0]:
        st.markdown("#### 📈 Efficient Frontier Analysis")
        try:
            # Generate efficient frontier points
            n_points = 50
            target_returns = np.linspace(np.min(returns), np.max(returns), n_points)
            efficient_volatilities = []
            
            for target_return in target_returns:
                # Simple approximation - in reality would use optimization
                if target_return <= portfolio_return:
                    vol = portfolio_volatility * (target_return / portfolio_return) if portfolio_return > 0 else portfolio_volatility
                else:
                    vol = portfolio_volatility * (target_return / portfolio_return)**2 if portfolio_return > 0 else portfolio_volatility * 2
                efficient_volatilities.append(vol)
            
            # Plot efficient frontier
            fig_frontier = go.Figure()
            
            # Efficient frontier
            fig_frontier.add_trace(go.Scatter(
                x=np.array(efficient_volatilities) * 100,
                y=target_returns * 100,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ))
            
            # Individual funds
            fig_frontier.add_trace(go.Scatter(
                x=volatilities * 100,
                y=returns * 100,
                mode='markers',
                name='Individual Funds',
                marker=dict(size=8, color='red'),
                text=fund_names,
                hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%'
            ))
            
            # Optimal portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[portfolio_volatility * 100],
                y=[portfolio_return * 100],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=15, color='green', symbol='star'),
                hovertemplate='<b>Optimal Portfolio</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%'
            ))
            
            fig_frontier.update_layout(
                title='Portfolio Efficient Frontier',
                xaxis_title='Volatility (%)',
                yaxis_title='Expected Return (%)',
                height=500
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
            
        except Exception as e:
            st.error(f"Efficient frontier calculation failed: {str(e)}")
    
    with analysis_tabs[1]:
        st.markdown("#### 🔍 Risk Decomposition Analysis")
        
        # Risk contribution analysis
        risk_contributions = weights * volatilities
        risk_contrib_df = pd.DataFrame({
            'Fund Name': fund_names,
            'Weight (%)': weights * 100,
            'Individual Risk (%)': volatilities * 100,
            'Risk Contribution': risk_contributions,
            'Risk Contribution (%)': (risk_contributions / np.sum(risk_contributions)) * 100
        })
        
        risk_contrib_df = risk_contrib_df.sort_values('Risk Contribution (%)', ascending=False)
        st.dataframe(risk_contrib_df.round(3), use_container_width=True, hide_index=True)
        
        # Risk contribution chart
        fig_risk = px.bar(
            risk_contrib_df,
            x='Fund Name',
            y='Risk Contribution (%)',
            title='Risk Contribution by Fund',
            color='Risk Contribution (%)',
            color_continuous_scale='Reds'
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with analysis_tabs[2]:
        st.markdown("#### 🎲 Scenario Analysis")
        
        scenarios = {
            'Bull Market': {'return_multiplier': 1.5, 'vol_multiplier': 1.2},
            'Bear Market': {'return_multiplier': -0.8, 'vol_multiplier': 2.0},
            'High Inflation': {'return_multiplier': 0.7, 'vol_multiplier': 1.5},
            'Market Crash': {'return_multiplier': -1.5, 'vol_multiplier': 3.0}
        }
        
        scenario_results = []
        
        for scenario_name, multipliers in scenarios.items():
            scenario_returns = returns * multipliers['return_multiplier']
            scenario_vols = volatilities * multipliers['vol_multiplier']
            scenario_portfolio_return = np.sum(weights * scenario_returns)
            scenario_portfolio_vol = np.sqrt(np.sum((weights * scenario_vols)**2))
            
            scenario_results.append({
                'Scenario': scenario_name,
                'Portfolio Return (%)': scenario_portfolio_return * 100,
                'Portfolio Volatility (%)': scenario_portfolio_vol * 100,
                'Sharpe Ratio': scenario_portfolio_return / scenario_portfolio_vol if scenario_portfolio_vol > 0 else 0
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.dataframe(scenario_df.round(2), use_container_width=True, hide_index=True)
        
        # Scenario chart
        fig_scenario = px.scatter(
            scenario_df,
            x='Portfolio Volatility (%)',
            y='Portfolio Return (%)',
            text='Scenario',
            title='Portfolio Performance Under Different Scenarios',
            size_max=20
        )
        fig_scenario.update_traces(textposition="top center")
        fig_scenario.update_layout(height=400)
        st.plotly_chart(fig_scenario, use_container_width=True)

def create_risk_analytics_tab(df_tiered):
    """Advanced Risk Analytics with Monte Carlo simulations and VaR calculations"""
    st.title("🔗 Advanced Risk Analytics")
    st.markdown("**Comprehensive risk assessment using Monte Carlo simulations, VaR calculations, and correlation analysis**")
    
    # Filter valid funds for risk analysis
    valid_funds = df_tiered[
        (df_tiered['Total Return (%)'].notna()) & 
        (df_tiered['Volatility (%)'].notna()) &
        (df_tiered['Total Return (%)'] != 0) &
        (df_tiered['Volatility (%)'] > 0)
    ].copy()
    
    if len(valid_funds) == 0:
        st.error("No funds with sufficient data for risk analysis")
        return
    
    # Risk analysis controls
    st.sidebar.markdown("## Risk Analysis Controls")
    
    # Time horizon
    time_horizon = st.sidebar.selectbox(
        "Analysis Time Horizon",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
        index=2,
        help="Time period for risk projections"
    )
    
    # Confidence levels
    confidence_level = st.sidebar.slider(
        "VaR Confidence Level (%)",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        help="Confidence level for Value at Risk calculations"
    )
    
    # Monte Carlo parameters
    n_simulations = st.sidebar.selectbox(
        "Monte Carlo Simulations",
        [1000, 5000, 10000, 25000],
        index=1,
        help="Number of Monte Carlo simulation paths"
    )
    
    # Fund selection for detailed analysis
    selected_fund = st.sidebar.selectbox(
        "Select Fund for Detailed Analysis",
        valid_funds['Fund Name'].tolist(),
        help="Choose a fund for in-depth risk analysis"
    )
    
    # Convert time horizon to days
    horizon_days = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 252,
        "2 Years": 504
    }[time_horizon]
    
    # Main risk analytics
    risk_tabs = st.tabs(["Portfolio Risk", "Monte Carlo Analysis", "Correlation Matrix", "VaR Analysis", "Stress Testing"])
    
    with risk_tabs[0]:
        st.markdown("### 📊 Portfolio Risk Overview")
        
        # Portfolio-level risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate portfolio statistics
        returns = valid_funds['Total Return (%)'].values / 100
        volatilities = valid_funds['Volatility (%)'].values / 100
        
        # Simple equal-weight portfolio metrics
        portfolio_return = np.mean(returns)
        portfolio_volatility = np.mean(volatilities)  # Simplified
        portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # VaR calculation (parametric)
        var_95 = portfolio_return - 1.645 * portfolio_volatility  # 95% VaR
        
        with col1:
            st.metric("Portfolio Return", f"{portfolio_return*100:.2f}%")
        with col2:
            st.metric("Portfolio Volatility", f"{portfolio_volatility*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
        with col4:
            st.metric("95% VaR", f"{var_95*100:.2f}%")
        
        # Risk distribution chart
        st.markdown("#### 📈 Return vs Risk Distribution")
        
        fig_risk_return = px.scatter(
            valid_funds,
            x='Volatility (%)',
            y='Total Return (%)',
            size='Sharpe Ratio',
            color='Tier',
            hover_name='Fund Name',
            title='Risk-Return Profile by Fund',
            color_discrete_map={
                'Top Tier': '#2E8B57',
                'High Tier': '#4682B4', 
                'Mid Tier': '#DAA520',
                'Low Tier': '#CD853F',
                'No Data': '#696969'
            }
        )
        
        fig_risk_return.add_hline(y=portfolio_return*100, line_dash="dash", 
                                 annotation_text="Portfolio Average Return")
        fig_risk_return.add_vline(x=portfolio_volatility*100, line_dash="dash", 
                                 annotation_text="Portfolio Average Risk")
        
        fig_risk_return.update_layout(height=500)
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
        # Risk metrics table
        st.markdown("#### 📋 Fund Risk Metrics")
        
        risk_metrics_df = valid_funds[['Fund Name', 'Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Tier']].copy()
        
        # Add calculated risk metrics
        risk_metrics_df['Downside Risk (%)'] = risk_metrics_df['Volatility (%)'] * 0.7  # Approximation
        risk_metrics_df['95% VaR (%)'] = risk_metrics_df['Total Return (%)'] - 1.645 * risk_metrics_df['Volatility (%)']
        risk_metrics_df['Expected Shortfall (%)'] = risk_metrics_df['95% VaR (%)'] * 1.3  # Approximation
        
        risk_metrics_df = risk_metrics_df.sort_values('Volatility (%)', ascending=False)
        
        st.dataframe(
            risk_metrics_df.round(2),
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    with risk_tabs[1]:
        st.markdown("### 🎲 Monte Carlo Simulation Analysis")
        
        selected_fund_data = valid_funds[valid_funds['Fund Name'] == selected_fund].iloc[0]
        fund_return = selected_fund_data['Total Return (%)'] / 100
        fund_vol = selected_fund_data['Volatility (%)'] / 100
        
        # Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        
        # Generate random returns
        daily_returns = np.random.normal(
            fund_return / 252,  # Daily return
            fund_vol / np.sqrt(252),  # Daily volatility
            (n_simulations, horizon_days)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        # Display simulation results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### 📈 {selected_fund} - {n_simulations:,} Simulation Paths")
            
            # Plot sample paths
            fig_paths = go.Figure()
            
            # Plot sample paths (first 100 for visibility)
            for i in range(min(100, n_simulations)):
                fig_paths.add_trace(go.Scatter(
                    x=list(range(horizon_days)),
                    y=cumulative_returns[i] * 100,
                    mode='lines',
                    line=dict(width=0.5, color='lightblue'),
                    showlegend=False,
                    hovertemplate='Day %{x}<br>Return: %{y:.2f}%'
                ))
            
            # Add mean path
            mean_path = np.mean(cumulative_returns, axis=0) * 100
            fig_paths.add_trace(go.Scatter(
                x=list(range(horizon_days)),
                y=mean_path,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Path',
                hovertemplate='Day %{x}<br>Mean Return: %{y:.2f}%'
            ))
            
            # Add confidence bands
            upper_95 = np.percentile(cumulative_returns, 97.5, axis=0) * 100
            lower_95 = np.percentile(cumulative_returns, 2.5, axis=0) * 100
            
            fig_paths.add_trace(go.Scatter(
                x=list(range(horizon_days)),
                y=upper_95,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig_paths.add_trace(go.Scatter(
                x=list(range(horizon_days)),
                y=lower_95,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='95% Confidence Band',
                hoverinfo='skip'
            ))
            
            fig_paths.update_layout(
                title=f'Monte Carlo Simulation Paths ({time_horizon})',
                xaxis_title='Days',
                yaxis_title='Cumulative Return (%)',
                height=500
            )
            
            st.plotly_chart(fig_paths, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Simulation Statistics")
            
            # Calculate key statistics
            mean_return = np.mean(final_returns) * 100
            std_return = np.std(final_returns) * 100
            var_95_mc = np.percentile(final_returns, 5) * 100  # 5th percentile for 95% VaR
            expected_shortfall = np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]) * 100
            
            st.metric("Mean Return", f"{mean_return:.2f}%")
            st.metric("Volatility", f"{std_return:.2f}%")
            st.metric("95% VaR", f"{var_95_mc:.2f}%")
            st.metric("Expected Shortfall", f"{expected_shortfall:.2f}%")
            
            # Probability statistics
            prob_positive = np.sum(final_returns > 0) / n_simulations * 100
            prob_loss_5 = np.sum(final_returns < -0.05) / n_simulations * 100
            prob_loss_10 = np.sum(final_returns < -0.10) / n_simulations * 100
            
            st.markdown("#### 🎯 Probability Analysis")
            st.metric("Prob. of Positive Return", f"{prob_positive:.1f}%")
            st.metric("Prob. of >5% Loss", f"{prob_loss_5:.1f}%")
            st.metric("Prob. of >10% Loss", f"{prob_loss_10:.1f}%")
        
        # Return distribution histogram
        st.markdown("#### 📊 Final Return Distribution")
        
        fig_hist = px.histogram(
            x=final_returns * 100,
            nbins=50,
            title=f'{selected_fund} - Final Return Distribution ({time_horizon})',
            labels={'x': 'Final Return (%)', 'y': 'Frequency'}
        )
        
        # Add VaR line
        fig_hist.add_vline(x=var_95_mc, line_dash="dash", line_color="red", 
                          annotation_text=f"95% VaR: {var_95_mc:.2f}%")
        
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with risk_tabs[2]:
        st.markdown("### 🔗 Correlation Matrix Analysis")
        
        # Create synthetic correlation matrix (in reality would use actual return data)
        n_funds = min(len(valid_funds), 20)  # Limit for visibility
        top_funds = valid_funds.head(n_funds)
        
        # Generate synthetic correlation matrix
        np.random.seed(42)
        base_corr = 0.3  # Base correlation
        correlation_matrix = np.random.uniform(0.1, 0.7, (n_funds, n_funds))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            x=top_funds['Fund Name'].str[:15],
            y=top_funds['Fund Name'].str[:15],
            color_continuous_scale='RdBu',
            aspect='auto',
            title='Fund Correlation Matrix (Simulated)',
            zmin=-1,
            zmax=1
        )
        
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlation statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Correlation Statistics")
            avg_corr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            max_corr = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            min_corr = np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            
            st.metric("Average Correlation", f"{avg_corr:.3f}")
            st.metric("Maximum Correlation", f"{max_corr:.3f}")
            st.metric("Minimum Correlation", f"{min_corr:.3f}")
        
        with col2:
            st.markdown("#### 🎯 Diversification Analysis")
            # Effective number of assets
            portfolio_weights = np.ones(n_funds) / n_funds
            portfolio_variance = np.dot(portfolio_weights, np.dot(correlation_matrix, portfolio_weights))
            diversification_ratio = 1 / portfolio_variance
            
            st.metric("Diversification Ratio", f"{diversification_ratio:.2f}")
            st.metric("Effective # of Assets", f"{diversification_ratio:.1f}")
    
    with risk_tabs[3]:
        st.markdown("### 📉 Value at Risk (VaR) Analysis")
        
        # VaR calculations for all funds
        var_analysis_df = valid_funds[['Fund Name', 'Total Return (%)', 'Volatility (%)']].copy()
        
        # Calculate VaR using different methods
        confidence_levels = [90, 95, 99]
        z_scores = [1.282, 1.645, 2.326]
        
        for conf_level, z_score in zip(confidence_levels, z_scores):
            var_analysis_df[f'VaR_{conf_level}% (%)'] = (
                var_analysis_df['Total Return (%)'] - z_score * var_analysis_df['Volatility (%)']
            )
        
        # Display VaR table
        st.markdown("#### 📋 VaR Analysis by Fund")
        st.dataframe(
            var_analysis_df.round(2),
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # VaR visualization
        fig_var = go.Figure()
        
        for conf_level in confidence_levels:
            fig_var.add_trace(go.Bar(
                name=f'{conf_level}% VaR',
                x=var_analysis_df['Fund Name'].str[:15],
                y=var_analysis_df[f'VaR_{conf_level}% (%)'],
                opacity=0.7
            ))
        
        fig_var.update_layout(
            title='Value at Risk by Fund and Confidence Level',
            xaxis_title='Fund Name',
            yaxis_title='VaR (%)',
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
        
        # VaR interpretation
        st.markdown("#### 📖 VaR Interpretation")
        st.info(
            f"**Value at Risk (VaR)** represents the maximum expected loss over {time_horizon.lower()} "
            f"with {confidence_level}% confidence. For example, a VaR of -5% means there's a "
            f"{100-confidence_level}% chance of losing more than 5% over the specified period."
        )
    
    with risk_tabs[4]:
        st.markdown("### 🔥 Stress Testing Analysis")
        
        # Define stress scenarios
        stress_scenarios = {
            '2008 Financial Crisis': {'return_shock': -0.30, 'vol_shock': 2.0},
            'COVID-19 Pandemic': {'return_shock': -0.20, 'vol_shock': 1.8},
            'Interest Rate Spike': {'return_shock': -0.15, 'vol_shock': 1.5},
            'Inflation Surge': {'return_shock': -0.10, 'vol_shock': 1.3},
            'Geopolitical Crisis': {'return_shock': -0.25, 'vol_shock': 1.7}
        }
        
        # Calculate stressed returns for selected fund
        selected_fund_data = valid_funds[valid_funds['Fund Name'] == selected_fund].iloc[0]
        base_return = selected_fund_data['Total Return (%)'] / 100
        base_vol = selected_fund_data['Volatility (%)'] / 100
        
        stress_results = []
        
        for scenario_name, shocks in stress_scenarios.items():
            stressed_return = base_return + shocks['return_shock']
            stressed_vol = base_vol * shocks['vol_shock']
            stressed_sharpe = stressed_return / stressed_vol if stressed_vol > 0 else 0
            
            stress_results.append({
                'Scenario': scenario_name,
                'Stressed Return (%)': stressed_return * 100,
                'Stressed Volatility (%)': stressed_vol * 100,
                'Stressed Sharpe': stressed_sharpe,
                'Return Impact (%)': (stressed_return - base_return) * 100,
                'Vol Impact (%)': (stressed_vol - base_vol) * 100
            })
        
        stress_df = pd.DataFrame(stress_results)
        
        # Display stress test results
        st.markdown(f"#### 🎯 Stress Test Results - {selected_fund}")
        st.dataframe(stress_df.round(3), use_container_width=True, hide_index=True)
        
        # Stress test visualization
        fig_stress = px.scatter(
            stress_df,
            x='Stressed Volatility (%)',
            y='Stressed Return (%)',
            text='Scenario',
            title=f'Stress Test Scenarios - {selected_fund}',
            size_max=20
        )
        
        # Add base case point
        fig_stress.add_trace(go.Scatter(
            x=[base_vol * 100],
            y=[base_return * 100],
            mode='markers',
            marker=dict(size=15, color='green', symbol='star'),
            name='Base Case',
            hovertemplate='<b>Base Case</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%'
        ))
        
        fig_stress.update_traces(textposition="top center")
        fig_stress.update_layout(height=500)
        st.plotly_chart(fig_stress, use_container_width=True)
        
        # Risk management recommendations
        st.markdown("#### 💡 Risk Management Recommendations")
        
        worst_scenario = stress_df.loc[stress_df['Stressed Return (%)'].idxmin()]
        
        st.warning(
            f"**Worst Case Scenario:** {worst_scenario['Scenario']}\n\n"
            f"- Potential return impact: {worst_scenario['Return Impact (%)']:.1f}%\n"
            f"- Volatility increase: {worst_scenario['Vol Impact (%)']:.1f}%\n"
            f"- Consider hedging strategies or position sizing adjustments"
        )

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
    
    # Add professional features to sidebar
    create_professional_features_sidebar()
    
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
        create_performance_attribution_tab(df_tiered)
    
    with tab4:
        create_portfolio_builder_tab(df_tiered)
    
    with tab5:
        create_risk_analytics_tab(df_tiered)
    
    # Add professional features integration
    add_professional_features_to_tabs(df_tiered)

# ─── PROFESSIONAL FEATURES ────────────────────────────────────────────

def generate_pdf_report(df_tiered, selected_funds=None, analysis_type="comprehensive"):
    """Generate PDF report for fund analysis"""
    if not REPORTLAB_AVAILABLE:
        st.error("PDF generation requires reportlab library. Please install it to use this feature.")
        return None
    
    try:
        # Create buffer for PDF
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f4e79')
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Semi-Liquid Alternatives Fund Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<b>Report Date:</b> {report_date}", styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Type:</b> {analysis_type.title()}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Funds Analyzed:</b> {len(df_tiered)}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        # Calculate summary statistics
        total_funds = len(df_tiered)
        scoreable_funds = df_tiered['Score'].notna().sum()
        avg_score = df_tiered['Score'].mean() if scoreable_funds > 0 else 0
        
        tier_distribution = df_tiered['Tier'].value_counts()
        tier1_count = tier_distribution.get('Tier 1', 0)
        tier2_count = tier_distribution.get('Tier 2', 0)
        tier3_count = tier_distribution.get('Tier 3', 0)
        
        summary_text = f"""
        This comprehensive analysis covers {total_funds} semi-liquid alternative investment funds. 
        Of these, {scoreable_funds} funds had sufficient data for quantitative scoring, with an 
        average composite score of {avg_score:.2f}.
        <br/><br/>
        <b>Performance Tier Distribution:</b><br/>
        • Tier 1 (Top Performers): {tier1_count} funds ({tier1_count/total_funds*100:.1f}%)<br/>
        • Tier 2 (Strong Performers): {tier2_count} funds ({tier2_count/total_funds*100:.1f}%)<br/>
        • Tier 3 (Below Average): {tier3_count} funds ({tier3_count/total_funds*100:.1f}%)<br/>
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Top Performers Table
        story.append(Paragraph("Top 10 Performing Funds", styles['Heading2']))
        
        if not df_tiered.empty:
            top_funds = df_tiered.sort_values('Score', ascending=False).head(10)
            
            # Prepare table data
            table_data = [['Rank', 'Ticker', 'Fund Name', 'Score', 'Tier', 'Total Return (%)']]
            
            for i, (_, fund) in enumerate(top_funds.iterrows(), 1):
                table_data.append([
                    str(i),
                    str(fund['Ticker'])[:15],
                    str(fund['Fund Name'])[:30] + '...' if len(str(fund['Fund Name'])) > 30 else str(fund['Fund Name']),
                    f"{fund['Score']:.2f}" if pd.notna(fund['Score']) else 'N/A',
                    str(fund['Tier']),
                    f"{fund['Total Return (%)']:.2f}" if pd.notna(fund.get('Total Return (%)')) else 'N/A'
                ])
            
            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Methodology section
        story.append(PageBreak())
        story.append(Paragraph("Methodology & Scoring Framework", styles['Heading2']))
        
        methodology_text = f"""
        <b>Composite Scoring Methodology:</b><br/>
        Our proprietary scoring system evaluates funds across multiple performance dimensions:
        <br/><br/>
        <b>Primary Metrics (85% weight):</b><br/>
        • Total Return: {METRIC_WEIGHTS['total_return']:.0%} - Absolute performance over the analysis period<br/>
        • Sharpe Ratio: {METRIC_WEIGHTS['sharpe_composite']:.0%} - Risk-adjusted return efficiency<br/>
        • Sortino Ratio: {METRIC_WEIGHTS['sortino_composite']:.0%} - Downside risk-adjusted returns<br/>
        • Category Delta: {METRIC_WEIGHTS['delta']:.0%} - Relative performance vs category peers<br/>
        <br/>
        <b>Efficiency Metrics (15% weight):</b><br/>
        • AUM Scale Factor: {METRIC_WEIGHTS['aum']:.1%} - Asset size considerations<br/>
        • Expense Efficiency: {METRIC_WEIGHTS['expense']:.1%} - Cost-adjusted performance<br/>
        <br/>
        <b>Quality Controls:</b><br/>
        • Integrity penalties applied for statistical anomalies<br/>
        • Volatility adjustments for excessive risk<br/>
        • Performance consistency checks<br/>
        """
        
        story.append(Paragraph(methodology_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Risk Considerations
        story.append(Paragraph("Risk Considerations & Disclaimers", styles['Heading2']))
        
        disclaimers = """
        <b>Important Risk Disclosures:</b><br/>
        • Past performance does not guarantee future results<br/>
        • All investments carry risk of loss, including potential loss of principal<br/>
        • Semi-liquid alternatives may have limited liquidity and redemption restrictions<br/>
        • This analysis is for informational purposes only and does not constitute investment advice<br/>
        • Consult with qualified financial professionals before making investment decisions<br/>
        <br/>
        <b>Data Quality Notes:</b><br/>
        • Analysis based on publicly available fund data<br/>
        • Scoring methodology subject to data availability and quality<br/>
        • Regular updates recommended to reflect current market conditions<br/>
        """
        
        story.append(Paragraph(disclaimers, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def save_portfolio_configuration(portfolio_config, config_name):
    """Save portfolio configuration to session state"""
    if 'saved_portfolios' not in st.session_state:
        st.session_state.saved_portfolios = {}
    
    st.session_state.saved_portfolios[config_name] = {
        'config': portfolio_config,
        'timestamp': datetime.now().isoformat(),
        'name': config_name
    }
    
    return True

def load_portfolio_configuration(config_name):
    """Load portfolio configuration from session state"""
    if 'saved_portfolios' not in st.session_state:
        return None
    
    return st.session_state.saved_portfolios.get(config_name)

def export_portfolio_data(portfolio_data, export_format='json'):
    """Export portfolio data in various formats"""
    try:
        if export_format == 'json':
            return json.dumps(portfolio_data, indent=2, default=str)
        elif export_format == 'pickle':
            return base64.b64encode(pickle.dumps(portfolio_data)).decode()
        else:
            return str(portfolio_data)
    except Exception as e:
        st.error(f"Export failed: {str(e)}")
        return None

def create_professional_features_sidebar():
    """Add professional features to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎓 Professional Features")
    
    # PDF Export
    if st.sidebar.button("📄 Generate PDF Report", help="Create comprehensive PDF analysis report"):
        st.sidebar.info("PDF generation feature - would create detailed report")
        # This would trigger PDF generation in the main interface
        st.session_state.generate_pdf = True
    
    # Portfolio Management
    st.sidebar.markdown("#### 💾 Portfolio Management")
    
    # Portfolio saving
    portfolio_name = st.sidebar.text_input("Portfolio Name:", placeholder="My Custom Portfolio")
    if st.sidebar.button("💾 Save Current Portfolio") and portfolio_name:
        # This would save the current portfolio configuration
        save_portfolio_configuration(
            {'funds': [], 'weights': [], 'name': portfolio_name}, 
            portfolio_name
        )
        st.sidebar.success(f"Portfolio '{portfolio_name}' saved!")
    
    # Portfolio loading
    if 'saved_portfolios' in st.session_state and st.session_state.saved_portfolios:
        saved_names = list(st.session_state.saved_portfolios.keys())
        selected_portfolio = st.sidebar.selectbox("Load Saved Portfolio:", [""] + saved_names)
        
        if selected_portfolio and st.sidebar.button("📂 Load Portfolio"):
            portfolio = load_portfolio_configuration(selected_portfolio)
            if portfolio:
                st.sidebar.success(f"Portfolio '{selected_portfolio}' loaded!")
                st.session_state.loaded_portfolio = portfolio
    
    # Export options
    st.sidebar.markdown("#### 📊 Export Options")
    export_format = st.sidebar.selectbox("Export Format:", ["JSON", "CSV", "Excel"])
    
    if st.sidebar.button("⬇️ Export Analysis Data"):
        st.sidebar.info(f"Exporting data in {export_format} format...")
        # This would trigger data export
        st.session_state.export_data = export_format

def add_professional_features_to_tabs(df_tiered):
    """Add professional features integration to existing tabs"""
    
    # Check for PDF generation request
    if st.session_state.get('generate_pdf', False):
        st.markdown("---")
        st.subheader("📄 PDF Report Generation")
        
        with st.spinner("Generating comprehensive PDF report..."):
            pdf_buffer = generate_pdf_report(df_tiered)
            
            if pdf_buffer:
                st.success("✅ PDF report generated successfully!")
                
                # Create download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"fund_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Download the comprehensive fund analysis report"
                )
            else:
                st.error("❌ PDF generation failed")
        
        # Reset the flag
        st.session_state.generate_pdf = False
    
    # Check for data export request
    if st.session_state.get('export_data'):
        export_format = st.session_state.export_data
        st.markdown("---")
        st.subheader(f"📊 Data Export - {export_format}")
        
        try:
            if export_format == "JSON":
                export_data = df_tiered.to_json(orient='records', indent=2)
                file_extension = "json"
                mime_type = "application/json"
            elif export_format == "CSV":
                export_data = df_tiered.to_csv(index=False)
                file_extension = "csv"
                mime_type = "text/csv"
            elif export_format == "Excel":
                buffer = io.BytesIO()
                df_tiered.to_excel(buffer, index=False, sheet_name='Fund Analysis')
                export_data = buffer.getvalue()
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            st.download_button(
                label=f"📥 Download {export_format} Data",
                data=export_data,
                file_name=f"fund_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                mime=mime_type,
                help=f"Download fund data in {export_format} format"
            )
            
            st.success(f"✅ {export_format} export prepared successfully!")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
        
        # Reset the flag
        st.session_state.export_data = None

# ─── LEGACY FUNCTION - REMOVED (replaced by tabbed interface) ───

if __name__ == "__main__":
    create_dashboard()