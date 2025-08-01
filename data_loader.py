"""
Data Loader Module for Semi-Liquid Fund Dashboard

This module handles:
- Google Sheets API connection and data loading
- Data cleaning and preprocessing
- Caching and performance optimization
- Error handling for data loading operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import logging
from typing import Optional, Dict, Any

from data_models import StandardColumns, DataQuality, DataFrameSchema
from error_handler import handle_errors, SafeDataAccess, ErrorMessages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── GOOGLE SHEETS CONFIGURATION ────────────────────────────────────────────

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

class GoogleSheetsLoader:
    """Handles Google Sheets data loading with error handling and caching"""
    
    def __init__(self, sheet_id: str):
        self.sheet_id = sheet_id
        self._client = None
    
    @st.cache_resource
    def _get_gspread_client(_self):
        """Get authenticated gspread client with caching"""
        try:
            # Try to load credentials from Streamlit secrets
            # First check if credentials are nested under gcp_service_account
            if "gcp_service_account" in st.secrets:
                credentials_dict = st.secrets["gcp_service_account"]
            # Otherwise, assume credentials are at the top level
            elif all(key in st.secrets for key in ["type", "project_id", "private_key", "client_email"]):
                credentials_dict = dict(st.secrets)
            else:
                st.error("❌ Google Sheets credentials not found in secrets")
                st.info("Please add Google service account credentials to your Streamlit secrets")
                return None
            
            # Create credentials
            credentials = Credentials.from_service_account_info(
                credentials_dict, scopes=SCOPES
            )
            return gspread.authorize(credentials)
                
        except Exception as e:
            logger.error(f"Failed to create gspread client: {str(e)}")
            st.error("❌ Failed to connect to Google Sheets")
            return None
    
    @property
    def client(self):
        """Lazy loading of gspread client"""
        if self._client is None:
            self._client = self._get_gspread_client()
        return self._client
    
    @st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
    def load_worksheet_data(_self, tab_keyword: str) -> Optional[pd.DataFrame]:
        """
        Load data from Google Sheets worksheet
        
        Args:
            tab_keyword: Keyword to find worksheet by title
            
        Returns:
            DataFrame with loaded and processed data, or None if failed
        """
        try:
            if _self.client is None:
                return None
                
            # Find worksheet by keyword
            sheet = _self.client.open_by_key(_self.sheet_id)
            ws = next((w for w in sheet.worksheets() if tab_keyword in w.title), None)
            
            if ws is None:
                available_sheets = [w.title for w in sheet.worksheets()]
                st.error(f"⚠️ No worksheet with '{tab_keyword}' found")
                st.error(f"Available sheets: {available_sheets}")
                return None
            
            # Get all values
            raw_values = ws.get_all_values()
            if not raw_values:
                st.error(f"⚠️ Worksheet '{ws.title}' is empty")
                return None
            
            # Process raw data
            df = _self._process_raw_data(raw_values, ws.title)
            
            if df is not None:
                logger.info(f"Successfully loaded {len(df)} rows from {ws.title}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading worksheet data: {str(e)}")
            st.error(f"❌ Failed to load data from Google Sheets: {str(e)}")
            return None
    
    def _process_raw_data(self, raw_values: list, sheet_title: str) -> Optional[pd.DataFrame]:
        """
        Process raw values from Google Sheets into cleaned DataFrame
        
        Args:
            raw_values: Raw values from Google Sheets
            sheet_title: Name of the sheet (for logging)
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            if len(raw_values) < 2:
                st.error(f"⚠️ Insufficient data in '{sheet_title}' (need header + data)")
                return None
            
            # Separate header and data
            header = [str(col).strip() for col in raw_values[0] if str(col).strip()]
            data_rows = raw_values[1:]
            
            if not header:
                st.error(f"⚠️ No valid headers found in '{sheet_title}'")
                return None
            
            if not data_rows:
                st.error(f"⚠️ No data rows found in '{sheet_title}'")
                return None
            
            # Process each data row
            processed_data = []
            header_len = len(header)
            
            for row in data_rows:
                # Convert to strings and clean
                clean_row = [str(val).strip() if val else '' for val in row]
                
                # Ensure consistent length
                if len(clean_row) < header_len:
                    clean_row.extend([''] * (header_len - len(clean_row)))
                elif len(clean_row) > header_len:
                    clean_row = clean_row[:header_len]
                
                processed_data.append(clean_row)
            
            # Create DataFrame
            df = pd.DataFrame(processed_data, columns=header)
            
            # Replace empty strings with None
            df = df.replace('', None)
            
            # Apply data standardization and cleaning
            df = DataQuality.clean_fund_data(df)
            
            # Convert numeric columns
            df = self._convert_numeric_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing raw data: {str(e)}")
            st.error(f"❌ Failed to process data from '{sheet_title}': {str(e)}")
            return None
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert appropriate columns to numeric types
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with converted numeric columns
        """
        if df is None or df.empty:
            return df
            
        df_converted = df.copy()
        
        # Columns that should be numeric
        numeric_columns = [
            StandardColumns.TOTAL_RETURN, StandardColumns.VOLATILITY,
            StandardColumns.SHARPE_RATIO, StandardColumns.SORTINO_RATIO,
            StandardColumns.AUM, StandardColumns.EXPENSE_RATIO,
            StandardColumns.SCORE, StandardColumns.MAX_DRAWDOWN,
            'Sharpe (1Y)', 'Sharpe (3Y)', 'Sharpe (5Y)',
            'Sortino (1Y)', 'Sortino (3Y)', 'Sortino (5Y)',
            'Total Return', 'Volatility', 'Score'
        ]
        
        for col in numeric_columns:
            if col in df_converted.columns:
                try:
                    # Handle percentage columns (remove % and convert)
                    if df_converted[col].dtype == 'object':
                        # Remove % signs and convert to float
                        df_converted[col] = df_converted[col].astype(str).str.replace('%', '').str.replace(',', '')
                    
                    # Convert to numeric
                    df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                    
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to numeric: {str(e)}")
        
        return df_converted

# ─── DATA VALIDATION AND QUALITY CHECKS ────────────────────────────────────

class DataValidator:
    """Validates loaded data quality and completeness"""
    
    @staticmethod
    def validate_fund_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of fund data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'data_quality_score': 100
        }
        
        if df is None or df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No data available")
            return validation_result
        
        # Check required columns
        required_cols = [StandardColumns.FUND, StandardColumns.TICKER]
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check data completeness
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > 50:
            validation_result['warnings'].append(f"High missing data rate: {missing_percentage:.1f}%")
            validation_result['data_quality_score'] -= 30
        elif missing_percentage > 20:
            validation_result['warnings'].append(f"Moderate missing data: {missing_percentage:.1f}%")
            validation_result['data_quality_score'] -= 15
        
        # Check for duplicate funds
        if StandardColumns.FUND in df.columns:
            duplicates = df[StandardColumns.FUND].duplicated().sum()
            if duplicates > 0:
                validation_result['warnings'].append(f"Found {duplicates} duplicate fund names")
                validation_result['data_quality_score'] -= 10
        
        # Check numeric data quality
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        for col in numeric_cols:
            infinite_values = np.isinf(df[col]).sum()
            if infinite_values > 0:
                validation_result['warnings'].append(f"Column '{col}' has {infinite_values} infinite values")
                validation_result['data_quality_score'] -= 5
        
        # Generate recommendations
        if missing_percentage > 20:
            validation_result['recommendations'].append("Consider improving data collection processes")
        
        if len(df) < 10:
            validation_result['recommendations'].append("Small dataset may limit analysis accuracy")
        
        return validation_result
    
    @staticmethod
    def show_validation_summary(validation_result: Dict[str, Any]):
        """Display validation summary in Streamlit"""
        
        if validation_result['is_valid']:
            st.success(f"✅ Data validation passed (Quality Score: {validation_result['data_quality_score']}/100)")
        else:
            st.error("❌ Data validation failed")
            for error in validation_result['errors']:
                st.error(f"• {error}")
        
        # Show warnings
        if validation_result['warnings']:
            with st.expander("⚠️ Data Quality Warnings", expanded=False):
                for warning in validation_result['warnings']:
                    st.warning(f"• {warning}")
        
        # Show recommendations
        if validation_result['recommendations']:
            with st.expander("💡 Recommendations", expanded=False):
                for rec in validation_result['recommendations']:
                    st.info(f"• {rec}")

# ─── MAIN DATA LOADING INTERFACE ────────────────────────────────────────────

@handle_errors("Failed to load fund data")
def load_fund_data(sheet_id: str, tab_keyword: str = "Fund") -> Optional[pd.DataFrame]:
    """
    Main function to load and validate fund data
    
    Args:
        sheet_id: Google Sheets ID
        tab_keyword: Keyword to find the worksheet
        
    Returns:
        Cleaned and validated DataFrame or None if failed
    """
    with st.spinner("Loading fund data from Google Sheets..."):
        loader = GoogleSheetsLoader(sheet_id)
        df = loader.load_worksheet_data(tab_keyword)
        
        if df is not None:
            # Validate data quality
            validation_result = DataValidator.validate_fund_data(df)
            DataValidator.show_validation_summary(validation_result)
            
            # Return data even if there are warnings (but not errors)
            if validation_result['is_valid']:
                return df
        
        return None

@handle_errors("Failed to load configuration")
def load_app_config() -> Dict[str, Any]:
    """
    Load application configuration
    
    Returns:
        Configuration dictionary
    """
    default_config = {
        'sheet_id': st.secrets.get("sheet_id", ""),
        'cache_ttl': 300,  # 5 minutes
        'max_rows_display': 1000,
        'performance_mode': True
    }
    
    return default_config

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def get_available_worksheets(sheet_id: str) -> list:
    """Get list of available worksheets in the Google Sheet"""
    try:
        loader = GoogleSheetsLoader(sheet_id)
        if loader.client:
            sheet = loader.client.open_by_key(sheet_id)
            return [ws.title for ws in sheet.worksheets()]
        return []
    except Exception as e:
        logger.error(f"Error getting worksheets: {str(e)}")
        return []

def refresh_data_cache():
    """Clear cached data to force refresh"""
    st.cache_data.clear()
    st.success("🔄 Data cache cleared. Data will be refreshed on next load.")