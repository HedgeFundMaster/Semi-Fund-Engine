"""
Data Models and Schema Validation for Semi-Liquid Fund Dashboard

This module provides:
- Standardized column name definitions
- DataFrame schema validation
- Column mapping and transformation functions
- Data quality checks and error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── STANDARDIZED COLUMN DEFINITIONS ────────────────────────────────────────

class StandardColumns:
    """Standardized column names for the fund dashboard"""
    
    # Core identification columns
    FUND = 'Fund'  # Main fund identifier (standardized from 'Fund Name')
    TICKER = 'Ticker'
    CATEGORY = 'Category'
    
    # Performance metrics
    TOTAL_RETURN = 'Total Return (%)'
    SHARPE_RATIO = 'Sharpe Ratio'
    SORTINO_RATIO = 'Sortino Ratio'
    VOLATILITY = 'Volatility (%)'
    
    # Risk metrics
    MAX_DRAWDOWN = 'Max Drawdown (%)'
    VAR_95 = 'VaR 95%'
    BETA = 'Beta'
    
    # Fund characteristics
    AUM = 'AUM'
    EXPENSE_RATIO = 'Expense Ratio (%)'
    INCEPTION_DATE = 'Inception Date'
    
    # Scoring and ranking
    SCORE = 'Score'
    TIER = 'Tier'
    RANK = 'Rank'
    
    # Time-specific metrics
    RETURN_1Y = '1Y Return (%)'
    RETURN_3Y = '3Y Return (%)'
    RETURN_5Y = '5Y Return (%)'
    
    SHARPE_1Y = 'Sharpe (1Y)'
    SHARPE_3Y = 'Sharpe (3Y)'
    SHARPE_5Y = 'Sharpe (5Y)'
    
    SORTINO_1Y = 'Sortino (1Y)'
    SORTINO_3Y = 'Sortino (3Y)'
    SORTINO_5Y = 'Sortino (5Y)'

# ─── COLUMN MAPPING CONFIGURATIONS ──────────────────────────────────────────

class ColumnMapping:
    """Maps various possible column names to standardized names"""
    
    # Common variations of fund name columns
    FUND_NAME_VARIATIONS = [
        'Fund Name', 'Fund', 'Name', 'Fund_Name', 'fund_name',
        'FundName', 'FUND_NAME', 'Fund Title', 'Product Name'
    ]
    
    # Common variations of ticker columns
    TICKER_VARIATIONS = [
        'Ticker', 'Symbol', 'Code', 'Fund Code', 'TICKER', 'ticker'
    ]
    
    # Performance metric variations
    RETURN_VARIATIONS = [
        'Total Return (%)', 'Total Return', 'Return (%)', 'Return',
        'Performance (%)', 'Performance', 'Net Return (%)'
    ]
    
    # Risk metric variations
    VOLATILITY_VARIATIONS = [
        'Volatility (%)', 'Volatility', 'Vol (%)', 'Vol', 'Std Dev (%)',
        'Standard Deviation (%)', 'Risk (%)'
    ]
    
    # AUM variations
    AUM_VARIATIONS = [
        'AUM', 'Assets Under Management', 'Total Assets', 'Fund Size',
        'AUM ($M)', 'AUM (M)', 'Assets ($M)'
    ]

# ─── SCHEMA VALIDATION ──────────────────────────────────────────────────────

class DataFrameSchema:
    """Validates and standardizes DataFrame schemas"""
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that required columns exist in DataFrame
        
        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        if df is None or df.empty:
            return False, required_cols
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        return len(missing_cols) == 0, missing_cols
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to match StandardColumns definitions
        
        Args:
            df: DataFrame with potentially non-standard column names
            
        Returns:
            DataFrame with standardized column names
        """
        if df is None or df.empty:
            return df
            
        df_copy = df.copy()
        column_map = {}
        
        # Map fund name variations
        for col in df_copy.columns:
            if col in ColumnMapping.FUND_NAME_VARIATIONS:
                column_map[col] = StandardColumns.FUND
            elif col in ColumnMapping.TICKER_VARIATIONS:
                column_map[col] = StandardColumns.TICKER
            elif col in ColumnMapping.RETURN_VARIATIONS:
                column_map[col] = StandardColumns.TOTAL_RETURN
            elif col in ColumnMapping.VOLATILITY_VARIATIONS:
                column_map[col] = StandardColumns.VOLATILITY
            elif col in ColumnMapping.AUM_VARIATIONS:
                column_map[col] = StandardColumns.AUM
        
        # Apply column mapping
        if column_map:
            df_copy = df_copy.rename(columns=column_map)
            logger.info(f"Standardized columns: {column_map}")
        
        return df_copy
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate data types for key columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'numeric_issues': [],
            'missing_data': [],
            'type_conversions': []
        }
        
        if df is None or df.empty:
            return validation_results
        
        # Check numeric columns
        numeric_columns = [
            StandardColumns.TOTAL_RETURN, StandardColumns.VOLATILITY,
            StandardColumns.SHARPE_RATIO, StandardColumns.SORTINO_RATIO,
            StandardColumns.AUM, StandardColumns.EXPENSE_RATIO, StandardColumns.SCORE
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Check if column can be converted to numeric
                    pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for missing data
                    missing_pct = df[col].isna().sum() / len(df) * 100
                    if missing_pct > 50:
                        validation_results['missing_data'].append(f"{col}: {missing_pct:.1f}% missing")
                        
                except Exception as e:
                    validation_results['numeric_issues'].append(f"{col}: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def validate_aum_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Specific validation for AUM data quality and scoring fairness
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with AUM validation results
        """
        aum_validation = {
            'total_funds': len(df) if df is not None else 0,
            'missing_aum_count': 0,
            'missing_aum_funds': [],
            'invalid_aum_count': 0,
            'invalid_aum_funds': [],
            'valid_aum_count': 0,
            'missing_percentage': 0.0,
            'recommendations': []
        }
        
        if df is None or df.empty:
            return aum_validation
            
        if StandardColumns.AUM not in df.columns:
            aum_validation['recommendations'].append("AUM column not found - funds will get neutral (0.5) AUM scores")
            return aum_validation
        
        aum_series = df[StandardColumns.AUM]
        fund_names = df.get(StandardColumns.FUND, df.index)
        
        # Check for missing AUM data
        missing_mask = aum_series.isna()
        aum_validation['missing_aum_count'] = missing_mask.sum()
        aum_validation['missing_aum_funds'] = fund_names[missing_mask].tolist()
        
        # Check for invalid AUM data (negative or zero values)
        valid_aum = pd.to_numeric(aum_series, errors='coerce')
        invalid_mask = (valid_aum <= 0) & (~valid_aum.isna())
        aum_validation['invalid_aum_count'] = invalid_mask.sum()
        aum_validation['invalid_aum_funds'] = fund_names[invalid_mask].tolist()
        
        # Count valid AUM data
        valid_mask = (valid_aum > 0)
        aum_validation['valid_aum_count'] = valid_mask.sum()
        
        # Calculate missing percentage
        total_funds = len(df)
        missing_total = aum_validation['missing_aum_count'] + aum_validation['invalid_aum_count']
        aum_validation['missing_percentage'] = (missing_total / total_funds) * 100 if total_funds > 0 else 0
        
        # Generate recommendations
        if aum_validation['missing_percentage'] > 30:
            aum_validation['recommendations'].append(f"High missing AUM rate ({aum_validation['missing_percentage']:.1f}%) - many funds will get neutral scores")
        if aum_validation['missing_percentage'] > 0:
            aum_validation['recommendations'].append("Funds with missing AUM will receive fair neutral scores (0.5) to avoid penalties")
        if aum_validation['invalid_aum_count'] > 0:
            aum_validation['recommendations'].append(f"{aum_validation['invalid_aum_count']} funds have invalid AUM values (≤0)")
            
        return aum_validation

# ─── DATA QUALITY FUNCTIONS ─────────────────────────────────────────────────

class DataQuality:
    """Data quality validation and cleaning functions"""
    
    @staticmethod
    def safe_column_access(df: pd.DataFrame, column: str, default_value: Any = None) -> pd.Series:
        """
        Safely access DataFrame column with fallback
        
        Args:
            df: DataFrame to access
            column: Column name to access
            default_value: Value to return if column doesn't exist
            
        Returns:
            Series or default value
        """
        if df is None or df.empty:
            return pd.Series([default_value] * (len(df) if df is not None else 0))
            
        if column in df.columns:
            return df[column]
        else:
            logger.warning(f"Column '{column}' not found. Using default value: {default_value}")
            return pd.Series([default_value] * len(df))
    
    @staticmethod
    def ensure_numeric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Ensure column is numeric, converting if necessary
        
        Args:
            df: DataFrame to modify
            column: Column name to ensure is numeric
            
        Returns:
            DataFrame with numeric column
        """
        if df is None or df.empty or column not in df.columns:
            return df
            
        df_copy = df.copy()
        try:
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        except Exception as e:
            logger.warning(f"Could not convert {column} to numeric: {str(e)}")
            
        return df_copy
    
    @staticmethod
    def clean_fund_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize fund data DataFrame
        
        Args:
            df: Raw fund data DataFrame
            
        Returns:
            Cleaned and standardized DataFrame
        """
        if df is None or df.empty:
            return df
            
        # Start with standardized column names
        df_clean = DataFrameSchema.standardize_column_names(df)
        
        # Ensure key numeric columns are properly typed
        numeric_columns = [
            StandardColumns.TOTAL_RETURN, StandardColumns.VOLATILITY,
            StandardColumns.SHARPE_RATIO, StandardColumns.SORTINO_RATIO,
            StandardColumns.AUM, StandardColumns.EXPENSE_RATIO, StandardColumns.SCORE
        ]
        
        for col in numeric_columns:
            df_clean = DataQuality.ensure_numeric_column(df_clean, col)
        
        # Clean text columns
        text_columns = [StandardColumns.FUND, StandardColumns.TICKER, StandardColumns.CATEGORY]
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        logger.info(f"Cleaned fund data: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        return df_clean

# ─── REQUIRED COLUMN SETS ───────────────────────────────────────────────────

class RequiredColumns:
    """Define required column sets for different operations"""
    
    # Minimum columns needed for basic display
    BASIC_DISPLAY = [
        StandardColumns.FUND,
        StandardColumns.TICKER
    ]
    
    # Columns needed for scoring
    SCORING = [
        StandardColumns.FUND,
        StandardColumns.TOTAL_RETURN,
        StandardColumns.SHARPE_RATIO
    ]
    
    # Columns needed for comparison tools
    COMPARISON = [
        StandardColumns.FUND,
        StandardColumns.TICKER,
        StandardColumns.TOTAL_RETURN,
        StandardColumns.VOLATILITY,
        StandardColumns.SHARPE_RATIO
    ]
    
    # Columns needed for portfolio optimization
    PORTFOLIO = [
        StandardColumns.FUND,
        StandardColumns.TOTAL_RETURN,
        StandardColumns.VOLATILITY
    ]
    
    # Columns needed for risk analysis
    RISK_ANALYSIS = [
        StandardColumns.FUND,
        StandardColumns.TOTAL_RETURN,
        StandardColumns.VOLATILITY,
        StandardColumns.MAX_DRAWDOWN
    ]

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def get_available_columns(df: pd.DataFrame) -> List[str]:
    """Get list of available columns in DataFrame"""
    if df is None or df.empty:
        return []
    return list(df.columns)

def validate_dataframe_for_operation(df: pd.DataFrame, operation: str) -> Tuple[bool, List[str], str]:
    """
    Validate DataFrame has required columns for specific operation
    
    Args:
        df: DataFrame to validate
        operation: Operation type ('basic', 'scoring', 'comparison', 'portfolio', 'risk')
        
    Returns:
        Tuple of (is_valid, missing_columns, error_message)
    """
    if df is None or df.empty:
        return False, [], "DataFrame is empty or None"
    
    operation_requirements = {
        'basic': RequiredColumns.BASIC_DISPLAY,
        'scoring': RequiredColumns.SCORING,
        'comparison': RequiredColumns.COMPARISON,
        'portfolio': RequiredColumns.PORTFOLIO,
        'risk': RequiredColumns.RISK_ANALYSIS
    }
    
    required_cols = operation_requirements.get(operation, RequiredColumns.BASIC_DISPLAY)
    is_valid, missing_cols = DataFrameSchema.validate_required_columns(df, required_cols)
    
    if not is_valid:
        available_cols = get_available_columns(df)
        error_msg = f"Missing columns for {operation}: {missing_cols}. Available: {available_cols}"
        return False, missing_cols, error_msg
    
    return True, [], "Validation passed"