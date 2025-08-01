"""
Error Handler for Semi-Liquid Fund Dashboard

This module provides:
- Graceful error handling for missing columns and data issues
- Fallback displays when charts and analysis fail
- User-friendly error messages
- Error recovery strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── DECORATOR FOR ERROR HANDLING ───────────────────────────────────────────

def handle_errors(fallback_message: str = "An error occurred", show_details: bool = False):
    """
    Decorator to handle errors gracefully in Streamlit functions
    
    Args:
        fallback_message: Message to show users when error occurs
        show_details: Whether to show technical details to users
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Show user-friendly error message
                st.error(f"⚠️ {fallback_message}")
                
                if show_details:
                    with st.expander("Technical Details (for debugging)"):
                        st.code(f"Function: {func.__name__}")
                        st.code(f"Error: {str(e)}")
                        st.code(f"Args: {args}")
                        st.code(f"Kwargs: {kwargs}")
                
                return None
        return wrapper
    return decorator

# ─── ERROR TYPES AND MESSAGES ───────────────────────────────────────────────

class ErrorMessages:
    """Standardized user-friendly error messages"""
    
    # Data-related errors
    NO_DATA = "📊 No data available. Please check your data source or try refreshing."
    INSUFFICIENT_DATA = "📈 Insufficient data for analysis. At least 3 data points are required."
    MISSING_COLUMNS = "📋 Required data columns are missing. Please check your data format."
    
    # Chart and visualization errors
    CHART_FAILED = "📈 Unable to create chart. Displaying data in table format instead."
    PLOT_ERROR = "📊 Visualization error occurred. Showing alternative display."
    
    # Analysis errors
    CALCULATION_ERROR = "🧮 Analysis calculation failed. Using simplified method."
    SCORING_ERROR = "⭐ Scoring calculation failed. Using basic ranking instead."
    
    # Selection and filtering errors
    NO_SELECTION = "👆 Please select items from the sidebar to continue."
    INVALID_FILTER = "🔍 Invalid filter criteria. Showing all available data."
    
    # Data quality errors
    DATA_QUALITY = "⚠️ Data quality issues detected. Results may be incomplete."
    MISSING_VALUES = "❓ Some data values are missing. Analysis continues with available data."

class ErrorTypes:
    """Categories of errors for different handling strategies"""
    
    CRITICAL = "critical"  # Prevents operation entirely
    WARNING = "warning"   # Shows warning but continues
    INFO = "info"         # Informational message only

# ─── SAFE DATA ACCESS FUNCTIONS ─────────────────────────────────────────────

class SafeDataAccess:
    """Safe methods for accessing DataFrame data with error handling"""
    
    @staticmethod
    def safe_column_check(df: pd.DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Safely check if columns exist in DataFrame
        
        Args:
            df: DataFrame to check
            columns: List of column names to verify
            
        Returns:
            Tuple of (all_exist, missing_columns)
        """
        if df is None or df.empty:
            return False, columns
            
        missing = [col for col in columns if col not in df.columns]
        return len(missing) == 0, missing
    
    @staticmethod
    def safe_get_column(df: pd.DataFrame, column: str, default_value: Any = None) -> pd.Series:
        """
        Safely get column from DataFrame with fallback
        
        Args:
            df: DataFrame to access
            column: Column name
            default_value: Value to use if column missing
            
        Returns:
            Series or series of default values
        """
        if df is None or df.empty:
            return pd.Series(dtype=object)
            
        if column in df.columns:
            return df[column]
        else:
            logger.warning(f"Column '{column}' not found, using default: {default_value}")
            return pd.Series([default_value] * len(df))
    
    @staticmethod
    def safe_filter_dataframe(df: pd.DataFrame, filter_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Safely filter DataFrame with error handling
        
        Args:
            df: DataFrame to filter
            filter_dict: Dictionary of column:value pairs for filtering
            
        Returns:
            Filtered DataFrame or original if filtering fails
        """
        if df is None or df.empty:
            return df
            
        try:
            filtered_df = df.copy()
            for column, value in filter_dict.items():
                if column in filtered_df.columns:
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
                else:
                    logger.warning(f"Filter column '{column}' not found")
                    
            return filtered_df
            
        except Exception as e:
            logger.error(f"DataFrame filtering failed: {str(e)}")
            st.warning(ErrorMessages.INVALID_FILTER)
            return df

# ─── FALLBACK DISPLAY FUNCTIONS ─────────────────────────────────────────────

class FallbackDisplays:
    """Fallback display methods when primary visualizations fail"""
    
    @staticmethod
    def show_data_table(df: pd.DataFrame, title: str = "Data Table", max_rows: int = 100):
        """
        Show data as a table when charts fail
        
        Args:
            df: DataFrame to display
            title: Table title
            max_rows: Maximum number of rows to show
        """
        if df is None or df.empty:
            st.info(ErrorMessages.NO_DATA)
            return
            
        st.subheader(f"📋 {title}")
        
        # Limit rows for performance
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        if len(df) > max_rows:
            st.info(f"Showing first {max_rows} of {len(df)} rows")
            
        st.dataframe(display_df, use_container_width=True)
    
    @staticmethod
    def show_summary_metrics(df: pd.DataFrame, numeric_columns: List[str]):
        """
        Show summary statistics when detailed analysis fails
        
        Args:
            df: DataFrame to summarize
            numeric_columns: List of numeric columns to analyze
        """
        if df is None or df.empty:
            st.info(ErrorMessages.NO_DATA)
            return
            
        st.subheader("📊 Summary Statistics")
        
        # Calculate safe statistics
        for col in numeric_columns:
            if col in df.columns:
                try:
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(values) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"{col} Mean", f"{values.mean():.2f}")
                        with col2:
                            st.metric(f"{col} Median", f"{values.median():.2f}")
                        with col3:
                            st.metric(f"{col} Min", f"{values.min():.2f}")
                        with col4:
                            st.metric(f"{col} Max", f"{values.max():.2f}")
                except Exception as e:
                    logger.error(f"Summary calculation failed for {col}: {str(e)}")
    
    @staticmethod
    def show_simple_list(items: List[str], title: str = "Items", max_items: int = 50):
        """
        Show simple list when complex displays fail
        
        Args:
            items: List of items to display
            title: List title
            max_items: Maximum items to show
        """
        if not items:
            st.info(f"No {title.lower()} available")
            return
            
        st.subheader(f"📝 {title}")
        
        display_items = items[:max_items] if len(items) > max_items else items
        
        if len(items) > max_items:
            st.info(f"Showing first {max_items} of {len(items)} items")
            
        for i, item in enumerate(display_items, 1):
            st.write(f"{i}. {item}")

# ─── CHART ERROR HANDLING ───────────────────────────────────────────────────

class ChartErrorHandler:
    """Handle chart and visualization errors gracefully"""
    
    @staticmethod
    def safe_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Scatter Plot", **kwargs):
        """
        Safely create scatter plot with fallback to table
        
        Args:
            df: DataFrame for plotting
            x_col: X-axis column
            y_col: Y-axis column
            title: Plot title
            **kwargs: Additional plot arguments
            
        Returns:
            Plotly figure or None if failed
        """
        try:
            # Import plotly here to avoid issues if not available
            import plotly.express as px
            
            if df is None or df.empty:
                st.warning(ErrorMessages.NO_DATA)
                return None
                
            # Check required columns
            missing_cols = [col for col in [x_col, y_col] if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns for chart: {missing_cols}")
                FallbackDisplays.show_data_table(df, f"{title} - Data Table")
                return None
                
            # Clean data for plotting
            plot_df = df[[x_col, y_col]].copy()
            plot_df = plot_df.dropna()
            
            if len(plot_df) == 0:
                st.warning("No valid data points for visualization")
                return None
                
            # Create scatter plot
            fig = px.scatter(plot_df, x=x_col, y=y_col, title=title, **kwargs)
            st.plotly_chart(fig, use_container_width=True)
            return fig
            
        except Exception as e:
            logger.error(f"Scatter plot failed: {str(e)}")
            st.error(ErrorMessages.CHART_FAILED)
            FallbackDisplays.show_data_table(df[[x_col, y_col]] if x_col in df.columns and y_col in df.columns else df, 
                                           f"{title} - Data Table")
            return None
    
    @staticmethod
    def safe_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Bar Chart", **kwargs):
        """
        Safely create bar chart with fallback
        
        Args:
            df: DataFrame for plotting
            x_col: X-axis column
            y_col: Y-axis column
            title: Chart title
            **kwargs: Additional chart arguments
        """
        try:
            if df is None or df.empty:
                st.warning(ErrorMessages.NO_DATA)
                return None
                
            # Check columns exist
            if x_col not in df.columns or y_col not in df.columns:
                missing = [col for col in [x_col, y_col] if col not in df.columns]
                st.error(f"Missing columns: {missing}")
                FallbackDisplays.show_data_table(df, f"{title} - Data")
                return None
                
            # Use Streamlit's built-in bar chart as fallback
            chart_data = df.set_index(x_col)[y_col]
            st.bar_chart(chart_data)
            
        except Exception as e:
            logger.error(f"Bar chart failed: {str(e)}")
            st.error(ErrorMessages.CHART_FAILED)
            FallbackDisplays.show_data_table(df, f"{title} - Data")

# ─── VALIDATION FUNCTIONS ───────────────────────────────────────────────────

class ValidationHandler:
    """Handle data validation with user-friendly messages"""
    
    @staticmethod
    def validate_selection(selected_items: List[Any], item_type: str = "items") -> bool:
        """
        Validate user selection with appropriate messaging
        
        Args:
            selected_items: List of selected items
            item_type: Type of items being selected (for error message)
            
        Returns:
            True if valid selection, False otherwise
        """
        if not selected_items:
            st.info(f"👆 Please select {item_type} from the sidebar to continue.")
            return False
        return True
    
    @staticmethod
    def validate_dataframe_basic(df: pd.DataFrame, operation_name: str = "operation") -> bool:
        """
        Basic DataFrame validation with user messaging
        
        Args:
            df: DataFrame to validate
            operation_name: Name of operation for error message
            
        Returns:
            True if valid, False otherwise
        """
        if df is None:
            st.error(f"❌ No data available for {operation_name}")
            return False
            
        if df.empty:
            st.warning(f"📊 No data found for {operation_name}")
            return False
            
        return True
    
    @staticmethod
    def validate_numeric_data(df: pd.DataFrame, columns: List[str], min_valid_rows: int = 3) -> Tuple[bool, str]:
        """
        Validate numeric data quality
        
        Args:
            df: DataFrame to validate
            columns: List of numeric columns to check
            min_valid_rows: Minimum number of valid rows required
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "No data available"
            
        valid_rows = 0
        for _, row in df.iterrows():
            if all(pd.notna(row.get(col)) and pd.to_numeric(row.get(col), errors='coerce') is not pd.NA 
                   for col in columns if col in df.columns):
                valid_rows += 1
                
        if valid_rows < min_valid_rows:
            return False, f"Insufficient valid data rows. Found {valid_rows}, need at least {min_valid_rows}"
            
        return True, "Data validation passed"

# ─── CONTEXT MANAGERS FOR ERROR HANDLING ───────────────────────────────────

class StreamlitErrorContext:
    """Context manager for handling Streamlit operations with error recovery"""
    
    def __init__(self, operation_name: str, fallback_func: Optional[Callable] = None):
        self.operation_name = operation_name
        self.fallback_func = fallback_func
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in {self.operation_name}: {exc_val}")
            st.error(f"⚠️ {self.operation_name} failed: {str(exc_val)}")
            
            if self.fallback_func:
                try:
                    st.info("Attempting alternative approach...")
                    self.fallback_func()
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    st.error("Alternative approach also failed. Please check your data or contact support.")
            
            return True  # Suppress the exception

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def log_error_context(df: pd.DataFrame, operation: str, additional_context: Dict[str, Any] = None):
    """
    Log error context for debugging
    
    Args:
        df: DataFrame involved in error
        operation: Operation that failed
        additional_context: Additional context information
    """
    context = {
        'operation': operation,
        'dataframe_shape': df.shape if df is not None else None,
        'dataframe_columns': list(df.columns) if df is not None and not df.empty else None,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if additional_context:
        context.update(additional_context)
        
    logger.error(f"Error context: {context}")

def show_data_quality_warnings(df: pd.DataFrame):
    """
    Show data quality warnings to users
    
    Args:
        df: DataFrame to check for quality issues
    """
    if df is None or df.empty:
        return
        
    warnings = []
    
    # Check for missing data
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 10:
        warnings.append(f"⚠️ {missing_pct:.1f}% of data values are missing")
    
    # Check for duplicate rows
    duplicate_pct = df.duplicated().sum() / len(df) * 100
    if duplicate_pct > 5:
        warnings.append(f"⚠️ {duplicate_pct:.1f}% of rows are duplicates")
    
    # Show warnings if any
    if warnings:
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)