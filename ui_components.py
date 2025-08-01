"""
UI Components Module for Semi-Liquid Fund Dashboard

This module provides:
- Reusable Streamlit UI components
- Chart and visualization functions
- Interactive widgets and controls
- Consistent styling and layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import logging

from data_models import StandardColumns
from error_handler import ChartErrorHandler, SafeDataAccess, ValidationHandler
from scoring_engine import ScoringUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── STYLING AND THEME ──────────────────────────────────────────────────────

class DashboardTheme:
    """Consistent theme and styling for the dashboard"""
    
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8',
        'tier_1': '#d4edda',
        'tier_2': '#fff3cd',
        'tier_3': '#f8d7da'
    }
    
    TIER_COLORS = {
        'Tier 1': '#28a745',
        'Tier 2': '#ffc107', 
        'Tier 3': '#dc3545'
    }
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling to the dashboard"""
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

# ─── HEADER AND NAVIGATION COMPONENTS ───────────────────────────────────────

class HeaderComponents:
    """Header and navigation components"""
    
    @staticmethod
    def display_main_header():
        """Display the main dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏦 Semi-Liquid Alternatives Fund Dashboard</h1>
            <p>Professional Fund Analysis & Portfolio Construction Tool</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_page_header(title: str, subtitle: str = None, icon: str = "📊"):
        """Display page-specific header"""
        st.markdown(f"# {icon} {title}")
        if subtitle:
            st.markdown(f"*{subtitle}*")
        st.markdown("---")
    
    @staticmethod
    def display_metrics_row(metrics: Dict[str, Any]):
        """Display a row of key metrics"""
        if not metrics:
            return
            
        cols = st.columns(len(metrics))
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, dict):
                    st.metric(label, value.get('value', 'N/A'), 
                             delta=value.get('delta'), 
                             delta_color=value.get('delta_color', 'normal'))
                else:
                    st.metric(label, value)

# ─── DATA DISPLAY COMPONENTS ────────────────────────────────────────────────

class DataDisplayComponents:
    """Components for displaying tabular data"""
    
    @staticmethod
    def display_fund_table(df: pd.DataFrame, 
                          columns: List[str] = None,
                          title: str = "Fund Data",
                          max_rows: int = 50,
                          interactive: bool = True):
        """
        Display fund data in a formatted table
        
        Args:
            df: DataFrame to display
            columns: Specific columns to show
            title: Table title
            max_rows: Maximum rows to display
            interactive: Whether to make table interactive
        """
        if not ValidationHandler.validate_dataframe_basic(df, "table display"):
            return
            
        st.subheader(f"📋 {title}")
        
        # Select columns to display
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            if not available_cols:
                st.error("None of the requested columns are available")
                return
            display_df = df[available_cols].copy()
        else:
            display_df = df.copy()
        
        # Limit rows for performance
        if len(display_df) > max_rows:
            display_df = display_df.head(max_rows)
            st.info(f"Showing first {max_rows} of {len(df)} rows")
        
        # Format numeric columns
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'Score' in col or 'Return' in col or 'Ratio' in col:
                display_df[col] = display_df[col].round(2)
        
        # Display table
        if interactive:
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.table(display_df)
    
    @staticmethod
    def display_fund_cards(df: pd.DataFrame, max_cards: int = 6):
        """Display fund information as cards"""
        
        if not ValidationHandler.validate_dataframe_basic(df, "card display"):
            return
            
        # Limit number of cards
        display_df = df.head(max_cards)
        
        for idx, fund in display_df.iterrows():
            fund_name = SafeDataAccess.safe_get_column(pd.DataFrame([fund]), StandardColumns.FUND, 'Unknown').iloc[0]
            score = fund.get(StandardColumns.SCORE, 0)
            tier = fund.get(StandardColumns.TIER, 'N/A')
            
            # Determine score class
            score_class = 'score-high' if score >= 75 else 'score-medium' if score >= 50 else 'score-low'
            
            st.markdown(f"""
            <div class="fund-card">
                <h4>{fund_name}</h4>
                <p><strong>Ticker:</strong> {fund.get('Ticker', 'N/A')}</p>
                <p><strong>Score:</strong> <span class="{score_class}">{score:.1f}</span></p>
                <p><strong>Tier:</strong> <span class="tier-badge tier-{tier.lower().replace(' ', '-')}">{tier}</span></p>
                <p><strong>Category:</strong> {fund.get('Category', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

# ─── CHART COMPONENTS ───────────────────────────────────────────────────────

class ChartComponents:
    """Reusable chart components with error handling"""
    
    @staticmethod
    def create_performance_scatter(df: pd.DataFrame, 
                                 x_col: str = StandardColumns.VOLATILITY,
                                 y_col: str = StandardColumns.TOTAL_RETURN,
                                 color_col: str = StandardColumns.SCORE,
                                 size_col: str = None,
                                 title: str = "Risk vs Return Analysis"):
        """
        Create performance scatter plot with safe error handling
        
        Args:
            df: DataFrame with fund data
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Color mapping column
            size_col: Size mapping column
            title: Chart title
        """
        try:
            if not ValidationHandler.validate_dataframe_basic(df, "scatter plot"):
                return None
                
            # Check required columns
            required_cols = [x_col, y_col]
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) < 2:
                st.error(f"Missing required columns for scatter plot: {[col for col in required_cols if col not in df.columns]}")
                return None
            
            # Prepare data
            plot_df = df.copy()
            
            # Clean data
            plot_df = plot_df.dropna(subset=[x_col, y_col])
            if len(plot_df) == 0:
                st.warning("No valid data points for scatter plot")
                return None
            
            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col in plot_df.columns else None,
                size=size_col if size_col and size_col in plot_df.columns else None,
                hover_name=StandardColumns.FUND if StandardColumns.FUND in plot_df.columns else None,
                title=title,
                labels={
                    x_col: x_col.replace('(%)', '').strip(),
                    y_col: y_col.replace('(%)', '').strip()
                }
            )
            
            fig.update_layout(
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return fig
            
        except Exception as e:
            logger.error(f"Scatter plot creation failed: {str(e)}")
            ChartErrorHandler.safe_scatter_plot(df, x_col, y_col, title)
            return None
    
    @staticmethod
    def create_tier_distribution(df: pd.DataFrame, title: str = "Fund Tier Distribution"):
        """Create tier distribution chart"""
        
        try:
            if not ValidationHandler.validate_dataframe_basic(df, "tier distribution"):
                return None
                
            if StandardColumns.TIER not in df.columns:
                st.error("Tier information not available")
                return None
            
            # Calculate tier distribution
            tier_counts = df[StandardColumns.TIER].value_counts()
            
            if tier_counts.empty:
                st.warning("No tier data available")
                return None
            
            # Create pie chart
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title=title,
                color_discrete_map=DashboardTheme.TIER_COLORS
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            return fig
            
        except Exception as e:
            logger.error(f"Tier distribution chart failed: {str(e)}")
            # Fallback to simple metrics
            if StandardColumns.TIER in df.columns:
                tier_counts = df[StandardColumns.TIER].value_counts()
                st.subheader(title)
                for tier, count in tier_counts.items():
                    st.write(f"**{tier}:** {count} funds")
            return None
    
    @staticmethod
    def create_performance_bar_chart(df: pd.DataFrame, 
                                   metric_col: str = StandardColumns.TOTAL_RETURN,
                                   n_funds: int = 10,
                                   title: str = "Top Performing Funds"):
        """Create horizontal bar chart of top performers"""
        
        try:
            if not ValidationHandler.validate_dataframe_basic(df, "performance bar chart"):
                return None
                
            if metric_col not in df.columns:
                st.error(f"Column '{metric_col}' not found")
                return None
            
            # Get top performers
            top_funds = df.nlargest(n_funds, metric_col)
            
            if len(top_funds) == 0:
                st.warning("No performance data available")
                return None
            
            # Create horizontal bar chart
            fig = px.bar(
                top_funds,
                x=metric_col,
                y=StandardColumns.FUND if StandardColumns.FUND in top_funds.columns else 'Ticker',
                orientation='h',
                title=title,
                color=metric_col,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=max(400, len(top_funds) * 30),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return fig
            
        except Exception as e:
            logger.error(f"Performance bar chart failed: {str(e)}")
            # Fallback to simple table
            if metric_col in df.columns:
                top_funds = df.nlargest(n_funds, metric_col)[
                    [StandardColumns.FUND if StandardColumns.FUND in df.columns else 'Ticker', metric_col]
                ]
                st.subheader(title)
                st.dataframe(top_funds, use_container_width=True)
            return None

# ─── INTERACTIVE COMPONENTS ─────────────────────────────────────────────────

class InteractiveComponents:
    """Interactive widgets and controls"""
    
    @staticmethod
    def fund_selector(df: pd.DataFrame, 
                     key: str = "fund_selector",
                     multi: bool = True,
                     default_selection: List[str] = None) -> List[str]:
        """
        Fund selection widget
        
        Args:
            df: DataFrame with fund data
            key: Unique key for widget
            multi: Whether to allow multiple selection
            default_selection: Default selected funds
            
        Returns:
            List of selected fund identifiers
        """
        if not ValidationHandler.validate_dataframe_basic(df, "fund selection"):
            return []
        
        # Use Fund column if available, otherwise Ticker
        if StandardColumns.FUND in df.columns:
            options = SafeDataAccess.safe_get_column(df, StandardColumns.FUND, 'Unknown').dropna().unique().tolist()
            label = "Select Funds"
        elif 'Ticker' in df.columns:
            options = df['Ticker'].dropna().unique().tolist()
            label = "Select Funds (by Ticker)"
        else:
            st.error("No fund identifier columns available")
            return []
        
        if not options:
            st.warning("No funds available for selection")
            return []
        
        # Create selection widget
        if multi:
            selected = st.multiselect(
                label,
                options=options,
                default=default_selection if default_selection else [],
                key=key
            )
        else:
            selected = st.selectbox(
                label,
                options=[''] + options,
                key=key
            )
            selected = [selected] if selected else []
        
        return selected
    
    @staticmethod
    def metric_selector(df: pd.DataFrame, 
                       key: str = "metric_selector",
                       label: str = "Select Metrics") -> List[str]:
        """Metric selection widget"""
        
        if not ValidationHandler.validate_dataframe_basic(df, "metric selection"):
            return []
        
        # Get numeric columns that are likely metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_keywords = ['Return', 'Ratio', 'Score', 'Volatility', 'AUM']
        
        suggested_metrics = []
        for col in numeric_cols:
            if any(keyword in col for keyword in metric_keywords):
                suggested_metrics.append(col)
        
        if not suggested_metrics:
            suggested_metrics = numeric_cols[:5]  # Fallback to first 5 numeric columns
        
        selected_metrics = st.multiselect(
            label,
            options=suggested_metrics,
            default=suggested_metrics[:3],  # Default to first 3
            key=key
        )
        
        return selected_metrics
    
    @staticmethod
    def tier_filter(df: pd.DataFrame, key: str = "tier_filter") -> List[str]:
        """Tier filtering widget"""
        
        if not ValidationHandler.validate_dataframe_basic(df, "tier filtering"):
            return []
        
        if StandardColumns.TIER not in df.columns:
            return []
        
        available_tiers = df[StandardColumns.TIER].dropna().unique().tolist()
        if not available_tiers:
            return []
        
        selected_tiers = st.multiselect(
            "Filter by Tier",
            options=available_tiers,
            default=available_tiers,
            key=key
        )
        
        return selected_tiers

# ─── LAYOUT COMPONENTS ──────────────────────────────────────────────────────

class LayoutComponents:
    """Layout and structure components"""
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive sidebar filters"""
        
        filters = {}
        
        if not ValidationHandler.validate_dataframe_basic(df, "sidebar filters"):
            return filters
        
        st.sidebar.header("🔍 Filters & Options")
        
        # Fund selection
        filters['selected_funds'] = InteractiveComponents.fund_selector(
            df, key="sidebar_fund_selector", multi=True
        )
        
        # Tier filter
        if StandardColumns.TIER in df.columns:
            filters['selected_tiers'] = InteractiveComponents.tier_filter(
                df, key="sidebar_tier_filter"
            )
        
        # Score range filter
        if StandardColumns.SCORE in df.columns:
            score_range = st.sidebar.slider(
                "Score Range",
                min_value=float(df[StandardColumns.SCORE].min()) if df[StandardColumns.SCORE].notna().any() else 0.0,
                max_value=float(df[StandardColumns.SCORE].max()) if df[StandardColumns.SCORE].notna().any() else 100.0,
                value=(0.0, 100.0),
                key="score_range_filter"
            )
            filters['score_range'] = score_range
        
        # Category filter
        if 'Category' in df.columns:
            categories = df['Category'].dropna().unique().tolist()
            if categories:
                filters['selected_categories'] = st.sidebar.multiselect(
                    "Filter by Category",
                    options=categories,
                    default=categories,
                    key="category_filter"
                )
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        
        if not ValidationHandler.validate_dataframe_basic(df, "filter application"):
            return df
        
        filtered_df = df.copy()
        
        # Apply fund selection
        if filters.get('selected_funds'):
            if StandardColumns.FUND in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[StandardColumns.FUND].isin(filters['selected_funds'])]
            elif 'Ticker' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ticker'].isin(filters['selected_funds'])]
        
        # Apply tier filter
        if filters.get('selected_tiers') and StandardColumns.TIER in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[StandardColumns.TIER].isin(filters['selected_tiers'])]
        
        # Apply score range filter
        if filters.get('score_range') and StandardColumns.SCORE in filtered_df.columns:
            min_score, max_score = filters['score_range']
            filtered_df = filtered_df[
                (filtered_df[StandardColumns.SCORE] >= min_score) & 
                (filtered_df[StandardColumns.SCORE] <= max_score)
            ]
        
        # Apply category filter
        if filters.get('selected_categories') and 'Category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Category'].isin(filters['selected_categories'])]
        
        return filtered_df
    
    @staticmethod
    def create_summary_stats(df: pd.DataFrame):
        """Create summary statistics display"""
        
        if not ValidationHandler.validate_dataframe_basic(df, "summary statistics"):
            return
        
        st.subheader("📈 Summary Statistics")
        
        metrics = {}
        
        # Basic counts
        metrics["Total Funds"] = len(df)
        
        # Score statistics
        if StandardColumns.SCORE in df.columns:
            valid_scores = df[StandardColumns.SCORE].dropna()
            if len(valid_scores) > 0:
                metrics["Average Score"] = f"{valid_scores.mean():.1f}"
                metrics["Top Score"] = f"{valid_scores.max():.1f}"
        
        # Performance statistics
        if StandardColumns.TOTAL_RETURN in df.columns:
            valid_returns = df[StandardColumns.TOTAL_RETURN].dropna()
            if len(valid_returns) > 0:
                metrics["Avg Return"] = f"{valid_returns.mean():.1f}%"
        
        HeaderComponents.display_metrics_row(metrics)