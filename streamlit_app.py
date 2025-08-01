"""
Semi-Liquid Alternatives Fund Dashboard - Main Application

This is the main orchestration file that coordinates all dashboard functionality
using the modular architecture with enhanced institutional-grade UI.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

# Import our custom modules
from data_models import StandardColumns
from error_handler import handle_errors, ValidationHandler
from data_loader import load_fund_data, load_app_config, refresh_data_cache
from scoring_engine import score_fund_universe
from ui_components import (
    DashboardTheme, HeaderComponents, DataDisplayComponents, 
    ChartComponents, InteractiveComponents, LayoutComponents, FundDetailComponents
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── PAGE CONFIGURATION ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semi Liquid Alternatives Fund Dashboard", 
    layout="wide", 
    page_icon="🏦"
)

# ─── ENHANCED CUSTOM CSS STYLING ────────────────────────────────────────────

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
    
    .phase-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
        color: white; 
        padding: 8px 16px; 
        border-radius: 8px; 
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Apply additional custom styling from modular components
DashboardTheme.apply_custom_css()

# ─── APPLICATION STATE MANAGEMENT ───────────────────────────────────────────

@st.cache_data(ttl=300)
def get_app_state() -> Dict[str, Any]:
    """Initialize and manage application state"""
    return {
        'data_loaded': False,
        'funds_data': None,
        'scored_data': None,
        'config': load_app_config()
    }

# ─── MAIN DATA LOADING ──────────────────────────────────────────────────────

@handle_errors("Failed to initialize dashboard")
def initialize_dashboard():
    """Initialize the dashboard with data loading and processing"""
    
    # Load configuration
    config = load_app_config()
    sheet_id = config.get('sheet_id', '')
    
    if not sheet_id:
        st.error("❌ Google Sheets ID not configured. Please check your settings.")
        st.info("Add 'sheet_id' to your Streamlit secrets.")
        return None, None
    
    # Load raw data
    with st.spinner("Loading fund data..."):
        raw_data = load_fund_data(sheet_id, tab_keyword="Fund")
    
    if raw_data is None or raw_data.empty:
        st.error("❌ Failed to load fund data")
        return None, None
    
    # Score the fund universe
    with st.spinner("Calculating scores and rankings..."):
        scored_data = score_fund_universe(raw_data)
    
    if scored_data is None:
        st.warning("⚠️ Scoring failed, using raw data")
        scored_data = raw_data
    
    logger.info(f"Dashboard initialized with {len(scored_data)} funds")
    return raw_data, scored_data

# ─── TAB IMPLEMENTATIONS ────────────────────────────────────────────────────

def create_main_rankings_tab(df: pd.DataFrame):
    """Main rankings and overview tab"""
    
    HeaderComponents.display_page_header(
        "Fund Rankings & Overview", 
        "Comprehensive ranking of all funds with key performance metrics"
    )
    
    if not ValidationHandler.validate_dataframe_basic(df, "rankings display"):
        return
    
    # Summary statistics
    LayoutComponents.create_summary_stats(df)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top performers chart
        ChartComponents.create_performance_bar_chart(
            df, 
            metric_col=StandardColumns.SCORE,
            n_funds=10,
            title="Top 10 Funds by Composite Score"
        )
        
    with col2:
        # Tier distribution
        ChartComponents.create_tier_distribution(df)
    
    # Risk vs Return scatter plot
    st.subheader("📊 Risk vs Return Analysis")
    ChartComponents.create_performance_scatter(
        df,
        x_col=StandardColumns.VOLATILITY,
        y_col=StandardColumns.TOTAL_RETURN,
        color_col=StandardColumns.SCORE,
        title="Risk vs Return by Fund Score"
    )
    
    # Detailed fund table
    display_columns = [
        StandardColumns.FUND, 'Ticker', StandardColumns.SCORE, 
        StandardColumns.TIER, StandardColumns.TOTAL_RETURN,
        StandardColumns.VOLATILITY, StandardColumns.SHARPE_RATIO
    ]
    
    DataDisplayComponents.display_fund_table(
        df, 
        columns=display_columns,
        title="All Funds - Detailed Rankings",
        max_rows=100
    )

def create_data_quality_tab(df: pd.DataFrame):
    """Data quality and validation tab"""
    
    HeaderComponents.display_page_header(
        "Data Quality Dashboard",
        "Comprehensive data quality analysis and validation results"
    )
    
    if not ValidationHandler.validate_dataframe_basic(df, "data quality analysis"):
        return
    
    # Data overview metrics
    total_funds = len(df)
    total_columns = len(df.columns)
    missing_data_pct = (df.isnull().sum().sum() / (total_funds * total_columns)) * 100
    
    metrics = {
        "Total Funds": total_funds,
        "Data Columns": total_columns,
        "Missing Data": f"{missing_data_pct:.1f}%",
        "Data Quality Score": f"{max(0, 100 - missing_data_pct):.0f}/100"
    }
    
    HeaderComponents.display_metrics_row(metrics)
    
    # Column completeness analysis
    st.subheader("📋 Column Completeness Analysis")
    
    completeness_data = []
    for col in df.columns:
        total_values = len(df)
        missing_values = df[col].isnull().sum()
        completeness_pct = ((total_values - missing_values) / total_values) * 100
        
        completeness_data.append({
            'Column': col,
            'Total Values': total_values,
            'Missing Values': missing_values,
            'Completeness (%)': completeness_pct
        })
    
    completeness_df = pd.DataFrame(completeness_data)
    completeness_df = completeness_df.sort_values('Completeness (%)', ascending=False)
    
    st.dataframe(completeness_df, use_container_width=True)
    
    # Data quality warnings
    st.subheader("⚠️ Data Quality Issues")
    
    issues_found = False
    
    # Check for high missing data columns
    high_missing_cols = completeness_df[completeness_df['Completeness (%)'] < 50]['Column'].tolist()
    if high_missing_cols:
        st.warning(f"Columns with >50% missing data: {', '.join(high_missing_cols)}")
        issues_found = True
    
    # Check for duplicate funds
    if StandardColumns.FUND in df.columns:
        duplicates = df[StandardColumns.FUND].duplicated().sum()
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate fund names")
            issues_found = True
    
    # Check for data type issues
    numeric_cols = [StandardColumns.TOTAL_RETURN, StandardColumns.VOLATILITY, StandardColumns.SCORE]
    for col in numeric_cols:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isnull().sum() - df[col].isnull().sum()
            if non_numeric > 0:
                st.warning(f"Column '{col}' has {non_numeric} non-numeric values")
                issues_found = True
    
    if not issues_found:
        st.success("✅ No significant data quality issues detected")

def create_fund_comparison_tab(df: pd.DataFrame):
    """Fund comparison and analysis tab"""
    
    HeaderComponents.display_page_header(
        "Fund Comparison Tool",
        "Side-by-side comparison of selected funds across key metrics"
    )
    
    if not ValidationHandler.validate_dataframe_basic(df, "fund comparison"):
        return
    
    # Fund selection
    selected_funds = InteractiveComponents.fund_selector(
        df, 
        key="comparison_fund_selector", 
        multi=True
    )
    
    if not ValidationHandler.validate_selection(selected_funds, "funds"):
        st.info("👆 Select 2 or more funds from above to start comparison")
        return
    
    if len(selected_funds) < 2:
        st.info("Please select at least 2 funds for comparison")
        return
    
    # Filter data for selected funds
    if StandardColumns.FUND in df.columns:
        comparison_df = df[df[StandardColumns.FUND].isin(selected_funds)].copy()
    else:
        comparison_df = df[df['Ticker'].isin(selected_funds)].copy()
    
    if comparison_df.empty:
        st.error("No data found for selected funds")
        return
    
    # Metrics selection
    available_metrics = InteractiveComponents.metric_selector(
        comparison_df,
        key="comparison_metrics",
        label="Select metrics to compare"
    )
    
    if not available_metrics:
        st.warning("No metrics selected for comparison")
        return
    
    # Comparison table
    st.subheader("📊 Fund Comparison Table")
    
    display_cols = [StandardColumns.FUND if StandardColumns.FUND in comparison_df.columns else 'Ticker']
    display_cols.extend([col for col in available_metrics if col in comparison_df.columns])
    
    DataDisplayComponents.display_fund_table(
        comparison_df,
        columns=display_cols,
        title="Selected Funds Comparison",
        interactive=True
    )
    
    # Comparison charts
    if len(available_metrics) >= 2:
        st.subheader("📈 Visual Comparison")
        
        # Scatter plot comparison
        ChartComponents.create_performance_scatter(
            comparison_df,
            x_col=available_metrics[0] if available_metrics[0] in comparison_df.columns else StandardColumns.VOLATILITY,
            y_col=available_metrics[1] if available_metrics[1] in comparison_df.columns else StandardColumns.TOTAL_RETURN,
            title=f"Comparison: {available_metrics[0] if len(available_metrics) > 0 else 'Metric 1'} vs {available_metrics[1] if len(available_metrics) > 1 else 'Metric 2'}"
        )

def create_portfolio_builder_tab(df: pd.DataFrame):
    """Portfolio construction and optimization tab"""
    
    HeaderComponents.display_page_header(
        "Portfolio Builder",
        "Construct and analyze optimal fund portfolios"
    )
    
    if not ValidationHandler.validate_dataframe_basic(df, "portfolio building"):
        return
    
    st.info("🚧 Portfolio Builder - Enhanced Version Coming Soon")
    st.write("Current functionality provides basic portfolio analysis capabilities.")
    
    # Fund selection for portfolio
    portfolio_funds = InteractiveComponents.fund_selector(
        df,
        key="portfolio_fund_selector",
        multi=True
    )
    
    if not ValidationHandler.validate_selection(portfolio_funds, "funds for portfolio"):
        return
    
    # Filter selected funds
    if StandardColumns.FUND in df.columns:
        selected_df = df[df[StandardColumns.FUND].isin(portfolio_funds)].copy()
    else:
        selected_df = df[df['Ticker'].isin(portfolio_funds)].copy()
    
    if selected_df.empty:
        st.error("No data found for selected funds")
        return
    
    # Basic portfolio statistics
    st.subheader("📊 Portfolio Analysis")
    
    # Calculate basic portfolio metrics
    if StandardColumns.TOTAL_RETURN in selected_df.columns and StandardColumns.VOLATILITY in selected_df.columns:
        
        avg_return = selected_df[StandardColumns.TOTAL_RETURN].mean()
        avg_volatility = selected_df[StandardColumns.VOLATILITY].mean()
        
        portfolio_metrics = {
            "Selected Funds": len(selected_df),
            "Avg Return": f"{avg_return:.2f}%",
            "Avg Volatility": f"{avg_volatility:.2f}%",
            "Avg Score": f"{selected_df[StandardColumns.SCORE].mean():.1f}" if StandardColumns.SCORE in selected_df.columns else "N/A"
        }
        
        HeaderComponents.display_metrics_row(portfolio_metrics)
        
        # Portfolio visualization
        ChartComponents.create_performance_scatter(
            selected_df,
            x_col=StandardColumns.VOLATILITY,
            y_col=StandardColumns.TOTAL_RETURN,
            color_col=StandardColumns.SCORE,
            title="Portfolio Funds - Risk vs Return"
        )
    
    # Selected funds table
    DataDisplayComponents.display_fund_table(
        selected_df,
        title="Selected Portfolio Funds",
        max_rows=20
    )

def create_risk_analytics_tab(df: pd.DataFrame):
    """Risk analysis and stress testing tab"""
    
    HeaderComponents.display_page_header(
        "Risk Analytics",
        "Comprehensive risk analysis and stress testing"
    )
    
    if not ValidationHandler.validate_dataframe_basic(df, "risk analytics"):
        return
    
    st.info("🚧 Risk Analytics - Enhanced Version Coming Soon")
    st.write("Current functionality provides basic risk analysis capabilities.")
    
    # Risk metrics overview
    st.subheader("⚠️ Risk Metrics Overview")
    
    if StandardColumns.VOLATILITY in df.columns:
        
        # Risk statistics
        vol_data = df[StandardColumns.VOLATILITY].dropna()
        if len(vol_data) > 0:
            risk_metrics = {
                "Avg Volatility": f"{vol_data.mean():.2f}%",
                "Min Volatility": f"{vol_data.min():.2f}%",
                "Max Volatility": f"{vol_data.max():.2f}%",
                "Risk Spread": f"{vol_data.max() - vol_data.min():.2f}%"
            }
            
            HeaderComponents.display_metrics_row(risk_metrics)
            
            # Risk distribution chart
            st.subheader("📊 Risk Distribution")
            
            # Create risk buckets
            risk_buckets = pd.cut(vol_data, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            risk_distribution = risk_buckets.value_counts()
            
            st.bar_chart(risk_distribution)
    
    # Risk vs Return analysis
    if StandardColumns.TOTAL_RETURN in df.columns and StandardColumns.VOLATILITY in df.columns:
        st.subheader("📈 Risk-Return Profile")
        
        ChartComponents.create_performance_scatter(
            df,
            x_col=StandardColumns.VOLATILITY,
            y_col=StandardColumns.TOTAL_RETURN,
            color_col=StandardColumns.TIER,
            title="Risk-Return Analysis by Tier"
        )

# ─── MAIN APPLICATION FLOW ──────────────────────────────────────────────────

def create_enhanced_header():
    """Create enhanced professional header with Phase 3 indicators"""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Semi-Liquid Alternatives Fund Selection Dashboard</h1>
        <p>Advanced quantitative analysis and performance scoring for alternative investments</p>
        <div class="phase-banner">
            <strong>🚀 Phase 3 - Institutional Analytics Platform</strong> | 
            Advanced Portfolio Construction & Risk Management
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_professional_sidebar_features(df):
    """Add enhanced professional features to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Phase 3 - Enhanced Analytics**")
    st.sidebar.caption("✨ Advanced scoring, comparisons, and interactive features")
    
    # Scoring Methodology Explanation
    st.sidebar.markdown("---")
    with st.sidebar.expander("📊 **Scoring Methodology**", expanded=False):
        st.markdown("### 📈 **Metric Weights**")
        st.markdown("""
        **Core Performance:**
        - 📊 Total Return: **30%**
        - ⚡ Sharpe Ratio: **25%** 
        - 🛡️ Sortino Ratio: **20%**
        
        **Additional Factors:**
        - 📈 Category Delta: **10%**
        - 💰 AUM Scale: **7.5%**
        - 💸 Expense Efficiency: **7.5%**
        """)
        
        st.markdown("### 🎯 **Tier Classification**")
        st.markdown("""
        - **Tier 1**: Top 25% performers
        - **Tier 2**: Middle 25% performers  
        - **Tier 3**: Bottom 50% performers
        """)
    
    # Fund Comparison Tool
    with st.sidebar.expander("🔍 **Fund Comparison Tool**", expanded=False):
        st.markdown("### Compare Funds Side-by-Side")
        
        # Get available funds for comparison
        available_funds = df['Ticker'].dropna().unique().tolist() if 'Ticker' in df.columns else []
        
        col1, col2 = st.columns(2)
        with col1:
            fund1 = st.selectbox("Select Fund 1:", ["None"] + available_funds, key="fund1")
        with col2:
            fund2 = st.selectbox("Select Fund 2:", ["None"] + available_funds, key="fund2")
        
        # Optional third fund
        fund3 = st.selectbox("Select Fund 3 (Optional):", ["None"] + available_funds, key="fund3")
        
        # Display comparison if funds selected
        selected_funds = [f for f in [fund1, fund2, fund3] if f != "None"]
        
        if len(selected_funds) >= 2:
            st.markdown("**Selected Funds:**")
            for i, fund in enumerate(selected_funds, 1):
                fund_data = df[df['Ticker'] == fund].iloc[0] if 'Ticker' in df.columns else None
                if fund_data is not None:
                    score = fund_data.get('Score', 0)
                    tier = fund_data.get('Tier', 'N/A')
                    st.write(f"{i}. **{fund}** - Score: {score:.2f}, {tier}")
    
    # Professional Features
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎓 Professional Features")
    
    # CSV Export
    if st.sidebar.button("📊 Generate CSV Report", help="Create comprehensive CSV analysis report"):
        st.sidebar.success("✅ CSV report generated - check main interface")
        st.session_state.generate_csv = True
    
    # Phase 3 Roadmap Preview
    with st.sidebar.expander("🚀 **Phase 3 Roadmap**", expanded=False):
        st.markdown("""
        **📊 Coming Soon:**
        • Advanced portfolio optimization
        • Risk factor analysis
        • Performance attribution
        • Correlation analysis
        • Monte Carlo simulations
        """)

def main():
    """Main application entry point"""
    
    # Display enhanced header
    create_enhanced_header()
    
    # Initialize data
    raw_data, scored_data = initialize_dashboard()
    
    if scored_data is None:
        st.error("❌ Dashboard initialization failed")
        st.stop()
    
    # Create enhanced sidebar features
    create_professional_sidebar_features(scored_data)
    
    # Create sidebar filters
    filters = LayoutComponents.create_sidebar_filters(scored_data)
    
    # Apply filters to data
    filtered_data = LayoutComponents.apply_filters(scored_data, filters)
    
    # Fund Detail Breakdown Feature
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Fund Detail Analysis")
    
    # Fund selection for detailed analysis
    fund_analysis_option = st.sidebar.selectbox(
        "Search by:",
        ["Fund Name", "Ticker"],
        help="Choose how to search for funds"
    )
    
    if fund_analysis_option == "Fund Name" and StandardColumns.FUND in scored_data.columns:
        available_funds = sorted(scored_data[StandardColumns.FUND].dropna().unique().tolist())
        identifier_type = "fund_name"
    elif fund_analysis_option == "Ticker" and 'Ticker' in scored_data.columns:
        available_funds = sorted(scored_data['Ticker'].dropna().unique().tolist())
        identifier_type = "ticker"
    else:
        available_funds = []
        identifier_type = "fund_name"
    
    if available_funds:
        selected_fund = st.sidebar.selectbox(
            f"Select Fund ({len(available_funds)} available):",
            [""] + available_funds,
            help="Choose a fund for detailed score breakdown and analysis"
        )
        
        if selected_fund:
            if st.sidebar.button(f"📊 Analyze {selected_fund}", use_container_width=True):
                st.session_state.selected_fund_analysis = {
                    'fund': selected_fund,
                    'type': identifier_type
                }
    else:
        st.sidebar.info("No funds available for detailed analysis")
    
    # Data refresh option
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        refresh_data_cache()
        st.rerun()
    
    # Handle CSV export request
    if st.session_state.get('generate_csv', False):
        create_csv_export(filtered_data)
        st.session_state.generate_csv = False
    
    # Enhanced Main content tabs with Phase 3 indicators
    tabs = st.tabs([
        "🏆 Fund Rankings",
        "🛡️ Data Quality", 
        "📊 Fund Comparison",
        "🎯 Portfolio Builder*",
        "⚠️ Risk Analytics*"
    ])
    
    # Add note about Phase 3 features
    st.caption("* Phase 3 Features - Enhanced versions coming soon")
    
    with tabs[0]:
        # Add enhanced metrics display to rankings tab
        create_enhanced_metrics_display(filtered_data)
        st.markdown("---")
        create_main_rankings_tab(filtered_data)
    
    with tabs[1]:
        create_data_quality_tab(filtered_data)
    
    with tabs[2]:
        create_fund_comparison_tab(filtered_data)
    
    with tabs[3]:
        create_portfolio_builder_tab(filtered_data)
    
    with tabs[4]:
        create_risk_analytics_tab(filtered_data)
    
    # Fund Detail Breakdown Display
    if 'selected_fund_analysis' in st.session_state:
        fund_analysis = st.session_state.selected_fund_analysis
        st.markdown("---")
        
        # Display comprehensive fund breakdown
        success = FundDetailComponents.fund_detail_breakdown(
            scored_data, 
            fund_analysis['fund'], 
            fund_analysis['type']
        )
        
        if success:
            # Add option to clear analysis
            if st.button("🔙 Clear Fund Analysis", type="secondary"):
                del st.session_state.selected_fund_analysis
                st.rerun()
    
    # Enhanced Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin-top: 2rem;">
        <h4>🏦 Semi-Liquid Alternatives Fund Dashboard</h4>
        <p><strong>Phase 3 - Institutional Analytics Platform</strong></p>
        <p>Professional fund analysis and portfolio construction tool</p>
        <small>Data updated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M') + """ | 
        Built with advanced quantitative analysis framework</small>
    </div>
    """, unsafe_allow_html=True)

def create_csv_export(df):
    """Create and offer CSV export of fund data"""
    try:
        csv_buffer = df.to_csv(index=False)
        
        st.success("📊 **CSV Report Generated Successfully!**")
        
        # Create download button
        st.download_button(
            label="📥 Download Fund Analysis CSV",
            data=csv_buffer,
            file_name=f"semi_liquid_funds_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download comprehensive fund analysis data"
        )
        
        # Show preview
        with st.expander("📋 **CSV Preview**", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 rows of {len(df)} total funds")
            
    except Exception as e:
        st.error(f"❌ Error generating CSV: {str(e)}")

def create_enhanced_metrics_display(df):
    """Create enhanced metric cards with professional styling"""
    if df is None or df.empty:
        return
    
    # Calculate key metrics
    total_funds = len(df)
    tier_dist = df['Tier'].value_counts() if 'Tier' in df.columns else {}
    avg_score = df['Score'].mean() if 'Score' in df.columns else 0
    
    # Create styled metric containers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Total Funds</h3>
            <h2>{total_funds}</h2>
            <small>Quality funds analyzed</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tier1_count = tier_dist.get('Tier 1', 0)
        st.markdown(f"""
        <div class="metric-container">
            <h3>🥇 Tier 1 Funds</h3>
            <h2 class="score-high">{tier1_count}</h2>
            <small>Top performers</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📈 Avg Score</h3>
            <h2 class="score-medium">{avg_score:.1f}</h2>
            <small>Composite performance</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown(f"""
        <div class="metric-container">
            <h3>✅ Data Quality</h3>
            <h2 class="score-high">{data_quality:.0f}%</h2>
            <small>Completeness score</small>
        </div>
        """, unsafe_allow_html=True)

# ─── APPLICATION ENTRY POINT ────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("❌ Application encountered an error. Please refresh the page or contact support.")
        st.error(f"Error details: {str(e)}")