"""
Scoring Engine Module for Semi-Liquid Fund Dashboard

This module handles:
- Fund scoring and ranking calculations
- Performance metrics computation
- Risk-adjusted returns analysis
- Tier classification system
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

from data_models import StandardColumns, DataFrameSchema
from error_handler import SafeDataAccess, show_aum_scoring_warnings, log_aum_scoring_summary
from error_handler import handle_errors, ValidationHandler, ErrorMessages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── SCORING CONFIGURATION ──────────────────────────────────────────────────

class ScoringConfig:
    """Configuration for scoring system"""
    
    # Metric weights for composite scoring
    METRIC_WEIGHTS = {
        'total_return': 0.25,      # 25% weight for total return
        'sharpe_composite': 0.20,   # 20% weight for Sharpe ratio
        'sortino_composite': 0.15,  # 15% weight for Sortino ratio
        'delta': 0.15,             # 15% weight for category delta
        'aum': 0.10,               # 10% weight for AUM factor
        'expense': 0.10,           # 10% weight for expense efficiency
        'consistency': 0.05        # 5% weight for consistency
    }
    
    # Risk-free rate for calculations
    RISK_FREE_RATE = 0.02
    
    # Trading days per year
    TRADING_DAYS = 252
    
    # Tier thresholds (percentiles)
    TIER_THRESHOLDS = {
        'tier_1': 0.75,  # Top 25%
        'tier_2': 0.50,  # Middle 25%
        'tier_3': 0.00   # Bottom 50%
    }

# ─── PERFORMANCE METRICS CALCULATOR ─────────────────────────────────────────

class PerformanceCalculator:
    """Calculate various performance and risk metrics"""
    
    @staticmethod
    @handle_errors("Performance metric calculation failed")
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = ScoringConfig.RISK_FREE_RATE) -> float:
        """
        Calculate Sharpe ratio with error handling
        
        Args:
            returns: Series of returns (as decimals, not percentages)
            risk_free_rate: Risk-free rate for calculation
            
        Returns:
            Sharpe ratio or NaN if calculation fails
        """
        try:
            if returns is None or returns.empty or returns.std() == 0:
                return np.nan
                
            excess_return = returns.mean() - risk_free_rate
            return excess_return / returns.std()
            
        except Exception as e:
            logger.warning(f"Sharpe ratio calculation failed: {str(e)}")
            return np.nan
    
    @staticmethod
    @handle_errors("Sortino ratio calculation failed")
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = ScoringConfig.RISK_FREE_RATE) -> float:
        """
        Calculate Sortino ratio (downside deviation)
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio or NaN if calculation fails
        """
        try:
            if returns is None or returns.empty:
                return np.nan
                
            excess_return = returns.mean() - risk_free_rate
            downside_returns = returns[returns < risk_free_rate]
            
            if len(downside_returns) == 0:
                return np.inf if excess_return > 0 else np.nan
                
            downside_deviation = downside_returns.std()
            if downside_deviation == 0:
                return np.inf if excess_return > 0 else np.nan
                
            return excess_return / downside_deviation
            
        except Exception as e:
            logger.warning(f"Sortino ratio calculation failed: {str(e)}")
            return np.nan
    
    @staticmethod
    @handle_errors("Information ratio calculation failed")
    def calculate_information_ratio(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio (active return / tracking error)
        
        Args:
            fund_returns: Fund return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Information ratio or NaN if calculation fails
        """
        try:
            if fund_returns is None or benchmark_returns is None:
                return np.nan
                
            if len(fund_returns) != len(benchmark_returns):
                return np.nan
                
            active_returns = fund_returns - benchmark_returns
            tracking_error = active_returns.std()
            
            if tracking_error == 0:
                return np.nan
                
            return active_returns.mean() / tracking_error
            
        except Exception as e:
            logger.warning(f"Information ratio calculation failed: {str(e)}")
            return np.nan
    
    @staticmethod
    @handle_errors("Maximum drawdown calculation failed")
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            cumulative_returns: Cumulative return series
            
        Returns:
            Maximum drawdown as decimal or NaN if calculation fails
        """
        try:
            if cumulative_returns is None or cumulative_returns.empty:
                return np.nan
                
            # Calculate running maximum
            peak = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - peak) / peak
            
            return drawdown.min()
            
        except Exception as e:
            logger.warning(f"Max drawdown calculation failed: {str(e)}")
            return np.nan

# ─── COMPOSITE SCORING ENGINE ───────────────────────────────────────────────

class CompositeScorer:
    """Main scoring engine for fund evaluation"""
    
    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
    
    @handle_errors("Composite scoring failed")
    def calculate_composite_score(self, fund_data: pd.Series) -> float:
        """
        Calculate composite score for a single fund
        
        Args:
            fund_data: Series containing fund metrics
            
        Returns:
            Composite score (0-100) or NaN if calculation fails
        """
        try:
            score_components = {}
            
            # Total return component
            total_return = fund_data.get(StandardColumns.TOTAL_RETURN, np.nan)
            if pd.notna(total_return):
                score_components['total_return'] = self._normalize_metric(total_return, 0, 30)
            
            # Sharpe ratio component
            sharpe = fund_data.get(StandardColumns.SHARPE_RATIO, np.nan)
            if pd.notna(sharpe):
                score_components['sharpe'] = self._normalize_metric(sharpe, -2, 3)
            
            # Sortino ratio component
            sortino = fund_data.get(StandardColumns.SORTINO_RATIO, np.nan)
            if pd.notna(sortino):
                score_components['sortino'] = self._normalize_metric(sortino, -2, 4)
            
            # AUM component (logarithmic scaling)
            aum = fund_data.get(StandardColumns.AUM, np.nan)
            if pd.notna(aum) and aum > 0:
                log_aum = np.log10(aum)
                score_components['aum'] = self._normalize_metric(log_aum, 0, 4)
            else:
                # Assign neutral score (0.5) for missing AUM to avoid unfair penalty
                score_components['aum'] = 0.5
                if pd.isna(aum):
                    logger.info(f"Fund '{fund_data.get(StandardColumns.FUND, 'Unknown')}' has missing AUM, assigned neutral score (0.5)")
            
            # Expense ratio component (inverted - lower is better)
            expense_ratio = fund_data.get(StandardColumns.EXPENSE_RATIO, np.nan)
            if pd.notna(expense_ratio):
                score_components['expense'] = self._normalize_metric(5 - expense_ratio, 0, 5)
            
            # Calculate weighted composite score
            if not score_components:
                return np.nan
                
            weighted_score = 0
            total_weight = 0
            
            weight_mapping = {
                'total_return': self.config.METRIC_WEIGHTS['total_return'],
                'sharpe': self.config.METRIC_WEIGHTS['sharpe_composite'],
                'sortino': self.config.METRIC_WEIGHTS['sortino_composite'],
                'aum': self.config.METRIC_WEIGHTS['aum'],
                'expense': self.config.METRIC_WEIGHTS['expense']
            }
            
            for component, value in score_components.items():
                if pd.notna(value) and component in weight_mapping:
                    weight = weight_mapping[component]
                    weighted_score += value * weight
                    total_weight += weight
            
            if total_weight == 0:
                return np.nan
                
            # Normalize to 0-100 scale
            final_score = (weighted_score / total_weight) * 100
            return max(0, min(100, final_score))
            
        except Exception as e:
            logger.error(f"Composite scoring failed: {str(e)}")
            return np.nan
    
    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize metric to 0-1 scale
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value (0-1)
        """
        if pd.isna(value):
            return np.nan
            
        if max_val == min_val:
            return 0.5
            
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))
    
    @handle_errors("Batch scoring failed")
    def score_fund_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score entire universe of funds
        
        Args:
            df: DataFrame containing fund data
            
        Returns:
            DataFrame with added scoring columns
        """
        if df is None or df.empty:
            st.warning(ErrorMessages.NO_DATA)
            return df
            
        df_scored = df.copy()
        
        # Validate AUM data and show warnings for transparency
        aum_validation = DataFrameSchema.validate_aum_data(df_scored)
        show_aum_scoring_warnings(aum_validation)
        log_aum_scoring_summary(aum_validation)
        
        # Calculate composite scores
        scores = []
        for idx, fund in df_scored.iterrows():
            score = self.calculate_composite_score(fund)
            scores.append(score)
        
        df_scored[StandardColumns.SCORE] = scores
        
        # Calculate rankings
        df_scored[StandardColumns.RANK] = df_scored[StandardColumns.SCORE].rank(
            method='dense', ascending=False, na_option='bottom'
        )
        
        # Assign tiers
        df_scored[StandardColumns.TIER] = self._assign_tiers(df_scored[StandardColumns.SCORE])
        
        # Calculate percentiles
        df_scored['Percentile'] = df_scored[StandardColumns.SCORE].rank(pct=True) * 100
        
        logger.info(f"Scored {len(df_scored)} funds successfully")
        return df_scored
    
    def _assign_tiers(self, scores: pd.Series) -> pd.Series:
        """
        Assign tier classifications based on score percentiles
        
        Args:
            scores: Series of composite scores
            
        Returns:
            Series of tier assignments
        """
        if scores.empty or scores.isna().all():
            return pd.Series(['N/A'] * len(scores))
        
        # Calculate percentile thresholds
        valid_scores = scores.dropna()
        if len(valid_scores) == 0:
            return pd.Series(['N/A'] * len(scores))
        
        tier_1_threshold = valid_scores.quantile(self.config.TIER_THRESHOLDS['tier_1'])
        tier_2_threshold = valid_scores.quantile(self.config.TIER_THRESHOLDS['tier_2'])
        
        # Assign tiers
        tiers = []
        for score in scores:
            if pd.isna(score):
                tiers.append('N/A')
            elif score >= tier_1_threshold:
                tiers.append('Tier 1')
            elif score >= tier_2_threshold:
                tiers.append('Tier 2')
            else:
                tiers.append('Tier 3')
        
        return pd.Series(tiers)

# ─── SPECIALIZED SCORING METHODS ────────────────────────────────────────────

class AdvancedScoring:
    """Advanced scoring methods for specialized analysis"""
    
    @staticmethod
    @handle_errors("Risk-adjusted scoring failed")
    def calculate_risk_adjusted_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk-adjusted performance scores
        
        Args:
            df: DataFrame with fund data
            
        Returns:
            DataFrame with risk-adjusted scores
        """
        if df is None or df.empty:
            return df
            
        df_risk = df.copy()
        
        # Risk-adjusted return calculation
        returns = SafeDataAccess.safe_get_column(df_risk, StandardColumns.TOTAL_RETURN, 0)
        volatility = SafeDataAccess.safe_get_column(df_risk, StandardColumns.VOLATILITY, 1)
        
        # Handle division by zero
        risk_adjusted_returns = []
        for ret, vol in zip(returns, volatility):
            if pd.notna(ret) and pd.notna(vol) and vol > 0:
                risk_adjusted_returns.append(ret / vol)
            else:
                risk_adjusted_returns.append(np.nan)
        
        df_risk['Risk_Adjusted_Return'] = risk_adjusted_returns
        
        # Normalize risk-adjusted returns to 0-100 scale
        valid_scores = [score for score in risk_adjusted_returns if pd.notna(score)]
        if valid_scores:
            min_score, max_score = min(valid_scores), max(valid_scores)
            if max_score != min_score:
                normalized_scores = []
                for score in risk_adjusted_returns:
                    if pd.notna(score):
                        normalized = ((score - min_score) / (max_score - min_score)) * 100
                        normalized_scores.append(normalized)
                    else:
                        normalized_scores.append(np.nan)
                df_risk['Risk_Adjusted_Score'] = normalized_scores
            else:
                df_risk['Risk_Adjusted_Score'] = [50] * len(df_risk)
        else:
            df_risk['Risk_Adjusted_Score'] = [np.nan] * len(df_risk)
        
        return df_risk
    
    @staticmethod
    @handle_errors("Consistency scoring failed")
    def calculate_consistency_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance consistency scores
        
        Args:
            df: DataFrame with multi-period performance data
            
        Returns:
            DataFrame with consistency scores
        """
        if df is None or df.empty:
            return df
            
        df_consistency = df.copy()
        
        # Look for multi-period returns
        period_columns = ['Sharpe (1Y)', 'Sharpe (3Y)', 'Sharpe (5Y)']
        available_periods = [col for col in period_columns if col in df_consistency.columns]
        
        if len(available_periods) < 2:
            df_consistency['Consistency_Score'] = np.nan
            return df_consistency
        
        consistency_scores = []
        for idx, fund in df_consistency.iterrows():
            period_values = [fund.get(col, np.nan) for col in available_periods]
            valid_values = [val for val in period_values if pd.notna(val)]
            
            if len(valid_values) >= 2:
                # Calculate coefficient of variation (lower is more consistent)
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    # Convert to consistency score (higher is better)
                    consistency_score = max(0, 100 - (cv * 50))
                    consistency_scores.append(consistency_score)
                else:
                    consistency_scores.append(50)  # Neutral score
            else:
                consistency_scores.append(np.nan)
        
        df_consistency['Consistency_Score'] = consistency_scores
        return df_consistency

# ─── SCORING UTILITIES ──────────────────────────────────────────────────────

class ScoringUtils:
    """Utility functions for scoring operations"""
    
    @staticmethod
    def get_scoring_summary(df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary statistics for scored fund universe
        
        Args:
            df: DataFrame with scored funds
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None or df.empty or StandardColumns.SCORE not in df.columns:
            return {'error': 'No scoring data available'}
        
        valid_scores = df[StandardColumns.SCORE].dropna()
        
        if len(valid_scores) == 0:
            return {'error': 'No valid scores found'}
        
        tier_counts = df[StandardColumns.TIER].value_counts().to_dict()
        
        summary = {
            'total_funds': len(df),
            'scored_funds': len(valid_scores),
            'average_score': valid_scores.mean(),
            'median_score': valid_scores.median(),
            'score_std': valid_scores.std(),
            'min_score': valid_scores.min(),
            'max_score': valid_scores.max(),
            'tier_distribution': tier_counts,
            'top_decile_threshold': valid_scores.quantile(0.9),
            'bottom_decile_threshold': valid_scores.quantile(0.1)
        }
        
        return summary
    
    @staticmethod
    def display_scoring_summary(summary: Dict[str, any]):
        """Display scoring summary in Streamlit"""
        
        if 'error' in summary:
            st.error(f"❌ {summary['error']}")
            return
        
        st.subheader("📊 Scoring Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Funds", summary['total_funds'])
            
        with col2:
            st.metric("Scored Funds", summary['scored_funds'])
            
        with col3:
            st.metric("Average Score", f"{summary['average_score']:.1f}")
            
        with col4:
            st.metric("Score Range", f"{summary['min_score']:.1f} - {summary['max_score']:.1f}")
        
        # Tier distribution
        if summary['tier_distribution']:
            st.subheader("🏆 Tier Distribution")
            tier_df = pd.DataFrame(list(summary['tier_distribution'].items()), 
                                 columns=['Tier', 'Count'])
            st.bar_chart(tier_df.set_index('Tier'))

# ─── MAIN SCORING INTERFACE ─────────────────────────────────────────────────

@handle_errors("Fund scoring process failed")
def score_fund_universe(df: pd.DataFrame, config: ScoringConfig = None) -> Optional[pd.DataFrame]:
    """
    Main function to score fund universe
    
    Args:
        df: DataFrame with fund data
        config: Scoring configuration (optional)
        
    Returns:
        DataFrame with scores and rankings
    """
    if not ValidationHandler.validate_dataframe_basic(df, "scoring"):
        return None
    
    with st.spinner("Calculating fund scores and rankings..."):
        scorer = CompositeScorer(config)
        scored_df = scorer.score_fund_universe(df)
        
        # Add advanced scoring
        advanced_scorer = AdvancedScoring()
        scored_df = advanced_scorer.calculate_risk_adjusted_score(scored_df)
        scored_df = advanced_scorer.calculate_consistency_score(scored_df)
        
        # Display summary
        summary = ScoringUtils.get_scoring_summary(scored_df)
        ScoringUtils.display_scoring_summary(summary)
        
        return scored_df

def get_top_performers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N performing funds by composite score"""
    
    if df is None or df.empty or StandardColumns.SCORE not in df.columns:
        return pd.DataFrame()
    
    return df.nlargest(n, StandardColumns.SCORE)

def get_funds_by_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Get funds filtered by tier classification"""
    
    if df is None or df.empty or StandardColumns.TIER not in df.columns:
        return pd.DataFrame()
    
    return df[df[StandardColumns.TIER] == tier]