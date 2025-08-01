"""
Statistical analysis module for PhenoAI package.

This module provides statistical utilities for analyzing phenological data,
including uncertainty quantification, correlation analysis, and trend testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from ..core.logger import LoggerMixin
from ..core.exceptions import AnalysisError, ValidationError

@dataclass
class CorrelationResults:
    """Data class to store correlation analysis results."""
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    method: str
    n_samples: int

@dataclass
class TrendAnalysis:
    """Data class to store trend analysis results."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # 'increasing', 'decreasing', 'no_trend'
    significance_level: float

class StatisticalAnalyzer(LoggerMixin):
    """
    Statistical analyzer for phenological data.
    
    Provides various statistical analysis methods including correlation,
    trend analysis, and uncertainty quantification.
    """
    
    def __init__(self, config=None):
        """
        Initialize statistical analyzer.
        
        Args:
            config: Configuration object with statistical settings
        """
        if config is None:
            self.confidence_level = 0.95
        else:
            self.confidence_level = config.confidence_level
    
    def calculate_pearson_correlation(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> CorrelationResults:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            CorrelationResults object
        """
        try:
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 3:
                raise ValidationError("Need at least 3 valid data points for correlation")
            
            # Calculate correlation
            correlation = float(np.corrcoef(x_clean, y_clean)[0, 1])
            
            # Calculate p-value using t-test
            n = len(x_clean)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            
            # Approximate p-value (would need scipy.stats for exact)
            # This is a simplified approximation
            df = n - 2
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
            
            # Calculate confidence interval (Fisher transformation)
            alpha = 1 - self.confidence_level
            z_score = 1.96  # Approximate for 95% CI
            
            # Fisher transformation
            z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))
            se_z = 1 / np.sqrt(n - 3)
            
            z_lower = z_r - z_score * se_z
            z_upper = z_r + z_score * se_z
            
            # Transform back
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return CorrelationResults(
                correlation_coefficient=correlation,
                p_value=p_value,
                confidence_interval=(float(r_lower), float(r_upper)),
                method='pearson',
                n_samples=n
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Pearson correlation: {str(e)}")
            raise AnalysisError(f"Failed to calculate Pearson correlation: {str(e)}")
    
    def calculate_spearman_correlation(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> CorrelationResults:
        """
        Calculate Spearman rank correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            CorrelationResults object
        """
        try:
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 3:
                raise ValidationError("Need at least 3 valid data points for correlation")
            
            # Calculate ranks
            x_ranks = self._calculate_ranks(x_clean)
            y_ranks = self._calculate_ranks(y_clean)
            
            # Calculate Pearson correlation on ranks
            correlation = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
            
            # Approximate p-value
            n = len(x_clean)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            df = n - 2
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
            
            # Simplified confidence interval
            se = 1 / np.sqrt(n - 3)
            margin = 1.96 * se
            
            return CorrelationResults(
                correlation_coefficient=correlation,
                p_value=p_value,
                confidence_interval=(correlation - margin, correlation + margin),
                method='spearman',
                n_samples=n
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Spearman correlation: {str(e)}")
            raise AnalysisError(f"Failed to calculate Spearman correlation: {str(e)}")
    
    def _calculate_ranks(self, data: np.ndarray) -> np.ndarray:
        """Calculate ranks for data array."""
        sorted_indices = np.argsort(data)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(len(data)) + 1
        
        # Handle ties by averaging ranks
        unique_values, counts = np.unique(data, return_counts=True)
        for value, count in zip(unique_values, counts):
            if count > 1:
                mask = data == value
                avg_rank = np.mean(ranks[mask])
                ranks[mask] = avg_rank
        
        return ranks
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # This is a very rough approximation
        # In practice, you would use scipy.stats.t.cdf
        if df > 30:
            # Use normal approximation for large df
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-2 * t**2 / np.pi)))
        else:
            # Very rough approximation for small df
            x = t / np.sqrt(df)
            return 0.5 + 0.5 * np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2))
    
    def analyze_trend(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        significance_level: float = 0.05
    ) -> TrendAnalysis:
        """
        Analyze trend in data using linear regression.
        
        Args:
            x: Independent variable (e.g., time)
            y: Dependent variable (e.g., vegetation index)
            significance_level: Significance level for trend testing
            
        Returns:
            TrendAnalysis object
        """
        try:
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 3:
                raise ValidationError("Need at least 3 valid data points for trend analysis")
            
            # Perform linear regression
            coefficients = np.polyfit(x_clean, y_clean, 1)
            slope = float(coefficients[0])
            intercept = float(coefficients[1])
            
            # Calculate R-squared
            y_pred = slope * x_clean + intercept
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
            
            # Calculate standard error of slope
            n = len(x_clean)
            x_mean = np.mean(x_clean)
            
            # Residual standard error
            mse = ss_res / (n - 2) if n > 2 else 0
            se_slope = np.sqrt(mse / np.sum((x_clean - x_mean) ** 2)) if np.sum((x_clean - x_mean) ** 2) != 0 else 0
            
            # Calculate t-statistic and p-value
            if se_slope != 0:
                t_stat = slope / se_slope
                df = n - 2
                p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
            else:
                p_value = 1.0
            
            # Calculate confidence interval for slope
            t_critical = 1.96  # Approximate for 95% CI
            margin = t_critical * se_slope
            confidence_interval = (slope - margin, slope + margin)
            
            # Determine trend direction
            if p_value < significance_level:
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'no_trend'
            
            return TrendAnalysis(
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                p_value=p_value,
                confidence_interval=confidence_interval,
                trend_direction=trend_direction,
                significance_level=significance_level
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            raise AnalysisError(f"Failed to analyze trend: {str(e)}")
    
    def calculate_confidence_intervals(
        self, 
        data: np.ndarray, 
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence intervals for data.
        
        Args:
            data: Data array
            confidence_level: Confidence level (if None, uses instance default)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        try:
            if confidence_level is None:
                confidence_level = self.confidence_level
            
            # Remove NaN values
            clean_data = data[~np.isnan(data)]
            
            if len(clean_data) == 0:
                return 0.0, 0.0, 0.0
            
            mean_value = float(np.mean(clean_data))
            std_error = float(np.std(clean_data) / np.sqrt(len(clean_data)))
            
            # Use t-distribution for small samples, normal for large
            if len(clean_data) > 30:
                z_score = 1.96  # Approximate for 95% CI
            else:
                # Rough approximation of t-distribution
                df = len(clean_data) - 1
                z_score = 2.0 + 0.5 / df  # Rough approximation
            
            margin = z_score * std_error
            
            return mean_value, mean_value - margin, mean_value + margin
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def detect_outliers_statistical(
        self, 
        data: np.ndarray, 
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect outliers using statistical methods.
        
        Args:
            data: Data array
            method: Outlier detection method ('zscore', 'iqr', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        try:
            data = np.array(data)
            outliers = np.zeros(len(data), dtype=bool)
            
            if method == 'zscore':
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                outliers = z_scores > threshold
                
            elif method == 'iqr':
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'modified_zscore':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
                outliers = np.abs(modified_z_scores) > threshold
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            return np.zeros(len(data), dtype=bool)
    
    def calculate_summary_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive summary statistics.
        
        Args:
            data: Data array
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            # Remove NaN values
            clean_data = data[~np.isnan(data)]
            
            if len(clean_data) == 0:
                return {key: 0.0 for key in [
                    'count', 'mean', 'std', 'min', 'max', 'median',
                    'q25', 'q75', 'skewness', 'kurtosis'
                ]}
            
            stats = {
                'count': float(len(clean_data)),
                'mean': float(np.mean(clean_data)),
                'std': float(np.std(clean_data)),
                'min': float(np.min(clean_data)),
                'max': float(np.max(clean_data)),
                'median': float(np.median(clean_data)),
                'q25': float(np.percentile(clean_data, 25)),
                'q75': float(np.percentile(clean_data, 75)),
            }
            
            # Calculate skewness and kurtosis
            mean = stats['mean']
            std = stats['std']
            
            if std != 0 and len(clean_data) > 2:
                # Sample skewness
                skewness = np.mean(((clean_data - mean) / std) ** 3)
                stats['skewness'] = float(skewness)
                
                # Sample excess kurtosis
                kurtosis = np.mean(((clean_data - mean) / std) ** 4) - 3
                stats['kurtosis'] = float(kurtosis)
            else:
                stats['skewness'] = 0.0
                stats['kurtosis'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating summary statistics: {str(e)}")
            return {key: 0.0 for key in [
                'count', 'mean', 'std', 'min', 'max', 'median',
                'q25', 'q75', 'skewness', 'kurtosis'
            ]}
    
    def test_normality(self, data: np.ndarray) -> Dict[str, float]:
        """
        Test data for normality (simplified test).
        
        Args:
            data: Data array
            
        Returns:
            Dictionary with normality test results
        """
        try:
            # Remove NaN values
            clean_data = data[~np.isnan(data)]
            
            if len(clean_data) < 3:
                return {'test_statistic': 0.0, 'p_value': 1.0, 'is_normal': False}
            
            # Simple normality test based on skewness and kurtosis
            stats = self.calculate_summary_statistics(clean_data)
            skewness = abs(stats['skewness'])
            kurtosis = abs(stats['kurtosis'])
            
            # Rough criteria for normality
            is_normal = skewness < 2.0 and kurtosis < 2.0
            
            # Combined test statistic
            test_statistic = skewness + kurtosis
            
            # Rough p-value approximation
            if is_normal:
                p_value = 0.8 - test_statistic * 0.2  # Rough approximation
            else:
                p_value = 0.1 - test_statistic * 0.02  # Rough approximation
            
            p_value = max(0.0, min(1.0, p_value))  # Clamp to [0, 1]
            
            return {
                'test_statistic': test_statistic,
                'p_value': p_value,
                'is_normal': is_normal
            }
            
        except Exception as e:
            self.logger.error(f"Error testing normality: {str(e)}")
            return {'test_statistic': 0.0, 'p_value': 1.0, 'is_normal': False}
