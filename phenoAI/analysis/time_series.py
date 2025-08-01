"""
Time series analysis module for PhenoAI package.

This module implements advanced time series analysis methods for phenological data,
including smoothing algorithms, trend detection, and seasonal decomposition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import UnivariateSpline
import warnings

from ..core.logger import LoggerMixin
from ..core.exceptions import AnalysisError, ValidationError

@dataclass
class TimeSeriesResults:
    """Data class to store time series analysis results."""
    dates: List[str]
    original_values: List[float]
    smoothed_values: List[float]
    trend: List[float]
    seasonal: List[float]
    residual: List[float]
    outliers: List[bool]
    confidence_intervals: Optional[Tuple[List[float], List[float]]] = None

class TimeSeriesAnalyzer(LoggerMixin):
    """
    Time series analyzer for phenological data.
    
    Implements various smoothing and decomposition methods for analyzing
    temporal patterns in vegetation indices.
    """
    
    def __init__(self, config=None):
        """
        Initialize time series analyzer.
        
        Args:
            config: Configuration object with time series settings
        """
        if config is None:
            # Default configuration
            self.smoothing_algorithm = 'savgol'
            self.smoothing_window = 15
            self.polynomial_order = 3
            self.loess_frac = 0.3
            self.confidence_level = 0.95
            self.outlier_detection_method = 'zscore'
        else:
            self.smoothing_algorithm = config.smoothing_algorithm
            self.smoothing_window = config.smoothing_window
            self.polynomial_order = config.polynomial_order
            self.loess_frac = config.loess_frac
            self.confidence_level = config.confidence_level
            self.outlier_detection_method = config.outlier_detection_method
    
    def detect_outliers(self, values: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in time series data.
        
        Args:
            values: Time series values
            method: Outlier detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        try:
            values = np.array(values)
            outliers = np.zeros(len(values), dtype=bool)
            
            if method == 'zscore':
                z_scores = np.abs((values - np.mean(values)) / np.std(values))
                outliers = z_scores > threshold
                
            elif method == 'iqr':
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = (values < lower_bound) | (values > upper_bound)
                
            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(values.reshape(-1, 1))
                    outliers = outlier_labels == -1
                except ImportError:
                    self.logger.warning("sklearn not available, falling back to zscore method")
                    return self.detect_outliers(values, method='zscore', threshold=threshold)
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            return outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            raise AnalysisError(f"Failed to detect outliers: {str(e)}")
    
    def savitzky_golay_smoothing(self, values: np.ndarray, window_length: int, poly_order: int) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing filter.
        
        Args:
            values: Time series values
            window_length: Length of the smoothing window (must be odd)
            poly_order: Order of polynomial for fitting
            
        Returns:
            Smoothed values
        """
        try:
            # Ensure window length is odd and valid
            if window_length % 2 == 0:
                window_length += 1
            
            window_length = min(window_length, len(values))
            if window_length <= poly_order:
                window_length = poly_order + 1
                if window_length % 2 == 0:
                    window_length += 1
            
            smoothed = signal.savgol_filter(values, window_length, poly_order, mode='nearest')
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Error in Savitzky-Golay smoothing: {str(e)}")
            raise AnalysisError(f"Failed to apply Savitzky-Golay smoothing: {str(e)}")
    
    def loess_smoothing(self, values: np.ndarray, frac: float = 0.3) -> np.ndarray:
        """
        Apply LOESS (locally estimated scatterplot smoothing).
        
        Args:
            values: Time series values
            frac: Fraction of data to use for smoothing
            
        Returns:
            Smoothed values
        """
        try:
            # Simple implementation using scipy's UnivariateSpline
            x = np.arange(len(values))
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            if not np.any(valid_mask):
                return values
            
            x_valid = x[valid_mask]
            values_valid = values[valid_mask]
            
            # Calculate smoothing parameter
            n = len(values_valid)
            s = n * frac
            
            # Create spline
            spline = UnivariateSpline(x_valid, values_valid, s=s)
            smoothed = spline(x)
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Error in LOESS smoothing: {str(e)}")
            # Fallback to moving average
            return self.moving_average_smoothing(values, int(len(values) * frac))
    
    def moving_average_smoothing(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply moving average smoothing.
        
        Args:
            values: Time series values
            window_size: Size of the moving window
            
        Returns:
            Smoothed values
        """
        try:
            window_size = min(window_size, len(values))
            if window_size <= 1:
                return values
            
            # Pad the array for edge handling
            pad_width = window_size // 2
            padded_values = np.pad(values, pad_width, mode='edge')
            
            # Apply moving average
            smoothed = np.convolve(padded_values, np.ones(window_size) / window_size, mode='valid')
            
            # Ensure output length matches input
            if len(smoothed) != len(values):
                # Trim or pad as necessary
                diff = len(values) - len(smoothed)
                if diff > 0:
                    smoothed = np.pad(smoothed, (diff//2, diff - diff//2), mode='edge')
                else:
                    start = (-diff) // 2
                    smoothed = smoothed[start:start + len(values)]
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Error in moving average smoothing: {str(e)}")
            raise AnalysisError(f"Failed to apply moving average smoothing: {str(e)}")
    
    def gaussian_smoothing(self, values: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian smoothing.
        
        Args:
            values: Time series values
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Smoothed values
        """
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(values, sigma=sigma, mode='nearest')
            return smoothed
            
        except ImportError:
            self.logger.warning("scipy.ndimage not available, falling back to moving average")
            window_size = int(6 * sigma)  # Approximate Gaussian window size
            return self.moving_average_smoothing(values, window_size)
        except Exception as e:
            self.logger.error(f"Error in Gaussian smoothing: {str(e)}")
            raise AnalysisError(f"Failed to apply Gaussian smoothing: {str(e)}")
    
    def apply_smoothing(self, values: np.ndarray, method: str = None) -> np.ndarray:
        """
        Apply smoothing to time series data.
        
        Args:
            values: Time series values
            method: Smoothing method to use (if None, uses configured method)
            
        Returns:
            Smoothed values
        """
        try:
            if method is None:
                method = self.smoothing_algorithm
            
            values = np.array(values)
            
            if method == 'savgol':
                return self.savitzky_golay_smoothing(values, self.smoothing_window, self.polynomial_order)
            elif method == 'loess':
                return self.loess_smoothing(values, self.loess_frac)
            elif method == 'moving_average':
                return self.moving_average_smoothing(values, self.smoothing_window)
            elif method == 'gaussian':
                sigma = self.smoothing_window / 6  # Convert window to sigma
                return self.gaussian_smoothing(values, sigma)
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error applying smoothing: {str(e)}")
            raise AnalysisError(f"Failed to apply smoothing: {str(e)}")
    
    def seasonal_decomposition(self, values: np.ndarray, period: int = 365) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform seasonal decomposition of time series.
        
        Args:
            values: Time series values
            period: Period for seasonal decomposition (default: 365 days)
            
        Returns:
            Tuple of (trend, seasonal, residual)
        """
        try:
            # Simple seasonal decomposition
            n = len(values)
            
            # Calculate trend using moving average
            if n >= period:
                trend = self.moving_average_smoothing(values, period)
            else:
                trend = self.moving_average_smoothing(values, n // 4)
            
            # Remove trend to get detrended series
            detrended = values - trend
            
            # Calculate seasonal component
            if n >= period:
                # Average seasonal pattern
                seasonal_matrix = detrended[:n//period * period].reshape(-1, period)
                seasonal_pattern = np.mean(seasonal_matrix, axis=0)
                
                # Tile pattern to match data length
                seasonal = np.tile(seasonal_pattern, n // period + 1)[:n]
            else:
                # For short series, use simple sinusoidal approximation
                x = np.arange(n)
                seasonal = np.sin(2 * np.pi * x / period) * np.std(detrended) * 0.5
            
            # Calculate residual
            residual = values - trend - seasonal
            
            return trend, seasonal, residual
            
        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition: {str(e)}")
            # Return simple decomposition
            trend = self.apply_smoothing(values, 'moving_average')
            seasonal = np.zeros_like(values)
            residual = values - trend
            return trend, seasonal, residual
    
    def calculate_confidence_intervals(self, values: np.ndarray, smoothed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for smoothed time series.
        
        Args:
            values: Original time series values
            smoothed: Smoothed time series values
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        try:
            # Calculate residuals
            residuals = values - smoothed
            
            # Estimate standard error
            std_error = np.std(residuals)
            
            # Calculate confidence interval using t-distribution approximation
            from scipy import stats
            alpha = 1 - self.confidence_level
            dof = len(values) - 1
            t_value = stats.t.ppf(1 - alpha/2, dof)
            
            margin_error = t_value * std_error
            
            lower_bound = smoothed - margin_error
            upper_bound = smoothed + margin_error
            
            return lower_bound, upper_bound
            
        except ImportError:
            # Fallback without scipy.stats
            std_error = np.std(values - smoothed)
            margin_error = 1.96 * std_error  # Approximate 95% CI
            return smoothed - margin_error, smoothed + margin_error
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {str(e)}")
            # Return smoothed values as both bounds
            return smoothed, smoothed
    
    def analyze_time_series(
        self, 
        dates: List[str], 
        values: List[float],
        remove_outliers: bool = True
    ) -> TimeSeriesResults:
        """
        Perform comprehensive time series analysis.
        
        Args:
            dates: List of date strings
            values: List of time series values
            remove_outliers: Whether to remove outliers before analysis
            
        Returns:
            TimeSeriesResults object with all analysis results
        """
        try:
            if len(dates) != len(values):
                raise ValidationError("Dates and values must have the same length")
            
            if len(values) < 3:
                raise ValidationError("Need at least 3 data points for time series analysis")
            
            # Convert to numpy arrays
            values_array = np.array(values, dtype=float)
            
            # Detect outliers
            outliers = self.detect_outliers(values_array, self.outlier_detection_method)
            
            # Remove outliers if requested
            if remove_outliers and np.any(outliers):
                # Simple outlier removal by interpolation
                clean_values = values_array.copy()
                outlier_indices = np.where(outliers)[0]
                
                for idx in outlier_indices:
                    # Find nearest non-outlier values for interpolation
                    left_idx = idx - 1
                    right_idx = idx + 1
                    
                    while left_idx >= 0 and outliers[left_idx]:
                        left_idx -= 1
                    while right_idx < len(values) and outliers[right_idx]:
                        right_idx += 1
                    
                    if left_idx >= 0 and right_idx < len(values):
                        # Linear interpolation
                        weight = (idx - left_idx) / (right_idx - left_idx)
                        clean_values[idx] = (1 - weight) * values_array[left_idx] + weight * values_array[right_idx]
                    elif left_idx >= 0:
                        clean_values[idx] = values_array[left_idx]
                    elif right_idx < len(values):
                        clean_values[idx] = values_array[right_idx]
                
                analysis_values = clean_values
            else:
                analysis_values = values_array
            
            # Apply smoothing
            smoothed_values = self.apply_smoothing(analysis_values)
            
            # Seasonal decomposition
            trend, seasonal, residual = self.seasonal_decomposition(analysis_values)
            
            # Calculate confidence intervals
            lower_bound, upper_bound = self.calculate_confidence_intervals(analysis_values, smoothed_values)
            
            return TimeSeriesResults(
                dates=dates,
                original_values=values,
                smoothed_values=smoothed_values.tolist(),
                trend=trend.tolist(),
                seasonal=seasonal.tolist(),
                residual=residual.tolist(),
                outliers=outliers.tolist(),
                confidence_intervals=(lower_bound.tolist(), upper_bound.tolist())
            )
            
        except Exception as e:
            self.logger.error(f"Error in time series analysis: {str(e)}")
            raise AnalysisError(f"Failed to analyze time series: {str(e)}")
    
    def detect_change_points(self, values: np.ndarray, method: str = 'pelt', penalty: float = 10.0) -> List[int]:
        """
        Detect change points in time series data.
        
        Args:
            values: Time series values
            method: Change point detection method
            penalty: Penalty parameter for change point detection
            
        Returns:
            List of change point indices
        """
        try:
            # Simple change point detection using variance
            # This is a basic implementation - in practice, you might want to use
            # specialized libraries like ruptures
            
            n = len(values)
            if n < 10:
                return []
            
            change_points = []
            window_size = max(5, n // 20)  # Adaptive window size
            
            for i in range(window_size, n - window_size):
                # Calculate variance before and after potential change point
                before = values[i-window_size:i]
                after = values[i:i+window_size]
                
                var_before = np.var(before)
                var_after = np.var(after)
                var_combined = np.var(values[i-window_size:i+window_size])
                
                # Test for significant change in variance
                if var_combined > (var_before + var_after) * (1 + penalty/100):
                    change_points.append(i)
            
            # Remove change points that are too close to each other
            if len(change_points) > 1:
                filtered_points = [change_points[0]]
                for cp in change_points[1:]:
                    if cp - filtered_points[-1] > window_size:
                        filtered_points.append(cp)
                change_points = filtered_points
            
            return change_points
            
        except Exception as e:
            self.logger.error(f"Error detecting change points: {str(e)}")
            return []
