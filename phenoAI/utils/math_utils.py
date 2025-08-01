"""
Mathematical utilities for PhenoAI package.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from scipy import stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from ..core.logger import LoggerMixin
from ..core.exceptions import AnalysisError

class MathUtils(LoggerMixin):
    """Mathematical utilities for phenological analysis."""
    
    @staticmethod
    def safe_divide(
        numerator: Union[np.ndarray, float], 
        denominator: Union[np.ndarray, float],
        default_value: float = 0.0
    ) -> Union[np.ndarray, float]:
        """
        Safe division avoiding division by zero.
        
        Args:
            numerator: Numerator values
            denominator: Denominator values
            default_value: Value to use when denominator is zero
            
        Returns:
            Division result with safe handling of zero division
        """
        try:
            if isinstance(denominator, np.ndarray):
                result = np.divide(
                    numerator, 
                    denominator, 
                    out=np.full_like(denominator, default_value, dtype=float),
                    where=(denominator != 0)
                )
                return result
            else:
                return numerator / denominator if denominator != 0 else default_value
                
        except Exception:
            return default_value
    
    @staticmethod
    def remove_outliers(
        data: np.ndarray, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from data.
        
        Args:
            data: Input data array
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (clean_data, outlier_indices)
        """
        try:
            if len(data) == 0:
                return data, np.array([])
            
            if method == 'iqr':
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            clean_data = data[~outliers]
            outlier_indices = np.where(outliers)[0]
            
            return clean_data, outlier_indices
            
        except Exception as e:
            # Return original data if outlier detection fails
            return data, np.array([])
    
    @staticmethod
    def smooth_data(
        data: np.ndarray,
        method: str = 'savgol',
        window_size: int = 5,
        **kwargs
    ) -> np.ndarray:
        """
        Smooth data using various methods.
        
        Args:
            data: Input data array
            method: Smoothing method ('savgol', 'moving_average', 'gaussian')
            window_size: Size of smoothing window
            **kwargs: Additional parameters for smoothing methods
            
        Returns:
            Smoothed data array
        """
        try:
            if len(data) < window_size:
                return data
            
            if method == 'savgol':
                poly_order = kwargs.get('poly_order', min(3, window_size - 1))
                if window_size % 2 == 0:
                    window_size += 1  # Ensure odd window size
                return savgol_filter(data, window_size, poly_order)
                
            elif method == 'moving_average':
                return np.convolve(data, np.ones(window_size) / window_size, mode='same')
                
            elif method == 'gaussian':
                sigma = kwargs.get('sigma', window_size / 4)
                from scipy.ndimage import gaussian_filter1d
                return gaussian_filter1d(data, sigma)
                
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
                
        except Exception as e:
            # Return original data if smoothing fails
            return data
    
    @staticmethod
    def interpolate_missing(
        data: np.ndarray,
        x: Optional[np.ndarray] = None,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Interpolate missing values in data.
        
        Args:
            data: Data array with potential NaN values
            x: X coordinates (if None, use indices)
            method: Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
            Data array with interpolated values
        """
        try:
            if x is None:
                x = np.arange(len(data))
            
            # Find non-NaN values
            mask = ~np.isnan(data)
            if not np.any(mask):
                return data  # All NaN, cannot interpolate
            
            if np.all(mask):
                return data  # No NaN values, no interpolation needed
            
            # Interpolate
            f = interp1d(
                x[mask], 
                data[mask], 
                kind=method, 
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            interpolated_data = data.copy()
            interpolated_data[~mask] = f(x[~mask])
            
            return interpolated_data
            
        except Exception as e:
            return data
    
    @staticmethod
    def find_peaks_and_valleys(
        data: np.ndarray,
        prominence: float = 0.1,
        distance: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks and valleys in data.
        
        Args:
            data: Input data array
            prominence: Required prominence of peaks
            distance: Minimum distance between peaks
            
        Returns:
            Tuple of (peak_indices, valley_indices)
        """
        try:
            from scipy.signal import find_peaks
            
            # Find peaks
            peaks, _ = find_peaks(data, prominence=prominence, distance=distance)
            
            # Find valleys (peaks in inverted data)
            valleys, _ = find_peaks(-data, prominence=prominence, distance=distance)
            
            return peaks, valleys
            
        except Exception as e:
            return np.array([]), np.array([])
    
    @staticmethod
    def calculate_trend(
        data: np.ndarray,
        x: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate trend statistics for data.
        
        Args:
            data: Input data array
            x: X coordinates (if None, use indices)
            
        Returns:
            Dictionary with trend statistics
        """
        try:
            if x is None:
                x = np.arange(len(data))
            
            # Remove NaN values
            mask = ~np.isnan(data)
            if np.sum(mask) < 2:
                return {
                    'slope': 0.0,
                    'intercept': 0.0,
                    'r_value': 0.0,
                    'p_value': 1.0,
                    'std_err': 0.0
                }
            
            x_clean = x[mask]
            data_clean = data[mask]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, data_clean)
            
            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_value': float(r_value),
                'p_value': float(p_value),
                'std_err': float(std_err)
            }
            
        except Exception as e:
            return {
                'slope': 0.0,
                'intercept': 0.0,
                'r_value': 0.0,
                'p_value': 1.0,
                'std_err': 0.0
            }
    
    @staticmethod
    def rolling_statistics(
        data: np.ndarray,
        window_size: int,
        statistics: List[str] = ['mean', 'std', 'min', 'max']
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling statistics for data.
        
        Args:
            data: Input data array
            window_size: Size of rolling window
            statistics: List of statistics to calculate
            
        Returns:
            Dictionary with rolling statistics
        """
        try:
            import pandas as pd
            
            series = pd.Series(data)
            rolling = series.rolling(window=window_size, min_periods=1)
            
            results = {}
            for stat in statistics:
                if hasattr(rolling, stat):
                    results[stat] = getattr(rolling, stat)().values
                else:
                    results[stat] = np.full_like(data, np.nan)
            
            return results
            
        except Exception as e:
            # Fallback to simple implementations
            results = {}
            for stat in statistics:
                if stat == 'mean':
                    results[stat] = np.convolve(
                        data, 
                        np.ones(window_size) / window_size, 
                        mode='same'
                    )
                else:
                    results[stat] = np.full_like(data, np.nan)
            
            return results

class StatisticalTests(LoggerMixin):
    """Statistical tests for phenological analysis."""
    
    @staticmethod
    def seasonal_mann_kendall(
        data: np.ndarray,
        seasons: np.ndarray
    ) -> Dict[str, float]:
        """
        Seasonal Mann-Kendall test for trend detection.
        
        Args:
            data: Time series data
            seasons: Season indicators (e.g., month numbers)
            
        Returns:
            Dictionary with test statistics
        """
        try:
            from scipy.stats import norm
            
            # Group data by season
            unique_seasons = np.unique(seasons)
            s_values = []
            
            for season in unique_seasons:
                season_mask = seasons == season
                season_data = data[season_mask]
                
                if len(season_data) < 3:
                    continue
                
                # Calculate S for this season
                n = len(season_data)
                s = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        if season_data[j] > season_data[i]:
                            s += 1
                        elif season_data[j] < season_data[i]:
                            s -= 1
                
                s_values.append(s)
            
            # Combine seasonal S values
            total_s = sum(s_values)
            
            # Calculate variance (simplified)
            n_total = len(data)
            var_s = n_total * (n_total - 1) * (2 * n_total + 5) / 18
            
            # Calculate Z statistic
            if total_s > 0:
                z = (total_s - 1) / np.sqrt(var_s)
            elif total_s < 0:
                z = (total_s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # Calculate p-value
            p_value = 2 * (1 - norm.cdf(abs(z)))
            
            return {
                's_statistic': float(total_s),
                'z_statistic': float(z),
                'p_value': float(p_value),
                'trend': 'increasing' if z > 0 else 'decreasing' if z < 0 else 'no trend'
            }
            
        except Exception as e:
            return {
                's_statistic': 0.0,
                'z_statistic': 0.0,
                'p_value': 1.0,
                'trend': 'no trend'
            }

# Convenience functions
def safe_divide(numerator, denominator, default_value=0.0):
    """Safe division avoiding division by zero."""
    return MathUtils.safe_divide(numerator, denominator, default_value)

def remove_outliers(data, method='iqr', threshold=1.5):
    """Remove outliers from data."""
    return MathUtils.remove_outliers(data, method, threshold)

def smooth_data(data, method='savgol', window_size=5, **kwargs):
    """Smooth data using various methods."""
    return MathUtils.smooth_data(data, method, window_size, **kwargs)
