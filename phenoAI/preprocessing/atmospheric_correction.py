"""
Atmospheric correction module for PhenoAI package.
"""

import numpy as np
import cv2
from typing import Tuple, Optional

from ..core.logger import LoggerMixin
from ..core.exceptions import ProcessingError

class AtmosphericCorrector(LoggerMixin):
    """Atmospheric correction utilities for PhenoCam images."""
    
    def __init__(self, config=None):
        """Initialize atmospheric corrector with configuration."""
        self.config = config
    
    def dark_object_subtraction(
        self, 
        image: np.ndarray, 
        percentile: float = 1.0
    ) -> np.ndarray:
        """
        Apply dark object subtraction for atmospheric correction.
        
        Args:
            image: Input image
            percentile: Percentile for dark object detection
            
        Returns:
            Atmospherically corrected image
        """
        try:
            corrected = image.copy().astype(np.float32)
            
            for channel in range(image.shape[2]):
                # Find dark object value (lowest percentile)
                dark_value = np.percentile(image[:, :, channel], percentile)
                
                # Subtract dark object value
                corrected[:, :, channel] = corrected[:, :, channel] - dark_value
                
                # Clip negative values
                corrected[:, :, channel] = np.clip(corrected[:, :, channel], 0, 255)
            
            return corrected.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in dark object subtraction: {str(e)}")
            raise ProcessingError(f"Failed to apply dark object subtraction: {str(e)}")
    
    def haze_removal(self, image: np.ndarray, omega: float = 0.95) -> np.ndarray:
        """
        Simple haze removal using dark channel prior.
        
        Args:
            image: Input image
            omega: Transmission estimation parameter
            
        Returns:
            Dehazed image
        """
        try:
            image_float = image.astype(np.float32) / 255.0
            
            # Calculate dark channel
            dark_channel = np.min(image_float, axis=2)
            
            # Estimate atmospheric light
            flat_dark = dark_channel.flatten()
            flat_image = image_float.reshape(-1, 3)
            indices = np.argsort(flat_dark)
            top_pixels = indices[-int(len(indices) * 0.001):]
            atmospheric_light = np.max(flat_image[top_pixels], axis=0)
            
            # Estimate transmission
            transmission = 1 - omega * (dark_channel / np.max(atmospheric_light))
            transmission = np.maximum(transmission, 0.1)  # Avoid division by very small numbers
            
            # Recover scene radiance
            dehazed = np.zeros_like(image_float)
            for c in range(3):
                dehazed[:, :, c] = (image_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]
            
            # Clip and convert back
            dehazed = np.clip(dehazed, 0, 1)
            return (dehazed * 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in haze removal: {str(e)}")
            raise ProcessingError(f"Failed to remove haze: {str(e)}")
    
    def white_balance_correction(self, image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
        """
        Apply white balance correction.
        
        Args:
            image: Input image
            method: White balance method ('gray_world', 'max_rgb')
            
        Returns:
            White balanced image
        """
        try:
            image_float = image.astype(np.float32)
            
            if method == 'gray_world':
                # Gray world assumption
                mean_values = np.mean(image_float, axis=(0, 1))
                gray_mean = np.mean(mean_values)
                scaling_factors = gray_mean / mean_values
                
            elif method == 'max_rgb':
                # Max RGB method
                max_values = np.max(image_float, axis=(0, 1))
                max_overall = np.max(max_values)
                scaling_factors = max_overall / max_values
                
            else:
                raise ValueError(f"Unknown white balance method: {method}")
            
            # Apply scaling factors
            corrected = image_float * scaling_factors
            corrected = np.clip(corrected, 0, 255)
            
            return corrected.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in white balance correction: {str(e)}")
            raise ProcessingError(f"Failed to correct white balance: {str(e)}")
    
    def apply_atmospheric_correction(
        self, 
        image: np.ndarray, 
        correction_type: str = 'auto'
    ) -> np.ndarray:
        """
        Apply comprehensive atmospheric correction.
        
        Args:
            image: Input image
            correction_type: Type of correction ('auto', 'dos', 'haze', 'white_balance')
            
        Returns:
            Corrected image
        """
        try:
            if correction_type == 'auto':
                # Apply multiple corrections
                corrected = self.white_balance_correction(image, method='gray_world')
                corrected = self.dark_object_subtraction(corrected, percentile=1.0)
                return corrected
                
            elif correction_type == 'dos':
                return self.dark_object_subtraction(image)
                
            elif correction_type == 'haze':
                return self.haze_removal(image)
                
            elif correction_type == 'white_balance':
                return self.white_balance_correction(image)
                
            else:
                raise ValueError(f"Unknown correction type: {correction_type}")
                
        except Exception as e:
            self.logger.error(f"Error in atmospheric correction: {str(e)}")
            raise ProcessingError(f"Failed to apply atmospheric correction: {str(e)}")
