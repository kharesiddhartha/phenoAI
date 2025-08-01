"""
Image enhancement module for PhenoAI package.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from ..core.logger import LoggerMixin
from ..core.exceptions import ProcessingError

class ImageEnhancer(LoggerMixin):
    """Image enhancement utilities for PhenoCam images."""
    
    def __init__(self, config=None):
        """Initialize image enhancer with configuration."""
        self.config = config
    
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            clip_limit: Clipping limit for CLAHE
            
        Returns:
            Contrast enhanced image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing contrast: {str(e)}")
            raise ProcessingError(f"Failed to enhance contrast: {str(e)}")
    
    def adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Adjust image gamma for brightness correction.
        
        Args:
            image: Input image
            gamma: Gamma value (< 1 = brighter, > 1 = darker)
            
        Returns:
            Gamma corrected image
        """
        try:
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            
            # Apply gamma correction
            return cv2.LUT(image, table)
            
        except Exception as e:
            self.logger.error(f"Error adjusting gamma: {str(e)}")
            raise ProcessingError(f"Failed to adjust gamma: {str(e)}")
    
    def denoise_image(self, image: np.ndarray, method: str = 'nlm') -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image
            method: Denoising method ('nlm', 'bilateral', 'gaussian')
            
        Returns:
            Denoised image
        """
        try:
            if method == 'nlm':
                # Non-local means denoising
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            elif method == 'bilateral':
                # Bilateral filtering
                return cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'gaussian':
                # Gaussian blur
                return cv2.GaussianBlur(image, (5, 5), 0)
            else:
                raise ValueError(f"Unknown denoising method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error denoising image: {str(e)}")
            raise ProcessingError(f"Failed to denoise image: {str(e)}")
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        
        Args:
            image: Input image
            strength: Sharpening strength
            
        Returns:
            Sharpened image
        """
        try:
            # Create unsharp mask
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
            
            return unsharp_mask
            
        except Exception as e:
            self.logger.error(f"Error sharpening image: {str(e)}")
            raise ProcessingError(f"Failed to sharpen image: {str(e)}")
    
    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically enhance image using multiple techniques.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Start with contrast enhancement
            enhanced = self.enhance_contrast(image, clip_limit=2.0)
            
            # Slight denoising
            enhanced = self.denoise_image(enhanced, method='bilateral')
            
            # Subtle sharpening
            enhanced = self.sharpen_image(enhanced, strength=0.5)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in auto enhancement: {str(e)}")
            raise ProcessingError(f"Failed to auto enhance image: {str(e)}")
