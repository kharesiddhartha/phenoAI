"""
Vegetation Indices Calculation Module for PhenoAI

Provides simplified vegetation index calculations compatible with the clean package.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List, Any

class VegetationIndexCalculator:
    """
    Simplified vegetation index calculator for PhenoAI.
    
    Calculates basic vegetation indices: GCC, RCC, BCC, ExG, CIVE
    """
    
    def __init__(self):
        """Initialize vegetation calculator."""
        pass
    
    def calculate_all(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """
        Calculate all vegetation indices for an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Dictionary with vegetation index values
        """
        try:
            # Apply ROI if specified
            if roi is not None:
                x, y, w, h = roi
                image = image[y:y+h, x:x+w]
            
            # Calculate individual indices
            gcc_val = self.gcc(image)
            rcc_val = self.rcc(image)
            bcc_val = self.bcc(image)
            exg_val = self.exg(image)
            
            return {
                'gcc': gcc_val,
                'rcc': rcc_val,
                'bcc': bcc_val,
                'exg': exg_val
            }
            
        except Exception as e:
            print(f"Error calculating vegetation indices: {e}")
            return {
                'gcc': 0.0,
                'rcc': 0.0,
                'bcc': 0.0,
                'exg': 0.0
            }
    
    def gcc(self, image: np.ndarray) -> float:
        """
        Calculate Green Chromatic Coordinate (GCC).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            GCC value
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                b, g, r = cv2.split(image)
            else:
                return 0.0
            
            # Convert to float to avoid overflow
            r = r.astype(np.float64)
            g = g.astype(np.float64)
            b = b.astype(np.float64)
            
            # Calculate total
            total = r + g + b
            
            # Avoid division by zero
            mask = total > 0
            if not np.any(mask):
                return 0.0
            
            # Calculate GCC
            gcc_values = np.zeros_like(total)
            gcc_values[mask] = g[mask] / total[mask]
            
            return float(np.mean(gcc_values[mask]))
            
        except Exception as e:
            print(f"Error calculating GCC: {e}")
            return 0.0
    
    def rcc(self, image: np.ndarray) -> float:
        """
        Calculate Red Chromatic Coordinate (RCC).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            RCC value
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                b, g, r = cv2.split(image)
            else:
                return 0.0
            
            # Convert to float
            r = r.astype(np.float64)
            g = g.astype(np.float64)
            b = b.astype(np.float64)
            
            # Calculate total
            total = r + g + b
            
            # Avoid division by zero
            mask = total > 0
            if not np.any(mask):
                return 0.0
            
            # Calculate RCC
            rcc_values = np.zeros_like(total)
            rcc_values[mask] = r[mask] / total[mask]
            
            return float(np.mean(rcc_values[mask]))
            
        except Exception as e:
            print(f"Error calculating RCC: {e}")
            return 0.0
    
    def bcc(self, image: np.ndarray) -> float:
        """
        Calculate Blue Chromatic Coordinate (BCC).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            BCC value
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                b, g, r = cv2.split(image)
            else:
                return 0.0
            
            # Convert to float
            r = r.astype(np.float64)
            g = g.astype(np.float64)
            b = b.astype(np.float64)
            
            # Calculate total
            total = r + g + b
            
            # Avoid division by zero
            mask = total > 0
            if not np.any(mask):
                return 0.0
            
            # Calculate BCC
            bcc_values = np.zeros_like(total)
            bcc_values[mask] = b[mask] / total[mask]
            
            return float(np.mean(bcc_values[mask]))
            
        except Exception as e:
            print(f"Error calculating BCC: {e}")
            return 0.0
    
    def exg(self, image: np.ndarray) -> float:
        """
        Calculate Excess Green Index (ExG).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            ExG value
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                b, g, r = cv2.split(image)
            else:
                return 0.0
            
            # Convert to float and normalize to [0, 1]
            r = r.astype(np.float64) / 255.0
            g = g.astype(np.float64) / 255.0
            b = b.astype(np.float64) / 255.0
            
            # Calculate ExG: 2*G - R - B
            exg_values = 2 * g - r - b
            
            return float(np.mean(exg_values))
            
        except Exception as e:
            print(f"Error calculating ExG: {e}")
            return 0.0
    """
    Calculator for various vegetation indices from PhenoCam images.
    
    Implements multiple vegetation indices including chromatic coordinates
    and advanced indices used in phenological studies.
    """
    
    def __init__(self, config=None):
        """
        Initialize vegetation index calculator.
        
        Args:
            config: Configuration object with vegetation indices settings
        """
        if config is None:
            # Default configuration
            self.indices = ['gcc', 'rcc', 'bcc', 'exg', 'cive', 'vf', 'hue', 'saturation']
            self.roi_size = (50, 50)
            self.num_rois = 3
            self.roi_selection_method = 'automated'
            self.color_space = 'RGB'
        else:
            self.indices = config.indices
            self.roi_size = config.roi_size
            self.num_rois = config.num_rois
            self.roi_selection_method = config.roi_selection_method
            self.color_space = config.color_space
    
    def calculate_chromatic_coordinates(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Calculate Red, Green, and Blue Chromatic Coordinates.
        
        Args:
            image: Input image in BGR format
            mask: Optional mask to limit calculation to specific regions
            
        Returns:
            Tuple of (RCC, GCC, BCC)
        """
        try:
            if mask is not None:
                # Apply mask
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                pixels = masked_image[mask > 0]
            else:
                pixels = image.reshape(-1, 3)
            
            # Remove black pixels (likely background)
            non_black = np.sum(pixels, axis=1) > 30
            pixels = pixels[non_black]
            
            if len(pixels) == 0:
                self.logger.warning("No valid pixels found for chromatic coordinate calculation")
                return 0.0, 0.0, 0.0
            
            # Calculate chromatic coordinates
            # Note: OpenCV uses BGR format
            b_channel = pixels[:, 0].astype(np.float32)
            g_channel = pixels[:, 1].astype(np.float32)
            r_channel = pixels[:, 2].astype(np.float32)
            
            # Calculate denominator (total digital numbers)
            total_dn = r_channel + g_channel + b_channel
            
            # Avoid division by zero
            valid_pixels = total_dn > 0
            total_dn = total_dn[valid_pixels]
            r_channel = r_channel[valid_pixels]
            g_channel = g_channel[valid_pixels]
            b_channel = b_channel[valid_pixels]
            
            if len(total_dn) == 0:
                return 0.0, 0.0, 0.0
            
            # Calculate chromatic coordinates
            rcc = np.mean(r_channel / total_dn)
            gcc = np.mean(g_channel / total_dn)
            bcc = np.mean(b_channel / total_dn)
            
            return float(rcc), float(gcc), float(bcc)
            
        except Exception as e:
            self.logger.error(f"Error calculating chromatic coordinates: {str(e)}")
            raise AnalysisError(f"Failed to calculate chromatic coordinates: {str(e)}")
    
    def calculate_excess_green(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Excess Green Index (ExG).
        
        ExG = 2 * G - R - B (normalized)
        
        Args:
            image: Input image in BGR format
            mask: Optional mask to limit calculation
            
        Returns:
            Excess Green Index value
        """
        try:
            if mask is not None:
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                pixels = masked_image[mask > 0]
            else:
                pixels = image.reshape(-1, 3)
            
            # Remove black pixels
            non_black = np.sum(pixels, axis=1) > 30
            pixels = pixels[non_black]
            
            if len(pixels) == 0:
                return 0.0
            
            # Convert to float and normalize
            pixels = pixels.astype(np.float32) / 255.0
            
            # Calculate ExG: 2*G - R - B
            b_channel = pixels[:, 0]
            g_channel = pixels[:, 1]
            r_channel = pixels[:, 2]
            
            exg = 2 * g_channel - r_channel - b_channel
            
            return float(np.mean(exg))
            
        except Exception as e:
            self.logger.error(f"Error calculating ExG: {str(e)}")
            raise AnalysisError(f"Failed to calculate ExG: {str(e)}")
    
    def calculate_cive(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Color Index of Vegetation (CIVE).
        
        CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
        
        Args:
            image: Input image in BGR format
            mask: Optional mask to limit calculation
            
        Returns:
            CIVE value
        """
        try:
            if mask is not None:
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                pixels = masked_image[mask > 0]
            else:
                pixels = image.reshape(-1, 3)
            
            # Remove black pixels
            non_black = np.sum(pixels, axis=1) > 30
            pixels = pixels[non_black]
            
            if len(pixels) == 0:
                return 0.0
            
            # Convert to float
            pixels = pixels.astype(np.float32)
            
            # Calculate CIVE
            b_channel = pixels[:, 0]
            g_channel = pixels[:, 1]
            r_channel = pixels[:, 2]
            
            cive = 0.441 * r_channel - 0.811 * g_channel + 0.385 * b_channel + 18.78745
            
            return float(np.mean(cive))
            
        except Exception as e:
            self.logger.error(f"Error calculating CIVE: {str(e)}")
            raise AnalysisError(f"Failed to calculate CIVE: {str(e)}")
    
    def calculate_vegetation_fraction(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Vegetation Fraction (VF) using a simple green threshold.
        
        Args:
            image: Input image in BGR format
            mask: Optional mask to limit calculation
            
        Returns:
            Vegetation fraction (0-1)
        """
        try:
            if mask is not None:
                work_image = cv2.bitwise_and(image, image, mask=mask)
                total_pixels = np.sum(mask > 0)
            else:
                work_image = image.copy()
                total_pixels = image.shape[0] * image.shape[1]
            
            # Convert to HSV for better vegetation detection
            hsv = cv2.cvtColor(work_image, cv2.COLOR_BGR2HSV)
            
            # Define green color range in HSV
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            
            # Create mask for green vegetation
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            if mask is not None:
                green_mask = cv2.bitwise_and(green_mask, mask)
            
            # Calculate vegetation fraction
            vegetation_pixels = np.sum(green_mask > 0)
            
            if total_pixels == 0:
                return 0.0
            
            return float(vegetation_pixels / total_pixels)
            
        except Exception as e:
            self.logger.error(f"Error calculating vegetation fraction: {str(e)}")
            raise AnalysisError(f"Failed to calculate vegetation fraction: {str(e)}")
    
    def calculate_hue_saturation(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Calculate average hue and saturation values.
        
        Args:
            image: Input image in BGR format
            mask: Optional mask to limit calculation
            
        Returns:
            Tuple of (hue, saturation)
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            if mask is not None:
                pixels = hsv[mask > 0]
            else:
                pixels = hsv.reshape(-1, 3)
            
            # Remove low saturation pixels (likely background)
            high_sat = pixels[:, 1] > 30
            pixels = pixels[high_sat]
            
            if len(pixels) == 0:
                return 0.0, 0.0
            
            # Calculate average hue and saturation
            hue = np.mean(pixels[:, 0]) * 2  # Convert from 0-179 to 0-360
            saturation = np.mean(pixels[:, 1]) / 255.0  # Normalize to 0-1
            
            return float(hue), float(saturation)
            
        except Exception as e:
            self.logger.error(f"Error calculating hue/saturation: {str(e)}")
            raise AnalysisError(f"Failed to calculate hue/saturation: {str(e)}")
    
    def select_vegetation_rois(
        self, 
        image: np.ndarray, 
        vegetation_mask: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Select ROIs (Regions of Interest) for vegetation analysis.
        
        Args:
            image: Input image
            vegetation_mask: Optional pre-computed vegetation mask
            
        Returns:
            List of ROI coordinates (x, y, width, height)
        """
        try:
            if vegetation_mask is None:
                # Create vegetation mask using green color threshold
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 50, 50])
                upper_green = np.array([85, 255, 255])
                vegetation_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            rois = []
            
            if self.roi_selection_method == 'automated':
                # Find contours of vegetation areas
                contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Select top N contours
                for i, contour in enumerate(contours[:self.num_rois]):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Ensure minimum size
                    if w >= self.roi_size[0] and h >= self.roi_size[1]:
                        rois.append((x, y, w, h))
                
            elif self.roi_selection_method == 'grid':
                # Grid-based ROI selection
                h, w = image.shape[:2]
                roi_w, roi_h = self.roi_size
                
                # Calculate grid positions
                rows = int(np.sqrt(self.num_rois))
                cols = self.num_rois // rows
                
                for i in range(rows):
                    for j in range(cols):
                        if len(rois) >= self.num_rois:
                            break
                        
                        x = int((w - roi_w) * (j + 0.5) / cols)
                        y = int((h - roi_h) * (i + 0.5) / rows)
                        
                        # Check if ROI contains vegetation
                        roi_mask = vegetation_mask[y:y+roi_h, x:x+roi_w]
                        if np.sum(roi_mask > 0) > (roi_w * roi_h * 0.3):  # At least 30% vegetation
                            rois.append((x, y, roi_w, roi_h))
            
            # If no ROIs found, use center region
            if not rois:
                h, w = image.shape[:2]
                roi_w, roi_h = self.roi_size
                x = (w - roi_w) // 2
                y = (h - roi_h) // 2
                rois.append((x, y, roi_w, roi_h))
                self.logger.warning("No vegetation ROIs found, using center region")
            
            return rois
            
        except Exception as e:
            self.logger.error(f"Error selecting ROIs: {str(e)}")
            raise AnalysisError(f"Failed to select vegetation ROIs: {str(e)}")
    
    def calculate_all_indices(
        self, 
        image: np.ndarray, 
        date: str = "", 
        filename: str = "",
        vegetation_mask: Optional[np.ndarray] = None
    ) -> VegetationIndices:
        """
        Calculate all vegetation indices for an image.
        
        Args:
            image: Input image in BGR format
            date: Date string for the image
            filename: Filename for tracking
            vegetation_mask: Optional vegetation mask
            
        Returns:
            VegetationIndices object with all calculated indices
        """
        try:
            # Select ROIs for analysis
            rois = self.select_vegetation_rois(image, vegetation_mask)
            
            # Create combined mask for all ROIs
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for x, y, w, h in rois:
                combined_mask[y:y+h, x:x+w] = 255
            
            # If vegetation mask is provided, intersect with ROI mask
            if vegetation_mask is not None:
                combined_mask = cv2.bitwise_and(combined_mask, vegetation_mask)
            
            # Calculate chromatic coordinates
            rcc, gcc, bcc = self.calculate_chromatic_coordinates(image, combined_mask)
            
            # Calculate additional indices
            exg = self.calculate_excess_green(image, combined_mask) if 'exg' in self.indices else 0.0
            cive = self.calculate_cive(image, combined_mask) if 'cive' in self.indices else 0.0
            vf = self.calculate_vegetation_fraction(image, combined_mask) if 'vf' in self.indices else 0.0
            
            # Calculate hue and saturation
            hue, saturation = self.calculate_hue_saturation(image, combined_mask) if 'hue' in self.indices or 'saturation' in self.indices else (0.0, 0.0)
            
            # Calculate brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if combined_mask is not None:
                brightness = float(np.mean(gray[combined_mask > 0]) / 255.0)
            else:
                brightness = float(np.mean(gray) / 255.0)
            
            # Calculate GCC standard deviation for uncertainty
            if combined_mask is not None:
                masked_pixels = image[combined_mask > 0]
                if len(masked_pixels) > 0:
                    # Calculate GCC for each pixel
                    pixel_total = np.sum(masked_pixels, axis=1).astype(np.float32)
                    valid_pixels = pixel_total > 0
                    if np.any(valid_pixels):
                        pixel_gcc = masked_pixels[valid_pixels, 1] / pixel_total[valid_pixels]
                        gcc_std = float(np.std(pixel_gcc))
                        pixel_count = len(pixel_gcc)
                    else:
                        gcc_std = 0.0
                        pixel_count = 0
                else:
                    gcc_std = 0.0
                    pixel_count = 0
            else:
                gcc_std = 0.0
                pixel_count = 0
            
            return VegetationIndices(
                date=date,
                filename=filename,
                gcc=gcc,
                rcc=rcc,
                bcc=bcc,
                exg=exg,
                cive=cive,
                vf=vf,
                hue=hue,
                saturation=saturation,
                brightness=brightness,
                gcc_std=gcc_std,
                pixel_count=pixel_count,
                roi_coordinates=rois
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating vegetation indices: {str(e)}")
            raise AnalysisError(f"Failed to calculate vegetation indices: {str(e)}")
    
    def batch_calculate_indices(
        self, 
        images: List[Tuple[np.ndarray, str, str]], 
        vegetation_masks: Optional[List[np.ndarray]] = None
    ) -> List[VegetationIndices]:
        """
        Calculate vegetation indices for multiple images.
        
        Args:
            images: List of (image, date, filename) tuples
            vegetation_masks: Optional list of vegetation masks
            
        Returns:
            List of VegetationIndices objects
        """
        results = []
        
        for i, (image, date, filename) in enumerate(images):
            try:
                mask = vegetation_masks[i] if vegetation_masks and i < len(vegetation_masks) else None
                indices = self.calculate_all_indices(image, date, filename, mask)
                results.append(indices)
                
                self.logger.debug(f"Calculated indices for {filename}: GCC={indices.gcc:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")
                # Continue with other images
                continue
        
        self.logger.info(f"Calculated vegetation indices for {len(results)} images")
        return results
