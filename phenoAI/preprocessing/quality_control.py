"""
Quality control module for PhenoAI.
"""

import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    """Quality assessment metrics for an image."""
    overall_score: float
    brightness: float
    contrast: float
    sharpness: float
    has_fog: bool
    has_snow: bool
    is_blurry: bool
    is_too_dark: bool

class QualityController:
    """Image quality assessment controller."""
    
    def __init__(self, fog_threshold: float = 0.15, snow_threshold: float = 0.8,
                 blur_threshold: float = 100.0, darkness_threshold: float = 50.0):
        """
        Initialize quality controller.
        
        Args:
            fog_threshold: Threshold for fog detection
            snow_threshold: Threshold for snow detection
            blur_threshold: Threshold for blur detection
            darkness_threshold: Threshold for darkness detection
        """
        self.fog_threshold = fog_threshold
        self.snow_threshold = snow_threshold
        self.blur_threshold = blur_threshold
        self.darkness_threshold = darkness_threshold
    
    def assess(self, image: np.ndarray, filename: str = "") -> QualityMetrics:
        """
        Assess image quality.
        
        Args:
            image: Input image
            filename: Image filename for logging
            
        Returns:
            QualityMetrics object with assessment results
        """
        try:
            # Calculate brightness
            brightness = self._calculate_brightness(image)
            
            # Calculate contrast
            contrast = self._calculate_contrast(image)
            
            # Calculate sharpness
            sharpness = self._calculate_sharpness(image)
            
            # Detect weather conditions
            has_fog = self._detect_fog(image)
            has_snow = self._detect_snow(image)
            
            # Quality checks
            is_blurry = sharpness < self.blur_threshold
            is_too_dark = brightness < self.darkness_threshold
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                brightness, contrast, sharpness, has_fog, has_snow, is_blurry, is_too_dark
            )
            
            return QualityMetrics(
                overall_score=overall_score,
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                has_fog=has_fog,
                has_snow=has_snow,
                is_blurry=is_blurry,
                is_too_dark=is_too_dark
            )
            
        except Exception as e:
            print(f"Error assessing quality for {filename}: {e}")
            return QualityMetrics(
                overall_score=0.0,
                brightness=0.0,
                contrast=0.0,
                sharpness=0.0,
                has_fog=True,
                has_snow=True,
                is_blurry=True,
                is_too_dark=True
            )
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate image brightness."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        except:
            return 0.0
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return float(np.std(gray))
        except:
            return 0.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except:
            return 0.0
    
    def _detect_fog(self, image: np.ndarray) -> bool:
        """Detect fog in image."""
        try:
            # Simple fog detection based on low contrast and high brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray) / 255.0
            brightness = np.mean(gray) / 255.0
            
            # Fog typically has low contrast but not too dark
            return contrast < self.fog_threshold and brightness > 0.3
        except:
            return False
    
    def _detect_snow(self, image: np.ndarray) -> bool:
        """Detect snow in image."""
        try:
            # Simple snow detection based on high brightness in all channels
            b, g, r = cv2.split(image)
            avg_brightness = (np.mean(r) + np.mean(g) + np.mean(b)) / (3 * 255.0)
            
            return avg_brightness > self.snow_threshold
        except:
            return False
    
    def _calculate_overall_score(self, brightness: float, contrast: float, sharpness: float,
                               has_fog: bool, has_snow: bool, is_blurry: bool, is_too_dark: bool) -> float:
        """Calculate overall quality score."""
        try:
            score = 1.0
            
            # Penalize poor conditions
            if has_fog:
                score *= 0.3
            if has_snow:
                score *= 0.4
            if is_blurry:
                score *= 0.2
            if is_too_dark:
                score *= 0.3
            
            # Normalize brightness (0-255 -> 0-1, optimal around 0.5)
            brightness_norm = min(brightness / 255.0, 1.0)
            brightness_score = 1.0 - abs(brightness_norm - 0.5) * 2
            
            # Normalize contrast (higher is generally better up to a point)
            contrast_score = min(contrast / 100.0, 1.0)
            
            # Normalize sharpness (higher is better up to a point)
            sharpness_score = min(sharpness / 1000.0, 1.0)
            
            # Combine scores
            score *= (brightness_score * 0.3 + contrast_score * 0.3 + sharpness_score * 0.4)
            
            return max(0.0, min(1.0, score))
            
        except:
            return 0.0
            self.cloud_threshold = 0.3
            self.fog_threshold = 0.1
            self.snow_threshold = 0.1
            self.enable_brightness_check = True
            self.enable_contrast_check = True
            self.enable_blur_check = True
            self.enable_weather_filtering = True
        else:
            self.min_brightness = config.min_brightness
            self.max_brightness = config.max_brightness
            self.min_contrast = config.min_contrast
            self.blur_threshold = config.blur_threshold
            self.cloud_threshold = config.cloud_threshold
            self.fog_threshold = config.fog_threshold
            self.snow_threshold = config.snow_threshold
            self.enable_brightness_check = config.enable_brightness_check
            self.enable_contrast_check = config.enable_contrast_check
            self.enable_blur_check = config.enable_blur_check
            self.enable_weather_filtering = config.enable_weather_filtering
    
    def detect_fog(self, image: np.ndarray) -> bool:
        """
        Detect if an image is foggy/hazy.
        
        Based on the algorithm from Image_Quality_Control_final.ipynb.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image is foggy, False otherwise
        """
        try:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and enhance fog features
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)

            # Compute the absolute difference between the original and blurred images
            diff = cv2.absdiff(gray, blurred)

            # Threshold the difference image to identify foggy regions
            _, fog_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

            # Additional features for fog detection
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            mean_blur = cv2.mean(blurred)[0]

            # Adjust the fog detection based on mean intensity, standard deviation, and mean blur
            is_foggy = (
                mean_intensity > 100 and
                std_intensity < 20 and
                mean_blur < 90 and
                np.sum(fog_mask == 255) / fog_mask.size > self.fog_threshold
            )

            return is_foggy
            
        except Exception as e:
            self.logger.error(f"Error in fog detection: {str(e)}")
            return False
    
    def detect_snow(self, image: np.ndarray) -> bool:
        """
        Detect if an image has snow coverage.
        
        Based on the algorithm from Image_Quality_Control_final.ipynb.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image has significant snow coverage, False otherwise
        """
        try:
            # Convert the image to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define a range for snow color in HSV
            lower_snow = np.array([0, 0, 180], dtype=np.uint8)
            upper_snow = np.array([180, 25, 255], dtype=np.uint8)

            # Create a mask to extract the lower vegetation part
            height, width, _ = image.shape
            lower_vegetation_mask = np.zeros((height, width), dtype=np.uint8)
            lower_vegetation_mask[int(height / 2):, :] = 255

            # Apply the mask to the HSV image
            hsv_lower_vegetation = cv2.bitwise_and(hsv, hsv, mask=lower_vegetation_mask)

            # Create a mask for snow in the lower vegetation part
            snow_mask = cv2.inRange(hsv_lower_vegetation, lower_snow, upper_snow)

            # Use morphology to enhance snow regions
            kernel = np.ones((5, 5), np.uint8)
            snow_mask = cv2.morphologyEx(snow_mask, cv2.MORPH_CLOSE, kernel)

            # Calculate the percentage of the lower vegetation covered by snow
            snowy_percentage = np.sum(snow_mask == 255) / snow_mask.size

            return snowy_percentage > self.snow_threshold
            
        except Exception as e:
            self.logger.error(f"Error in snow detection: {str(e)}")
            return False
    
    def detect_blur(self, image: np.ndarray) -> bool:
        """
        Detect if an image is blurred.
        
        Based on the algorithm from Image_Quality_Control_final.ipynb.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image is blurred, False otherwise
        """
        try:
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate the Laplacian of the image to detect edges
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            # Calculate the variance of the Laplacian
            laplacian_var = laplacian.var()

            # Apply a threshold to detect blurred images
            is_blurred = laplacian_var < self.blur_threshold

            return is_blurred
            
        except Exception as e:
            self.logger.error(f"Error in blur detection: {str(e)}")
            return False
    
    def detect_darkness(self, image: np.ndarray) -> bool:
        """
        Detect if an image is too dark (nighttime or low light).
        
        Based on the algorithm from Image_Quality_Control_final.ipynb.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image is too dark, False otherwise
        """
        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate the mean and standard deviation of pixel intensities
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)

            # Additional conditions for nighttime detection
            is_dark_mean = mean_intensity < 70
            is_low_contrast = std_intensity < 30

            return is_dark_mean and is_low_contrast
            
        except Exception as e:
            self.logger.error(f"Error in darkness detection: {str(e)}")
            return False
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate image brightness.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized brightness value (0-1)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.mean(gray) / 255.0
        except Exception as e:
            self.logger.error(f"Error calculating brightness: {str(e)}")
            return 0.0
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized contrast value
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.std(gray) / 255.0
        except Exception as e:
            self.logger.error(f"Error calculating contrast: {str(e)}")
            return 0.0
    
    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Blur score (higher = sharper)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()
        except Exception as e:
            self.logger.error(f"Error calculating blur score: {str(e)}")
            return 0.0
    
    def assess_image_quality(self, image: np.ndarray, filename: str = "") -> QualityMetrics:
        """
        Perform comprehensive quality assessment on an image.
        
        Args:
            image: Input image as numpy array
            filename: Optional filename for tracking
            
        Returns:
            QualityMetrics object with all quality assessments
        """
        try:
            # Calculate basic metrics
            brightness = self.calculate_brightness(image)
            contrast = self.calculate_contrast(image)
            blur_score = self.calculate_blur_score(image)
            
            # Detect quality issues
            is_foggy = self.detect_fog(image) if self.enable_weather_filtering else False
            is_snowy = self.detect_snow(image) if self.enable_weather_filtering else False
            is_blurred = self.detect_blur(image) if self.enable_blur_check else False
            is_dark = self.detect_darkness(image)
            
            # Calculate overall quality score (0-1, higher is better)
            quality_score = 0.0
            total_weight = 0.0
            
            if self.enable_brightness_check:
                # Brightness penalty (optimal range is 0.2-0.8)
                if 0.2 <= brightness <= 0.8:
                    brightness_score = 1.0
                else:
                    brightness_score = max(0, 1.0 - abs(brightness - 0.5) * 2)
                quality_score += brightness_score * 0.25
                total_weight += 0.25
            
            if self.enable_contrast_check:
                # Contrast score (higher is generally better, but cap at reasonable level)
                contrast_score = min(1.0, contrast * 5)  # Scale contrast appropriately
                quality_score += contrast_score * 0.25
                total_weight += 0.25
            
            if self.enable_blur_check:
                # Blur score (normalize and invert so higher is better)
                blur_score_norm = min(1.0, blur_score / 1000.0)  # Normalize blur score
                quality_score += blur_score_norm * 0.3
                total_weight += 0.3
            
            if self.enable_weather_filtering:
                # Weather penalties
                weather_score = 1.0
                if is_foggy:
                    weather_score -= 0.5
                if is_snowy:
                    weather_score -= 0.3
                if is_dark:
                    weather_score -= 0.4
                weather_score = max(0.0, weather_score)
                quality_score += weather_score * 0.2
                total_weight += 0.2
            
            # Normalize quality score
            if total_weight > 0:
                quality_score /= total_weight
            
            # Determine if image passes quality check
            passed_quality_check = True
            
            if self.enable_brightness_check:
                if brightness < self.min_brightness or brightness > self.max_brightness:
                    passed_quality_check = False
            
            if self.enable_contrast_check:
                if contrast < self.min_contrast:
                    passed_quality_check = False
            
            if self.enable_blur_check and is_blurred:
                passed_quality_check = False
            
            if self.enable_weather_filtering and (is_foggy or is_snowy or is_dark):
                passed_quality_check = False
            
            return QualityMetrics(
                filename=filename,
                brightness=brightness,
                contrast=contrast,
                blur_score=blur_score,
                is_foggy=is_foggy,
                is_snowy=is_snowy,
                is_blurred=is_blurred,
                is_dark=is_dark,
                quality_score=quality_score,
                passed_quality_check=passed_quality_check
            )
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {str(e)}")
            raise QualityControlError(f"Failed to assess image quality: {str(e)}")
    
    def filter_images_by_quality(
        self, 
        image_paths: List[Union[str, Path]], 
        min_quality_score: float = 0.5
    ) -> Tuple[List[str], List[QualityMetrics]]:
        """
        Filter a list of images based on quality criteria.
        
        Args:
            image_paths: List of image file paths
            min_quality_score: Minimum quality score to pass filtering
            
        Returns:
            Tuple of (passed_image_paths, all_quality_metrics)
        """
        passed_images = []
        all_metrics = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    self.logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Assess quality
                metrics = self.assess_image_quality(image, str(image_path))
                all_metrics.append(metrics)
                
                # Check if image passes quality filters
                if metrics.passed_quality_check and metrics.quality_score >= min_quality_score:
                    passed_images.append(str(image_path))
                    self.logger.debug(f"Image passed quality check: {image_path} (score: {metrics.quality_score:.3f})")
                else:
                    self.logger.debug(f"Image failed quality check: {image_path} (score: {metrics.quality_score:.3f})")
                
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {str(e)}")
        
        self.logger.info(f"Quality filtering: {len(passed_images)}/{len(image_paths)} images passed")
        
        return passed_images, all_metrics
    
    def generate_quality_report(self, quality_metrics: List[QualityMetrics]) -> Dict:
        """
        Generate a comprehensive quality assessment report.
        
        Args:
            quality_metrics: List of QualityMetrics objects
            
        Returns:
            Dictionary containing quality statistics
        """
        if not quality_metrics:
            return {}
        
        # Calculate statistics
        total_images = len(quality_metrics)
        passed_images = sum(1 for m in quality_metrics if m.passed_quality_check)
        
        # Calculate averages
        avg_brightness = np.mean([m.brightness for m in quality_metrics])
        avg_contrast = np.mean([m.contrast for m in quality_metrics])
        avg_blur_score = np.mean([m.blur_score for m in quality_metrics])
        avg_quality_score = np.mean([m.quality_score for m in quality_metrics])
        
        # Count quality issues
        foggy_count = sum(1 for m in quality_metrics if m.is_foggy)
        snowy_count = sum(1 for m in quality_metrics if m.is_snowy)
        blurred_count = sum(1 for m in quality_metrics if m.is_blurred)
        dark_count = sum(1 for m in quality_metrics if m.is_dark)
        
        report = {
            'total_images': total_images,
            'passed_images': passed_images,
            'pass_rate': passed_images / total_images if total_images > 0 else 0,
            'average_metrics': {
                'brightness': avg_brightness,
                'contrast': avg_contrast,
                'blur_score': avg_blur_score,
                'quality_score': avg_quality_score
            },
            'quality_issues': {
                'foggy': foggy_count,
                'snowy': snowy_count,
                'blurred': blurred_count,
                'dark': dark_count
            },
            'quality_issue_rates': {
                'foggy_rate': foggy_count / total_images if total_images > 0 else 0,
                'snowy_rate': snowy_count / total_images if total_images > 0 else 0,
                'blurred_rate': blurred_count / total_images if total_images > 0 else 0,
                'dark_rate': dark_count / total_images if total_images > 0 else 0
            }
        }
        
        return report
