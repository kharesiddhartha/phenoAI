"""
Batch processing module for PhenoAI package.

This module provides batch processing capabilities for analyzing
multiple images and time series datasets.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import json

from ..core.logger import LoggerMixin
from ..core.config import Config
from ..core.exceptions import ProcessingError, ValidationError
from ..preprocessing.quality_control import ImageQualityController, QualityMetrics
from ..analysis.vegetation_indices import VegetationIndexCalculator, VegetationIndices
from ..analysis.phenology import PhenologyAnalyzer, PhenologicalEvents
from ..analysis.time_series import TimeSeriesAnalyzer, TimeSeriesResults

@dataclass
class ProcessingProgress:
    """Data class to track processing progress."""
    total_files: int
    processed_files: int
    failed_files: int
    current_file: str
    start_time: float
    estimated_time_remaining: Optional[float] = None

class BatchProcessor(LoggerMixin):
    """
    Batch processor for analyzing multiple PhenoCam images and time series.
    
    Provides parallel processing capabilities and progress tracking for
    large-scale phenological analysis.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Configuration object
        """
        self.config = config if config is not None else Config()
        
        # Initialize components
        self.quality_controller = ImageQualityController(self.config.quality_control)
        self.vi_calculator = VegetationIndexCalculator(self.config.vegetation_indices)
        self.phenology_analyzer = PhenologyAnalyzer(self.config.phenology)
        self.ts_analyzer = TimeSeriesAnalyzer(self.config.phenology)
        
        # Processing state
        self.progress = None
        self.stop_processing = False
    
    def extract_date_from_filename(self, filename: str, date_pattern: str) -> Optional[str]:
        """
        Extract date from filename using pattern.
        
        Args:
            filename: Image filename
            date_pattern: Date pattern (e.g., "*yyyy_mm_dd*")
            
        Returns:
            Extracted date string or None if not found
        """
        try:
            import re
            
            # Convert pattern to regex
            # Replace yyyy, mm, dd with regex groups
            pattern = date_pattern.replace('*', '.*')
            pattern = pattern.replace('yyyy', r'(\d{4})')
            pattern = pattern.replace('mm', r'(\d{1,2})')
            pattern = pattern.replace('dd', r'(\d{1,2})')
            
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    year, month, day = groups[:3]
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting date from {filename}: {str(e)}")
            return None
    
    def process_single_image(
        self, 
        image_path: Union[str, Path],
        date_pattern: Optional[str] = None,
        apply_quality_control: bool = True
    ) -> Tuple[Optional[VegetationIndices], Optional[QualityMetrics]]:
        """
        Process a single image to extract vegetation indices.
        
        Args:
            image_path: Path to image file
            date_pattern: Pattern for extracting date from filename
            apply_quality_control: Whether to apply quality control
            
        Returns:
            Tuple of (VegetationIndices, QualityMetrics) or (None, None) if failed
        """
        try:
            image_path = Path(image_path)
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return None, None
            
            # Extract date from filename
            if date_pattern:
                date_str = self.extract_date_from_filename(image_path.name, date_pattern)
            else:
                date_str = image_path.stem  # Use filename without extension
            
            if not date_str:
                date_str = image_path.stem
            
            # Apply quality control
            quality_metrics = None
            if apply_quality_control:
                quality_metrics = self.quality_controller.assess_image_quality(
                    image, str(image_path)
                )
                
                if not quality_metrics.passed_quality_check:
                    self.logger.debug(f"Image failed quality control: {image_path}")
                    return None, quality_metrics
            
            # Calculate vegetation indices
            vegetation_indices = self.vi_calculator.calculate_all_indices(
                image, date_str, image_path.name
            )
            
            return vegetation_indices, quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None, None
    
    def process_image_directory(
        self,
        image_directory: Union[str, Path],
        output_directory: Union[str, Path],
        date_pattern: Optional[str] = None,
        file_pattern: str = "*.jpg",
        apply_quality_control: bool = True,
        save_intermediate: bool = False,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> Tuple[List[VegetationIndices], List[QualityMetrics]]:
        """
        Process all images in a directory.
        
        Args:
            image_directory: Directory containing images
            output_directory: Directory to save results
            date_pattern: Pattern for extracting dates from filenames
            file_pattern: Pattern for matching image files
            apply_quality_control: Whether to apply quality control
            save_intermediate: Whether to save intermediate results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (vegetation_indices_list, quality_metrics_list)
        """
        try:
            image_dir = Path(image_directory)
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all image files
            if file_pattern.startswith("*."):
                extension = file_pattern[2:]
                image_files = list(image_dir.glob(f"*.{extension}"))
                # Also try other common extensions
                for ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp']:
                    if ext != extension:
                        image_files.extend(image_dir.glob(f"*.{ext}"))
            else:
                image_files = list(image_dir.glob(file_pattern))
            
            if not image_files:
                raise ValidationError(f"No image files found matching pattern {file_pattern}")
            
            self.logger.info(f"Found {len(image_files)} images to process")
            
            # Initialize progress tracking
            self.progress = ProcessingProgress(
                total_files=len(image_files),
                processed_files=0,
                failed_files=0,
                current_file="",
                start_time=time.time()
            )
            
            vegetation_indices_list = []
            quality_metrics_list = []
            
            # Process images
            if self.config.processing.parallel_processing and len(image_files) > 1:
                # Parallel processing
                vegetation_indices_list, quality_metrics_list = self._process_images_parallel(
                    image_files, date_pattern, apply_quality_control, progress_callback
                )
            else:
                # Sequential processing
                vegetation_indices_list, quality_metrics_list = self._process_images_sequential(
                    image_files, date_pattern, apply_quality_control, progress_callback
                )
            
            # Save results
            if vegetation_indices_list:
                self._save_vegetation_indices(vegetation_indices_list, output_dir)
            
            if quality_metrics_list:
                self._save_quality_metrics(quality_metrics_list, output_dir)
            
            # Save processing summary
            self._save_processing_summary(output_dir, len(image_files), len(vegetation_indices_list))
            
            self.logger.info(f"Processed {len(vegetation_indices_list)} images successfully")
            
            return vegetation_indices_list, quality_metrics_list
            
        except Exception as e:
            self.logger.error(f"Error processing image directory: {str(e)}")
            raise ProcessingError(f"Failed to process image directory: {str(e)}")
    
    def _process_images_sequential(
        self,
        image_files: List[Path],
        date_pattern: Optional[str],
        apply_quality_control: bool,
        progress_callback: Optional[Callable[[ProcessingProgress], None]]
    ) -> Tuple[List[VegetationIndices], List[QualityMetrics]]:
        """Process images sequentially."""
        vegetation_indices_list = []
        quality_metrics_list = []
        
        for i, image_path in enumerate(image_files):
            if self.stop_processing:
                break
            
            self.progress.current_file = image_path.name
            self.progress.processed_files = i
            
            # Estimate remaining time
            if i > 0:
                elapsed_time = time.time() - self.progress.start_time
                avg_time_per_file = elapsed_time / i
                remaining_files = len(image_files) - i
                self.progress.estimated_time_remaining = avg_time_per_file * remaining_files
            
            # Call progress callback
            if progress_callback:
                progress_callback(self.progress)
            
            # Process image
            vegetation_indices, quality_metrics = self.process_single_image(
                image_path, date_pattern, apply_quality_control
            )
            
            if vegetation_indices:
                vegetation_indices_list.append(vegetation_indices)
            else:
                self.progress.failed_files += 1
            
            if quality_metrics:
                quality_metrics_list.append(quality_metrics)
        
        self.progress.processed_files = len(image_files)
        if progress_callback:
            progress_callback(self.progress)
        
        return vegetation_indices_list, quality_metrics_list
    
    def _process_images_parallel(
        self,
        image_files: List[Path],
        date_pattern: Optional[str],
        apply_quality_control: bool,
        progress_callback: Optional[Callable[[ProcessingProgress], None]]
    ) -> Tuple[List[VegetationIndices], List[QualityMetrics]]:
        """Process images in parallel."""
        vegetation_indices_list = []
        quality_metrics_list = []
        
        max_workers = min(self.config.processing.max_workers, len(image_files))
        
        def process_wrapper(image_path):
            return self.process_single_image(image_path, date_pattern, apply_quality_control)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_wrapper, image_path): image_path 
                for image_path in image_files
            }
            
            # Process completed tasks
            for i, future in enumerate(future_to_file):
                if self.stop_processing:
                    break
                
                try:
                    vegetation_indices, quality_metrics = future.result()
                    
                    if vegetation_indices:
                        vegetation_indices_list.append(vegetation_indices)
                    else:
                        self.progress.failed_files += 1
                    
                    if quality_metrics:
                        quality_metrics_list.append(quality_metrics)
                    
                    # Update progress
                    self.progress.processed_files = i + 1
                    self.progress.current_file = future_to_file[future].name
                    
                    if i > 0:
                        elapsed_time = time.time() - self.progress.start_time
                        avg_time_per_file = elapsed_time / (i + 1)
                        remaining_files = len(image_files) - (i + 1)
                        self.progress.estimated_time_remaining = avg_time_per_file * remaining_files
                    
                    if progress_callback:
                        progress_callback(self.progress)
                
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {str(e)}")
                    self.progress.failed_files += 1
        
        return vegetation_indices_list, quality_metrics_list
    
    def _save_vegetation_indices(self, vegetation_indices: List[VegetationIndices], output_dir: Path):
        """Save vegetation indices to file."""
        try:
            # Convert to DataFrame
            data = []
            for vi in vegetation_indices:
                row = {
                    'date': vi.date,
                    'filename': vi.filename,
                    'gcc': vi.gcc,
                    'rcc': vi.rcc,
                    'bcc': vi.bcc,
                    'exg': vi.exg,
                    'cive': vi.cive,
                    'vf': vi.vf,
                    'hue': vi.hue,
                    'saturation': vi.saturation,
                    'brightness': vi.brightness,
                    'gcc_std': vi.gcc_std,
                    'pixel_count': vi.pixel_count
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save in configured format
            output_format = self.config.processing.output_format
            if output_format == 'xlsx':
                df.to_excel(output_dir / 'vegetation_indices.xlsx', index=False)
            elif output_format == 'csv':
                df.to_csv(output_dir / 'vegetation_indices.csv', index=False)
            elif output_format == 'json':
                df.to_json(output_dir / 'vegetation_indices.json', orient='records', indent=2)
            
            self.logger.info(f"Saved vegetation indices to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving vegetation indices: {str(e)}")
    
    def _save_quality_metrics(self, quality_metrics: List[QualityMetrics], output_dir: Path):
        """Save quality metrics to file."""
        try:
            # Convert to DataFrame
            data = []
            for qm in quality_metrics:
                row = {
                    'filename': qm.filename,
                    'brightness': qm.brightness,
                    'contrast': qm.contrast,
                    'blur_score': qm.blur_score,
                    'is_foggy': qm.is_foggy,
                    'is_snowy': qm.is_snowy,
                    'is_blurred': qm.is_blurred,
                    'is_dark': qm.is_dark,
                    'quality_score': qm.quality_score,
                    'passed_quality_check': qm.passed_quality_check
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save in configured format
            output_format = self.config.processing.output_format
            if output_format == 'xlsx':
                df.to_excel(output_dir / 'quality_metrics.xlsx', index=False)
            elif output_format == 'csv':
                df.to_csv(output_dir / 'quality_metrics.csv', index=False)
            elif output_format == 'json':
                df.to_json(output_dir / 'quality_metrics.json', orient='records', indent=2)
            
            self.logger.info(f"Saved quality metrics to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving quality metrics: {str(e)}")
    
    def _save_processing_summary(self, output_dir: Path, total_files: int, successful_files: int):
        """Save processing summary."""
        try:
            summary = {
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': total_files - successful_files,
                'success_rate': successful_files / total_files if total_files > 0 else 0,
                'processing_time_seconds': time.time() - self.progress.start_time if self.progress else 0,
                'config': self.config.to_dict()
            }
            
            with open(output_dir / 'processing_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving processing summary: {str(e)}")
    
    def analyze_time_series(
        self,
        vegetation_indices: List[VegetationIndices],
        index_name: str = 'gcc',
        output_directory: Optional[Union[str, Path]] = None
    ) -> Tuple[TimeSeriesResults, PhenologicalEvents]:
        """
        Analyze time series of vegetation indices.
        
        Args:
            vegetation_indices: List of VegetationIndices objects
            index_name: Name of vegetation index to analyze
            output_directory: Optional directory to save results
            
        Returns:
            Tuple of (TimeSeriesResults, PhenologicalEvents)
        """
        try:
            # Sort by date
            sorted_indices = sorted(vegetation_indices, key=lambda x: x.date)
            
            # Perform time series analysis
            ts_results, pheno_events = self.phenology_analyzer.analyze_vegetation_indices_time_series(
                sorted_indices, index_name
            )
            
            # Save results if output directory provided
            if output_directory:
                output_dir = Path(output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save time series results
                ts_df = pd.DataFrame({
                    'date': ts_results.dates,
                    'original_values': ts_results.original_values,
                    'smoothed_values': ts_results.smoothed_values,
                    'trend': ts_results.trend,
                    'seasonal': ts_results.seasonal,
                    'residual': ts_results.residual,
                    'outliers': ts_results.outliers
                })
                
                if ts_results.confidence_intervals:
                    ts_df['lower_ci'] = ts_results.confidence_intervals[0]
                    ts_df['upper_ci'] = ts_results.confidence_intervals[1]
                
                ts_df.to_csv(output_dir / f'{index_name}_time_series.csv', index=False)
                
                # Save phenological events
                pheno_dict = {
                    'min_value': pheno_events.min_value,
                    'max_value': pheno_events.max_value,
                    'amplitude': pheno_events.amplitude,
                    'start_of_season': pheno_events.start_of_season,
                    'end_of_season': pheno_events.end_of_season,
                    'peak_of_season': pheno_events.peak_of_season,
                    'length_of_season': pheno_events.length_of_season,
                    'spring_slope': pheno_events.spring_slope,
                    'autumn_slope': pheno_events.autumn_slope,
                    'max_spring_rate': pheno_events.max_spring_rate,
                    'max_autumn_rate': pheno_events.max_autumn_rate,
                    'extraction_method': pheno_events.extraction_method,
                    'r_squared': pheno_events.r_squared,
                    'rmse': pheno_events.rmse
                }
                
                with open(output_dir / f'{index_name}_phenological_events.json', 'w') as f:
                    json.dump(pheno_dict, f, indent=2)
            
            return ts_results, pheno_events
            
        except Exception as e:
            self.logger.error(f"Error analyzing time series: {str(e)}")
            raise ProcessingError(f"Failed to analyze time series: {str(e)}")
    
    def stop(self):
        """Stop batch processing."""
        self.stop_processing = True
        self.logger.info("Stopping batch processing...")
    
    def get_progress(self) -> Optional[ProcessingProgress]:
        """Get current processing progress."""
        return self.progress
