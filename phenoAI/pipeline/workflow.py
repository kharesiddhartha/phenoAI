"""
Comprehensive workflow module for PhenoAI package.

This module provides end-to-end workflows for phenological analysis,
combining all components into easy-to-use pipelines.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import time

from ..core.logger import LoggerMixin
from ..core.config import Config
from ..core.exceptions import ProcessingError, ValidationError
from .batch_processor import BatchProcessor, ProcessingProgress
from ..analysis.vegetation_indices import VegetationIndices
from ..analysis.phenology import PhenologicalEvents
from ..analysis.time_series import TimeSeriesResults

class PhenologyWorkflow(LoggerMixin):
    """
    Complete phenology analysis workflow.
    
    Provides high-level workflows that combine image processing,
    quality control, vegetation index calculation, time series analysis,
    and phenological event extraction.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize phenology workflow.
        
        Args:
            config: Configuration object
        """
        self.config = config if config is not None else Config()
        self.batch_processor = BatchProcessor(self.config)
        
        # Results storage
        self.results = {}
    
    def run_full_analysis(
        self,
        image_directory: Union[str, Path],
        output_directory: Union[str, Path],
        date_pattern: Optional[str] = None,
        indices_to_analyze: Optional[List[str]] = None,
        generate_reports: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, any]:
        """
        Run complete phenological analysis workflow.
        
        Args:
            image_directory: Directory containing time series images
            output_directory: Directory to save all results
            date_pattern: Pattern for extracting dates from filenames
            indices_to_analyze: List of vegetation indices to analyze
            generate_reports: Whether to generate analysis reports
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            start_time = time.time()
            
            self.logger.info("Starting full phenological analysis workflow")
            
            # Create output directory
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Default indices to analyze
            if indices_to_analyze is None:
                indices_to_analyze = ['gcc', 'rcc', 'bcc', 'exg']
            
            # Step 1: Process all images and extract vegetation indices
            self.logger.info("Step 1: Processing images and extracting vegetation indices")
            vegetation_indices, quality_metrics = self.batch_processor.process_image_directory(
                image_directory=image_directory,
                output_directory=output_dir,
                date_pattern=date_pattern,
                apply_quality_control=True,
                progress_callback=progress_callback
            )
            
            if not vegetation_indices:
                raise ProcessingError("No vegetation indices extracted from images")
            
            self.logger.info(f"Extracted vegetation indices from {len(vegetation_indices)} images")
            
            # Step 2: Analyze time series for each vegetation index
            self.logger.info("Step 2: Analyzing time series for vegetation indices")
            time_series_results = {}
            phenological_events = {}
            
            for index_name in indices_to_analyze:
                try:
                    self.logger.info(f"Analyzing time series for {index_name}")
                    
                    # Create subdirectory for this index
                    index_output_dir = output_dir / index_name
                    index_output_dir.mkdir(exist_ok=True)
                    
                    # Analyze time series
                    ts_results, pheno_events = self.batch_processor.analyze_time_series(
                        vegetation_indices=vegetation_indices,
                        index_name=index_name,
                        output_directory=index_output_dir
                    )
                    
                    time_series_results[index_name] = ts_results
                    phenological_events[index_name] = pheno_events
                    
                    self.logger.info(f"Completed time series analysis for {index_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {index_name}: {str(e)}")
                    continue
            
            # Step 3: Generate comprehensive summary
            self.logger.info("Step 3: Generating analysis summary")
            summary = self._generate_analysis_summary(
                vegetation_indices=vegetation_indices,
                quality_metrics=quality_metrics,
                time_series_results=time_series_results,
                phenological_events=phenological_events,
                processing_time=time.time() - start_time
            )
            
            # Save summary
            with open(output_dir / 'analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Step 4: Generate reports if requested
            if generate_reports:
                self.logger.info("Step 4: Generating analysis reports")
                try:
                    self._generate_reports(
                        output_dir=output_dir,
                        vegetation_indices=vegetation_indices,
                        quality_metrics=quality_metrics,
                        time_series_results=time_series_results,
                        phenological_events=phenological_events
                    )
                except Exception as e:
                    self.logger.warning(f"Error generating reports: {str(e)}")
            
            # Store results
            self.results = {
                'vegetation_indices': vegetation_indices,
                'quality_metrics': quality_metrics,
                'time_series_results': time_series_results,
                'phenological_events': phenological_events,
                'summary': summary
            }
            
            self.logger.info(f"Workflow completed successfully in {time.time() - start_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in full analysis workflow: {str(e)}")
            raise ProcessingError(f"Workflow failed: {str(e)}")
    
    def run_quality_control_analysis(
        self,
        image_directory: Union[str, Path],
        output_directory: Union[str, Path],
        generate_quality_report: bool = True
    ) -> Dict[str, any]:
        """
        Run quality control analysis on images.
        
        Args:
            image_directory: Directory containing images
            output_directory: Directory to save results
            generate_quality_report: Whether to generate quality report
            
        Returns:
            Dictionary containing quality control results
        """
        try:
            self.logger.info("Starting quality control analysis")
            
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process images with quality control only
            _, quality_metrics = self.batch_processor.process_image_directory(
                image_directory=image_directory,
                output_directory=output_dir,
                apply_quality_control=True
            )
            
            if not quality_metrics:
                raise ProcessingError("No quality metrics generated")
            
            # Generate quality control report
            quality_report = self.batch_processor.quality_controller.generate_quality_report(quality_metrics)
            
            # Save quality report
            with open(output_dir / 'quality_control_report.json', 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            # Generate detailed quality report if requested
            if generate_quality_report:
                self._generate_quality_control_report(output_dir, quality_metrics, quality_report)
            
            return {
                'quality_metrics': quality_metrics,
                'quality_report': quality_report
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality control analysis: {str(e)}")
            raise ProcessingError(f"Quality control analysis failed: {str(e)}")
    
    def run_vegetation_index_extraction(
        self,
        image_directory: Union[str, Path],
        output_directory: Union[str, Path],
        date_pattern: Optional[str] = None,
        indices: Optional[List[str]] = None
    ) -> List[VegetationIndices]:
        """
        Extract vegetation indices from images without time series analysis.
        
        Args:
            image_directory: Directory containing images
            output_directory: Directory to save results
            date_pattern: Pattern for extracting dates
            indices: List of indices to calculate
            
        Returns:
            List of VegetationIndices objects
        """
        try:
            self.logger.info("Starting vegetation index extraction")
            
            # Temporarily modify config if specific indices requested
            original_indices = self.config.vegetation_indices.indices
            if indices:
                self.config.vegetation_indices.indices = indices
            
            try:
                vegetation_indices, _ = self.batch_processor.process_image_directory(
                    image_directory=image_directory,
                    output_directory=output_directory,
                    date_pattern=date_pattern,
                    apply_quality_control=False
                )
            finally:
                # Restore original config
                self.config.vegetation_indices.indices = original_indices
            
            return vegetation_indices
            
        except Exception as e:
            self.logger.error(f"Error in vegetation index extraction: {str(e)}")
            raise ProcessingError(f"Vegetation index extraction failed: {str(e)}")
    
    def _generate_analysis_summary(
        self,
        vegetation_indices: List[VegetationIndices],
        quality_metrics: List,
        time_series_results: Dict,
        phenological_events: Dict,
        processing_time: float
    ) -> Dict[str, any]:
        """Generate comprehensive analysis summary."""
        try:
            summary = {
                'analysis_info': {
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'processing_time_seconds': processing_time,
                    'phenoai_version': '2.0.0',
                    'config': self.config.to_dict()
                },
                'data_summary': {
                    'total_images_processed': len(vegetation_indices),
                    'total_images_with_quality_metrics': len(quality_metrics),
                    'date_range': {
                        'start': min(vi.date for vi in vegetation_indices) if vegetation_indices else None,
                        'end': max(vi.date for vi in vegetation_indices) if vegetation_indices else None
                    },
                    'vegetation_indices_analyzed': list(time_series_results.keys())
                },
                'quality_control_summary': {},
                'vegetation_indices_summary': {},
                'phenological_events_summary': {}
            }
            
            # Quality control summary
            if quality_metrics:
                passed_qc = sum(1 for qm in quality_metrics if qm.passed_quality_check)
                summary['quality_control_summary'] = {
                    'total_images': len(quality_metrics),
                    'passed_quality_control': passed_qc,
                    'quality_pass_rate': passed_qc / len(quality_metrics),
                    'average_quality_score': np.mean([qm.quality_score for qm in quality_metrics]),
                    'quality_issues': {
                        'foggy_images': sum(1 for qm in quality_metrics if qm.is_foggy),
                        'snowy_images': sum(1 for qm in quality_metrics if qm.is_snowy),
                        'blurred_images': sum(1 for qm in quality_metrics if qm.is_blurred),
                        'dark_images': sum(1 for qm in quality_metrics if qm.is_dark)
                    }
                }
            
            # Vegetation indices summary
            if vegetation_indices:
                summary['vegetation_indices_summary'] = {
                    'gcc': {
                        'mean': float(np.mean([vi.gcc for vi in vegetation_indices])),
                        'std': float(np.std([vi.gcc for vi in vegetation_indices])),
                        'min': float(np.min([vi.gcc for vi in vegetation_indices])),
                        'max': float(np.max([vi.gcc for vi in vegetation_indices]))
                    },
                    'rcc': {
                        'mean': float(np.mean([vi.rcc for vi in vegetation_indices])),
                        'std': float(np.std([vi.rcc for vi in vegetation_indices])),
                        'min': float(np.min([vi.rcc for vi in vegetation_indices])),
                        'max': float(np.max([vi.rcc for vi in vegetation_indices]))
                    },
                    'bcc': {
                        'mean': float(np.mean([vi.bcc for vi in vegetation_indices])),
                        'std': float(np.std([vi.bcc for vi in vegetation_indices])),
                        'min': float(np.min([vi.bcc for vi in vegetation_indices])),
                        'max': float(np.max([vi.bcc for vi in vegetation_indices]))
                    }
                }
            
            # Phenological events summary
            for index_name, events in phenological_events.items():
                summary['phenological_events_summary'][index_name] = {
                    'start_of_season': events.start_of_season,
                    'end_of_season': events.end_of_season,
                    'peak_of_season': events.peak_of_season,
                    'length_of_season': events.length_of_season,
                    'amplitude': events.amplitude,
                    'spring_slope': events.spring_slope,
                    'autumn_slope': events.autumn_slope,
                    'extraction_method': events.extraction_method,
                    'r_squared': events.r_squared
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {str(e)}")
            return {'error': str(e)}
    
    def _generate_reports(
        self,
        output_dir: Path,
        vegetation_indices: List[VegetationIndices],
        quality_metrics: List,
        time_series_results: Dict,
        phenological_events: Dict
    ):
        """Generate comprehensive analysis reports."""
        try:
            # Create reports directory
            reports_dir = output_dir / 'reports'
            reports_dir.mkdir(exist_ok=True)
            
            # Generate HTML report
            self._generate_html_report(
                reports_dir,
                vegetation_indices,
                quality_metrics,
                time_series_results,
                phenological_events
            )
            
            # Generate CSV summary tables
            self._generate_csv_summaries(
                reports_dir,
                vegetation_indices,
                quality_metrics,
                phenological_events
            )
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")
    
    def _generate_html_report(
        self,
        output_dir: Path,
        vegetation_indices: List[VegetationIndices],
        quality_metrics: List,
        time_series_results: Dict,
        phenological_events: Dict
    ):
        """Generate HTML analysis report."""
        try:
            html_content = self._create_html_report_content(
                vegetation_indices,
                quality_metrics,
                time_series_results,
                phenological_events
            )
            
            with open(output_dir / 'analysis_report.html', 'w') as f:
                f.write(html_content)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
    
    def _create_html_report_content(
        self,
        vegetation_indices: List[VegetationIndices],
        quality_metrics: List,
        time_series_results: Dict,
        phenological_events: Dict
    ) -> str:
        """Create HTML report content."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PhenoAI Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2E8B57; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary-box {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-label {{ font-weight: bold; color: #555; }}
                .metric-value {{ font-size: 1.2em; color: #2E8B57; }}
            </style>
        </head>
        <body>
            <h1>PhenoAI Phenological Analysis Report</h1>
            <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary-box">
                <h2>Analysis Summary</h2>
                <div class="metric">
                    <span class="metric-label">Total Images:</span>
                    <span class="metric-value">{len(vegetation_indices)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quality Passed:</span>
                    <span class="metric-value">{sum(1 for qm in quality_metrics if qm.passed_quality_check) if quality_metrics else 'N/A'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Indices Analyzed:</span>
                    <span class="metric-value">{len(time_series_results)}</span>
                </div>
            </div>
            
            <h2>Vegetation Indices Statistics</h2>
            <table>
                <tr>
                    <th>Index</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
        """
        
        # Add vegetation indices statistics
        if vegetation_indices:
            for index_name in ['gcc', 'rcc', 'bcc']:
                values = [getattr(vi, index_name) for vi in vegetation_indices]
                html += f"""
                <tr>
                    <td>{index_name.upper()}</td>
                    <td>{np.mean(values):.4f}</td>
                    <td>{np.std(values):.4f}</td>
                    <td>{np.min(values):.4f}</td>
                    <td>{np.max(values):.4f}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>Phenological Events</h2>
            <table>
                <tr>
                    <th>Index</th>
                    <th>Start of Season (DOY)</th>
                    <th>End of Season (DOY)</th>
                    <th>Peak of Season (DOY)</th>
                    <th>Season Length (days)</th>
                    <th>Amplitude</th>
                </tr>
        """
        
        # Add phenological events
        for index_name, events in phenological_events.items():
            html += f"""
            <tr>
                <td>{index_name.upper()}</td>
                <td>{events.start_of_season:.1f if events.start_of_season else 'N/A'}</td>
                <td>{events.end_of_season:.1f if events.end_of_season else 'N/A'}</td>
                <td>{events.peak_of_season:.1f if events.peak_of_season else 'N/A'}</td>
                <td>{events.length_of_season:.1f if events.length_of_season else 'N/A'}</td>
                <td>{events.amplitude:.4f}</td>
            </tr>
            """
        
        html += """
            </table>
            
            <h2>Quality Control Summary</h2>
        """
        
        if quality_metrics:
            passed_qc = sum(1 for qm in quality_metrics if qm.passed_quality_check)
            pass_rate = passed_qc / len(quality_metrics) * 100
            
            html += f"""
            <div class="summary-box">
                <p><strong>Quality Pass Rate:</strong> {pass_rate:.1f}% ({passed_qc}/{len(quality_metrics)} images)</p>
                <p><strong>Average Quality Score:</strong> {np.mean([qm.quality_score for qm in quality_metrics]):.3f}</p>
            </div>
            
            <table>
                <tr>
                    <th>Quality Issue</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td>Foggy Images</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_foggy)}</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_foggy) / len(quality_metrics) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Snowy Images</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_snowy)}</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_snowy) / len(quality_metrics) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Blurred Images</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_blurred)}</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_blurred) / len(quality_metrics) * 100:.1f}%</td>
                </tr>
                <tr>
                    <td>Dark Images</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_dark)}</td>
                    <td>{sum(1 for qm in quality_metrics if qm.is_dark) / len(quality_metrics) * 100:.1f}%</td>
                </tr>
            </table>
            """
        
        html += """
            <footer>
                <p><em>Generated by PhenoAI v2.0.0</em></p>
            </footer>
        </body>
        </html>
        """
        
        return html
    
    def _generate_csv_summaries(
        self,
        output_dir: Path,
        vegetation_indices: List[VegetationIndices],
        quality_metrics: List,
        phenological_events: Dict
    ):
        """Generate CSV summary tables."""
        try:
            # Phenological events summary
            if phenological_events:
                events_data = []
                for index_name, events in phenological_events.items():
                    events_data.append({
                        'index': index_name,
                        'start_of_season': events.start_of_season,
                        'end_of_season': events.end_of_season,
                        'peak_of_season': events.peak_of_season,
                        'length_of_season': events.length_of_season,
                        'amplitude': events.amplitude,
                        'spring_slope': events.spring_slope,
                        'autumn_slope': events.autumn_slope,
                        'max_spring_rate': events.max_spring_rate,
                        'max_autumn_rate': events.max_autumn_rate,
                        'r_squared': events.r_squared,
                        'extraction_method': events.extraction_method
                    })
                
                events_df = pd.DataFrame(events_data)
                events_df.to_csv(output_dir / 'phenological_events_summary.csv', index=False)
            
            # Quality metrics summary
            if quality_metrics:
                quality_data = []
                for qm in quality_metrics:
                    quality_data.append({
                        'filename': qm.filename,
                        'brightness': qm.brightness,
                        'contrast': qm.contrast,
                        'blur_score': qm.blur_score,
                        'quality_score': qm.quality_score,
                        'passed_quality_check': qm.passed_quality_check,
                        'is_foggy': qm.is_foggy,
                        'is_snowy': qm.is_snowy,
                        'is_blurred': qm.is_blurred,
                        'is_dark': qm.is_dark
                    })
                
                quality_df = pd.DataFrame(quality_data)
                quality_df.to_csv(output_dir / 'quality_metrics_summary.csv', index=False)
            
        except Exception as e:
            self.logger.error(f"Error generating CSV summaries: {str(e)}")
    
    def _generate_quality_control_report(self, output_dir: Path, quality_metrics: List, quality_report: Dict):
        """Generate detailed quality control report."""
        try:
            # Create quality control specific report
            qc_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PhenoAI Quality Control Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #DC143C; }}
                    .summary {{ background-color: #fff5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                    .good {{ color: #228B22; }}
                    .warning {{ color: #FF8C00; }}
                    .poor {{ color: #DC143C; }}
                </style>
            </head>
            <body>
                <h1>Quality Control Analysis Report</h1>
                <div class="summary">
                    <h2>Overall Quality Statistics</h2>
                    <div class="metric">Pass Rate: <strong>{quality_report.get('pass_rate', 0) * 100:.1f}%</strong></div>
                    <div class="metric">Average Quality Score: <strong>{quality_report.get('average_metrics', {}).get('quality_score', 0):.3f}</strong></div>
                    <div class="metric">Total Images: <strong>{quality_report.get('total_images', 0)}</strong></div>
                </div>
                
                <h2>Quality Issues Breakdown</h2>
                <ul>
                    <li>Foggy Images: {quality_report.get('quality_issues', {}).get('foggy', 0)} ({quality_report.get('quality_issue_rates', {}).get('foggy_rate', 0) * 100:.1f}%)</li>
                    <li>Snowy Images: {quality_report.get('quality_issues', {}).get('snowy', 0)} ({quality_report.get('quality_issue_rates', {}).get('snowy_rate', 0) * 100:.1f}%)</li>
                    <li>Blurred Images: {quality_report.get('quality_issues', {}).get('blurred', 0)} ({quality_report.get('quality_issue_rates', {}).get('blurred_rate', 0) * 100:.1f}%)</li>
                    <li>Dark Images: {quality_report.get('quality_issues', {}).get('dark', 0)} ({quality_report.get('quality_issue_rates', {}).get('dark_rate', 0) * 100:.1f}%)</li>
                </ul>
            </body>
            </html>
            """
            
            with open(output_dir / 'quality_control_report.html', 'w') as f:
                f.write(qc_html)
            
        except Exception as e:
            self.logger.error(f"Error generating quality control report: {str(e)}")
    
    def get_results(self) -> Dict[str, any]:
        """Get stored analysis results."""
        return self.results
    
    def save_config(self, output_path: Union[str, Path]):
        """Save current configuration to file."""
        self.config.save_to_file(str(output_path))
    
    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        self.config.load_from_file(str(config_path))
        # Recreate batch processor with new config
        self.batch_processor = BatchProcessor(self.config)
