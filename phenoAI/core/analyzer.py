"""
PhenoAI Core Analyzer Module

Main analysis engine providing comprehensive phenological analysis
including quality control, vegetation indices, time series analysis,
and curve fitting.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import simplified modules
from phenoAI.analysis.phenology import PhenologyAnalyzer

class PhenoAIAnalyzer:
    """
    Main PhenoAI analysis engine with comprehensive workflow capabilities.
    
    Provides quality control, vegetation indices calculation, time series analysis,
    phenological parameter extraction, and export functionality.
    """
    
    def __init__(self, config=None):
        """
        Initialize PhenoAI analyzer.
        
        Args:
            config: Configuration object (optional, uses defaults if None)
        """
        self.config = config
        self.phenology_analyzer = PhenologyAnalyzer()
        
    def _calculate_basic_vegetation_indices(self, image_path: str) -> Dict[str, float]:
        """Calculate basic vegetation indices using simple RGB analysis"""
        try:
            import cv2
            
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Could not load image: {image_path}'}
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate basic indices
            r = np.mean(image_rgb[:, :, 0].astype(float))
            g = np.mean(image_rgb[:, :, 1].astype(float))
            b = np.mean(image_rgb[:, :, 2].astype(float))
            
            total = r + g + b
            if total == 0:
                return {'error': 'Invalid image - all pixels are black'}
            
            gcc = g / total
            rcc = r / total
            bcc = b / total
            
            # ExG index
            exg = 2 * g - r - b
            exg_norm = (exg - exg.min()) / (exg.max() - exg.min()) if exg.max() != exg.min() else 0
            
            # CIVE index  
            cive = 0.441 * r - 0.881 * g + 0.385 * b + 18.78745
            cive_norm = (cive - cive.min()) / (cive.max() - cive.min()) if cive.max() != cive.min() else 0
            
            return {
                'gcc': float(gcc),
                'rcc': float(rcc),
                'bcc': float(bcc),
                'exg': float(np.mean(exg_norm)),
                'cive': float(np.mean(cive_norm)),
                'r_mean': float(r),
                'g_mean': float(g),
                'b_mean': float(b)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _basic_quality_assessment(self, image_path: str) -> Dict[str, Any]:
        """Perform basic image quality assessment"""
        try:
            import cv2
            
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Could not load image: {image_path}'}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic quality metrics
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Simple blur detection using Laplacian variance
            blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Simple fog/haze detection using brightness uniformity
            fog_metric = float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0
            
            # Quality scoring
            quality_score = 1.0
            
            if brightness < 50 or brightness > 200:
                quality_score -= 0.2  # Poor brightness
            
            if contrast < 20:
                quality_score -= 0.3  # Poor contrast
                
            if blur_metric < 100:
                quality_score -= 0.3  # Blurry image
                
            if fog_metric < 0.3:
                quality_score -= 0.2  # Possible fog/haze
            
            quality_score = max(0.0, quality_score)
            
            return {
                'quality_score': quality_score,
                'brightness': brightness,
                'contrast': contrast,
                'blur_metric': float(blur_metric),
                'fog_metric': fog_metric,
                'acceptable': quality_score >= 0.5
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename using common patterns"""
        import re
        from datetime import datetime
        
        # Common date patterns in filenames
        patterns = [
            r'(\d{4})[_-]?(\d{2})[_-]?(\d{2})',  # YYYY-MM-DD or YYYY_MM_DD
            r'(\d{2})[_-]?(\d{2})[_-]?(\d{4})',  # MM-DD-YYYY or MM_DD_YYYY
            r'(\d{4})(\d{2})(\d{2})',            # YYYYMMDD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    groups = match.groups()
                    if len(groups[0]) == 4:  # Year first
                        year, month, day = groups
                    else:  # Month first
                        month, day, year = groups
                    
                    # Validate date
                    date_obj = datetime(int(year), int(month), int(day))
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    continue
        
        # If no date pattern found, return current date
        return datetime.now().strftime('%Y-%m-%d')
    
    def analyze_single_image(self, image_path: str, date: str = None) -> Dict[str, Any]:
        """
        Analyze a single image for vegetation indices and quality.
        
        Args:
            image_path: Path to the image file
            date: Date string (if None, extracted from filename)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not os.path.exists(image_path):
                return {'error': f'Image file not found: {image_path}'}
            
            # Extract date if not provided
            if date is None:
                filename = os.path.basename(image_path)
                date = self._extract_date_from_filename(filename)
            
            # Quality assessment
            quality_results = self._basic_quality_assessment(image_path)
            
            # Vegetation indices calculation
            vegetation_results = self._calculate_basic_vegetation_indices(image_path)
            
            return {
                'image_path': image_path,
                'date': date,
                'quality_assessment': quality_results,
                'vegetation_indices': vegetation_results,
                'success': 'error' not in quality_results and 'error' not in vegetation_results
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_image_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Analyze all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            List of analysis results for each image
        """
        try:
            if not os.path.exists(folder_path):
                return [{'error': f'Folder not found: {folder_path}'}]
            
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            image_files = []
            
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(folder_path, file))
            
            if not image_files:
                return [{'error': 'No image files found in folder'}]
            
            # Sort by filename for chronological order
            image_files.sort()
            
            # Analyze each image
            results = []
            for image_path in image_files:
                result = self.analyze_single_image(image_path)
                results.append(result)
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def create_time_series_dataframe(self, analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame from analysis results for time series analysis.
        
        Args:
            analysis_results: List of image analysis results
            
        Returns:
            DataFrame with time series data
        """
        try:
            data = []
            
            for result in analysis_results:
                if result.get('success', False):
                    vi_data = result.get('vegetation_indices', {})
                    qa_data = result.get('quality_assessment', {})
                    
                    if 'error' not in vi_data and 'error' not in qa_data:
                        row = {
                            'date': result.get('date'),
                            'image_path': result.get('image_path'),
                            'gcc': vi_data.get('gcc', 0),
                            'rcc': vi_data.get('rcc', 0),
                            'bcc': vi_data.get('bcc', 0),
                            'exg': vi_data.get('exg', 0),
                            'cive': vi_data.get('cive', 0),
                            'quality_score': qa_data.get('quality_score', 0),
                            'brightness': qa_data.get('brightness', 0),
                            'contrast': qa_data.get('contrast', 0)
                        }
                        data.append(row)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def run_complete_analysis(self, input_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Run complete phenological analysis workflow.
        
        Args:
            input_path: Path to image or folder of images
            output_dir: Directory for output files (optional)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            print("Starting PhenoAI complete analysis...")
            
            # Set up output directory
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(input_path), 'phenoai_results')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Analyze images
            print("Step 1: Analyzing images...")
            if os.path.isfile(input_path):
                analysis_results = [self.analyze_single_image(input_path)]
            else:
                analysis_results = self.analyze_image_folder(input_path)
            
            if not analysis_results or all('error' in result for result in analysis_results):
                return {'error': 'No successful image analysis results'}
            
            # Step 2: Create time series DataFrame
            print("Step 2: Creating time series...")
            df = self.create_time_series_dataframe(analysis_results)
            
            if df.empty:
                return {'error': 'No valid time series data created'}
            
            # Step 3: Phenological analysis
            print("Step 3: Extracting phenological parameters...")
            phenology_results = self.phenology_analyzer.extract_parameters(df)
            
            # Step 4: Create visualizations
            print("Step 4: Creating visualizations...")
            try:
                import matplotlib.pyplot as plt
                
                # Time series plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # GCC time series
                axes[0, 0].plot(df['date'], df['gcc'], 'g-', linewidth=2)
                axes[0, 0].set_title('Green Chromatic Coordinate (GCC)')
                axes[0, 0].set_ylabel('GCC')
                axes[0, 0].grid(True, alpha=0.3)
                
                # RCC time series
                axes[0, 1].plot(df['date'], df['rcc'], 'r-', linewidth=2)
                axes[0, 1].set_title('Red Chromatic Coordinate (RCC)')
                axes[0, 1].set_ylabel('RCC')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Quality scores
                axes[1, 0].plot(df['date'], df['quality_score'], 'b-', linewidth=2)
                axes[1, 0].set_title('Image Quality Score')
                axes[1, 0].set_ylabel('Quality Score')
                axes[1, 0].grid(True, alpha=0.3)
                
                # ExG index
                axes[1, 1].plot(df['date'], df['exg'], 'orange', linewidth=2)
                axes[1, 1].set_title('Excess Green Index (ExG)')
                axes[1, 1].set_ylabel('ExG')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(output_dir, 'phenoai_time_series.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Plot saved: {plot_path}")
                
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")
                plot_path = None
            
            # Step 5: Export data to Excel
            print("Step 5: Exporting results...")
            try:
                excel_path = os.path.join(output_dir, 'phenoai_analysis_results.xlsx')
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    # Time series data
                    df.to_excel(writer, sheet_name='Time_Series_Data', index=False)
                    
                    # Phenological parameters
                    if phenology_results.get('success', False):
                        params_df = pd.DataFrame([phenology_results.get('gcc_parameters', {})])
                        params_df.to_excel(writer, sheet_name='Phenological_Parameters', index=False)
                        
                        # Statistical parameters
                        stats_df = pd.DataFrame([phenology_results.get('statistical_parameters', {})])
                        stats_df.to_excel(writer, sheet_name='Statistical_Summary', index=False)
                
                print(f"Excel file saved: {excel_path}")
                
            except Exception as e:
                print(f"Warning: Could not create Excel file: {e}")
                excel_path = None
            
            # Compile final results
            results = {
                'success': True,
                'input_path': input_path,
                'output_directory': output_dir,
                'total_images_processed': len(analysis_results),
                'successful_analyses': sum(1 for r in analysis_results if r.get('success', False)),
                'time_series_data': df.to_dict('records'),
                'phenology_analysis': phenology_results,
                'output_files': {
                    'excel_file': excel_path,
                    'plot_file': plot_path
                }
            }
            
            print("PhenoAI analysis completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error in complete analysis: {e}")
            return {'error': str(e)}
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a text summary report of the analysis.
        
        Args:
            analysis_results: Results from run_complete_analysis
            
        Returns:
            Formatted summary report string
        """
        try:
            if not analysis_results.get('success', False):
                return f"Analysis failed: {analysis_results.get('error', 'Unknown error')}"
            
            report = []
            report.append("=" * 60)
            report.append("PhenoAI Analysis Summary Report")
            report.append("=" * 60)
            report.append("")
            
            # Basic info
            report.append(f"Input Path: {analysis_results.get('input_path', 'N/A')}")
            report.append(f"Output Directory: {analysis_results.get('output_directory', 'N/A')}")
            report.append(f"Total Images Processed: {analysis_results.get('total_images_processed', 0)}")
            report.append(f"Successful Analyses: {analysis_results.get('successful_analyses', 0)}")
            report.append("")
            
            # Phenological parameters
            pheno_results = analysis_results.get('phenology_analysis', {})
            if pheno_results.get('success', False):
                gcc_params = pheno_results.get('gcc_parameters', {})
                
                report.append("Phenological Parameters:")
                report.append("-" * 25)
                report.append(f"  Minimum GCC: {gcc_params.get('min_gcc', 'N/A'):.4f}")
                report.append(f"  Maximum GCC: {gcc_params.get('max_gcc', 'N/A'):.4f}")
                report.append(f"  GCC Amplitude: {gcc_params.get('amplitude', 'N/A'):.4f}")
                report.append(f"  Peak Day: {gcc_params.get('peak_doy', 'N/A')}")
                report.append(f"  Start of Season: Day {gcc_params.get('start_of_season', 'N/A')}")
                report.append(f"  End of Season: Day {gcc_params.get('end_of_season', 'N/A')}")
                report.append(f"  Season Length: {gcc_params.get('season_length', 'N/A')} days")
                report.append(f"  Spring Slope: {gcc_params.get('spring_slope', 'N/A'):.6f}")
                report.append(f"  Autumn Slope: {gcc_params.get('autumn_slope', 'N/A'):.6f}")
                report.append("")
            
            # Output files
            output_files = analysis_results.get('output_files', {})
            report.append("Output Files Generated:")
            report.append("-" * 23)
            if output_files.get('excel_file'):
                report.append(f"  Excel Data: {os.path.basename(output_files['excel_file'])}")
            if output_files.get('plot_file'):
                report.append(f"  Time Series Plot: {os.path.basename(output_files['plot_file'])}")
            
            report.append("")
            report.append("Analysis completed successfully!")
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error generating summary report: {e}"
    
    def __init__(self, config: Optional[PhenoAIConfig] = None):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration object or None for default
        """
        self.config = config or PhenoAIConfig()
        self.vegetation_calc = VegetationIndexCalculator()
        self.quality_controller = ImageQualityController()
        
        self.results = []
    
    def run_complete_analysis(self, input_dir: str, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run complete phenological analysis with all features.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for results  
            config: Analysis configuration options
            
        Returns:
            Comprehensive analysis results
        """
        analysis_config = config or {}
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 Starting comprehensive phenological analysis...")
        print(f"📂 Input: {input_dir}")
        print(f"📁 Output: {output_dir}")
        
        results = {
            'success': False,
            'summary': {},
            'files_generated': [],
            'errors': []
        }
        
        try:
            # 1. Get image files
            image_files = get_image_files(input_dir)
            if not image_files:
                results['errors'].append('No image files found')
                return results
            
            print(f"📸 Found {len(image_files)} images to process")
            
            # 2. Process images
            processed_results = []
            quality_results = []
            
            for i, image_file in enumerate(image_files):
                filename = os.path.basename(image_file)
                print(f"Processing {i+1}/{len(image_files)}: {filename}")
                
                try:
                    # Load image
                    image = load_image(image_file)
                    if image is None:
                        continue
                    
                    # Extract date
                    date_pattern = analysis_config.get('date_pattern', '*yyyy-mm-dd*')
                    date_str = extract_date(filename, date_pattern)
                    
                    result = {
                        'filename': filename,
                        'date': date_str,
                        'image_path': image_file
                    }
                    
                    # Quality control
                    if analysis_config.get('include_quality_control', True):
                        quality = self.quality_controller.assess_quality(image, filename)
                        
                        result.update({
                            'quality_score': quality.overall_score,
                            'has_fog': quality.has_fog,
                            'has_snow': quality.has_snow,
                            'is_blurry': quality.is_blurry,
                            'is_too_dark': quality.is_too_dark,
                            'brightness': quality.brightness,
                            'contrast': quality.contrast,
                            'sharpness': quality.sharpness
                        })
                        
                        quality_results.append(result.copy())
                    
                    # Vegetation indices
                    if analysis_config.get('include_vegetation_indices', True):
                        indices = self.vegetation_calc.calculate_all(image)
                        result.update(indices)
                    
                    processed_results.append(result)
                    
                except Exception as e:
                    print(f"  ❌ Error processing {filename}: {str(e)}")
                    results['errors'].append(f"Error processing {filename}: {str(e)}")
            
            if not processed_results:
                results['errors'].append('No images successfully processed')
                return results
            
            # 3. Create DataFrame
            df = pd.DataFrame(processed_results)
            
            # Sort by date if possible
            if 'date' in df.columns and df['date'].notna().any():
                df = df.sort_values('date')
            
            # 4. Time series analysis
            if analysis_config.get('include_time_series', True):
                print("📈 Performing time series analysis...")
                timeseries_results = self._analyze_timeseries(df, output_dir)
                results['timeseries'] = timeseries_results
            
            # 5. Phenological parameter extraction
            if analysis_config.get('include_phenology', True):
                print("🌱 Extracting phenological parameters...")
                phenology_results = self._extract_phenology_parameters(df, output_dir)
                results['phenology'] = phenology_results
            
            # 6. Generate plots and visualizations
            if analysis_config.get('generate_plots', True):
                print("📊 Generating visualizations...")
                plot_files = self._generate_visualizations(df, output_dir)
                results['files_generated'].extend(plot_files)
            
            # 7. Export to Excel
            if analysis_config.get('export_excel', True):
                print("📋 Exporting to Excel...")
                excel_files = self._export_to_excel(df, output_dir)
                results['files_generated'].extend(excel_files)
            
            # 8. Generate summary report
            print("📄 Creating summary report...")
            summary_file = self._generate_summary_report(df, output_dir, analysis_config)
            results['files_generated'].append(summary_file)
            
            # 9. Save configuration used
            config_file = os.path.join(output_dir, 'analysis_config.json')
            with open(config_file, 'w') as f:
                json.dump(analysis_config, f, indent=2)
            results['files_generated'].append(config_file)
            
            # Create summary
            results['summary'] = {
                'total_images': len(image_files),
                'processed_images': len(processed_results),
                'high_quality_images': len([r for r in processed_results if r.get('quality_score', 0) > 0.7]),
                'avg_gcc': float(df['gcc'].mean()) if 'gcc' in df.columns else None,
                'start_of_season': self._get_phenology_metric(df, 'sos'),
                'end_of_season': self._get_phenology_metric(df, 'eos'),
                'processing_date': datetime.now().isoformat()
            }
            
            results['success'] = True
            print("✅ Analysis completed successfully!")
            
        except Exception as e:
            results['errors'].append(f"Analysis failed: {str(e)}")
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _analyze_timeseries(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """Analyze time series data"""
        try:
            from ..analysis.time_series import TimeSeriesAnalyzer
            
            analyzer = TimeSeriesAnalyzer()
            results = analyzer.analyze(df)
            
            # Save time series results
            ts_file = os.path.join(output_dir, 'timeseries_analysis.csv')
            if 'timeseries_data' in results:
                pd.DataFrame(results['timeseries_data']).to_csv(ts_file, index=False)
            
            return results
            
        except Exception as e:
            print(f"Warning: Time series analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _extract_phenology_parameters(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """Extract phenological parameters"""
        try:
            from phenoAI.analysis.phenology_clean import PhenologyAnalyzer
            
            analyzer = PhenologyAnalyzer()
            results = analyzer.extract_parameters(df)
            
            # Save phenology results
            pheno_file = os.path.join(output_dir, 'phenology_parameters.json')
            with open(pheno_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            print(f"Warning: Phenology extraction failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_visualizations(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """Generate plots and visualizations"""
        plot_files = []
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. GCC time series plot
            if 'gcc' in df.columns and 'date' in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Convert dates for plotting
                dates = pd.to_datetime(df['date'], errors='coerce')
                valid_data = df[dates.notna()]
                
                if len(valid_data) > 0:
                    plt.plot(pd.to_datetime(valid_data['date']), valid_data['gcc'], 
                            'o-', linewidth=2, markersize=4, alpha=0.8)
                    plt.title('Green Chromatic Coordinate (GCC) Time Series', fontsize=14, fontweight='bold')
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('GCC', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    gcc_plot = os.path.join(plots_dir, 'gcc_timeseries.png')
                    plt.savefig(gcc_plot, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(gcc_plot)
            
            # 2. Vegetation indices comparison
            indices = ['gcc', 'rcc', 'bcc', 'exg', 'cive']
            available_indices = [idx for idx in indices if idx in df.columns]
            
            if len(available_indices) >= 2:
                plt.figure(figsize=(14, 8))
                
                for i, idx in enumerate(available_indices):
                    plt.subplot(2, 3, i+1)
                    plt.hist(df[idx].dropna(), bins=30, alpha=0.7, color=f'C{i}')
                    plt.title(f'{idx.upper()} Distribution')
                    plt.xlabel(idx.upper())
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                indices_plot = os.path.join(plots_dir, 'vegetation_indices_distribution.png')
                plt.savefig(indices_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(indices_plot)
            
            # 3. Quality assessment plot
            if 'quality_score' in df.columns:
                plt.figure(figsize=(10, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(df['quality_score'].dropna(), bins=20, alpha=0.7, color='skyblue')
                plt.title('Image Quality Score Distribution')
                plt.xlabel('Quality Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                quality_categories = ['High (>0.7)', 'Medium (0.4-0.7)', 'Low (<0.4)']
                quality_counts = [
                    (df['quality_score'] > 0.7).sum(),
                    ((df['quality_score'] >= 0.4) & (df['quality_score'] <= 0.7)).sum(),
                    (df['quality_score'] < 0.4).sum()
                ]
                
                plt.pie(quality_counts, labels=quality_categories, autopct='%1.1f%%', startangle=90)
                plt.title('Image Quality Categories')
                
                plt.tight_layout()
                quality_plot = os.path.join(plots_dir, 'quality_assessment.png')
                plt.savefig(quality_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(quality_plot)
            
            # 4. Correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f')
                plt.title('Correlation Matrix of Vegetation Indices and Quality Metrics')
                plt.tight_layout()
                
                corr_plot = os.path.join(plots_dir, 'correlation_matrix.png')
                plt.savefig(corr_plot, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(corr_plot)
            
        except Exception as e:
            print(f"Warning: Visualization generation failed: {str(e)}")
        
        return plot_files
    
    def _export_to_excel(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        """Export results to Excel with multiple sheets"""
        excel_files = []
        
        try:
            excel_file = os.path.join(output_dir, 'phenoai_complete_analysis.xlsx')
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Main results
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Summary statistics
                summary_stats = df.describe()
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
                
                # Quality analysis
                if 'quality_score' in df.columns:
                    quality_df = df[['filename', 'quality_score', 'has_fog', 'has_snow', 'is_blurry', 'is_too_dark']].copy()
                    quality_df.to_excel(writer, sheet_name='Quality_Analysis', index=False)
                
                # Vegetation indices only
                vegetation_cols = ['filename', 'date'] + [col for col in df.columns if col in ['gcc', 'rcc', 'bcc', 'exg', 'cive']]
                if len(vegetation_cols) > 2:
                    df[vegetation_cols].to_excel(writer, sheet_name='Vegetation_Indices', index=False)
            
            excel_files.append(excel_file)
            
        except Exception as e:
            print(f"Warning: Excel export failed: {str(e)}")
        
        return excel_files
    
    def _generate_summary_report(self, df: pd.DataFrame, output_dir: str, config: Dict) -> str:
        """Generate comprehensive summary report"""
        report_file = os.path.join(output_dir, 'analysis_report.txt')
        
        try:
            with open(report_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("PhenoAI - Comprehensive Analysis Report\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Images Processed: {len(df)}\n\n")
                
                # Date range
                if 'date' in df.columns and df['date'].notna().any():
                    f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n\n")
                
                # Vegetation indices statistics
                f.write("VEGETATION INDICES STATISTICS\n")
                f.write("-" * 40 + "\n")
                
                for idx in ['gcc', 'rcc', 'bcc', 'exg', 'cive']:
                    if idx in df.columns:
                        values = df[idx].dropna()
                        if len(values) > 0:
                            f.write(f"{idx.upper()}:\n")
                            f.write(f"  Mean: {values.mean():.4f}\n")
                            f.write(f"  Std:  {values.std():.4f}\n")
                            f.write(f"  Min:  {values.min():.4f}\n")
                            f.write(f"  Max:  {values.max():.4f}\n\n")
                
                # Quality statistics
                if 'quality_score' in df.columns:
                    f.write("IMAGE QUALITY STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    
                    quality_scores = df['quality_score'].dropna()
                    f.write(f"Mean Quality Score: {quality_scores.mean():.4f}\n")
                    f.write(f"High Quality Images (>0.7): {(quality_scores > 0.7).sum()}\n")
                    f.write(f"Medium Quality Images (0.4-0.7): {((quality_scores >= 0.4) & (quality_scores <= 0.7)).sum()}\n")
                    f.write(f"Low Quality Images (<0.4): {(quality_scores < 0.4).sum()}\n")
                    
                    if 'has_fog' in df.columns:
                        f.write(f"Images with Fog: {df['has_fog'].sum()}\n")
                    if 'has_snow' in df.columns:
                        f.write(f"Images with Snow: {df['has_snow'].sum()}\n")
                    if 'is_blurry' in df.columns:
                        f.write(f"Blurry Images: {df['is_blurry'].sum()}\n")
                    if 'is_too_dark' in df.columns:
                        f.write(f"Dark Images: {df['is_too_dark'].sum()}\n")
                    
                    f.write("\n")
                
                # Configuration used
                f.write("ANALYSIS CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Report generated by PhenoAI v1.0.0\n")
        
        except Exception as e:
            print(f"Warning: Report generation failed: {str(e)}")
        
        return report_file
    
    def _get_phenology_metric(self, df: pd.DataFrame, metric: str) -> Any:
        """Get phenological metric from DataFrame"""
        try:
            if metric in df.columns:
                return float(df[metric].dropna().iloc[0]) if len(df[metric].dropna()) > 0 else None
        except:
            pass
        return None

    def process_directory(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path (overrides config)
            output_dir: Output directory path (overrides config)
            
        Returns:
            Dictionary with processing results
        """
        try:
            input_dir = input_dir or self.config.input_directory
            output_dir = output_dir or self.config.output_directory
            
            print(f"Processing images from: {input_dir}")
            
            # Get image files
            image_files = get_image_files(input_dir)
            
            if not image_files:
                return {
                    'success': False,
                    'message': f'No image files found in {input_dir}',
                    'processed_count': 0
                }
            
            print(f"Found {len(image_files)} images")
            
            # Process images
            results = self._process_sequential(image_files)
            
            # Filter successful results
            successful_results = [r for r in results if r is not None]
            
            print(f"Successfully processed {len(successful_results)}/{len(image_files)} images")
            
            # Save results
            if successful_results:
                self._save_results(successful_results, output_dir)
            
            return {
                'success': True,
                'processed_count': len(successful_results),
                'total_count': len(image_files),
                'output_directory': output_dir,
                'results': successful_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Processing failed: {str(e)}',
                'processed_count': 0
            }
    
    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                return None
            
            filename = os.path.basename(image_path)
            
            # Extract date
            date_str = extract_date(filename)
            
            # Quality assessment
            quality_metrics = self.quality_controller.assess_quality(image, filename)
            
            # Calculate vegetation indices
            vegetation_indices = self.vegetation_calc.calculate_all(image)
            
            # Combine results
            result = {
                'filename': filename,
                'date': date_str,
                'image_path': image_path,
                **vegetation_indices
            }
            
            if quality_metrics:
                result.update({
                    'quality_score': quality_metrics.overall_score,
                    'has_fog': quality_metrics.has_fog,
                    'has_snow': quality_metrics.has_snow,
                    'is_blurry': quality_metrics.is_blurry,
                    'is_too_dark': quality_metrics.is_too_dark,
                    'brightness': quality_metrics.brightness,
                    'contrast': quality_metrics.contrast,
                    'sharpness': quality_metrics.sharpness
                })
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _process_sequential(self, image_files: List[str]) -> List[Optional[Dict]]:
        """Process images sequentially."""
        results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
            result = self.process_single_image(image_file)
            results.append(result)
        return results
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """Save analysis results to files."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Sort by date if available
            if 'date' in df.columns and df['date'].notna().any():
                df = df.sort_values('date')
            
            # Save as CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'phenoai_analysis_{timestamp}.csv')
            df.to_csv(output_file, index=False)
            
            print(f"Results saved to: {output_file}")
            
            # Save summary statistics
            self._save_summary(df, output_dir, timestamp)
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _save_summary(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """Save summary statistics."""
        try:
            summary_file = os.path.join(output_dir, f'analysis_summary_{timestamp}.txt')
            
            with open(summary_file, 'w') as f:
                f.write("PhenoAI Analysis Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Total images processed: {len(df)}\n")
                if 'date' in df.columns:
                    f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n\n")
                
                # Vegetation index statistics
                for idx in ['gcc', 'rcc', 'bcc', 'exg', 'cive']:
                    if idx in df.columns:
                        values = df[idx].dropna()
                        if len(values) > 0:
                            f.write(f"{idx.upper()} Statistics:\n")
                            f.write(f"  Mean: {values.mean():.4f}\n")
                            f.write(f"  Std:  {values.std():.4f}\n")
                            f.write(f"  Min:  {values.min():.4f}\n")
                            f.write(f"  Max:  {values.max():.4f}\n\n")
                
                # Quality statistics
                if 'quality_score' in df.columns:
                    quality_scores = df['quality_score'].dropna()
                    if len(quality_scores) > 0:
                        f.write(f"Quality Statistics:\n")
                        f.write(f"  Mean quality score: {quality_scores.mean():.4f}\n")
                        if 'has_fog' in df.columns:
                            f.write(f"  Images with fog: {df['has_fog'].sum()}\n")
                        if 'has_snow' in df.columns:
                            f.write(f"  Images with snow: {df['has_snow'].sum()}\n")
                        if 'is_blurry' in df.columns:
                            f.write(f"  Blurry images: {df['is_blurry'].sum()}\n")
                        if 'is_too_dark' in df.columns:
                            f.write(f"  Dark images: {df['is_too_dark'].sum()}\n")
            
            print(f"Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Error saving summary: {e}")
