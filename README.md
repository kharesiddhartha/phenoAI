# PhenoAI: A deep learning Python framework to process close-range time-lapse PhenoCam data

<div align="center">

![PhenoAI Logo](https://img.shields.io/badge/PhenoAI-v1.2.0-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**Advanced Deep Learning Framework for Automated Phenological Analysis of Close-Range Time-Lapse PhenoCam Data**

</div>

## 🌿 Overview

PhenoAI is a state-of-the-art deep learning framework designed for automated phenological analysis of vegetation from close-range time-lapse PhenoCam data. This framework implements cutting-edge computer vision techniques to provide comprehensive vegetation monitoring, segmentation, and temporal analysis capabilities.

> **Research Publication**: This work is part of the research paper:  
> **"PhenoAI: A deep learning Python framework to process close-range time-lapse PhenoCam data"**  
> *by Akash Kumar, Siddhartha Khare, and Sergio Rossi*  
> Published in **Ecological Informatics, 2025**

### 🎯 Key Features

- **🔬 Advanced Image Quality Control**: Automated detection and filtering of low-quality images
- **🌱 Vegetation Segmentation**: Deep learning-powered vegetation extraction using semantic segmentation
- **📊 ROI Analysis**: K-means clustering-based Region of Interest (ROI) generation and analysis
- **📈 Vegetation Indices**: Comprehensive calculation of multiple vegetation indices (NDVI, EVI, SAVI, etc.)
- **🖥️ Professional GUI**: Advanced parameter tuning interface with real-time preview
- **⚡ Batch Processing**: Efficient processing of large-scale time-lapse datasets
- **📋 Interactive CLI**: User-friendly command-line interface with guided workflows

## 🚀 Quick Start

### Installation

```bash
For Local Repository

Download the .zip from releases (recommended)
#Alternatively,
# Clone the repository (But have to move the trained segmented model in model/[you trained segmented model] folder manually) 
git clone https://github.com/your-username/phenoAI.git

cd phenoAI

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Launch interactive mode
python -m phenoAI.cli --interactive
(Recommended) --> user-friendly command line design

# Or use the installed command
phenoai --interactive
```

## 📋 Detailed Usage Guide

### 1. Interactive Command Line Interface

The easiest way to get started is using the interactive CLI:

```bash
python -m phenoAI.cli --interactive
```

This will present you with a menu-driven interface:

```
🌿 PhenoAI v1.2.0 - Phenological Analysis Framework
📚 Advanced Deep Learning for Close-Range PhenoCam Data Analysis

Select workflow:
1. 🔍 Image Quality Control
2. 🌱 Vegetation Segmentation  
3. 📊 ROI Generation and Analysis
4. 📈 Vegetation Index Calculation
5. 🎯 Complete Workflow (All Steps)
6. 🖥️ Parameter Tuning GUI
7. ❌ Exit

Enter your choice (1-7):
```

### 2. Step-by-Step Workflow

#### Step 1: Image Quality Control

```python
from phenoAI import PhenoAI

# Initialize framework
analyzer = PhenoAI()

# Set input directory containing your time-lapse images
analyzer.input_dir = "path/to/your/images"

# Run quality control
quality_results = analyzer.step1_image_quality_control()

# Results will be saved to:
# - Output/01_processing/high_quality_images/
# - Output/02_low_quality_images/
```

**Features:**
- Automated blur detection using Laplacian variance
- Brightness and contrast analysis
- Noise level assessment
- Histogram-based quality metrics

#### Step 2: Vegetation Segmentation

```python
# Run vegetation segmentation on quality-filtered images
segmentation_results = analyzer.step2_vegetation_segmentation()

# Results include:
# - Vegetation masks (binary)
# - Vegetation extraction (RGB)
# - Segmentation metrics and confidence scores
```

**Deep Learning Model:**
- Uses advanced semantic segmentation (DeepLabV3+ architecture)
- Pre-trained on diverse vegetation datasets
- Supports multiple vegetation types (grass, crops, trees, etc.)
- Confidence-based result filtering

#### Step 3: ROI Generation and Analysis

```python
# Generate ROIs using K-means clustering on vegetation areas
roi_results = analyzer.step3_roi_generation()

# Interactive ROI preview with professional GUI
analyzer.display_roi_preview_popup(segmentation_results)

# Select specific ROIs for analysis
analyzer.selected_rois = [0, 2, 4, 6]  # or use 'all'
```

**ROI Features:**
- K-means clustering on vegetation pixels
- Polygon-based ROI boundaries
- Interactive preview with colorful cluster visualization
- Customizable ROI selection

#### Step 4: Vegetation Index Calculation

```python
# Calculate comprehensive vegetation indices
vi_results = analyzer.step4_vegetation_indices()

# Available indices:
# - NDVI (Normalized Difference Vegetation Index)
# - EVI (Enhanced Vegetation Index)
# - SAVI (Soil-Adjusted Vegetation Index)
# - MSAVI (Modified SAVI)
# - NDWI (Normalized Difference Water Index)
# - CIG (Crop Index Green)
# - VEG (Vegetation Index)
```

### 3. Advanced Parameter Tuning

Launch the professional GUI for parameter optimization:

```python
from phenoAI.gui import run_parameter_tuning

# Launch parameter tuning interface
run_parameter_tuning()
```

**GUI Features:**
- Real-time parameter adjustment with live preview
- 5 sample images per filter type
- Instant detection status feedback
- Professional Tkinter interface
- Save/load parameter configurations

### 4. Batch Processing

```python
# Process multiple directories
import os
from phenoAI import PhenoAI

directories = [
    "dataset1/time_series_1/",
    "dataset1/time_series_2/",
    "dataset2/wheat_2024/"
]

for directory in directories:
    analyzer = PhenoAI()
    analyzer.input_dir = directory
    analyzer.output_dir = f"results/{os.path.basename(directory)}"
    
    # Run complete workflow
    analyzer.run_full_workflow()
```

## 🛠️ Advanced Configuration

### Custom Model Integration

```python
# Use custom segmentation model
analyzer = PhenoAI()
analyzer.segmentation_model_path = "path/to/your/custom_model.h5"

# Set custom vegetation labels
analyzer.selected_vegetation_label = "crop"  # or "grass", "tree", etc.
```

### ROI Configuration

```python
# Customize ROI generation parameters
roi_params = {
    "n_clusters": 8,        # Number of K-means clusters
    "min_roi_size": 100,    # Minimum ROI size in pixels
    "max_roi_size": 5000,   # Maximum ROI size in pixels
    "roi_shape": "polygon"  # ROI shape type
}

analyzer.set_roi_parameters(roi_params)
```

### Quality Control Thresholds

```python
# Adjust quality control parameters
quality_params = {
    "blur_threshold": 100.0,      # Laplacian variance threshold
    "brightness_min": 50,         # Minimum brightness
    "brightness_max": 200,        # Maximum brightness
    "contrast_threshold": 30      # Minimum contrast
}

analyzer.set_quality_parameters(quality_params)
```

## 📊 Output Structure

PhenoAI generates a comprehensive output directory structure:

```
Output/
├── 01_processing/
│   ├── high_quality_images/     # Quality-filtered images
│   └── processing_log.json      # Processing metadata
├── 02_low_quality_images/       # Rejected images with reasons
├── 03_segmentation_outputs/     # Vegetation masks and extractions
│   ├── masks/                   # Binary vegetation masks
│   ├── extractions/             # RGB vegetation extractions
│   └── metrics/                 # Segmentation confidence scores
├── 04_roi_analysis/             # ROI results and visualizations
│   ├── roi_boundaries/          # ROI polygon coordinates
│   ├── roi_previews/           # Visual ROI overlays
│   └── cluster_analysis/        # K-means clustering results
├── 05_vegetation_indices/       # Calculated indices
│   ├── ndvi_results/           # NDVI time series
│   ├── evi_results/            # EVI time series
│   └── comprehensive_indices/   # All indices combined
└── reports/
    ├── summary_report.html      # Comprehensive analysis report
    ├── quality_metrics.csv      # Quality control statistics
    ├── segmentation_metrics.csv # Segmentation performance
    └── temporal_analysis.csv    # Time-series vegetation indices
```

## 🧪 Example Workflows

### Workflow 1: Basic Vegetation Monitoring

```python
from phenoAI import PhenoAI

# Initialize analyzer
analyzer = PhenoAI()
analyzer.input_dir = "wheat_field_2024/"

# Run complete automated workflow
results = analyzer.run_full_workflow()

# Access results
print(f"Processed {len(results['high_quality_images'])} images")
print(f"Generated {len(results['rois'])} ROIs")
print(f"NDVI temporal trend: {results['vegetation_indices']['ndvi_trend']}")
```

### Workflow 2: Custom ROI Analysis

```python
# Step-by-step processing with custom ROI selection
analyzer = PhenoAI()
analyzer.input_dir = "crop_monitoring/"

# Quality control
quality_results = analyzer.step1_image_quality_control()

# Vegetation segmentation
seg_results = analyzer.step2_vegetation_segmentation(quality_results)

# Generate and preview ROIs
analyzer.step3_roi_generation(seg_results)
analyzer.display_roi_preview_popup(seg_results)

# Manual ROI selection
analyzer.selected_rois = [1, 3, 5, 7, 9]  # Select specific ROIs

# Calculate indices for selected ROIs only
vi_results = analyzer.step4_vegetation_indices(roi_subset=analyzer.selected_rois)
```

### Workflow 3: Parameter Optimization

```python
# Launch GUI for parameter tuning
from phenoAI.gui import run_parameter_tuning

# Interactive parameter optimization
run_parameter_tuning()

# Apply optimized parameters
analyzer = PhenoAI()
analyzer.load_parameters("optimized_params.json")
analyzer.run_full_workflow()
```

## 📈 Vegetation Indices Details

PhenoAI calculates multiple vegetation indices for comprehensive analysis:

### NDVI (Normalized Difference Vegetation Index)
```
NDVI = (NIR - Red) / (NIR + Red)
```
- **Range**: -1 to +1
- **Use**: General vegetation health and density
- **Interpretation**: Higher values indicate healthier vegetation

### EVI (Enhanced Vegetation Index)
```
EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
```
- **Range**: -1 to +1
- **Use**: Improved sensitivity in high biomass areas
- **Advantage**: Reduced atmospheric and soil background effects

### SAVI (Soil-Adjusted Vegetation Index)
```
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
```
- **Range**: -1 to +1
- **Use**: Areas with significant soil background
- **Parameter**: L = 0.5 (soil brightness correction factor)

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Issues
```python
# If segmentation model fails to load
analyzer = PhenoAI()
analyzer.download_pretrained_model()  # Download latest model
analyzer.segmentation_model_path = "Model/Basic_Vegetation_Model.h5"
```

#### 2. Memory Issues with Large Datasets
```python
# Process in smaller batches
analyzer.set_batch_size(32)  # Reduce batch size
analyzer.enable_memory_optimization(True)
```

#### 3. GUI Issues
```python
# If GUI fails to launch
import tkinter as tk
try:
    root = tk.Tk()
    root.destroy()
    print("✅ Tkinter available")
except:
    print("❌ Tkinter not available - install tkinter package")
```

### Performance Optimization

```python
# Enable GPU acceleration (if available)
analyzer.enable_gpu(True)

# Parallel processing
analyzer.set_num_workers(4)  # Adjust based on CPU cores

# Memory-efficient processing
analyzer.enable_lazy_loading(True)
```

## 📚 API Reference

### Core Classes

#### `PhenoAI`
Main analysis framework class.

**Methods:**
- `step1_image_quality_control()` - Image quality filtering
- `step2_vegetation_segmentation()` - Deep learning segmentation
- `step3_roi_generation()` - ROI clustering and generation
- `step4_vegetation_indices()` - Vegetation index calculation
- `run_full_workflow()` - Complete automated processing

#### `ParameterTuningGUI`
Advanced parameter optimization interface.

**Features:**
- Real-time parameter adjustment
- Multi-image preview
- Parameter persistence
- Export/import configurations

### Utility Functions

```python
from phenoAI.utils import (
    calculate_vegetation_index,
    apply_quality_filters,
    generate_roi_polygons,
    create_temporal_plots
)

# Calculate custom vegetation index
custom_vi = calculate_vegetation_index(
    image_path="sample.jpg",
    index_type="NDVI",
    roi_coordinates=[(100, 100), (200, 200)]
)
```

## 🤝 Contributing

We welcome contributions to PhenoAI! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** coding standards
3. **Add comprehensive tests** for new features
4. **Update documentation** for API changes
5. **Submit a pull request** with detailed description

### Development Setup

```bash
# Clone for development
git clone https://github.com/your-username/phenoAI.git
cd phenoAI

# Create virtual environment
python -m venv phenoai_dev
source phenoai_dev/bin/activate  # On Windows: phenoai_dev\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
python -m pytest tests/
```

## 📄 Citation

If you use PhenoAI in your research, please cite our paper:

```APA
Kumar, A., Khare, S., & Rossi, S. (2025). PhenoAI: A deep learning Python framework to process close-range time-lapse PhenoCam data. Ecological Informatics, 88, 103134. https://doi.org/10.1016/j.ecoinf.2025.103134

```bibtex
@article{KUMAR2025103134,
title = {PhenoAI: A deep learning Python framework to process close-range time-lapse PhenoCam data},
journal = {Ecological Informatics},
volume = {88},
pages = {103134},
year = {2025},
issn = {1574-9541},
doi = {https://doi.org/10.1016/j.ecoinf.2025.103134},
url = {https://www.sciencedirect.com/science/article/pii/S1574954125001438},
author = {Akash Kumar and Siddhartha Khare and Sergio Rossi},
keywords = {Close-range Remote Sensing, GCC, Forest Phenology, PhenoCam, Python, Deep Learning}
}
```

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/phenoAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/phenoAI/discussions)
- **Email**: [akash_k@ce.iitr.ac.in](mailto:akash_k@ce.iitr.ac.in), [siddhartha.khare@ce.iitr.ac.in](mailto:siddhartha.khare@ce.iitr.ac.in)

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Deep learning models based on DeepLabV3+ architecture
- Vegetation index calculations following established remote sensing protocols
- GUI framework built with Python Tkinter
- Segmentation pipeline inspired by computer vision best practices

---

<div align="center">

**PhenoAI v1.2.0** - Advancing Phenological Research with Deep Learning

*Developed by Akash Kumar, Siddhartha Khare and Sergio Rossi | Published in Ecological Informatics 2025*

[![GitHub stars](https://img.shields.io/github/stars/kharesiddhartha/phenoAI.svg?style=social&label=Star)](https://github.com/kharesiddhartha/phenoAI)
[![GitHub forks](https://img.shields.io/github/forks/kharesiddhartha/phenoAI.svg?style=social&label=Fork)](https://github.com/kharesiddhartha/phenoAI/fork)

</div>
