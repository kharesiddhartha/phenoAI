"""
Setup script for PhenoAI package - Advanced Deep Learning Framework for Phenological Analysis

PhenoAI is a comprehensive deep learning framework designed to automate the processing 
chain of phenological analysis of vegetation from close-range time-lapse PhenoCam data.

Authors: Akash Kumar, Siddhartha Khare, and Sergio Rossi
Published in: Ecological Informatics, 2025
DOI: https://doi.org/10.1016/j.ecoinf.2025.103134
License: MIT

This setup.py provides backward compatibility for older pip versions.
Modern configuration is primarily handled by pyproject.toml.
"""

import os
from setuptools import setup, find_packages

def read_version():
    """Read version from __init__.py"""
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(here, 'phenoAI', '__init__.py'), encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "1.2.0"

def read_long_description():
    """Read the README.md file for long description"""
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Advanced Deep Learning Framework for Automated Phenological Analysis of Close-Range Time-Lapse PhenoCam Data"

# Configuration primarily handled by pyproject.toml
# This setup.py ensures backward compatibility
setup(
    # Basic package information (complementing pyproject.toml)
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    # Ensure package data is included
    include_package_data=True,
    zip_safe=False,  # Package contains data files (Model/*.h5, assets/*)
    
    # Dynamic version reading (fallback if not in pyproject.toml)
    version=read_version(),
    
    # Long description (fallback)
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # Additional project URLs not in pyproject.toml
    project_urls={
        "Research Paper": "https://doi.org/10.1016/j.ecoinf.2025.103134",
        "Releases": "https://github.com/kharesiddhartha/phenoAI/releases",
        "Changelog": "https://github.com/kharesiddhartha/phenoAI/blob/main/CHANGELOG.md",
        "Citation": "https://doi.org/10.1016/j.ecoinf.2025.103134",
    },
    
    # Platform specification
    platforms=["any"],
    
    # Additional classifiers for better discoverability
    classifiers=[
        # Additional topics not in pyproject.toml
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization", 
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Environment :: Console",
        "Environment :: Win32 (MS Windows)",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows", 
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
    ],
    
    # Additional keywords for PyPI search
    keywords=[
        "forest phenology", "crop monitoring", "NDVI", "segmentation",
        "machine learning", "tensorflow", "scientific computing", 
        "environmental monitoring", "ecological analysis", "agriculture",
        "time-series analysis", "image processing", "automated analysis"
    ],
)
