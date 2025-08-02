"""
PhenoAI - Professional Phenological Analysis Package

A comprehensive Python package for automated phenological analysis from RGB imagery.
Implements advanced vegetation indices, quality control, time series analysis,
and phenological parameter extraction following published research methodologies.

Main Features:
- Vegetation indices (GCC, RCC, BCC, ExG, CIVE)
- Image quality control (fog, snow, blur detection)
- Time series analysis with curve fitting
- Phenological parameter extraction
- Interactive CLI interface
- Comprehensive reporting and visualization

Usage:
    # Import the comprehensive framework
    from phenoAI.cli_comprehensive import PhenoAI
    
    # Create analyzer instance
    analyzer = PhenoAI()
    
    # Run full workflow interactively
    analyzer.run_full_workflow()

Command Line:
    phenoai --version
    phenoai --interactive  # Run comprehensive workflow
    phenoai --simple      # Basic version info
"""

__version__ = "1.2.1"
__author__ = "PhenoAI Development Team"

# Try to import the main CLI framework
try:
    from .cli import PhenoAI
    _framework_version = "Standard"
    _comprehensive_available = True
    __all__ = ['PhenoAI', '__version__', '__author__']
    print(f"🌿 PhenoAI v{__version__} - Phenological Analysis Framework")
    print("📚 Type 'from phenoAI import PhenoAI' to get started")
    print("🖥️  Command line: 'python -m phenoAI.cli --interactive'")
except ImportError as e:
        _comprehensive_available = False
        __all__ = ['__version__', '__author__']
        import warnings
        warnings.warn(
            f"PhenoAI comprehensive framework requires additional dependencies.\n"
            f"Install with: pip install opencv-python scikit-learn matplotlib pandas scipy\n"
            f"Error: {e}",
            ImportWarning
        )
        print(f"PhenoAI v{__version__} - Basic Mode (dependencies missing)")

# Legacy compatibility
try:
    from .legacy import *
except ImportError:
    pass
