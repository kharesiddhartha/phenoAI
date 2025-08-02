#!/usr/bin/env python3
"""
PhenoAI Installation Test Script
===============================

This script tests the installation and basic functionality of the PhenoAI package.
Run this after installing the package to verify everything works correctly.

Usage:
    python test_installation.py
"""

import sys
import importlib
import subprocess

def test_import():
    """Test if PhenoAI can be imported successfully."""
    print("🔍 Testing package import...")
    try:
        import phenoAI
        print("✅ Successfully imported phenoAI")
        
        # Test specific modules
        from phenoAI import PhenoAI as PhenoAIClass
        print("✅ Successfully imported PhenoAI class")
        
        from phenoAI.analysis import vegetation_indices
        print("✅ Successfully imported vegetation_indices")
        
        from phenoAI.preprocessing import atmospheric_correction
        print("✅ Successfully imported atmospheric_correction")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cli():
    """Test if CLI interface is accessible."""
    print("\n🔍 Testing CLI interface...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "phenoAI.cli", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI interface is working")
            return True
        else:
            print(f"❌ CLI failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ CLI test timed out")
        return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_dependencies():
    """Test if core dependencies are available."""
    print("\n🔍 Testing core dependencies...")
    
    dependencies = [
        "numpy",
        "scipy", 
        "opencv-python",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "pillow",
        "rasterio",
        "shapely",
        "geopandas"
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep.replace("-", "_"))
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"⚠️  {dep} is missing")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("💡 Try installing with: pip install phenoai[all]")
        return False
    
    return True

def test_optional_dependencies():
    """Test optional dependencies."""
    print("\n🔍 Testing optional dependencies...")
    
    optional_deps = {
        "tensorflow": "TensorFlow (for deep learning features)",
        "torch": "PyTorch (alternative deep learning)",
        "jupyter": "Jupyter (for notebook support)",
        "sphinx": "Sphinx (for documentation)"
    }
    
    for dep, description in optional_deps.items():
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} is available - {description}")
        except ImportError:
            print(f"ℹ️  {dep} is not installed - {description}")

def check_version():
    """Check package version."""
    print("\n🔍 Checking package version...")
    try:
        import phenoAI
        if hasattr(phenoAI, '__version__'):
            print(f"📦 PhenoAI version: {phenoAI.__version__}")
        else:
            print("📦 PhenoAI version: Unable to determine")
    except Exception as e:
        print(f"❌ Version check failed: {e}")

def main():
    """Run all tests."""
    print("🚀 PhenoAI Installation Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 4
    
    # Test import
    if test_import():
        tests_passed += 1
    
    # Check version
    check_version()
    
    # Test CLI
    if test_cli():
        tests_passed += 1
    
    # Test dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Test optional dependencies (informational)
    test_optional_dependencies()
    tests_passed += 1  # This test always "passes"
    
    print("\n" + "=" * 40)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! PhenoAI is ready to use.")
        print("\n📚 Quick Start:")
        print("  from phenoAI import PhenoAI")
        print("  analyzer = PhenoAI()")
        print("  # Start analyzing your remote sensing data!")
        
        print("\n🖥️  CLI Usage:")
        print("  python -m phenoAI.cli --interactive")
        
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\n🔧 Try reinstalling with all dependencies:")
        print("  pip uninstall phenoai")
        print("  pip install phenoai[all]")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
