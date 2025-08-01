"""
Custom exceptions for PhenoAI package.
"""

class PhenoAIError(Exception):
    """Base exception class for PhenoAI package."""
    pass

class ValidationError(PhenoAIError):
    """Raised when input validation fails."""
    pass

class ProcessingError(PhenoAIError):
    """Raised when processing operations fail."""
    pass

class ModelError(PhenoAIError):
    """Raised when model operations fail."""
    pass

class DataError(PhenoAIError):
    """Raised when data operations fail."""
    pass

class ConfigurationError(PhenoAIError):
    """Raised when configuration is invalid."""
    pass

class QualityControlError(PhenoAIError):
    """Raised when quality control checks fail."""
    pass

class AnalysisError(PhenoAIError):
    """Raised when analysis operations fail."""
    pass

class SegmentationError(PhenoAIError):
    """Raised when segmentation operations fail."""
    pass

class TimeSeriesError(PhenoAIError):
    """Raised when time series analysis fails."""
    pass

class ROIError(PhenoAIError):
    """Raised when ROI selection or processing fails."""
    pass

class FileFormatError(PhenoAIError):
    """Raised when file format is not supported or invalid."""
    pass
