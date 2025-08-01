"""
Configuration management for PhenoAI.
"""

import os
import yaml
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration class for PhenoAI analysis."""
    
    # Input/Output settings
    input_directory: str = ""
    output_directory: str = "./phenoai_results"
    output_format: str = "csv"  # csv, excel, json
    
    # Processing settings
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Quality control settings
    quality_control_enabled: bool = True
    quality_threshold: float = 0.5
    fog_threshold: float = 0.15
    snow_threshold: float = 0.8
    blur_threshold: float = 100.0
    darkness_threshold: float = 50.0
    
    # Vegetation indices to calculate
    vegetation_indices: List[str] = field(default_factory=lambda: ['gcc', 'rcc', 'bcc', 'exg'])
    
    # ROI settings
    roi_enabled: bool = False
    roi_coordinates: Optional[Dict[str, int]] = None
    
    # Advanced settings
    atmospheric_correction: bool = False
    temporal_smoothing: bool = True
    smoothing_window: int = 7
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    def save_to_file(self, config_path: str):
        """Save configuration to file."""
        data = {
            'input_directory': self.input_directory,
            'output_directory': self.output_directory,
            'output_format': self.output_format,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'quality_control_enabled': self.quality_control_enabled,
            'quality_threshold': self.quality_threshold,
            'fog_threshold': self.fog_threshold,
            'snow_threshold': self.snow_threshold,
            'blur_threshold': self.blur_threshold,
            'darkness_threshold': self.darkness_threshold,
            'vegetation_indices': self.vegetation_indices,
            'roi_enabled': self.roi_enabled,
            'roi_coordinates': self.roi_coordinates,
            'atmospheric_correction': self.atmospheric_correction,
            'temporal_smoothing': self.temporal_smoothing,
            'smoothing_window': self.smoothing_window
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
