"""
Date utilities for PhenoAI package.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pathlib import Path

from ..core.logger import LoggerMixin
from ..core.exceptions import ValidationError

class DateParser(LoggerMixin):
    """Date parsing utilities for PhenoCam image filenames."""
    
    def __init__(self):
        """Initialize date parser."""
        # Common date patterns for PhenoCam images
        self.patterns = [
            (r'(\d{4})_(\d{1,2})_(\d{1,2})', '%Y_%m_%d'),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
            (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
            (r'(\d{2})_(\d{1,2})_(\d{1,2})', '%y_%m_%d'),
            (r'(\d{2})-(\d{1,2})-(\d{1,2})', '%y-%m-%d'),
            (r'(\d{2})(\d{2})(\d{2})', '%y%m%d'),
        ]
    
    def extract_date_from_filename(
        self, 
        filename: str, 
        pattern: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract date from filename.
        
        Args:
            filename: Image filename
            pattern: Custom date pattern (e.g., "*yyyy_mm_dd*")
            
        Returns:
            Extracted date string in YYYY-MM-DD format or None
        """
        try:
            if pattern:
                return self._extract_with_pattern(filename, pattern)
            else:
                return self._extract_with_default_patterns(filename)
                
        except Exception as e:
            self.logger.error(f"Error extracting date from {filename}: {str(e)}")
            return None
    
    def _extract_with_pattern(self, filename: str, pattern: str) -> Optional[str]:
        """Extract date using custom pattern."""
        try:
            # Convert pattern to regex
            regex_pattern = pattern.replace('*', '.*')
            regex_pattern = regex_pattern.replace('yyyy', r'(\d{4})')
            regex_pattern = regex_pattern.replace('mm', r'(\d{1,2})')
            regex_pattern = regex_pattern.replace('dd', r'(\d{1,2})')
            
            match = re.search(regex_pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    year, month, day = groups[:3]
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error with pattern {pattern}: {str(e)}")
            return None
    
    def _extract_with_default_patterns(self, filename: str) -> Optional[str]:
        """Extract date using default patterns."""
        for regex_pattern, date_format in self.patterns:
            try:
                match = re.search(regex_pattern, filename)
                if match:
                    groups = match.groups()
                    
                    # Handle 2-digit years
                    if date_format.startswith('%y'):
                        year = int(groups[0])
                        if year < 50:
                            year += 2000
                        else:
                            year += 1900
                        groups = (str(year),) + groups[1:]
                    
                    # Create date string
                    year, month, day = groups[:3]
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    
            except Exception:
                continue
        
        return None
    
    def parse_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string to datetime object.
        
        Args:
            date_str: Date string
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Try multiple formats
            formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%Y%m%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%m-%d-%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing date string {date_str}: {str(e)}")
            return None
    
    def date_to_doy(self, date_str: str) -> Optional[int]:
        """
        Convert date string to day of year.
        
        Args:
            date_str: Date string
            
        Returns:
            Day of year (1-366) or None if conversion fails
        """
        try:
            date_obj = self.parse_date_string(date_str)
            if date_obj:
                return date_obj.timetuple().tm_yday
            return None
            
        except Exception as e:
            self.logger.error(f"Error converting date to DOY: {str(e)}")
            return None
    
    def get_date_range(self, date_strings: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Get date range from list of date strings.
        
        Args:
            date_strings: List of date strings
            
        Returns:
            Tuple of (start_date, end_date) or (None, None) if no valid dates
        """
        try:
            valid_dates = []
            for date_str in date_strings:
                date_obj = self.parse_date_string(date_str)
                if date_obj:
                    valid_dates.append(date_obj)
            
            if not valid_dates:
                return None, None
            
            start_date = min(valid_dates).strftime('%Y-%m-%d')
            end_date = max(valid_dates).strftime('%Y-%m-%d')
            
            return start_date, end_date
            
        except Exception as e:
            self.logger.error(f"Error getting date range: {str(e)}")
            return None, None

def extract_date_from_filename(filename: str, pattern: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to extract date from filename.
    
    Args:
        filename: Image filename
        pattern: Optional date pattern
        
    Returns:
        Extracted date string or None
    """
    parser = DateParser()
    return parser.extract_date_from_filename(filename, pattern)

def create_date_pattern(example_filename: str, date_position: str = "auto") -> Optional[str]:
    """
    Create a date pattern from an example filename.
    
    Args:
        example_filename: Example filename containing date
        date_position: Position of date ("auto", "start", "middle", "end")
        
    Returns:
        Date pattern string or None if not found
    """
    parser = DateParser()
    
    # Try to find date in filename
    for regex_pattern, _ in parser.patterns:
        match = re.search(regex_pattern, example_filename)
        if match:
            # Found date, create pattern
            start, end = match.span()
            
            if date_position == "auto":
                # Create pattern based on position
                if start == 0:
                    # Date at start
                    if len(example_filename) > end:
                        return "yyyy_mm_dd*"
                    else:
                        return "yyyy_mm_dd"
                elif end == len(example_filename):
                    # Date at end
                    return "*yyyy_mm_dd"
                else:
                    # Date in middle
                    return "*yyyy_mm_dd*"
            else:
                # Use specified position
                if date_position == "start":
                    return "yyyy_mm_dd*"
                elif date_position == "end":
                    return "*yyyy_mm_dd"
                else:
                    return "*yyyy_mm_dd*"
    
    return None
