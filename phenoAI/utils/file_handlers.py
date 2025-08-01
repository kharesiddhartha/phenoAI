"""
File handling utilities for PhenoAI.
"""

import os
import cv2
import numpy as np
from typing import List, Optional
from PIL import Image

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array in BGR format, or None if failed
    """
    try:
        if not os.path.exists(image_path):
            return None
        
        # Try OpenCV first
        image = cv2.imread(image_path)
        if image is not None:
            return image
        
        # Try PIL as backup
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Output file path
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save using OpenCV
        success = cv2.imwrite(output_path, image)
        return success
        
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def get_image_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory path to search
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    image_files = []
    
    try:
        if not os.path.exists(directory):
            return image_files
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in extensions:
                    image_files.append(os.path.join(root, file))
        
        # Sort files for consistent processing order
        image_files.sort()
        
    except Exception as e:
        print(f"Error getting image files from {directory}: {e}")
    
    return image_files
        
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
            
        Raises:
            DataError: If image loading fails
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise DataError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() not in self.supported_formats:
            raise DataError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            # Use OpenCV for most formats
            image = cv2.imread(str(image_path))
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if image is None:
                raise DataError(f"Failed to load image: {image_path}")
            
            self.logger.debug(f"Loaded image {image_path} with shape {image.shape}")
            return image
            
        except Exception as e:
            raise DataError(f"Error loading image {image_path}: {str(e)}")
    
    def load_images_from_directory(
        self, 
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load all images from a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            Dictionary mapping filenames to image arrays
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise DataError(f"Directory not found: {directory}")
        
        if recursive:
            image_files = directory.rglob(pattern)
        else:
            image_files = directory.glob(pattern)
        
        # Filter by supported formats
        image_files = [
            f for f in image_files 
            if f.suffix.lower() in self.supported_formats
        ]
        
        if not image_files:
            raise DataError(f"No supported image files found in {directory}")
        
        images = {}
        for image_file in image_files:
            try:
                image = self.load_image(image_file)
                images[image_file.name] = image
            except DataError as e:
                self.logger.warning(f"Skipping image {image_file}: {e}")
        
        self.logger.info(f"Loaded {len(images)} images from {directory}")
        return images
    
    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise DataError(f"Image file not found: {image_path}")
        
        # Basic file info
        stat = image_path.stat()
        info = {
            'filename': image_path.name,
            'path': str(image_path),
            'size_bytes': stat.st_size,
            'modified_time': stat.st_mtime
        }
        
        # Image dimensions
        try:
            with Image.open(image_path) as img:
                info.update({
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format
                })
        except Exception as e:
            self.logger.warning(f"Could not get image info for {image_path}: {e}")
        
        return info

class DataSaver(LoggerMixin):
    """Class for saving analysis results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize DataSaver.
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataframe(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        format: str = 'xlsx',
        **kwargs
    ) -> Path:
        """
        Save pandas DataFrame to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            format: Output format ('xlsx', 'csv', 'json', 'pickle')
            **kwargs: Additional arguments for saving functions
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        if format.lower() == 'xlsx':
            if not filename.endswith('.xlsx'):
                output_path = output_path.with_suffix('.xlsx')
            df.to_excel(output_path, index=False, **kwargs)
            
        elif format.lower() == 'csv':
            if not filename.endswith('.csv'):
                output_path = output_path.with_suffix('.csv')
            df.to_csv(output_path, index=False, **kwargs)
            
        elif format.lower() == 'json':
            if not filename.endswith('.json'):
                output_path = output_path.with_suffix('.json')
            df.to_json(output_path, orient='records', **kwargs)
            
        elif format.lower() == 'pickle':
            if not filename.endswith('.pkl'):
                output_path = output_path.with_suffix('.pkl')
            df.to_pickle(output_path, **kwargs)
            
        else:
            raise ValidationError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved DataFrame to {output_path}")
        return output_path
    
    def save_json(self, data: Dict[str, Any], filename: str) -> Path:
        """
        Save dictionary as JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        if not filename.endswith('.json'):
            output_path = output_path.with_suffix('.json')
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved JSON to {output_path}")
        return output_path
    
    def save_pickle(self, data: Any, filename: str) -> Path:
        """
        Save data as pickle file.
        
        Args:
            data: Data to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        if not filename.endswith('.pkl'):
            output_path = output_path.with_suffix('.pkl')
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved pickle to {output_path}")
        return output_path
