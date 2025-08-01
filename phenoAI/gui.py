"""
PhenoAI: Improved Parameter Tuning GUI

An enhanced GUI for tuning image quality control parameters with:
- 5 sample images per filter
- Real-time updates when sliders change
- Clear detection status for each image
- Refresh buttons for each filter
- User-friendly descriptions
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import random
from typing import Dict, List, Optional, Any
import json

class ImprovedParameterTuningGUI:
    """Enhanced GUI for parameter tuning with multiple images and real-time updates."""

    def __init__(self, selected_filters: List[str], image_folder: str):
        self.selected_filters = selected_filters
        self.image_folder = image_folder
        self.images_per_filter = 5
        self.final_parameters = None
        
        # Initialize tkinter
        self.root = tk.Tk()
        self.root.title("PhenoAI - Parameter Tuning")
        self.root.geometry("2800x1600")  # MASSIVE window for huge images
        self.root.configure(bg='#f0f0f0')
        self.root.state('zoomed')  # Maximize window on Windows
        
        # Parameter storage
        self.parameters = {}
        self.sample_images = {}
        self.image_widgets = {}
        
        # Add delay mechanism for slider updates
        self.update_delays = {}  # Store delay timers for each filter
        
        # Load sample images for each filter
        self.load_sample_images()
        self.create_interface()

    def load_sample_images(self):
        """Load sample images for each filter with truly random selection."""
        try:
            all_images = [f for f in os.listdir(self.image_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(all_images) < self.images_per_filter:
                # If not enough images, repeat the list
                all_images = all_images * ((self.images_per_filter // len(all_images)) + 1)
            
            for filter_name in self.selected_filters:
                # Create a new shuffled copy for each filter to ensure variety
                shuffled_images = all_images.copy()
                random.shuffle(shuffled_images)
                # Select different images for each filter
                self.sample_images[filter_name] = shuffled_images[:self.images_per_filter]
                
        except Exception as e:
            print(f"Error loading sample images: {e}")
            # Fallback: use empty lists
            for filter_name in self.selected_filters:
                self.sample_images[filter_name] = []

    def create_interface(self):
        """Create the main interface."""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🌿 PhenoAI Parameter Tuning", 
                              font=('Arial', 16), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Add helpful info about defaults
        info_frame = tk.Frame(self.root, bg='#ecf0f1', height=30)
        info_frame.pack(fill='x')
        info_frame.pack_propagate(False)
        
        info_label = tk.Label(info_frame, 
                             text="💡 TIP: Default values are pre-set for typical images. Adjust only if needed. Focus on the LARGE IMAGES below to judge quality!", 
                             font=('Arial', 9, 'bold'), fg='#2c3e50', bg='#ecf0f1')
        info_label.pack(expand=True)

        # CREATE CONTROL PANEL FIRST so it gets space allocated
        self.create_control_panel()

        # Main container with scrollbar - now it won't cover the control panel
        main_container = tk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=(10, 0))  # No bottom padding

        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_container, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create filter sections
        for filter_name in self.selected_filters:
            self.create_filter_section(scrollable_frame, filter_name)

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def create_filter_section(self, parent, filter_name):
        """Create a complete section for one filter."""
        # Main frame for this filter
        filter_frame = tk.LabelFrame(parent, text=f"{filter_name.title()} Filter Detection", 
                                   font=('Arial', 14, 'bold'), fg='#2c3e50', bg='#f0f0f0',
                                   padx=10, pady=10)
        filter_frame.pack(fill='x', pady=10)

        # Description with smaller text and default value info
        descriptions = {
            'blur': "🔍 BLUR DETECTION: Finds blurry images | Default: Sensitivity=30, Quality=200",
            'fog': "🌫️ FOG DETECTION: Finds foggy/hazy images | Default: Sensitivity=30, Contrast=25", 
            'dark': "🌙 DARK DETECTION: Finds very dark images | Default: Sensitivity=30, Brightness=50",
            'snow': "❄️ SNOW DETECTION: Finds snowy images | Default: Sensitivity=30, Coverage=200"
        }
        
        desc_label = tk.Label(filter_frame, text=descriptions.get(filter_name, "Image quality filter"), 
                             font=('Arial', 8), fg='#34495e', bg='#f0f0f0', wraplength=1200)
        desc_label.pack(anchor='w', pady=(0, 10))

        # Controls frame
        controls_frame = tk.Frame(filter_frame, bg='#f0f0f0')
        controls_frame.pack(fill='x', pady=(0, 10))

        # Sensitivity control with smaller text and default values
        tk.Label(controls_frame, text="🎯 Detection Sensitivity (0=Lenient, 100=Strict) [Default: 30]:", 
                font=('Arial', 9), bg='#f0f0f0').pack(side='left', padx=(0, 10))
        
        # Set appropriate default values for each filter type
        default_sensitivity = 30.0  # More lenient default
        sensitivity_var = tk.DoubleVar(value=default_sensitivity)
        sensitivity_scale = tk.Scale(controls_frame, from_=0, to=100, orient='horizontal',
                                   variable=sensitivity_var, resolution=5, length=200,
                                   font=('Arial', 8),
                                   command=lambda v, f=filter_name: self.on_parameter_change(f))
        sensitivity_scale.pack(side='left', padx=(0, 20))

        # Add threshold controls with smaller text and default values
        threshold_var = None
        if filter_name == 'blur':
            tk.Label(controls_frame, text="📐 Sharpness Quality [Default: 200]:", 
                    font=('Arial', 9), bg='#f0f0f0').pack(side='left', padx=(20, 10))
            threshold_var = tk.DoubleVar(value=200.0)  # Good default for most images
            threshold_scale = tk.Scale(controls_frame, from_=50, to=500, orient='horizontal',
                                     variable=threshold_var, resolution=10, length=200,
                                     font=('Arial', 8),
                                     command=lambda v, f=filter_name: self.on_parameter_change(f))
            threshold_scale.pack(side='left', padx=(0, 20))

        elif filter_name == 'fog':
            tk.Label(controls_frame, text="🌫️ Contrast Level [Default: 25]:", 
                    font=('Arial', 9), bg='#f0f0f0').pack(side='left', padx=(20, 10))
            threshold_var = tk.DoubleVar(value=25.0)  # Good default for fog detection
            threshold_scale = tk.Scale(controls_frame, from_=5, to=50, orient='horizontal',
                                     variable=threshold_var, resolution=2, length=200,
                                     font=('Arial', 8),
                                     command=lambda v, f=filter_name: self.on_parameter_change(f))
            threshold_scale.pack(side='left', padx=(0, 20))

        elif filter_name == 'dark':
            tk.Label(controls_frame, text="💡 Brightness Level [Default: 50]:", 
                    font=('Arial', 9), bg='#f0f0f0').pack(side='left', padx=(20, 10))
            threshold_var = tk.DoubleVar(value=50.0)  # Good default for dark detection
            threshold_scale = tk.Scale(controls_frame, from_=10, to=100, orient='horizontal',
                                     variable=threshold_var, resolution=2, length=200,
                                     font=('Arial', 8),
                                     command=lambda v, f=filter_name: self.on_parameter_change(f))
            threshold_scale.pack(side='left', padx=(0, 20))

        elif filter_name == 'snow':
            tk.Label(controls_frame, text="❄️ Snow Coverage [Default: 200]:", 
                    font=('Arial', 9), bg='#f0f0f0').pack(side='left', padx=(20, 10))
            threshold_var = tk.DoubleVar(value=200.0)  # Good default for snow detection
            threshold_scale = tk.Scale(controls_frame, from_=180, to=255, orient='horizontal',
                                     variable=threshold_var, resolution=5, length=200,
                                     font=('Arial', 8),
                                     command=lambda v, f=filter_name: self.on_parameter_change(f))
            threshold_scale.pack(side='left', padx=(0, 20))

        # Refresh button with better styling
        refresh_btn = tk.Button(controls_frame, text="🔄 Update Images", 
                               command=lambda f=filter_name: self.refresh_filter_images(f),
                               bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
                               relief='raised', borderwidth=3, padx=20, pady=5)
        refresh_btn.pack(side='right', padx=(20, 0))

        # Store parameters
        self.parameters[filter_name] = {'sensitivity': sensitivity_var}
        if threshold_var:
            if filter_name == 'blur':
                self.parameters[filter_name]['laplacian_threshold'] = threshold_var
            elif filter_name == 'fog':
                self.parameters[filter_name]['contrast_threshold'] = threshold_var
            elif filter_name == 'dark':
                self.parameters[filter_name]['brightness_threshold'] = threshold_var
            elif filter_name == 'snow':
                self.parameters[filter_name]['white_threshold'] = threshold_var

        # Images grid
        images_frame = tk.Frame(filter_frame, bg='#f0f0f0')
        images_frame.pack(fill='x', pady=(10, 0))

        self.image_widgets[filter_name] = []
        for i in range(self.images_per_filter):
            img_container = tk.Frame(images_frame, bg='#ecf0f1', relief='raised', bd=2)
            img_container.pack(side='left', padx=5, pady=5)  # Minimal padding

            # Image label - Flexible size to accommodate original aspect ratios
            img_label = tk.Label(img_container, bg='#ecf0f1')
            img_label.pack(padx=5, pady=(5, 2))  # Minimal padding

            # Status label with smaller text 
            status_label = tk.Label(img_container, text="🔄 Loading samples...", 
                                  font=('Arial', 7), bg='#ecf0f1', fg='#3498db', 
                                  wraplength=200)  # Compact wrap width
            status_label.pack(pady=(2, 5))  # Minimal padding

            self.image_widgets[filter_name].append({
                'image': img_label,
                'status': status_label,
                'container': img_container
            })

        # Initial update
        self.update_filter_images(filter_name)

    def create_control_panel(self):
        """Create the bottom control panel with visible buttons."""
        # Make sure the panel is always visible at bottom
        panel = tk.Frame(self.root, bg='#2c3e50', height=100)
        panel.pack(fill='x', side='bottom', anchor='s')
        panel.pack_propagate(False)
        
        # Create two sections: info on left, buttons on right
        left_frame = tk.Frame(panel, bg='#2c3e50')
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(panel, bg='#2c3e50')
        right_frame.pack(side='right', padx=10, pady=10)
        
        # Instructions on the left
        instructions = "💡 TIP: Default values work for most images. Adjust only if needed. Focus on large images above!"
        instruction_label = tk.Label(left_frame, text=instructions, 
                                   font=('Arial', 9), fg='#bdc3c7', bg='#2c3e50',
                                   wraplength=1000, justify='left')
        instruction_label.pack(anchor='w', pady=10)
        
        # Buttons on the right - SIDE BY SIDE ARRANGEMENT
        buttons_frame = tk.Frame(right_frame, bg='#2c3e50')
        buttons_frame.pack(pady=10)
        
        # Save button (left side)
        save_btn = tk.Button(buttons_frame, text="💾 SAVE PARAMETERS", command=self.save_parameters,
                           bg='#27ae60', fg='white', font=('Arial', 10, 'bold'),
                           relief='raised', borderwidth=3, padx=20, pady=8, width=16, height=2,
                           cursor='hand2')
        save_btn.pack(side='left', padx=(0, 10))
        
        # Apply and Close button (right side) - VISIBLE SIDE BY SIDE
        apply_btn = tk.Button(buttons_frame, text="✅ APPLY & CLOSE", command=self.apply_and_close,
                            bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                            relief='raised', borderwidth=3, padx=20, pady=8, width=16, height=2,
                            cursor='hand2')
        apply_btn.pack(side='right')

    def save_parameters(self):
        """Save current parameters to file."""
        try:
            # Collect all current parameters
            current_params = {}
            for filter_name in self.selected_filters:
                current_params[filter_name] = {}
                for param_name, var in self.parameters[filter_name].items():
                    current_params[filter_name][param_name] = var.get()
            
            # Save to JSON file
            import json
            with open('tuned_parameters.json', 'w') as f:
                json.dump(current_params, f, indent=2)
            
            # Show success message
            import tkinter.messagebox as msgbox
            msgbox.showinfo("Success", "✅ Parameters saved successfully to 'tuned_parameters.json'!")
            
        except Exception as e:
            import tkinter.messagebox as msgbox
            msgbox.showerror("Error", f"❌ Failed to save parameters: {str(e)}")

    def apply_and_close(self):
        """Apply current parameters and close the GUI."""
        try:
            # Initialize final_parameters if not exists
            if not hasattr(self, 'final_parameters') or self.final_parameters is None:
                self.final_parameters = {}
                
            # Collect final parameters
            for filter_name in self.selected_filters:
                self.final_parameters[filter_name] = {}
                for param_name, var in self.parameters[filter_name].items():
                    self.final_parameters[filter_name][param_name] = var.get()
            
            # Auto-save before closing
            self.save_parameters()
            
            # Close the GUI
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            import tkinter.messagebox as msgbox
            msgbox.showerror("Error", f"❌ Failed to apply parameters: {str(e)}")

    def on_parameter_change(self, filter_name):
        """Called when a parameter slider changes - with delay to prevent lag."""
        # Cancel any existing timer for this filter
        if filter_name in self.update_delays:
            self.root.after_cancel(self.update_delays[filter_name])
        
        # Set a new timer with 500ms delay (increased from 100ms)
        self.update_delays[filter_name] = self.root.after(500, lambda: self.update_filter_images(filter_name))

    def refresh_filter_images(self, filter_name):
        """Refresh images for a specific filter with completely new random selection."""
        # Load new random images - ensure we get different ones each time
        try:
            all_images = [f for f in os.listdir(self.image_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Always shuffle the entire list to get different images
            random.shuffle(all_images)
            
            if len(all_images) >= self.images_per_filter:
                # Get a fresh random selection
                self.sample_images[filter_name] = all_images[:self.images_per_filter]
            else:
                # If not enough unique images, repeat but still shuffle
                repeated_images = all_images * ((self.images_per_filter // len(all_images)) + 1)
                random.shuffle(repeated_images)
                self.sample_images[filter_name] = repeated_images[:self.images_per_filter]
                
        except Exception as e:
            print(f"Error refreshing images for {filter_name}: {e}")
            return

        self.update_filter_images(filter_name)

    def update_filter_images(self, filter_name):
        """Update all images for a specific filter."""
        if filter_name not in self.sample_images:
            return

        for i, image_name in enumerate(self.sample_images[filter_name]):
            if i >= len(self.image_widgets[filter_name]):
                break

            img_path = os.path.join(self.image_folder, image_name)
            self.update_single_image(filter_name, i, img_path)

    def update_single_image(self, filter_name, index, img_path):
        """Update a single image widget."""
        try:
            # Load and analyze image
            img = cv2.imread(img_path)
            if img is None:
                self.set_image_status(filter_name, index, "Load Error", "red", None)
                return

            # Analyze with current parameters
            is_detected, status_text, analysis_value = self.analyze_image_with_filter(img, filter_name)

            # Create thumbnail - PRESERVE EXACT ORIGINAL ASPECT RATIO
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Get original dimensions
            original_width, original_height = img_pil.size
            original_aspect = original_width / original_height
            
            # Set maximum display size but preserve exact aspect ratio - ULTRA COMPACT
            max_display_width = 220  # Ultra compact to fit all screen sizes
            max_display_height = 150  # Proportionally smaller
            
            # Calculate new size keeping EXACT original aspect ratio
            if original_width > max_display_width or original_height > max_display_height:
                # Need to scale down
                scale_width = max_display_width / original_width
                scale_height = max_display_height / original_height
                scale_factor = min(scale_width, scale_height)
                
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
            else:
                # Image is already small enough, keep original size
                new_width = original_width
                new_height = original_height
            
            # Resize with high quality while preserving exact aspect ratio
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img_pil)

            # Update widget
            widget = self.image_widgets[filter_name][index]
            widget['image'].configure(image=photo)
            widget['image'].image = photo  # Keep reference

            # Update status with more descriptive and user-friendly messages
            color = "#e74c3c" if is_detected else "#27ae60"  # Red if detected, green if ok
            if is_detected:
                status = f"🚫 QUALITY ISSUE\n{filter_name.upper()} detected"
            else:
                status = f"✅ GOOD IMAGE\nPassed {filter_name} check"
            full_status = f"{status}\nScore: {analysis_value:.1f}"
            
            widget['status'].configure(text=full_status, fg=color)
            widget['container'].configure(bg='#ffebee' if is_detected else '#e8f5e8')

        except Exception as e:
            self.set_image_status(filter_name, index, f"Error: {str(e)[:20]}", "red", None)

    def set_image_status(self, filter_name, index, text, color, photo):
        """Set status for an image widget."""
        widget = self.image_widgets[filter_name][index]
        widget['status'].configure(text=text, fg=color)
        if photo:
            widget['image'].configure(image=photo)
            widget['image'].image = photo

    def analyze_image_with_filter(self, img, filter_name):
        """Analyze image with specific filter and current parameters."""
        params = self.parameters[filter_name]
        sensitivity = params['sensitivity'].get()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if filter_name == 'blur':
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            threshold = params['laplacian_threshold'].get() * (1 - sensitivity / 100.0)
            is_detected = laplacian_var < threshold
            status_text = f"Laplacian: {laplacian_var:.1f}\nThreshold: {threshold:.1f}"
            return is_detected, status_text, laplacian_var

        elif filter_name == 'fog':
            std_val = gray.std()
            threshold = params['contrast_threshold'].get() * (1 - sensitivity / 100.0)
            is_detected = std_val < threshold
            status_text = f"Contrast: {std_val:.1f}\nThreshold: {threshold:.1f}"
            return is_detected, status_text, std_val

        elif filter_name == 'dark':
            mean_val = gray.mean()
            threshold = params['brightness_threshold'].get() * (1 + sensitivity / 100.0)
            is_detected = mean_val < threshold
            status_text = f"Brightness: {mean_val:.1f}\nThreshold: {threshold:.1f}"
            return is_detected, status_text, mean_val

        elif filter_name == 'snow':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            white_threshold = params['white_threshold'].get()
            lower_white = np.array([0, 0, white_threshold])
            upper_white = np.array([180, 25, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            white_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
            detection_threshold = 0.1 + (sensitivity / 100) * 0.3
            is_detected = white_ratio > detection_threshold
            status_text = f"White: {white_ratio:.3f}\nThreshold: {detection_threshold:.3f}"
            return is_detected, status_text, white_ratio

        return False, "Unknown filter", 0

    def save_and_close(self):
        """Save parameters and close."""
        self.final_parameters = {}
        for filter_name, params in self.parameters.items():
            self.final_parameters[filter_name] = {}
            for param_name, var in params.items():
                self.final_parameters[filter_name][param_name] = var.get()
        
        self.root.quit()
        self.root.destroy()

    def cancel(self):
        """Cancel without saving."""
        self.final_parameters = None
        self.root.quit()
        self.root.destroy()

    def run(self) -> Optional[Dict[str, Any]]:
        """Run the GUI and return parameters."""
        try:
            self.root.mainloop()
            return self.final_parameters
        except Exception as e:
            print(f"GUI error: {e}")
            return None

def run_parameter_tuning(mode: str = 'normal', selected_filters: List[str] = None, image_folder: str = None) -> Optional[Dict[str, Any]]:
    """Entry point to launch the improved GUI."""
    try:
        if mode == 'quality_control':
            # For quality control mode, use default filters and first available image folder
            if image_folder is None:
                # Find first available image folder
                for folder in ['PhenoCam_Wheat', 'images', 'data']:
                    if os.path.exists(folder):
                        image_folder = folder
                        break
                else:
                    print("⚠️ No image folder found for quality control tuning")
                    return None
            
            if selected_filters is None:
                selected_filters = ['blur', 'brightness', 'contrast']
        
        gui = ImprovedParameterTuningGUI(selected_filters, image_folder)
        return gui.run()
    except Exception as e:
        print(f"Failed to run parameter tuning GUI: {e}")
        return None

def show_roi_preview(image_path: str, rois_data: List[Dict[str, Any]], vegetation_mask: Optional[np.ndarray] = None):
    """Show ROI preview with vegetation extraction and k-means clustered ROIs."""
    try:
        import tkinter as tk
        from tkinter import ttk
        import cv2
        from PIL import Image, ImageTk
        
        # Load the vegetation extraction image
        vegetation_img = cv2.imread(image_path)
        if vegetation_img is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Create preview image with ROI overlays
        preview_img = vegetation_img.copy()
        
        # Draw ROI polygons on the vegetation extraction
        colors = [
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta  
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Green
            (255, 165, 0),    # Orange
            (255, 0, 0),      # Red
            (0, 0, 255),      # Blue
            (128, 0, 128),    # Purple
        ]
        
        for i, roi in enumerate(rois_data):
            roi_id = roi['id']
            color = colors[i % len(colors)]
            
            if roi.get('type') == 'polygon' and 'polygon' in roi:
                # Draw polygon ROI
                polygon_points = roi['polygon']
                if len(polygon_points) >= 3:
                    pts = np.array(polygon_points, dtype=np.int32)
                    
                    # Draw filled polygon with transparency
                    overlay = preview_img.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, preview_img, 0.7, 0, preview_img)
                    
                    # Draw polygon outline
                    cv2.polylines(preview_img, [pts], True, color, 3)
                    
                    # Add ROI number with background
                    center = roi['center']
                    text = str(roi_id)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(preview_img, 
                                 (center[0] - text_size[0]//2 - 5, center[1] - text_size[1]//2 - 10),
                                 (center[0] + text_size[0]//2 + 5, center[1] + text_size[1]//2 + 5),
                                 (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(preview_img, text, 
                              (center[0] - text_size[0]//2, center[1] + text_size[1]//2), 
                              cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        # Create tkinter window for preview
        root = tk.Tk()
        root.title(f"ROI Preview - {len(rois_data)} K-means Clustered ROIs")
        root.geometry("1000x700")
        root.configure(bg='#2c3e50')
        
        # Title frame
        title_frame = tk.Frame(root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text=f"🌿 Vegetation ROI Preview - {len(rois_data)} K-means Clusters", 
                              font=('Arial', 14, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Info frame
        info_frame = tk.Frame(root, bg='#34495e', height=40)
        info_frame.pack(fill='x')
        info_frame.pack_propagate(False)
        
        info_text = f"🔍 Vegetation areas shown in original colors, non-vegetation in grayscale | 🎯 {len(rois_data)} ROI clusters overlaid"
        info_label = tk.Label(info_frame, text=info_text, 
                             font=('Arial', 9), fg='#ecf0f1', bg='#34495e')
        info_label.pack(expand=True)
        
        # Image frame
        image_frame = tk.Frame(root, bg='#2c3e50')
        image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Resize image for display while preserving aspect ratio
        h, w = preview_img.shape[:2]
        max_width, max_height = 900, 500
        
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            preview_img = cv2.resize(preview_img, (new_w, new_h))
        
        # Convert to PIL and display
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(preview_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        image_label = tk.Label(image_frame, image=photo, bg='#2c3e50')
        image_label.pack(expand=True)
        
        # Instructions frame
        instructions_frame = tk.Frame(root, bg='#2c3e50', height=80)
        instructions_frame.pack(fill='x')
        instructions_frame.pack_propagate(False)
        
        instructions_text = "💡 Each colored polygon represents a K-means vegetation cluster\n🔄 Close this window to continue with ROI selection"
        instructions_label = tk.Label(instructions_frame, text=instructions_text, 
                                    font=('Arial', 10), fg='#bdc3c7', bg='#2c3e50',
                                    justify='center')
        instructions_label.pack(expand=True)
        
        # Close button
        button_frame = tk.Frame(root, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        close_btn = tk.Button(button_frame, text="✅ Continue with ROI Selection", 
                             command=root.destroy,
                             bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                             padx=30, pady=10)
        close_btn.pack()
        
        # Keep reference to photo
        root.photo = photo
        
        # Make window modal and centered
        root.transient()
        root.grab_set()
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Show window
        root.mainloop()
        
    except Exception as e:
        print(f"ROI preview error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_filters = ['blur', 'fog', 'dark']
    test_image_folder = 'PhenoCam_Wheat'
    
    if os.path.exists(test_image_folder):
        print("Starting Improved Parameter Tuning GUI...")
        final_params = run_parameter_tuning(test_filters, test_image_folder)
        if final_params:
            print("\nTuning complete. Final parameters:")
            print(json.dumps(final_params, indent=2))
        else:
            print("\nTuning cancelled.")
    else:
        print(f"Error: Test image folder not found at '{test_image_folder}'")
