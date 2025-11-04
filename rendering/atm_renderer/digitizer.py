"""
Enhanced Atmosphere Generator with Automatic Bin Generation
Adds PixelValueMapper class to handle 2D->3D pixel value mapping
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings
import matplotlib.pyplot as plt

class PixelValueMapper:
    """
    Maps pixel values from 2D atmosphere images to 3D rendered sphere outputs.
    
    This class handles the transformation that occurs when PyVista renders a 2D 
    latitude-longitude map onto a 3D sphere, accounting for colormap application
    and lighting effects.
    
    Usage:
        # Calibration phase (do once per rendering setup)
        mapper = PixelValueMapper()
        im2d = model.generate_atmosphere(t=0)
        frame3d = visualizer.render_frame(...)
        mapper.calibrate(im2d, frame3d, background_threshold=60)
        
        # Generation phase (use for all subsequent frames)
        bins = mapper.generate_bins(nbins=10)
        digitized = mapper.digitize(frame3d, nbins=10)
    """
    
    def __init__(self):
        self.value_min_2d = None
        self.value_max_2d = None
        self.value_min_3d = None
        self.value_max_3d = None
        self.background_threshold = None
        self.is_calibrated = False
        
    def calibrate(self, 
                  im2d: np.ndarray, 
                  frame3d: np.ndarray, 
                  background_threshold: Optional[float] = None,
                  percentile_range: Tuple[float, float] = (1, 99)) -> dict:
        """
        Calibrate the mapper by comparing a 2D atmosphere image with its 3D rendering.
        
        Args:
            im2d: 2D atmosphere array from AtmosphereModel.generate_atmosphere()
            frame3d: 3D rendered frame from AtmosphereVisualizer.render_frame()
            background_threshold: Pixel value below which to consider as background.
                                 If None, auto-detected from bimodal distribution.
            percentile_range: Percentile range to use for robust min/max estimation
            
        Returns:
            dict: Calibration statistics including transformation parameters
        """
        # Get 2D image range
        self.value_min_2d = np.percentile(im2d, percentile_range[0])
        self.value_max_2d = np.percentile(im2d, percentile_range[1])
        
        # Detect background threshold if not provided
        if background_threshold is None:
            self.background_threshold = self._detect_background_threshold(frame3d)
        else:
            self.background_threshold = background_threshold
        
        # Get 3D sphere range (excluding background)
        sphere_pixels = frame3d[frame3d > self.background_threshold]
        
        if len(sphere_pixels) == 0:
            raise ValueError("No sphere pixels found above background threshold. "
                           "Try lowering background_threshold parameter.")
        
        self.value_min_3d = np.percentile(sphere_pixels, percentile_range[0])
        self.value_max_3d = np.percentile(sphere_pixels, percentile_range[1])
        
        self.is_calibrated = True
        
        # Compute transformation statistics
        stats = {
            '2d_range': (self.value_min_2d, self.value_max_2d),
            '3d_range': (self.value_min_3d, self.value_max_3d),
            'background_threshold': self.background_threshold,
            'scale_factor': (self.value_max_3d - self.value_min_3d) / 
                          (self.value_max_2d - self.value_min_2d),
            'offset': self.value_min_3d - self.value_min_2d,
            'n_background_pixels': np.sum(frame3d <= self.background_threshold),
            'n_sphere_pixels': len(sphere_pixels)
        }
        
        return stats
    
    def _detect_background_threshold(self, frame3d: np.ndarray) -> float:
        """
        Auto-detect background threshold using histogram analysis.
        Assumes bimodal distribution: low values (background) and high values (sphere).
        """
        # Flatten and get histogram
        flat = frame3d.flatten()
        hist, bin_edges = np.histogram(flat, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find the first local minimum (valley between two peaks)
        # This separates background from sphere
        peaks_found = 0
        for i in range(1, len(hist) - 1):
            # Detect peak
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks_found += 1
            # Find valley after first peak
            if peaks_found >= 1 and hist[i] < hist[i-1] and hist[i] < hist[i+1]:
                threshold = bin_centers[i]
                return threshold
        
        # Fallback: use median of lower half as threshold
        return np.percentile(flat, 30)
    
    def generate_bins(self, nbins: int, include_background: bool = True) -> np.ndarray:
        """
        Generate bin edges for digitization based on calibrated 3D value range.
        
        Args:
            nbins: Number of bins for the sphere region (excluding background)
            include_background: If True, adds background bin as first bin
            
        Returns:
            Array of bin edges suitable for np.digitize()
        """
        if not self.is_calibrated:
            raise RuntimeError("Mapper must be calibrated before generating bins. "
                             "Call calibrate() first.")
        
        # Generate evenly spaced bins across sphere value range
        sphere_bins = np.linspace(self.value_min_3d, self.value_max_3d, nbins + 1)
        
        if include_background:
            # Add background bin at the beginning
            bins = np.concatenate([[0, self.background_threshold], sphere_bins])
        else:
            bins = sphere_bins
        
        return bins
    
    def digitize(self, frame3d: np.ndarray, nbins: int, 
                 include_background: bool = True) -> np.ndarray:
        """
        Digitize a 3D rendered frame into bins.
        
        Args:
            frame3d: 3D rendered frame from AtmosphereVisualizer.render_frame()
            nbins: Number of bins for the sphere region
            include_background: If True, treats background as separate bin
            
        Returns:
            Digitized array with values from 0 to nbins (or nbins+1 if background included)
        """
        bins = self.generate_bins(nbins, include_background)
        digitized = np.digitize(frame3d, bins, right=True)
        return digitized
    
    def predict_3d_value(self, value_2d: float) -> float:
        """
        Predict what a 2D pixel value will become after 3D rendering.
        
        Args:
            value_2d: Pixel value from 2D atmosphere image
            
        Returns:
            Predicted pixel value in 3D rendered output
        """
        if not self.is_calibrated:
            raise RuntimeError("Mapper must be calibrated first.")
        
        # Linear interpolation from 2D range to 3D range
        normalized = (value_2d - self.value_min_2d) / (self.value_max_2d - self.value_min_2d)
        value_3d = self.value_min_3d + normalized * (self.value_max_3d - self.value_min_3d)
        return value_3d
    
    def get_bin_labels(self, nbins: int, include_background: bool = True) -> List[str]:
        """
        Generate human-readable labels for bins.
        
        Args:
            nbins: Number of bins
            include_background: Whether background bin is included
            
        Returns:
            List of string labels for each bin
        """
        bins = self.generate_bins(nbins, include_background)
        labels = []
        
        for i in range(len(bins) - 1):
            if i == 0 and include_background:
                labels.append(f"Background [0-{bins[1]:.0f}]")
            else:
                labels.append(f"Bin {i if not include_background else i-1} "
                            f"[{bins[i]:.0f}-{bins[i+1]:.0f}]")
        
        return labels
    
    def save_calibration(self, filepath: str):
        """Save calibration parameters to file."""
        import json
        
        if not self.is_calibrated:
            raise RuntimeError("No calibration to save.")
        
        data = {
            'value_min_2d': float(self.value_min_2d),
            'value_max_2d': float(self.value_max_2d),
            'value_min_3d': float(self.value_min_3d),
            'value_max_3d': float(self.value_max_3d),
            'background_threshold': float(self.background_threshold),
            'is_calibrated': self.is_calibrated
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_calibration(self, filepath: str):
        """Load calibration parameters from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.value_min_2d = data['value_min_2d']
        self.value_max_2d = data['value_max_2d']
        self.value_min_3d = data['value_min_3d']
        self.value_max_3d = data['value_max_3d']
        self.background_threshold = data['background_threshold']
        self.is_calibrated = data['is_calibrated']


def demonstrate_usage():
    """
    Demonstration of how to use PixelValueMapper in your workflow.
    """
    print("=" * 70)
    print("PixelValueMapper Usage Example")
    print("=" * 70)
    
    # Simulate some data
    print("\n1. Simulating 2D atmosphere and 3D rendering...")
    im2d = np.random.uniform(0.85, 1.05, size=(400, 400))

    # plot the 2d image and histogram
    plt.figure(), plt.imshow(im2d, cmap='inferno')
    plt.figure(), plt.hist(im2d)

    # Simulate 3D rendering: scale and shift values, add background
    frame3d = np.zeros((800, 800))
    # Background (dark pixels)
    frame3d[:200, :] = np.random.uniform(0, 25, size=(200, 800))
    # Sphere (transformed pixel values)
    sphere_region = (im2d - 0.85) / (1.05 - 0.85)  # Normalize to [0, 1]
    sphere_region = 60 + sphere_region * (150 - 60)  # Scale to [60, 150]
    # Place sphere in center of frame
    y_start, x_start = 200, 200
    frame3d[y_start:y_start+400, x_start:x_start+400] = sphere_region

    print(f"   2D image range: [{im2d.min():.3f}, {im2d.max():.3f}]")
    print(f"   3D frame range: [{frame3d.min():.3f}, {frame3d.max():.3f}]")
    
    # Plot the 3d image and history
    plt.figure(), plt.imshow(frame3d, cmap='inferno')
    plt.figure(), plt.hist(frame3d)

    # Create and calibrate mapper
    print("\n2. Calibrating mapper...")
    mapper = PixelValueMapper()
    stats = mapper.calibrate(im2d, frame3d)
    
    print(f"   Background threshold: {stats['background_threshold']:.2f}")
    print(f"   2D range: [{stats['2d_range'][0]:.3f}, {stats['2d_range'][1]:.3f}]")
    print(f"   3D range: [{stats['3d_range'][0]:.2f}, {stats['3d_range'][1]:.2f}]")
    print(f"   Scale factor: {stats['scale_factor']:.2f}")
    
    # Generate bins
    print("\n3. Generating bins...")
    nbins = 10
    bins = mapper.generate_bins(nbins=nbins)
    print(f"   Number of bins: {len(bins) - 1}")
    print(f"   Bin edges: {bins}")
    
    # Get bin labels
    print("\n4. Bin labels:")
    labels = mapper.get_bin_labels(nbins=nbins)
    for label in labels:
        print(f"   {label}")
    
    # Digitize the frame
    print("\n5. Digitizing 3D frame...")
    digitized = mapper.digitize(frame3d, nbins=nbins)
    print(f"   Digitized range: [{digitized.min()}, {digitized.max()}]")
    print(f"   Unique bins used: {np.unique(digitized)}")
    
    # Test prediction
    print("\n6. Testing 2D->3D value prediction...")
    test_values_2d = [0.85, 0.95, 1.05]
    for val_2d in test_values_2d:
        val_3d = mapper.predict_3d_value(val_2d)
        print(f"   2D value {val_2d:.2f} -> 3D value {val_3d:.2f}")
    
    print("\n" + "=" * 70)
    print("Complete! Ready to use in your simulation.")
    print("=" * 70)


# Integration example for your existing code
def integrate_with_simulation():
    """
    Example showing how to integrate PixelValueMapper into your existing workflow.
    """
    print("\nIntegration Example:")
    print("-" * 70)
    print("""
# In your main simulation code, after creating results:

# 1. Create mapper instance
mapper = PixelValueMapper()

# 2. Calibrate using first frame (do this once per simulation)
mesh = SphericalMesh(resolution=400)
model = AtmosphericModel(mesh, atmo_config)
im2d = model.generate_atmosphere(t=0)  # First frame 2D
frame3d = results[inclination]['gray_array'][0]  # First frame 3D

stats = mapper.calibrate(im2d, frame3d, background_threshold=60)
print(f"Calibration stats: {stats}")

# 3. Generate bins for all subsequent frames
nbins = 10
bins = mapper.generate_bins(nbins=nbins)
print(f"Generated bins: {bins}")

# 4. Use in your plot_frames function
def plot_frames_with_mapper(h5_path, inclination, t=0, mapper=None, nbins=10):
    with h5py.File(h5_path, 'r') as f:
        data = f[f'{inclination}/gray_array'][t]
        
        # Use mapper to digitize
        if mapper is not None:
            binned = mapper.digitize(data, nbins=nbins)
        else:
            # Fallback to manual bins
            bins = [0, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
            binned = np.digitize(data, bins, right=True)
        
        # Plot as before
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(data, vmin=0, vmax=150, cmap='inferno')
        axes[1].imshow(binned, cmap='viridis')
        plt.show()

# 5. Save calibration for later use
mapper.save_calibration('calibration.json')

# 6. Load in future runs
mapper_new = PixelValueMapper()
mapper_new.load_calibration('calibration.json')
bins = mapper_new.generate_bins(nbins=10)
    """)
    print("-" * 70)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_usage()
    
    # Show integration example
    # integrate_with_simulation()