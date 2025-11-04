import numpy as np
import json
import os
from datetime import datetime
from collections import Counter
import itertools
from scipy.ndimage import zoom

class CodecSystem:
    """
    A system for managing multi-dimensional codec mappings for spectral analysis.
    
    Handles arbitrary codec value spaces, generates unique IDs for all combinations,
    and manages spectral assignments for fractional area calculations.
    """
    
    def __init__(self):
        """Initialize empty codec registry."""
        self.codec_definitions = {}
        # Structure: {
        #     'codec_name': {
        #         'max_value': int,
        #         'values': list,
        #         'value_to_index': dict,
        #         'index_to_value': dict,
        #         'description': str
        #     }
        # }
        
        self.combination_map = {}
        # Structure: {codec_id: {'codec_name': value, ...}}
        
        self.reverse_combination_map = {}
        # Structure: {(value1, value2, ...): codec_id}
        
        self.spectra_assignments = {}
        # Structure: {codec_id: 'filepath'}
        
        self.codec_order = []
        # List of codec names in insertion order
        
        self._combinations_valid = False
    
    def add_codec_type(self, name, max_value=None, values=None, description=''):
        """
        Register a new codec type with flexible value specification.
        
        Parameters:
            name (str): Codec identifier (e.g., 'cloud_thickness')
            max_value (int, optional): If provided, assumes sequential [1, 2, ..., max_value]
            values (list/array, optional): Explicit list of allowed values (e.g., [100, 150, 200])
            description (str): Human-readable description
        
        Examples:
            >>> codec.add_codec_type('cloud_thickness', max_value=11)
            >>> codec.add_codec_type('feature_type', values=[150, 200, 250])
        """
        # Validation
        if name in self.codec_definitions:
            raise ValueError(f"Codec '{name}' already registered")
        
        if max_value is None and values is None:
            raise ValueError("Must provide either 'max_value' or 'values'")
        
        if max_value is not None and values is not None:
            raise ValueError("Provide only one of 'max_value' or 'values'")
        
        # Generate values list
        if max_value is not None:
            values = list(range(1, max_value + 1))
        else:
            # Clean up provided values
            values = sorted(list(set(values)))
            # Exclude 0 (background) if present
            values = [v for v in values if v != 0]
            
            if len(values) == 0:
                raise ValueError("No valid values after removing background (0)")
        
        # Warn if 0 found in original values
        if values is not None and 0 in values:
            print(f"WARNING: Codec '{name}' contains 0 (background value). "
                  f"This will be excluded from valid codec values.")
        
        # Create bidirectional mappings
        value_to_index = {val: idx for idx, val in enumerate(values)}
        index_to_value = {idx: val for idx, val in enumerate(values)}
        
        # Store definition
        self.codec_definitions[name] = {
            'max_value': len(values),
            'values': values,
            'value_to_index': value_to_index,
            'index_to_value': index_to_value,
            'description': description
        }
        
        # Track order
        self.codec_order.append(name)
        
        # Invalidate combinations (need regeneration)
        self._combinations_valid = False
        
        print(f"Added codec '{name}' with {len(values)} values: {values}")
    
    def generate_combinations(self):
        """
        Generate all possible codec combinations and assign unique IDs.
        
        Uses Cartesian product to create all combinations.
        Total IDs = product of all codec max_values.
        """
        if len(self.codec_definitions) == 0:
            raise ValueError("No codecs registered. Use add_codec_type() first.")
        
        # Extract value lists in order
        codec_names = self.codec_order
        value_lists = [self.codec_definitions[name]['values'] for name in codec_names]
        
        # Generate Cartesian product
        combinations = list(itertools.product(*value_lists))
        
        # Clear existing maps
        self.combination_map = {}
        self.reverse_combination_map = {}
        
        # Assign IDs
        for codec_id, combo in enumerate(combinations):
            # Forward map: ID → combination dict
            combo_dict = {codec_names[i]: combo[i] for i in range(len(codec_names))}
            self.combination_map[codec_id] = combo_dict
            
            # Reverse map: tuple → ID
            self.reverse_combination_map[combo] = codec_id
        
        self._combinations_valid = True
        
        print(f"Generated {len(self.combination_map)} codec combinations")
        print(f"ID range: 0-{len(self.combination_map)-1}")
    
    def assign_spectra(self, codec_id, filepath, validate=True):
        """
        Map a specific codec ID to a spectra file path.
        
        Parameters:
            codec_id (int): The unique combination ID
            filepath (str): Path to CSV file containing spectral data
            validate (bool): If True, check file exists and ID is valid
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() before assigning spectra")
        
        if validate:
            # Check ID exists
            if codec_id not in self.combination_map:
                raise ValueError(
                    f"Invalid codec_id {codec_id}. "
                    f"Valid range: 0-{len(self.combination_map)-1}"
                )
            
            # Check file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Spectra file not found: {filepath}")
        
        # Assign
        self.spectra_assignments[codec_id] = filepath
    
    def assign_spectra_bulk(self, assignments):
        """
        Assign multiple spectra at once.
        
        Parameters:
            assignments (dict): {codec_id: filepath, ...}
        """
        for codec_id, filepath in assignments.items():
            self.assign_spectra(codec_id, filepath, validate=True)
    
    def assign_spectra_pattern(self, pattern, start_id=0, end_id=None):
        """
        Assign spectra using a filename pattern.
        
        Parameters:
            pattern (str): Format string with {id} placeholder
                          e.g., 'spectra/cloud_{id}.csv'
            start_id (int): Starting ID for assignment
            end_id (int): Ending ID (None = all remaining)
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() first")
        
        if end_id is None:
            end_id = len(self.combination_map)
        
        for codec_id in range(start_id, end_id):
            filepath = pattern.format(id=codec_id)
            self.assign_spectra(codec_id, filepath, validate=False)
        
        print(f"Assigned spectra for IDs {start_id}-{end_id-1} using pattern")
    
    def create_composite_image(self, alignment_policy='largest', 
                               warn_threshold=2.0, **codec_arrays):
        """
        Convert multiple codec arrays into a single composite image with unique IDs.
        
        Parameters:
            alignment_policy (str): 'largest', 'smallest', 'most_common', or tuple
            warn_threshold (float): Warn if downsampling by this factor
            **codec_arrays: dict of {codec_name: 2D_array}
        
        Returns:
            composite (2D array): Array of codec IDs (NaN for background)
        
        Example:
            >>> composite = codec.create_composite_image(
            ...     cloud_thickness=results['40']['digitized'][0],
            ...     feature_type=results['40']['specmask']
            ... )
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() first")
        
        # Validate codec names
        for name in codec_arrays.keys():
            if name not in self.codec_definitions:
                raise ValueError(
                    f"Codec '{name}' not registered. Use add_codec_type() first."
                )
        
        # Check all required codecs provided
        for name in self.codec_order:
            if name not in codec_arrays:
                raise ValueError(f"Missing required codec: '{name}'")
        
        # Check and align dimensions
        shapes = [arr.shape for arr in codec_arrays.values()]
        
        if len(set(shapes)) > 1:
            print(f"WARNING: Mismatched dimensions detected: {set(shapes)}")
            target_shape = self._determine_target_shape(shapes, alignment_policy)
            print(f"Aligning all arrays to {target_shape}")
            codec_arrays = self._align_arrays(
                codec_arrays, target_shape, warn_threshold=warn_threshold
            )
        else:
            target_shape = shapes[0]
        
        # Create background mask (where ANY codec has value 0)
        background_mask = np.zeros(target_shape, dtype=bool)
        for name, array in codec_arrays.items():
            background_mask |= (array == 0)
        
        # Initialize composite
        composite = np.full(target_shape, np.nan, dtype=float)
        
        # Encode to codec IDs (optimized vectorized version)
        # Create index arrays for each codec
        index_arrays = []
        for name in self.codec_order:
            values = codec_arrays[name]
            value_to_index = self.codec_definitions[name]['value_to_index']
            
            # Map values to indices
            indices = np.zeros_like(values, dtype=int)
            for val, idx in value_to_index.items():
                indices[values == val] = idx
            index_arrays.append(indices)
        
        # Compute linear index using mixed-radix encoding
        # For [cloud_thickness, feature_type] with max values [11, 3]:
        # ID = cloud_idx * 3 + feature_idx
        max_values = [self.codec_definitions[name]['max_value'] 
                      for name in self.codec_order]
        strides = [1]
        for mv in reversed(max_values[1:]):
            strides.insert(0, strides[0] * mv)
        
        composite_int = np.zeros(target_shape, dtype=int)
        for idx_array, stride in zip(index_arrays, strides):
            composite_int += idx_array * stride
        
        # Convert to float and apply background mask
        composite = composite_int.astype(float)
        composite[background_mask] = np.nan
        
        return composite
    
    def calculate_spectral_fractions(self, composite_image, detailed=False):
        """
        Calculate the fractional area contribution of each codec ID.
        
        Parameters:
            composite_image (2D array): Array of codec IDs (with NaN for background)
            detailed (bool): If True, return additional metadata
        
        Returns:
            dict: {codec_id: fraction, ...} or detailed dict if detailed=True
        """
        # Flatten and remove NaN (background)
        valid_pixels = composite_image[~np.isnan(composite_image)]
        
        if len(valid_pixels) == 0:
            print("WARNING: No valid pixels found (all background)")
            return {} if not detailed else {
                'fractions': {},
                'counts': {},
                'total_pixels': 0,
                'background_pixels': composite_image.size,
                'coverage': 0.0
            }
        
        # Count occurrences of each codec ID
        unique_ids, counts = np.unique(valid_pixels, return_counts=True)
        
        # Total non-background pixels
        total_pixels = len(valid_pixels)
        
        # Calculate fractions
        fractions = {}
        for codec_id, count in zip(unique_ids, counts):
            fractions[int(codec_id)] = count / total_pixels
        
        if detailed:
            return {
                'fractions': fractions,
                'counts': dict(zip(unique_ids.astype(int), counts)),
                'total_pixels': int(total_pixels),
                'background_pixels': int(np.sum(np.isnan(composite_image))),
                'coverage': float(total_pixels / composite_image.size)
            }
        else:
            return fractions
    
    def get_combination(self, codec_id):
        """
        Get codec combination for a single ID.
        
        Parameters:
            codec_id (int): Codec ID
        
        Returns:
            dict: {codec_name: value, ...}
        
        Example:
            >>> codec.get_combination(5)
            {'cloud_thickness': 2, 'feature_type': 200}
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() first")
        
        if codec_id not in self.combination_map:
            raise ValueError(f"Invalid codec_id: {codec_id}")
        
        return self.combination_map[codec_id].copy()
    
    def get_combinations(self, codec_ids):
        """
        Get combinations for multiple IDs.
        
        Parameters:
            codec_ids: array-like of integers
        
        Returns:
            dict: {codec_id: combination_dict, ...}
        """
        return {cid: self.get_combination(cid) for cid in codec_ids}
    
    def get_codec_id(self, **codec_values):
        """
        Reverse lookup: get ID from codec values.
        
        Example:
            >>> codec.get_codec_id(cloud_thickness=5, feature_type=200)
            14
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() first")
        
        # Create tuple in correct order
        try:
            combo_tuple = tuple(codec_values[name] for name in self.codec_order)
        except KeyError as e:
            raise ValueError(f"Missing codec in query: {e}")
        
        if combo_tuple not in self.reverse_combination_map:
            raise ValueError(f"Invalid combination: {codec_values}")
        
        return self.reverse_combination_map[combo_tuple]
    
    def print_combination(self, codec_id, include_spectra=True):
        """
        Human-readable description of a codec ID.
        
        Parameters:
            codec_id (int): Codec ID to describe
            include_spectra (bool): Include assigned spectra path
        """
        combo = self.get_combination(codec_id)
        
        print(f"\nCodec ID {codec_id}:")
        for name, value in combo.items():
            desc = self.codec_definitions[name].get('description', '')
            if desc:
                print(f"  {name}: {value}  ({desc})")
            else:
                print(f"  {name}: {value}")
        
        if include_spectra:
            if codec_id in self.spectra_assignments:
                print(f"  Spectra: {self.spectra_assignments[codec_id]}")
            else:
                print(f"  Spectra: [not assigned]")
    
    def print_all_combinations(self, max_display=20):
        """
        Print all codec combinations (limited display).
        
        Parameters:
            max_display (int): Maximum number to display
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() first")
        
        total = len(self.combination_map)
        print(f"\nTotal combinations: {total}")
        print(f"Displaying first {min(max_display, total)}:\n")
        
        for codec_id in range(min(max_display, total)):
            self.print_combination(codec_id, include_spectra=False)
        
        if total > max_display:
            print(f"\n... and {total - max_display} more")
    
    def export_mapping(self, filepath):
        """
        Save codec configuration to JSON file.
        
        Parameters:
            filepath (str): Output file path
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() before exporting")
        
        export_data = {
            'version': '1.0',
            'codec_definitions': self._serialize_codec_definitions(),
            'codec_order': self.codec_order,
            'spectra_assignments': {str(k): v for k, v in self.spectra_assignments.items()},
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_combinations': len(self.combination_map),
                'assigned_spectra': len(self.spectra_assignments)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported codec mapping to {filepath}")
    
    def import_mapping(self, filepath, validate_files=True):
        """
        Load codec configuration from JSON file.
        
        Parameters:
            filepath (str): Input file path
            validate_files (bool): Check if assigned spectra files exist
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear current state
        self.__init__()
        
        # Restore codec definitions
        for name in data['codec_order']:
            defn = data['codec_definitions'][name]
            self.add_codec_type(
                name=name,
                values=defn['values'],
                description=defn.get('description', '')
            )
        
        # Regenerate combinations
        self.generate_combinations()
        
        # Restore spectra assignments
        for codec_id_str, filepath_spec in data['spectra_assignments'].items():
            codec_id = int(codec_id_str)
            self.assign_spectra(codec_id, filepath_spec, validate=validate_files)
        
        print(f"Loaded codec mapping from {filepath}")
        print(f"  Codec combinations: {len(self.combination_map)}")
        print(f"  Spectra assignments: {len(self.spectra_assignments)}")
    
    def process_time_series(self, time_variant_codecs, time_constant_codecs=None, 
                           frames=None, verbose=True):
        """
        Process multiple time frames with mixed time-variant/constant codecs.
        
        Parameters:
            time_variant_codecs (dict): {codec_name: 3D_array (time, y, x)}
            time_constant_codecs (dict): {codec_name: 2D_array (y, x)}
            frames (list): Frame indices to process (None = all)
            verbose (bool): Print progress
        
        Returns:
            list: [
                {'frame': 0, 'composite': array, 'fractions': {...}},
                {'frame': 1, 'composite': array, 'fractions': {...}},
                ...
            ]
        """
        if not self._combinations_valid:
            raise RuntimeError("Must call generate_combinations() first")
        
        if time_constant_codecs is None:
            time_constant_codecs = {}
        
        # Determine number of frames
        n_frames = next(iter(time_variant_codecs.values())).shape[0]
        if frames is None:
            frames = range(n_frames)
        
        results = []
        for i, frame_i in enumerate(frames):
            if verbose and i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frames)}")
            
            # Combine time-constant and current frame of time-variant
            codec_arrays = time_constant_codecs.copy()
            
            for name, array_3d in time_variant_codecs.items():
                codec_arrays[name] = array_3d[frame_i]
            
            # Process single frame
            composite = self.create_composite_image(**codec_arrays)
            fractions = self.calculate_spectral_fractions(composite, detailed=True)
            
            results.append({
                'frame': frame_i,
                'composite': composite,
                'fractions': fractions
            })
        
        if verbose:
            print(f"Completed processing {len(frames)} frames")
        
        return results
    
    # ========== Helper Methods ==========
    
    def _serialize_codec_definitions(self):
        """Convert codec_definitions to JSON-serializable format."""
        serialized = {}
        for name, defn in self.codec_definitions.items():
            serialized[name] = {
                'max_value': defn['max_value'],
                'values': defn['values'],
                'description': defn['description']
            }
        return serialized
    
    def _determine_target_shape(self, shapes, policy='largest'):
        """
        Determine target shape for mismatched arrays.
        
        Parameters:
            shapes (list): List of tuples
            policy: 'largest', 'smallest', 'most_common', or tuple
        
        Returns:
            tuple: Target (height, width)
        """
        if policy == 'largest':
            return tuple(max(s[i] for s in shapes) for i in range(2))
        
        elif policy == 'smallest':
            return tuple(min(s[i] for s in shapes) for i in range(2))
        
        elif policy == 'most_common':
            return Counter(shapes).most_common(1)[0][0]
        
        elif isinstance(policy, tuple):
            return policy
        
        else:
            raise ValueError(f"Unknown policy: {policy}")
    
    def _align_arrays(self, codec_arrays, target_shape, method='nearest', 
                     warn_threshold=2.0):
        """
        Resize all arrays to target shape.
        
        Parameters:
            codec_arrays (dict): {name: 2D_array}
            target_shape (tuple): (height, width)
            method (str): 'nearest' (order=0) for discrete values
            warn_threshold (float): Warn if downsampling by this factor
        
        Returns:
            dict: Resized arrays
        """
        aligned = {}
        for name, array in codec_arrays.items():
            if array.shape == target_shape:
                aligned[name] = array
                continue
            
            # Calculate zoom factors
            zoom_factors = (target_shape[0] / array.shape[0],
                           target_shape[1] / array.shape[1])
            
            # Warn if significant downsampling
            min_zoom = min(zoom_factors)
            if min_zoom < (1.0 / warn_threshold):
                print(f"  WARNING: Downsampling '{name}' by {1/min_zoom:.1f}x "
                      f"({array.shape} → {target_shape})")
            
            # Resize using nearest neighbor to preserve discrete values
            aligned[name] = zoom(array, zoom_factors, order=0)
        
        return aligned

# demo_codec.py
import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

# Import your existing readPhotometry function
def readPhotometry(targetPath):
    """Read photometry data: gray_array, specmask, metadata, time_array."""
    results = {}
    with h5.File(targetPath, 'r') as f:
        for inclin in f.keys():
            data = f[inclin]
            results[inclin] = {
                'gray_array': data['gray_array'][:],
                'specmask': data['specmask'][:],
                'metadata': json.loads(data['metadata'][()].decode('utf-8')),
                'time_array': data['time_array'][:]
            }
    return results

def generate_bins(a, b, nbin, type='linear', power=2):
    """Generate bins with power-law spacing."""
    if type == 'linear':
        bins = np.linspace(a, b, nbin)
    elif type == 'power':
        t = np.linspace(0, 1, nbin)
        normalized = t ** power
        bins = a + normalized * (b - a)
    return [0] + bins.tolist() 

def digitize_frames(results, bins):
    """Digitize all frames in gray_array for each inclination."""
    for inclin in results.keys():
        gray_array = results[inclin]['gray_array']
        digitized = np.digitize(gray_array, bins, right=True)
        results[inclin]['digitized'] = digitized

# ===== DEMO USAGE =====

if __name__ == "__main__":
    # Load your data
    path = '/Users/nguyendat/Documents/GitHub/polarVortexJwst/rendering/atm_renderer/output/test_discrete.h5'
    results = readPhotometry(path)
    
    # Generate bins and digitize
    cl1, cl2 = results['40']['metadata']['colorlim']
    Fpolar_var = results['40']['metadata']['Fpolar_var']
    Fband_var = results['40']['metadata']['Fband_var']
    var = max(Fpolar_var, Fband_var)
    v1, v2 = 1 - var, 1 + var
    slope = 255/(cl2 - cl1)
    intercept = 0 - slope * cl1
    a, b = slope*v1 + intercept, slope*v2 + intercept
    n = 4
    bins = generate_bins(a, b, nbin=n, type='linear')
    
    digitize_frames(results, bins)
    
    print("=" * 60)
    print("CODEC SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # ===== STEP 1: Initialize Codec System =====
    print("\n### STEP 1: Initialize Codec System ###")
    codec = CodecSystem()
    
    ### Search for .csv files in the specified directory
    specpath = '/Users/nguyendat/Documents/GitHub/polarVortexJwst/spec_module/bd_grid_20251017_150039/'
    # csv_files = glob.glob(f"{specpath}/*.csv")

    print(f"Found {len(csv_files)} .csv files in {specpath}")
    # Add cloud thickness codec
    codec.add_codec_type('cloud_thickness', max_value=n, 
                        description='Cloud optical depth levels')
    
    # Generate all combinations
    codec.generate_combinations()
    
    # ===== STEP 2: Assign Spectra (Manual) =====
    print("\n### STEP 2: Manual Spectra Assignment ###")
    print("For demonstration, we'll create dummy assignments:")
    
    # In real usage, you would do:
    # codec.assign_spectra(0, 'path/to/spectra_cloud_1.csv')
    # codec.assign_spectra(1, 'path/to/spectra_cloud_2.csv')
    # etc.
    
    # For demo, we'll use a pattern (files don't need to exist for demo)
    # for i in range(n):
    #     codec.assign_spectra(i, f'spectra/cloud_{i+1}.csv', validate=False)
    
    # Use the provided csv_files for assignment
    for i in range(n):
        codec.assign_spectra(i, specpath+csv_files[i], validate=True)

    print(f"Assigned {len(codec.spectra_assignments)} spectra files")
    
    # Display first few combinations
    print("\nCodec combinations:")
    for i in range(n):
        codec.print_combination(i)
    
    # ===== STEP 3: Create Composite Image (Single Frame) =====
    print("\n### STEP 3: Create Composite Image ###")
    frame_idx = 0
    
    composite = codec.create_composite_image(
        cloud_thickness=results['40']['digitized'][frame_idx]
    )
    
    print(f"Composite image shape: {composite.shape}")
    print(f"Unique codec IDs: {np.unique(composite[~np.isnan(composite)])}")
    
    # ===== STEP 4: Calculate Spectral Fractions =====
    print("\n### STEP 4: Calculate Spectral Fractions ###")
    fractions = codec.calculate_spectral_fractions(composite, detailed=True)
    
    print(f"\nFractional contributions (frame {frame_idx}):")
    for codec_id, frac in sorted(fractions['fractions'].items()):
        combo = codec.get_combination(codec_id)
        print(f"  ID {codec_id} (cloud={combo['cloud_thickness']}): {frac:.4f}")
    
    print(f"\nTotal non-background pixels: {fractions['total_pixels']}")
    print(f"Coverage: {fractions['coverage']:.2%}")
    
    # ===== STEP 5: Process Time Series =====
    print("\n### STEP 5: Process Time Series (first 10 frames) ###")
    
    results_ts = codec.process_time_series(
        time_variant_codecs={
            'cloud_thickness': results['40']['digitized']
        },
        frames=range(10),
        verbose=False
    )
    
    print(f"Processed {len(results_ts)} frames")
    print("\nFractions for frame 5:")
    for codec_id, frac in sorted(results_ts[5]['fractions']['fractions'].items()):
        print(f"  ID {codec_id}: {frac:.4f}")
    
    # ===== STEP 6: Export/Import Configuration =====
    print("\n### STEP 6: Export Configuration ###")
    codec.export_mapping('codec_config.json')
    
    # Test import
    codec_loaded = CodecSystem()
    codec_loaded.import_mapping('codec_config.json', validate_files=False)
    
    # ===== STEP 7: Visualization =====
    print("\n### STEP 7: Visualization ###")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot original digitized frames
    for i in range(3):
        ax = axes[0, i]
        im = results['40']['digitized'][i*4]
        ax.imshow(im, cmap='viridis')
        ax.set_title(f"Digitized Frame {i*4}")
        ax.axis('off')
    
    # Plot composite codec ID images
    for i in range(3):
        ax = axes[1, i]
        composite_i = results_ts[i*4]['composite']
        im = ax.imshow(composite_i, cmap='gist_rainbow', interpolation='nearest')
        ax.set_title(f"Codec IDs Frame {i*4}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('codec_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'codec_visualization.png'")
    plt.show()
    
    # ===== STEP 8: Summary Statistics =====
    print("\n### STEP 8: Summary Statistics Across Time ###")
    
    # Collect all fractions
    all_fractions = {}
    for ts_result in results_ts:
        for codec_id, frac in ts_result['fractions']['fractions'].items():
            if codec_id not in all_fractions:
                all_fractions[codec_id] = []
            all_fractions[codec_id].append(frac)
    
    print("\nAverage fractional contributions (first 10 frames):")
    total = 0
    for codec_id in sorted(all_fractions.keys()):
        avg_frac = np.mean(all_fractions[codec_id])
        combo = codec.get_combination(codec_id)
        total += avg_frac
        print(f"  ID {codec_id} (cloud={combo['cloud_thickness']}): {avg_frac:.4f}")
    
    if total != 1.0:
        print(f"\nWARNING: Total average fraction = {total:.4f} (should be ~1.0)")
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)