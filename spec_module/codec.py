# demo_codec.py
import numpy as np
import json
import os
from datetime import datetime
from collections import Counter
import itertools
from scipy.ndimage import zoom
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

#%%
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

    # ========== Spectral Loading and Composite Calculation ==========
    
    def load_spectrum(self, codec_id):
        """
        Load spectrum from assigned CSV file for a specific codec ID.
        
        Parameters:
            codec_id (int): Codec ID
        
        Returns:
            tuple: (wavelength, flux) as numpy arrays
        
        Raises:
            ValueError: If no spectrum assigned to this codec_id
            FileNotFoundError: If file doesn't exist
        """
        if codec_id not in self.spectra_assignments:
            raise ValueError(f"No spectrum assigned to codec_id {codec_id}")
        
        filepath = self.spectra_assignments[codec_id]
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Spectrum file not found: {filepath}")
        
        # Load CSV (skip header line)
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        
        wavelength = data[:, 0]
        flux = data[:, 1]
        
        return wavelength, flux

    def load_all_spectra(self, verbose=True):
        """
        Load all assigned spectra into memory.
        
        Stores in self.spectra_data = {codec_id: (wavelength, flux), ...}
        
        Parameters:
            verbose (bool): Print loading progress
        """
        if not self.spectra_assignments:
            raise RuntimeError("No spectra assigned. Use assign_spectra() first.")
        
        self.spectra_data = {}
        self.wavelength_grid = None
        
        for codec_id in sorted(self.spectra_assignments.keys()):
            if verbose:
                print(f"Loading spectrum for codec_id {codec_id}...", end=" ")
            
            wavelength, flux = self.load_spectrum(codec_id)
            self.spectra_data[codec_id] = (wavelength, flux)
            
            # Verify wavelength grid consistency
            if self.wavelength_grid is None:
                self.wavelength_grid = wavelength
            else:
                if not np.allclose(wavelength, self.wavelength_grid):
                    raise ValueError(
                        f"Wavelength grid mismatch for codec_id {codec_id}. "
                        f"All spectra must share the same wavelength grid."
                    )
            
            if verbose:
                print(f"✓ ({len(wavelength)} points)")
        
        if verbose:
            print(f"\nLoaded {len(self.spectra_data)} spectra")
            print(f"Wavelength range: {self.wavelength_grid[0]:.3f} - {self.wavelength_grid[-1]:.3f} μm")

    def calculate_composite_spectrum(self, fractions, normalize=False):
        """
        Compute weighted sum of spectra based on fractional areas.
        
        Parameters:
            fractions (dict): {codec_id: fractional_area, ...}
            normalize (bool): If True, normalize fractions to sum to 1.0
        
        Returns:
            tuple: (wavelength, composite_flux) as numpy arrays
        
        Example:
            >>> fractions = codec.calculate_spectral_fractions(composite_image)
            >>> wave, flux = codec.calculate_composite_spectrum(fractions)
        """
        if not hasattr(self, 'spectra_data') or not self.spectra_data:
            raise RuntimeError("No spectra loaded. Call load_all_spectra() first.")
        
        # Handle detailed fractions dict format
        if isinstance(fractions, dict) and 'fractions' in fractions:
            fractions = fractions['fractions']
        
        # Verify all codec_ids have loaded spectra
        missing = set(fractions.keys()) - set(self.spectra_data.keys())
        if missing:
            raise ValueError(f"Missing spectra for codec_ids: {missing}")
        
        # Normalize fractions if requested
        if normalize:
            total = sum(fractions.values())
            if total > 0:
                fractions = {k: v/total for k, v in fractions.items()}
        
        # Initialize composite flux
        composite_flux = np.zeros_like(self.wavelength_grid)
        
        # Weighted sum
        for codec_id, fraction in fractions.items():
            _, flux = self.spectra_data[codec_id]
            composite_flux += fraction * flux
        
        return self.wavelength_grid, composite_flux

    def calculate_composite_spectra_timeseries(self, time_series_results, 
                                            save_dir=None, verbose=True):
        """
        Calculate composite spectra for all frames in a time series.
        
        Parameters:
            time_series_results (list): Output from process_time_series()
            save_dir (str, optional): Directory to save individual spectra
            verbose (bool): Print progress
        
        Returns:
            list: [(frame_idx, wavelength, flux), ...] for each frame
        """
        if not hasattr(self, 'spectra_data') or not self.spectra_data:
            raise RuntimeError("No spectra loaded. Call load_all_spectra() first.")
        
        composite_spectra = []
        
        for i, result in enumerate(time_series_results):
            if verbose and i % 10 == 0:
                print(f"Computing composite spectrum for frame {i+1}/{len(time_series_results)}")
            
            frame_idx = result['frame']
            fractions = result['fractions']
            
            # Calculate composite spectrum
            wavelength, flux = self.calculate_composite_spectrum(fractions)
            
            composite_spectra.append((frame_idx, wavelength, flux))
            
            # Optionally save to file
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                output_file = os.path.join(save_dir, f"composite_frame_{frame_idx:04d}.csv")
                np.savetxt(
                    output_file,
                    np.column_stack((wavelength, flux)),
                    delimiter=",",
                    header="wavelength_um,flux_erg/cm2/s/hz",
                    comments="")
        
        if verbose:
            print(f"Computed {len(composite_spectra)} composite spectra")
            if save_dir:
                print(f"Saved to: {save_dir}")
        
        return composite_spectra

    # ======= Saving and Loading Output Imags and Spectra ======
    def save_codec_images_h5(self, results, filepath):
        """
        Save time series processing results to HDF5 file with compression.
        Parameters:
            results (list): List of dictionaries from process_time_series():
                [
                    {'frame': 0, 'composite': array, 'fractions': {...}},
                    {'frame': 1, 'composite': array, 'fractions': {...}},
                    ...
                ]
            filepath (str): Output file path (should end in .h5)
        """
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, dict):
                return {convert_to_serializable(k): convert_to_serializable(v) 
                        for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['n_frames'] = len(results)
            f.attrs['description'] = 'Time series processing results'
            
            # Save each frame with compression
            for i, result in enumerate(results):
                group = f.create_group(f'frame_{i}')
                
                # Store frame index
                group.create_dataset('frame_index', data=result['frame'])
                
                # Store composite image with compression
                group.create_dataset(
                    'composite',
                    data=result['composite'],
                    compression='gzip',
                    compression_opts=4
                )
                
                # Convert fractions to serializable format and store as JSON string
                fractions_serializable = convert_to_serializable(result['fractions'])
                fractions_json = json.dumps(fractions_serializable)
                group.create_dataset('fractions', data=fractions_json)

    def load_codec_images_h5(filepath, frame_index=None):
        """
        Load time series processing results from HDF5 file.
        
        Parameters:
            filepath (str): Path to HDF5 file
            frame_index (int, optional): If specified, load only this frame
        
        Returns:
            If frame_index is None: list of result dictionaries
            If frame_index is specified: single result dictionary
        """
        def load_single_frame(f, idx):
            """Helper to load a single frame from open file handle"""
            group = f[f'frame_{idx}']
            
            # Load fractions from JSON string
            fractions_json = group['fractions'][()]
            if isinstance(fractions_json, bytes):
                fractions_json = fractions_json.decode('utf-8')
            fractions = json.loads(fractions_json)
            
            return {
                'frame': int(group['frame_index'][()]),
                'composite': group['composite'][:],
                'fractions': fractions
            }
        
        with h5py.File(filepath, 'r') as f:
            if frame_index is not None:
                # Load single frame
                return load_single_frame(f, frame_index)
            else:
                # Load all frames
                n_frames = f.attrs['n_frames']
                results = []
                for i in range(n_frames):
                    results.append(load_single_frame(f, i))
                return results

    def save_composite_spectrum(self, wavelength, flux, filepath):
        """
        Save composite spectrum to CSV file in the same format as input.
        
        Parameters:
            wavelength (array): Wavelength array
            flux (array): Flux array
            filepath (str): Output file path
        """
        np.savetxt(
            filepath,
            np.column_stack((wavelength, flux)),
            delimiter=",",
            header="wavelength_um,flux_erg/cm2/s/hz",
            comments="")

    def save_composite_spectra_timeseries(self, composite_spectra_ts, filepath):
        """
        Save composite spectra time series to HDF5 file with compression.
        Parameters: filepath (str): Output file path (should end in .h5)
        
        The HDF5 file structure:
            - Each frame is stored as a group named 'frame_0', 'frame_1', etc.
            - Each group contains: 'time', 'wavelength', 'flux' datasets

        """
        with h5py.File(filepath, 'w') as f:
            # Store metadata
            f.attrs['n_frames'] = len(composite_spectra_ts)
            f.attrs['description'] = 'Composite spectra time series'
            
            # Save each frame with compression
            for i, (frame_time, wave_array, flux_array) in enumerate(composite_spectra_ts):
                group = f.create_group(f'frame_{i}')
                
                # Store time as scalar
                group.create_dataset('time', data=frame_time)
                # Store wavelength and flux with gzip compression
                group.create_dataset('wavelength', data=wave_array, compression='gzip', compression_opts=4)
                group.create_dataset('flux', data=flux_array, compression='gzip', compression_opts=4)
    
    def load_spectra_ts_h5(filepath, frame_index=None):
        """
        Load composite spectra time series from HDF5 file.
        Parameters:
            filepath (str): Path to HDF5 file
            frame_index (int, optional): If specified, load only this frame
            If frame_index is None: list of [frame_time, wave_array, flux_array]
        """
        with h5py.File(filepath, 'r') as f:
            if frame_index is not None:
                # Load single frame
                group = f[f'frame_{frame_index}']
                return [group['time'][()], group['wavelength'][:], group['flux'][:]
                ]
            else:
                # Load all frames
                n_frames = f.attrs['n_frames']
                composite_spectra_ts = []
                for i in range(n_frames):
                    group = f[f'frame_{i}']
                    composite_spectra_ts.append([group['time'][()], group['wavelength'][:], group['flux'][:]
                    ])
                return composite_spectra_ts

    def get_spectra_info(self):
        """
        Get summary information about loaded spectra.
        Returns:
            dict: Summary statistics
        """
        if not hasattr(self, 'spectra_data') or not self.spectra_data:
            return {"status": "No spectra loaded"}
        
        # Calculate flux statistics for each spectrum
        flux_stats = {}
        for codec_id, (wave, flux) in self.spectra_data.items():
            flux_stats[codec_id] = {
                'mean': float(np.mean(flux)),
                'min': float(np.min(flux)),
                'max': float(np.max(flux)),
                'std': float(np.std(flux))
            }
        
        return {
            'n_spectra': len(self.spectra_data),
            'n_wavelength_points': len(self.wavelength_grid),
            'wavelength_range': (float(self.wavelength_grid[0]), 
                                float(self.wavelength_grid[-1])),
            'flux_statistics': flux_stats
        }

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

# Import your existing readPhotometry function
def readPhotometry(targetPath):
    """Read photometry data: gray_array, specmask, metadata, time_array."""
    results = {}
    with h5py.File(targetPath, 'r') as f:
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
        bins[0] = bins[0]*0.8 # Slightly extend the first bin
        bins[-1] = bins[-1]*1.2 # Slightly extend the last bin
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
    
    runname = 'timeSeries_'+f'1200k_clouds=10'
    if False:
        runname += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    runpath = os.path.join(os.getcwd(), runname)
    if not os.path.exists(runpath):
        os.makedirs(runpath)
    
    # ===== Generate bins and digitize =======
    cl1, cl2 = results['40']['metadata']['colorlim']
    Fpolar_var = results['40']['metadata']['Fpolar_var']
    Fband_var = results['40']['metadata']['Fband_var']
    var = max(Fpolar_var, Fband_var)
    v1, v2 = 1 - var, 1 + var
    slope = 255/(cl2 - cl1)
    intercept = 0 - slope * cl1
    a, b = slope*v1 + intercept, slope*v2 + intercept
    n = 10
    bins = generate_bins(a, b, nbin=n, type='linear')
    digitize_frames(results, bins)
    
    # ===== STEP 1: Initialize Codec System =====
    print("\n### STEP 1: Initialize Codec System ###")
    codec = CodecSystem()
    
    ### Search for .csv files in the specified directory
    dir = '/Users/nguyendat/Documents/GitHub/polarVortexJwst/spec_module/'
    specpath = dir+'bd_grid_20251108_162457_all5condensates'

    csv_files = sorted(glob.glob(f"{specpath}/*.csv"))
    print(f"Found {len(csv_files)} .csv files in {specpath}")

    # Add cloud thickness codec
    codec.add_codec_type('cloud_thickness', max_value=n, 
                        description='Cloud optical depth levels')
    codec.generate_combinations() # Generate all combinations
    
    # ===== STEP 2: Assign Spectra (Manual) =====
    # Use the provided csv_files for assignment
    for i in range(n):
        codec.assign_spectra(i, csv_files[i], validate=True)
    print(f"Assigned {len(codec.spectra_assignments)} spectra files")
    
    print("\n### Loading Spectra ###")
    codec.load_all_spectra(verbose=True) # Load all spectra into memory 

    print("\nCodec combinations:")
    for i in range(n):
        codec.print_combination(i) # Display first few combinations

    # ===== Export/Import Configuration =====
    configPath = os.path.join(runpath, 'codec_config.json')
    codec.export_mapping(configPath)
    # Test import
    codec_loaded = CodecSystem()
    codec_loaded.import_mapping(configPath, validate_files=False)
    
    #%%
    # ===== Process Time Series for All Inclination =====
    print("\n### Processing Time Series ###")
    
    for inclin in results.keys():
        results_ts = None

        # Calculate Composite Spectra for All Frames
        results_ts = codec.process_time_series(
            time_variant_codecs={
                'cloud_thickness': results[inclin]['digitized']
            },
            frames=range(0, len(results[inclin]['digitized'])),
            verbose=False)
        # Save the codecs images time series
        codecPath = os.path.join(runpath, f'codec_images_i={inclin}.h5')
        codec.save_codec_images_h5(results_ts, codecPath)

        # Calculating Composite Spectra Time Series
        composite_spectra_ts = codec.calculate_composite_spectra_timeseries(
            results_ts, verbose=False)

        spectsPath = os.path.join(runpath, f'composite_spectra_i={inclin}.h5')
        # Save the spectra timeseries
        codec.save_composite_spectra_timeseries(composite_spectra_ts, spectsPath)
    
        # ===== Visualization: Compare Individual vs Composite =====
        print("\n### Visualizing Spectra ###")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ===== Plot 1: All individual spectra =====
        ax = axes[0, 0]
        for codec_id in range(n):  # Plot all codec
            wave_i, flux_i = codec.load_spectrum(codec_id)
            ax.plot(wave_i, flux_i, alpha=1, lw=0.5, label=f'Codec {codec_id}')
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux (erg/cm²/s/Hz)')
        ax.set_title('Individual Spectra')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # ===== Plot 2: Light curves for specific wavelength bin =====
        ax = axes[0,1]

        # Define wavelength bins
        wavebin = [(2.0, 2.1, 'red'),
                (2.5, 2.6, 'blue'),]

        # Extract light curves for each bin
        for wmin, wmax, color in wavebin:
            flux_timeseries = []
            time_indices = results[inclin]['time_array']
            
            for frame_i, wave_i, flux_i in composite_spectra_ts:
                # Find wavelength indices in the bin
                mask = (wave_i >= wmin) & (wave_i <= wmax)
                if np.sum(mask) > 0:
                    # Calculate median flux in this bin
                    median_flux = np.median(flux_i[mask])
                    flux_timeseries.append(median_flux)
            
            # Plot light curve
            ax.plot(time_indices, flux_timeseries, 
                    marker='o', linestyle='-', color=color, 
                    linewidth=2, markersize=4,
                    label=f'{wmin}-{wmax} μm')

            # Highlight specific timepoint
            for xid in [30, 60, 90]:
                ax.plot(time_indices[xid], flux_timeseries[xid], ls='', 
                        marker='*', markersize=12, color=color)

        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Median Flux (erg/cm²/s/Hz)')

        ax.set_title(f'Light Curves for Selected Wavelength Bins; i={inclin}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('linear')

        # ===== Plot 3: Fractional contributions =====

        # Process a Single Frame for Reference
        print(f"\n### Processing Single Frame; i={inclin} ###")
        frame_idx = 0
        composite = codec.create_composite_image(
            cloud_thickness=results[inclin]['digitized'][frame_idx]
        )
        fractions = codec.calculate_spectral_fractions(composite, detailed=True)
        # wave, flux = codec.calculate_composite_spectrum(fractions)

        ax = axes[1, 0]
        codec_ids = list(fractions['fractions'].keys())
        frac_values = list(fractions['fractions'].values())
        ax.bar(codec_ids, frac_values)
        ax.set_xlabel('Codec ID')
        ax.set_ylabel('Fractional Area')
        ax.set_title(f'Fractional Contributions (Frame {frame_idx}); i={inclin}')
        ax.grid(True, alpha=0.3, axis='y')

        ### ===== Add inset atm images to axes[1, 1] ======
        inset_ax = ax.inset_axes([0.05, 0.7, 0.17, 0.25])  # [x, y, width, height]
        inset_ax.imshow(results_ts[15]['composite'], cmap='viridis', aspect='auto')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title('Clouds-codec', fontsize=8)

        # ===== Plot 4: Composite spectra evolution over time =====
        ax = axes[1, 1]
        # for i in range(0, len(composite_spectra_ts), 10):  # Every 10th frame
        #     frame_i, wave_i, flux_i = composite_spectra_ts[i]
        #     ax.plot(wave_i, flux_i, alpha=1.0, lw=0.5, label=f'Frame {frame_i}')
        
        frameA, waveA, fluxA = composite_spectra_ts[30]
        frameB, waveB, fluxB = composite_spectra_ts[60]
        frameC, waveC, fluxC = composite_spectra_ts[90]

        ax.plot(waveA, fluxA/fluxB, lw=1, label=f'Frame A / Frame B')
        ax.plot(waveA, fluxA/fluxC, lw=1, label=f'Frame A / Frame C')
        ax.plot(waveA, fluxB/fluxC, lw=1, label=f'Frame B / Frame C')

        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux (erg/cm²/s/Hz)')
        ax.set_title(f'Composite Spectra Ratio; i={inclin}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        handle = f'spectra_analysis_i={inclin}.pdf'
        outpath = os.path.join(runpath, handle)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print("\nSaved visualization to ", handle)
        plt.show()
        plt.close()

        # ===== Summary Statistics =====
        print("\n### Spectra Information ###")
        info = codec.get_spectra_info()
        print(f"Loaded {info['n_spectra']} spectra")
        print(f"Wavelength range: {info['wavelength_range'][0]:.3f} - {info['wavelength_range'][1]:.3f} μm")
        print(f"Number of wavelength points: {info['n_wavelength_points']}")
        
# %%
