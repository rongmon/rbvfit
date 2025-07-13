#!/usr/bin/env python
"""
rbvfit 2.0 GUI IO Module

Simple file loading using rb_spectrum for all formats.
Includes project save/load functionality.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import datetime

from rbcodes.utils.rb_spectrum import rb_spectrum

try:
    from rbvfit import rb_setline as rb
    HAS_RB_SETLINE = True
except ImportError:
    HAS_RB_SETLINE = False


def load_spectrum_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectrum using rb_spectrum - handles all formats"""
    sp = rb_spectrum.from_file(filename)
    if sp.co_is_set:
        flux=sp.flux.value/sp.co.value 
        error=sp.sig.value/sp.co.value 
    else:
        flux=sp.flux.value
        error=sp.sig.value
    return sp.wavelength.value, flux, error


def load_multiple_files(file_list: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load multiple spectrum files"""
    spectra = {}
    for filename in file_list:
        try:
            wave, flux, error = load_spectrum_file(filename)
            spectra[filename] = {
                'wave': wave, 
                'flux': flux, 
                'error': error,
                'basename': Path(filename).name,
                'wave_original': wave.copy(),  # Keep original for reset
                'flux_original': flux.copy(),
                'error_original': error.copy()
            }
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return spectra


def slice_spectrum(spectra_dict: Dict[str, Dict[str, np.ndarray]], 
                  wave_min: float, wave_max: float) -> Dict[str, Dict[str, np.ndarray]]:
    """Apply wavelength slicing to spectra"""
    sliced_spectra = {}
    for filename, data in spectra_dict.items():
        wave_orig = data['wave_original']
        flux_orig = data['flux_original'] 
        error_orig = data['error_original']
        
        # Apply slice
        mask = (wave_orig >= wave_min) & (wave_orig <= wave_max)
        
        sliced_spectra[filename] = data.copy()
        sliced_spectra[filename].update({
            'wave': wave_orig[mask],
            'flux': flux_orig[mask],
            'error': error_orig[mask]
        })
    
    return sliced_spectra


def reset_spectrum_slice(spectra_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Reset spectra to original full range"""
    reset_spectra = {}
    for filename, data in spectra_dict.items():
        reset_spectra[filename] = data.copy()
        reset_spectra[filename].update({
            'wave': data['wave_original'].copy(),
            'flux': data['flux_original'].copy(), 
            'error': data['error_original'].copy()
        })
    
    return reset_spectra


def detect_ion_from_wavelength(wavelength: float) -> Optional[str]:
    """Auto-detect ion name from transition wavelength"""
    if not HAS_RB_SETLINE:
        return None
    
    try:
        line_info = rb.rb_setline(wavelength, 'closest')
        line_name = line_info['name'][0]
        # Extract ion name (e.g., "MgII 2796" -> "MgII")
        return line_name.split()[0]
    except Exception:
        return None


def get_spectrum_info(spectra_dict: Dict[str, Dict[str, np.ndarray]]) -> str:
    """Get summary info about loaded spectra"""
    info_lines = []
    for filename, data in spectra_dict.items():
        wave = data['wave']
        info_lines.append(f"{data['basename']}: {len(wave)} points, "
                         f"{wave.min():.1f}-{wave.max():.1f} Å")
    return "\n".join(info_lines)


def create_parameter_dataframe(n_components: int) -> pd.DataFrame:
    """Create pandas DataFrame for component parameters"""
    data = {
        'Component': [f"C{i+1}" for i in range(n_components)],
        'N': [13.5] * n_components,  # Default log column density
        'N_err': [2.0] * n_components,
        'b': [15.0] * n_components,  # Default b parameter (km/s)
        'b_err': [10.0] * n_components,
        'v': [0.0] * n_components,   # Default velocity (km/s)
        'v_err': [50.0] * n_components
    }
    return pd.DataFrame(data)


def save_configuration(config, filename: str):
    """Save fit configuration to JSON"""
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else str(config)
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_configuration(filename: str):
    """Load fit configuration from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)


def export_results_csv(results, filename: str):
    """Export fit results to CSV"""
    # TODO: Implement when results structure is defined
    pass


def export_results_latex(results, filename: str):
    """Export results to LaTeX table"""
    # TODO: Implement when results structure is defined
    pass


# ==================== PROJECT SAVE/LOAD FUNCTIONALITY ====================

def save_project_data(project_data: Dict[str, Any], filename: str) -> None:
    """
    Save complete project state to file.
    
    Parameters
    ----------
    project_data : dict
        Complete project state dictionary
    filename : str
        Output filename (.rbv or .json)
        
    Raises
    ------
    IOError
        If file cannot be written
    ValueError
        If project_data is invalid
    """
    # Validate project data
    _validate_project_data(project_data)
    
    # Add metadata
    project_data['saved_at'] = datetime.datetime.now().isoformat()
    project_data['rbvfit_version'] = '2.0'
    
    # Save to file with custom JSON encoder
    with open(filename, 'w') as f:
        json.dump(project_data, f, indent=2, default=_json_serializer)


def load_project_data(filename: str) -> Dict[str, Any]:
    """
    Load complete project state from file.
    
    Parameters
    ----------
    filename : str
        Project file to load (.rbv or .json)
        
    Returns
    -------
    dict
        Complete project state dictionary
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file format is invalid
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Project file not found: {filename}")
    
    # Load and validate
    with open(filename, 'r') as f:
        project_data = json.load(f)
    
    # Validate file format
    _validate_project_file(project_data, filename)
    
    return project_data


def validate_project_file(filename: str) -> Dict[str, Any]:
    """
    Validate project file and return metadata.
    
    Parameters
    ----------
    filename : str
        Project file to validate
        
    Returns
    -------
    dict
        File metadata and validation info
    """
    try:
        project_data = load_project_data(filename)
        
        # Count components
        n_configs = len(project_data.get('configurations', {}))
        n_systems = sum(len(systems) for systems in project_data.get('config_systems', {}).values())
        n_parameters = len(project_data.get('config_parameters', {}))
        has_master_theta = project_data.get('master_theta') is not None
        
        return {
            'valid': True,
            'version': project_data.get('version', 'unknown'),
            'created': project_data.get('created', 'unknown'),
            'saved_at': project_data.get('saved_at', 'unknown'),
            'n_configurations': n_configs,
            'n_systems': n_systems,
            'n_parameters': n_parameters,
            'has_master_theta': has_master_theta,
            'missing_files': check_missing_files(project_data)
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def check_missing_files(project_data: Dict[str, Any]) -> List[str]:
    """Check which data files referenced in project are missing - updated for new format"""
    missing = []
    configurations = project_data.get('configurations', {})
    
    for name, config in configurations.items():
        # Check multiple possible file path fields
        filepath = config.get('filepath', '') or config.get('filename', '')
        basename = config.get('basename', '')
        
        if filepath and not Path(filepath).exists():
            missing.append(basename or Path(filepath).name)
    
    return missing


def serialize_configurations(configurations: Dict[str, Dict]) -> Dict[str, Any]:
    """Serialize configurations with complete file paths and wavelength processing history"""
    serialized = {}
    for name, config in configurations.items():
        config_data = {
            'name': config['name'],
            'fwhm': config['fwhm'],
            'description': config.get('description', ''),
            
            # Multiple file path formats for robust restoration
            'filename': config.get('filename', ''),
            'filepath': str(Path(config.get('filename', '')).resolve()) if config.get('filename') else '',
            'basename': Path(config.get('filename', '')).name if config.get('filename') else '',
            'file_directory': str(Path(config.get('filename', '')).parent) if config.get('filename') else '',
            
            'has_data': config['wave'] is not None,
        }
        
        # Save complete wavelength processing information
        if config['wave'] is not None:
            # Current processed data range
            config_data['current_wavelength_range'] = [float(config['wave'].min()), float(config['wave'].max())]
            config_data['current_data_points'] = len(config['wave'])
            
            # Original data range (before any processing)
            if 'wave_original' in config:
                orig_wave = config['wave_original']
                config_data['original_wavelength_range'] = [float(orig_wave.min()), float(orig_wave.max())]
                config_data['original_data_points'] = len(orig_wave)
            else:
                # If no original saved, current = original
                config_data['original_wavelength_range'] = config_data['current_wavelength_range']
                config_data['original_data_points'] = config_data['current_data_points']
            
            # Wavelength processing history
            config_data['trim_range'] = config.get('trim_range', None)
            config_data['union_info'] = config.get('union_info', None)
            config_data['processing_steps'] = config.get('processing_steps', [])
            config_data['is_trimmed'] = config_data['current_data_points'] != config_data['original_data_points']
            
            # Additional processing metadata
            config_data['wave_step'] = config.get('wave_step', None)
            config_data['spectral_resolution'] = config.get('spectral_resolution', None)
            
        else:
            # No data loaded
            config_data['current_wavelength_range'] = None
            config_data['current_data_points'] = 0
            config_data['original_wavelength_range'] = None  
            config_data['original_data_points'] = 0
            config_data['trim_range'] = None
            config_data['union_info'] = None
            config_data['processing_steps'] = []
            config_data['is_trimmed'] = False
        
        serialized[name] = config_data
    return serialized


def deserialize_configurations(serialized: Dict[str, Any]) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Deserialize configurations with automatic data loading and processing
    
    Returns:
        configurations: Restored configuration dict
        missing_files: List of files that couldn't be loaded
    """
    configurations = {}
    missing_files = []
    
    for name, config_data in serialized.items():
        config = {
            'name': config_data['name'],
            'fwhm': config_data['fwhm'],
            'description': config_data.get('description', ''),
            'filename': config_data.get('filename', ''),
            'filepath': config_data.get('filepath', config_data.get('filename', '')),
            'basename': config_data.get('basename', ''),
            'file_directory': config_data.get('file_directory', ''),
            'wave': None,
            'flux': None,
            'error': None
        }
        
        # Restore processing metadata
        config['trim_range'] = config_data.get('trim_range')
        config['union_info'] = config_data.get('union_info')
        config['processing_steps'] = config_data.get('processing_steps', [])
        config['wave_step'] = config_data.get('wave_step')
        config['spectral_resolution'] = config_data.get('spectral_resolution')
        
        # Try to auto-reload data if file exists
        filepath = config_data.get('filepath', '') or config_data.get('filename', '')
        if filepath and Path(filepath).exists():
            try:
                # Load original spectrum
                wave_orig, flux_orig, error_orig = load_spectrum_file(filepath)
                
                # Store original data
                config['wave_original'] = wave_orig
                config['flux_original'] = flux_orig
                config['error_original'] = error_orig
                
                # Apply processing steps
                wave_proc = wave_orig.copy()
                flux_proc = flux_orig.copy() 
                error_proc = error_orig.copy()
                
                # Apply trim if it was saved
                trim_range = config_data.get('trim_range')
                if trim_range and len(trim_range) == 2:
                    trim_min, trim_max = trim_range
                    mask = (wave_proc >= trim_min) & (wave_proc <= trim_max)
                    wave_proc = wave_proc[mask]
                    flux_proc = flux_proc[mask]
                    error_proc = error_proc[mask]
                
                # Apply other processing steps
                processing_steps = config_data.get('processing_steps', [])
                for step in processing_steps:
                    # Apply each processing step as needed
                    # This can be extended for unions, resampling, etc.
                    pass
                
                # Set final processed data
                config['wave'] = wave_proc
                config['flux'] = flux_proc
                config['error'] = error_proc
                
            except Exception as e:
                print(f"Warning: Could not reload data for {name}: {e}")
                basename = config_data.get('basename', Path(filepath).name)
                missing_files.append(f"{name}: {basename}")
        
        elif filepath:
            # File specified but missing
            basename = config_data.get('basename', Path(filepath).name)
            missing_files.append(f"{name}: {basename}")
        
        configurations[name] = config
    
    return configurations, missing_files



def serialize_parameters(config_parameters: Dict[Tuple, pd.DataFrame]) -> Dict[str, Any]:
    """Serialize parameter DataFrames"""
    serialized = {}
    
    for key, df in config_parameters.items():
        # Convert tuple key to string
        key_str = f"{key[0]}___{key[1]}"  # Use triple underscore as separator
        serialized[key_str] = {
            'data': df.to_dict('records'),  # Convert DataFrame to list of dicts
            'columns': df.columns.tolist()
        }
    return serialized


def deserialize_parameters(serialized: Dict[str, Any]) -> Dict[Tuple, pd.DataFrame]:
    """Deserialize parameter DataFrames"""
    config_parameters = {}
    for key_str, param_data in serialized.items():
        # Convert string key back to tuple
        parts = key_str.split('___')
        if len(parts) == 2:
            key = (parts[0], parts[1])
            df = pd.DataFrame(param_data['data'], columns=param_data['columns'])
            config_parameters[key] = df
            
    return config_parameters


def serialize_master_theta(master_theta: Optional[np.ndarray]) -> Optional[List[float]]:
    """Serialize master theta array"""
    if master_theta is not None:
        return master_theta.tolist()  # Convert numpy array to list
    return None


def deserialize_master_theta(serialized: Optional[List[float]]) -> Optional[np.ndarray]:
    """Deserialize master theta array"""
    if serialized is not None:
        return np.array(serialized)
    return None


def serialize_collection_info(collection_result) -> Optional[Dict[str, Any]]:
    """Serialize minimal collection result info for restoration"""
    if collection_result is None:
        return None
        
    return {
        'n_systems': len(collection_result.master_systems),
        'n_params': len(collection_result.master_theta),
        'n_instruments': len(collection_result.instrument_mappings),
        'system_info': [
            {
                'z': sys.z,
                'ion': sys.ion,
                'transitions': sys.transitions,
                'components': sys.components,
                'source_instrument': sys.source_instrument
            }
            for sys in collection_result.master_systems
        ]
    }


def create_project_summary(project_data: Dict[str, Any]) -> str:
    """Create enhanced project summary with processing info"""
    n_configs = len(project_data.get('configurations', {}))
    n_systems = sum(len(systems) for systems in project_data.get('config_systems', {}).values())
    n_params = len(project_data.get('config_parameters', {}))
    missing_files = check_missing_files(project_data)
    
    # Count processing info
    configs = project_data.get('configurations', {})
    n_with_data = sum(1 for config in configs.values() if config.get('has_data', False))
    n_trimmed = sum(1 for config in configs.values() if config.get('is_trimmed', False))
    
    summary = f"Project Summary:\n"
    summary += f"  Configurations: {n_configs} ({n_with_data} with data)\n"
    summary += f"  Ion systems: {n_systems}\n" 
    summary += f"  Parameter sets: {n_params}\n"
    summary += f"  Master theta: {'Available' if project_data.get('master_theta') else 'Not compiled'}\n"
    
    if n_trimmed > 0:
        summary += f"  Processed data: {n_trimmed} configurations trimmed/filtered\n"
    
    if missing_files:
        summary += f"  ⚠️ Missing files: {len(missing_files)}\n"
        for f in missing_files[:3]:  # Show first 3
            summary += f"    • {f}\n"
        if len(missing_files) > 3:
            summary += f"    • ... and {len(missing_files)-3} more\n"
    
    return summary




# ==================== PRIVATE HELPER FUNCTIONS ====================

def _validate_project_data(project_data: Dict[str, Any]) -> None:
    """Validate project data structure"""
    required_keys = ['version', 'configurations']
    for key in required_keys:
        if key not in project_data:
            raise ValueError(f"Missing required key in project data: {key}")
    
    

def _validate_project_file(project_data: Dict[str, Any], filename: str) -> None:
    """Validate loaded project file"""
    if not isinstance(project_data, dict):
        raise ValueError(f"Invalid project file format: {filename}")
    
    if 'version' not in project_data:
        raise ValueError(f"Missing version in project file: {filename}")
    
    version = project_data['version']
    if version != '2.0':
        # Don't raise error, but will warn in GUI
        pass


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and other objects"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif hasattr(obj, 'isoformat'):  # datetime
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    return str(obj)