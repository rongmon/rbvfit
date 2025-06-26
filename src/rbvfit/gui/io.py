#!/usr/bin/env python
"""
rbvfit 2.0 GUI IO Module

Simple file loading using rb_spectrum for all formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

from rbcodes.utils.rb_spectrum import rb_spectrum

try:
    from rbvfit import rb_setline as rb
    HAS_RB_SETLINE = True
except ImportError:
    HAS_RB_SETLINE = False


def load_spectrum_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectrum using rb_spectrum - handles all formats"""
    sp = rb_spectrum.from_file(filename)
    return sp.wavelength.value, sp.flux.value, sp.sig.value


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