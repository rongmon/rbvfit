"""
Core Voigt profile model for rbvfit 2.0.

This module provides the VoigtModel class for evaluating Voigt profiles
with automatic ion parameter tying and multi-group support.

Optimized version with atomic parameter caching and fast evaluation path
for MCMC performance.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy.special import wofz
from astropy.convolution import convolve as astropy_convolve, Gaussian1DKernel, CustomKernel
from dataclasses import dataclass

# Import configuration and parameter management
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager
from rbvfit import rb_setline as rb

# Try to import linetools for COS LSF
try:
    from linetools.spectra.lsf import LSF
    HAS_LINETOOLS = True
except ImportError:
    HAS_LINETOOLS = False


@dataclass
class CachedAtomicData:
    """Container for cached atomic parameters."""
    wavelength: float
    lambda0: float
    f_osc: float
    gamma: float
    
    
@dataclass
class FastLineData:
    """Pre-computed line data for fast evaluation."""
    atomic_data: CachedAtomicData
    z_factor: float  # (1 + z)
    system_idx: int
    ion_name: str
    component_idx: int
    
    
class FastParameterMapping:
    """
    Pre-computed parameter mapping for fast theta conversion.
    
    This replaces the parameter manager in the MCMC hot path with
    direct array operations for maximum performance.
    """
    
    def __init__(self, config: FitConfiguration):
        """Pre-compute all parameter indices and mappings."""
        self.config = config
        
        # Calculate total components across all systems
        self.total_components = 0
        self.component_map = []  # Maps global component index to (sys_idx, ion_name, comp_idx)
        
        global_comp_idx = 0
        for sys_idx, system in enumerate(config.systems):
            for ion_group in system.ion_groups:
                for comp_idx in range(ion_group.components):
                    self.component_map.append((sys_idx, ion_group.ion_name, comp_idx))
                    global_comp_idx += 1
        
        self.total_components = global_comp_idx
        
        # Pre-compute line expansion mapping
        self.line_mappings = []  # Maps each line to its component
        
        for sys_idx, system in enumerate(config.systems):
            for ion_group in system.ion_groups:
                for wavelength in ion_group.transitions:
                    for comp_idx in range(ion_group.components):
                        # Find global component index
                        global_comp_idx = 0
                        for s_idx, sys in enumerate(config.systems):
                            if s_idx == sys_idx:
                                for ig in sys.ion_groups:
                                    if ig.ion_name == ion_group.ion_name:
                                        global_comp_idx += comp_idx
                                        break
                                    else:
                                        global_comp_idx += ig.components
                                break
                            else:
                                for ig in sys.ion_groups:
                                    global_comp_idx += ig.components
                        
                        self.line_mappings.append({
                            'wavelength': wavelength,
                            'z': system.redshift,
                            'system_idx': sys_idx,
                            'ion_name': ion_group.ion_name,
                            'component_idx': comp_idx,
                            'global_component_idx': global_comp_idx
                        })
    
    def theta_to_line_arrays(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Fast conversion of theta to line parameter arrays.
        
        Returns
        -------
        tuple
            N_values, b_values, v_values, line_info arrays
        """
        # Split theta into N, b, v sections
        N_section = theta[0:self.total_components]
        b_section = theta[self.total_components:2*self.total_components]
        v_section = theta[2*self.total_components:3*self.total_components]
        
        # Expand to line parameters using pre-computed mappings
        n_lines = len(self.line_mappings)
        N_values = np.zeros(n_lines)
        b_values = np.zeros(n_lines)
        v_values = np.zeros(n_lines)
        
        for i, line_mapping in enumerate(self.line_mappings):
            comp_idx = line_mapping['global_component_idx']
            N_values[i] = N_section[comp_idx]
            b_values[i] = b_section[comp_idx]
            v_values[i] = v_section[comp_idx]
        
        return N_values, b_values, v_values, self.line_mappings


class VoigtModel:
    """
    Voigt profile model with multi-group support and automatic ion tying.
    
    Optimized version with atomic parameter caching and fast evaluation path.
    This class provides both clean model evaluation for general use and an
    optimized fast path for MCMC performance.
    
    Parameters
    ----------
    config : FitConfiguration
        The fitting configuration defining systems and ions
    FWHM : str or float, optional
        Line spread function specification:
        - float: Gaussian FWHM in pixels (default: '6.5')
        - 'COS': Use HST/COS LSF (requires additional parameters)
    grating : str, optional
        HST grating name if FWHM='COS' (e.g., 'G130M')
    life_position : str, optional
        HST lifetime position if FWHM='COS' (default: '1')
    cen_wave : str, optional
        Central wavelength for COS LSF (e.g., '1300A')
        
    Examples
    --------
    >>> config = FitConfiguration()
    >>> config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    >>> model = VoigtModel(config)
    >>> flux = model.evaluate(theta, wavelength)
    """
    
    def __init__(self, config: FitConfiguration, FWHM: Union[str, float] = '6.5',
                 grating: str = 'G130M', life_position: str = '1', 
                 cen_wave: str = '1300A'):
        """Initialize the Voigt model with atomic parameter caching."""
        self.config = config
        self.config.validate()
        self.param_manager = ParameterManager(config)
        
        # Store LSF parameters
        self.FWHM = FWHM
        self.grating = grating
        self.life_position = life_position
        self.cen_wave = cen_wave
        
        # Set up convolution kernel
        self._setup_kernel()
        
        # OPTIMIZATION: Cache atomic parameters and create fast mapping
        self._cache_atomic_parameters()
        self._setup_fast_mapping()
        
    def _cache_atomic_parameters(self):
        """Cache all atomic parameters as flat arrays during initialization."""
        
        # Collect all lines in order
        line_data = []
        for system in self.config.systems:
            for ion_group in system.ion_groups:
                for wavelength in ion_group.transitions:
                    for comp_idx in range(ion_group.components):
                        line_info = rb.rb_setline(wavelength, 'closest')
                        line_data.append({
                            'lambda0': line_info['wave'][0],
                            'gamma': line_info['gamma'][0],
                            'f_osc': line_info['fval'][0],
                            'z': system.redshift
                        })
        
        # Convert to flat numpy arrays
        self.atomic_lambda0 = np.array([line['lambda0'] for line in line_data])
        self.atomic_gamma = np.array([line['gamma'] for line in line_data])
        self.atomic_f = np.array([line['f_osc'] for line in line_data])
        
        # Redshift factors as array
        self.z_factors = np.array([1.0 + line['z'] for line in line_data])
        
        # Store number of lines
        self.n_lines = len(line_data)
        
    def _setup_fast_mapping(self):
        """Set up fast parameter index arrays."""
        
        # Calculate total components
        self.total_components = sum(
            ion_group.components 
            for system in self.config.systems 
            for ion_group in system.ion_groups
        )
        
        # Create parameter index arrays
        indices = []
        global_comp_idx = 0
        
        for system in self.config.systems:
            for ion_group in system.ion_groups:
                # For each transition in this ion group
                for wavelength in ion_group.transitions:
                    # Each transition uses ALL components of this ion
                    for comp_idx in range(ion_group.components):
                        indices.append(global_comp_idx + comp_idx)
                
                # After processing all transitions, advance to next ion group
                global_comp_idx += ion_group.components
        
        # Convert to numpy arrays for direct indexing
        self.N_indices = np.array(indices)
        self.b_indices = np.array(indices) + self.total_components
        self.v_indices = np.array(indices) + 2 * self.total_components
    
    def _setup_kernel(self):
        """Set up the convolution kernel based on FWHM parameter."""
        if self.FWHM == 'COS':
            if not HAS_LINETOOLS:
                raise ImportError("COS LSF requires linetools package. "
                                  "Install with: pip install linetools")
            # Set up COS LSF
            instr_config = dict(
                name='COS',
                grating=self.grating,
                life_position=self.life_position,
                cen_wave=self.cen_wave
            )
            coslsf = LSF(instr_config)
            _, data = coslsf.load_COS_data()
            self.kernel = CustomKernel(data[self.cen_wave].data)
        else:
            # Gaussian kernel
            fwhm_pixels = float(self.FWHM)
            sigma = fwhm_pixels / 2.355  # Convert FWHM to sigma
            self.kernel = Gaussian1DKernel(stddev=sigma)
    

    def voigt_tau(self, lambda0, gamma, f, N, b, wv):
        """
        Voigt profile calculation that handles both scalars and arrays.
        
        Parameters
        ----------
        lambda0 : float or array
            Rest wavelength(s) in Angstroms
        gamma : float or array
            Damping parameter(s)
        f : float or array
            Oscillator strength(s)
        N : float or array
            Column density(ies) in linear scale
        b : float or array
            Doppler parameter(s) in km/s
        wv : np.ndarray
            Wavelength array in Angstroms (rest frame)
            Shape: (n_wavelength,) or (n_wavelength, n_lines)
            
        Returns
        -------
        np.ndarray
            Optical depth array
        """
        # Use same algorithm as before, but handle array broadcasting
        c = 29979245800.0  # cm/s
        constant = 448898479.507  # sqrt(pi) * e^2 / m_e in cm^3/s^2
        
        # Ensure arrays for vectorization
        lambda0 = np.asarray(lambda0)
        gamma = np.asarray(gamma) 
        f = np.asarray(f)
        N = np.asarray(N)
        b = np.asarray(b)
        
        # Handle broadcasting for multiple lines
        if lambda0.ndim > 0:  # Array inputs
            # Reshape for broadcasting
            lambda0 = lambda0[np.newaxis, :]
            gamma = gamma[np.newaxis, :]
            f = f[np.newaxis, :]
            N = N[np.newaxis, :]
            b = b[np.newaxis, :]
            wv = wv[:, np.newaxis]
        
        # Vectorized calculation (same physics as before)
        b_f = b / lambda0 * 10**13
        a = gamma / (4 * np.pi * b_f)
        
        freq0 = c / lambda0 * 10**8
        freq = c / wv * 10**8
        
        norm_const = constant / (freq0 * b * 10**5)
        
        x = (freq - freq0) / b_f
        H = np.real(wofz(x + 1j * a))
        
        tau = N * f * norm_const * H
        
        # Remove the extra dimension
        if tau.ndim == 3:
            tau = tau.squeeze(axis=1)  # (n_wave, 1, n_lines) → (n_wave, n_lines)
        
        return tau
                

    def evaluate(self, theta: np.ndarray, wavelength: np.ndarray, 
                 return_components: bool = False, return_unconvolved: bool = False,
                 validate_theta: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Evaluate the Voigt model using pre-computed arrays - no mapping overhead.
        """
        # Optional validation
        if validate_theta:
            self.param_manager.validate_theta(theta)
        
        # Direct array indexing - no mapping functions!
        N_linear = 10**theta[self.N_indices]
        b_values = theta[self.b_indices] 
        v_values = theta[self.v_indices]
        
        # Constants
        c = 299792.458  # km/s
        
        # Vectorized redshift calculation
        z_total = self.z_factors * (1 + v_values/c) - 1
        
        # Broadcast wavelength for vectorized calculation
        wave_rest = wavelength[:, np.newaxis] / (1 + z_total[np.newaxis, :])
        
        # Vectorized Voigt calculation for ALL lines simultaneously
        tau_all = self.voigt_tau(
            self.atomic_lambda0, self.atomic_gamma, self.atomic_f,
            N_linear, b_values, wave_rest
        )
        
        # Handle component tracking if needed (keep existing logic for now)
        if return_components:
            # This part can be optimized later - for now just return total
            component_fluxes = []
            component_info = []
        
        # Sum all optical depths
        tau_total = np.sum(tau_all, axis=1)
        
        # Convert to flux
        flux_total = np.exp(-tau_total)
        
        # Apply convolution unless unconvolved requested
        if not return_unconvolved and self.kernel is not None:
            flux_total = astropy_convolve(flux_total, self.kernel, boundary='extend')
        
        # Return based on what's requested
        if return_components:
            return {
                'total': flux_total,
                'components': component_fluxes,
                'component_info': component_info
            }
        else:
            return flux_total
            
    def _get_line_parameters_cached(self, theta: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert theta to line parameters using cached atomic data.
        
        This is an optimized version of param_manager.theta_to_line_parameters()
        that uses cached atomic data instead of calling rb_setline.
        """
        # Use parameter manager for structure but bypass atomic lookups
        line_params = self.param_manager.theta_to_line_parameters(theta)
        
        # The atomic parameters are already cached, so this is reasonably fast
        # In the future, could optimize this further by caching the parameter
        # conversion itself
        
        return line_params
    
    def _create_kernel(self, FWHM: Union[str, float]):
        """
        Create a kernel for the given FWHM specification.
        
        Parameters
        ----------
        FWHM : str or float
            FWHM specification (same as constructor)
            
        Returns
        -------
        Kernel object
        """
        if FWHM == 'COS':
            if not HAS_LINETOOLS:
                raise ImportError("COS LSF requires linetools package")
            # Use model's COS parameters
            instr_config = dict(
                name='COS',
                grating=self.grating,
                life_position=self.life_position,
                cen_wave=self.cen_wave
            )
            coslsf = LSF(instr_config)
            _, data = coslsf.load_COS_data()
            return CustomKernel(data[self.cen_wave].data)
        else:
            # Gaussian kernel
            fwhm_pixels = float(FWHM)
            sigma = fwhm_pixels / 2.355
            return Gaussian1DKernel(stddev=sigma)
    
    def _group_line_parameters(self, line_params: List[Dict[str, Any]]) -> Dict[Tuple, List[Dict[str, Any]]]:
        """
        Group line parameters by unique velocity components.
        
        Parameters
        ----------
        line_params : List[Dict[str, Any]]
            List of line parameters from theta_to_line_parameters
            
        Returns
        -------
        Dict[Tuple, List[Dict[str, Any]]]
            Dictionary mapping component keys to lists of line parameters
        """
        component_groups = {}
        
        for lp in line_params:
            # Create unique key for each velocity component
            comp_key = (lp['system_idx'], lp['ion'], lp['component_idx'])
            
            if comp_key not in component_groups:
                component_groups[comp_key] = []
            
            component_groups[comp_key].append(lp)
        
        return component_groups
    
    def get_info(self) -> str:
        """Get model configuration summary."""
        lines = ["VoigtModel Summary", "=" * 50]
        lines.append(self.config.summary())
        
        if self.FWHM == 'COS':
            lines.append(f"\nLSF: COS {self.grating} LP{self.life_position}")
        else:
            lines.append(f"\nLSF: Gaussian FWHM = {self.FWHM} pixels")
        
        # Add optimization info
        lines.append(f"\nOptimizations:")
        lines.append(f"  Cached atomic parameters: {len(self.atomic_cache)} transitions")
        lines.append(f"  Fast mapping components: {self.fast_mapping.total_components}")
        lines.append(f"  Fast mapping lines: {len(self.fast_mapping.line_mappings)}")
            
        return "\n".join(lines)
    
    def evaluate_unconvolved(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """
        Evaluate model without convolution.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter array
        wavelength : np.ndarray
            Wavelength array in Angstroms
            
        Returns
        -------
        np.ndarray
            Unconvolved flux array
        """
        return self.evaluate(theta, wavelength, FWHM='none')
    
    # Compatibility with v1.0 naming
    def model_flux(self, theta: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """Alias for evaluate() to match v1.0 interface."""
        return self.evaluate(theta, wave)
    
    def model_unconvolved(self, theta: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """Alias for evaluate_unconvolved() to match v1.0 interface."""
        return self.evaluate(theta, wave, FWHM='none')
        
