"""
Clean VoigtModel implementation for rbvfit 2.0 - Elegant single-model approach.

Each VoigtModel represents a single instrument with specific FWHM.
Multi-instrument fitting achieved by using multiple VoigtModel instances
with shared FitConfiguration.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy.special import wofz
from astropy.convolution import convolve as astropy_convolve, Gaussian1DKernel, CustomKernel

from dataclasses import dataclass
import copy

# Import configuration and parameter management
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager
from rbvfit import rb_setline as rb
from scipy import ndimage

# Try to import linetools for COS LSF
try:
    from linetools.spectra.lsf import LSF
    HAS_LINETOOLS = True
except ImportError:
    HAS_LINETOOLS = False


def mean_fwhm_pixels(FWHM_vel_kms: float, wave_obs_grid: np.ndarray) -> float:
    """
    Convert FWHM from velocity units (km/s) to pixels.
    
    Parameters
    ----------
    FWHM_vel_kms : float
        Full Width at Half Maximum in km/s
    wave_obs_grid : np.ndarray
        Observed wavelength grid in Angstroms
        
    Returns
    -------
    float
        Mean FWHM in pixels
    """
    if np.any(wave_obs_grid <= 0):
        raise ValueError("Wavelength grid must be strictly positive.")
    if len(wave_obs_grid) < 2:
        raise ValueError("Wavelength grid must have at least two points.")

    c_kms = 299792.458  # Speed of light in km/s
    delta_lambda = np.gradient(wave_obs_grid)
    fwhm_lambda = wave_obs_grid * FWHM_vel_kms / c_kms
    fwhm_pixels = fwhm_lambda / delta_lambda
    return np.mean(fwhm_pixels)


def voigt_tau(lambda0: float, gamma: float, f: float, N: float, b: float, 
              wv: np.ndarray) -> np.ndarray:
    """
    Voigt profile optical depth calculation.
    
    Parameters
    ----------
    lambda0 : float
        Rest wavelength in Angstroms.
    gamma : float
        Damping constant (s^-1).
    f : float
        Oscillator strength.
    N : float
        Column density (cm^-2).
    b : float
        Doppler parameter (km/s).
    wv : np.ndarray
        Wavelength array (Angstroms).
        
    Returns
    -------
    tau : np.ndarray
        Optical depth as a function of wavelength.
    """
    c = 2.99792458e10  # speed of light in cm/s
    b_f = b / lambda0 * 1e13  # Doppler width in Hz
    freq0 = c / lambda0 * 1e8  # Hz
    freq = c / wv * 1e8  # Hz
    constant = 448898479.507 / (freq0 * b * 1e5)
    a = gamma / (4 * np.pi * b_f)
    x = (freq - freq0) / b_f
    
    H = wofz(x + 1j * a).real
    tau = N * f * constant * H
    return tau


def _vectorized_voigt_tau(atomic_lambda0: np.ndarray, atomic_gamma: np.ndarray, 
                         atomic_f: np.ndarray, N_linear: np.ndarray, 
                         b_values: np.ndarray, wave_rest: np.ndarray) -> np.ndarray:
    """
    Vectorized Voigt profile calculation for all lines simultaneously.
    
    Parameters
    ----------
    atomic_lambda0 : np.ndarray, shape (n_lines,)
        Rest wavelengths for all lines
    atomic_gamma : np.ndarray, shape (n_lines,)
        Damping parameters for all lines
    atomic_f : np.ndarray, shape (n_lines,)
        Oscillator strengths for all lines
    N_linear : np.ndarray, shape (n_lines,)
        Linear column densities for all lines
    b_values : np.ndarray, shape (n_lines,)
        Doppler parameters for all lines
    wave_rest : np.ndarray, shape (n_lines, n_wavelengths)
        Rest-frame wavelength arrays for each line
        
    Returns
    -------
    np.ndarray, shape (n_lines, n_wavelengths)
        Optical depth arrays for all lines
    """
    # Physical constants
    c = 2.99792458e10  # speed of light in cm/s
    c_freq = 2.99792458e18      # c * 1e8
    atomic_constant = 4.48898479507e3  # 448898479.507 / 1e5
    
    
    # Vectorized calculations - broadcasting across all lines
    # Shape: (n_lines, 1) for broadcasting with wavelengths
    lambda0_bc = atomic_lambda0[:, np.newaxis]
    gamma_bc = atomic_gamma[:, np.newaxis]
    f_bc = atomic_f[:, np.newaxis]
    N_bc = N_linear[:, np.newaxis]
    b_bc = b_values[:, np.newaxis]
    
    # Vectorized frequency calculations
    b_f = b_bc / lambda0_bc * 1e13
    #freq0 = c / lambda0_bc * 1e8
    #freq = c / wave_rest * 1e8
    freq0 = c_freq / lambda0_bc
    freq = c_freq / wave_rest
    
    # Vectorized constants
    constant = atomic_constant / (freq0 * b_bc)#constant = 448898479.507 / (freq0 * b_bc * 1e5)
    
    # Vectorized complex argument calculation
    a = gamma_bc / (4 * np.pi * b_f)
    x = (freq - freq0) / b_f
    z = x + 1j * a
    
    # Vectorized wofz calculation - handles complex arrays natively
    H = wofz(z).real
    
    # Vectorized final calculation
    tau_all = N_bc * f_bc * constant * H
    
    return tau_all


def _evaluate_compiled_model(data_container, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """
    Global function to evaluate compiled Voigt model.
    
    Parameters
    ----------
    data_container : CompiledModelData
        Container with all model data
    theta : np.ndarray
        Parameter array
    wavelength : np.ndarray
        Wavelength array for evaluation
        
    Returns
    -------
    np.ndarray
        Model flux array
    """
    atomic_lambda0 = data_container.atomic_lambda0
    atomic_gamma = data_container.atomic_gamma
    atomic_f = data_container.atomic_f
    z_factors = data_container.z_factors
    N_indices = data_container.N_indices
    b_indices = data_container.b_indices
    v_indices = data_container.v_indices
    n_lines = data_container.n_lines

    # Extract parameters
    N_linear = 10**theta[N_indices]
    b_values = theta[b_indices]
    v_values = theta[v_indices]

    # VECTORIZED CALCULATION: All lines at once
    c = 299792.458  # km/s
    
    # Vectorized z_total calculation - shape: (n_lines,)
    z_total = z_factors * (1 + v_values/c) - 1
    
    # Broadcast wavelength calculation - shape: (n_lines, n_wavelengths)
    z_total_bc = z_total[:, np.newaxis]
    wave_rest = wavelength[np.newaxis, :] / (1 + z_total_bc)

    # Vectorized Voigt calculation for all lines simultaneously
    tau_all_lines = _vectorized_voigt_tau(
        atomic_lambda0, atomic_gamma, atomic_f,
        N_linear, b_values, wave_rest
    )
    
    # Sum optical depths across all lines
    tau_total = np.sum(tau_all_lines, axis=0)

    # Convert to flux
    flux = np.exp(-tau_total)

    # Apply convolution if kernel exists
    kernel = data_container.kernel
    if kernel is not None:
        if isinstance(kernel,Gaussian1DKernel ):
            # Use fast scipy for Gaussian kernels
            flux = ndimage.convolve1d(flux, kernel.array, mode='nearest')
        elif isinstance(kernel, CustomKernel):
            # Use astropy for CustomKernel (COS LSF)
            flux = astropy_convolve(flux, kernel, boundary="extend")
        else:
            # Fallback to astropy for unknown kernel types
            flux = astropy_convolve(flux, data_container.kernel, boundary="extend")


    return flux


@dataclass
class CompiledModelData:
    """
    Data container for compiled model. This class only holds data and is picklable.
    """
    atomic_lambda0: np.ndarray
    atomic_gamma: np.ndarray
    atomic_f: np.ndarray
    z_factors: np.ndarray
    N_indices: np.ndarray
    b_indices: np.ndarray
    v_indices: np.ndarray
    kernel: Optional[Any]
    n_lines: int
    total_components: int


class CompiledVoigtModel:
    """
    Compiled Voigt model that can be pickled for multiprocessing.
    
    This class is a thin wrapper around the data container and delegates
    computation to module-level functions.
    """
    
    def __init__(self, data_container: CompiledModelData):
        """Initialize with data container."""
        self.data = data_container
    
    def model_flux(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """
        Evaluate model flux.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter array
        wavelength : np.ndarray
            Wavelength array
            
        Returns
        -------
        np.ndarray
            Model flux
        """
        return _evaluate_compiled_model(self.data, theta, wavelength)
    
    def __call__(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """Make the object callable for convenience."""
        return self.model_flux(theta, wavelength)
    
    def __getstate__(self):
        """Custom pickling - only pickle the data."""
        return {'data': self.data}
    
    def __setstate__(self, state):
        """Custom unpickling."""
        self.data = state['data']


class VoigtModel:
    """
    Clean Voigt profile model for single instrument.
    
    For multi-instrument fitting, create multiple VoigtModel instances
    with the same FitConfiguration but different FWHM values.
    """
    
    def __init__(self, config: FitConfiguration, FWHM: Union[str, float] = '6.5',
                 grating: str = 'G130M', life_position: str = '1', 
                 cen_wave: str = '1300A'):
        """
        Initialize the Voigt model.
        
        Parameters
        ----------
        config : FitConfiguration
            Fitting configuration (defines physics)
        FWHM : str or float, optional
            FWHM for convolution kernel (defines instrumental response)
        grating : str, optional
            HST grating for COS LSF
        life_position : str, optional
            HST lifetime position for COS LSF
        cen_wave : str, optional
            Central wavelength for COS LSF
        """
        self.config = config
        self.config.validate()
        self.param_manager = ParameterManager(config)
        
        # Store LSF parameters
        self.FWHM = config.instrumental_params.get('FWHM', FWHM)
        self.grating = config.instrumental_params.get('grating', grating)
        self.life_position = config.instrumental_params.get('life_position', life_position)
        self.cen_wave = config.instrumental_params.get('cen_wave', cen_wave)
        
        # Set up convolution kernel
        self._setup_kernel()
        
        # Pre-compute arrays for fast evaluation
        self._cache_atomic_parameters()
        self._setup_fast_mapping()
        
        # Compilation state
        self._compiled = False
    
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
        if self.FWHM is None:
            # No convolution - return unconvolved line profiles
            self.kernel = None
        elif self.FWHM == 'COS':
            if not HAS_LINETOOLS:
                raise ImportError("COS LSF requires linetools package")
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
            fwhm_pixels = float(self.FWHM)
            sigma = fwhm_pixels / 2.355
            self.kernel = Gaussian1DKernel(stddev=sigma)

    def compile(self, verbose: bool = False) -> CompiledVoigtModel:
        """
        Compile the model for fast evaluation. Returns a picklable object.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print compilation information
            
        Returns
        -------
        CompiledVoigtModel
            Compiled model that can be pickled and used with multiprocessing
        """
        if verbose:
            structure = self.config.get_parameter_structure()
            print(f"Compiling VoigtModel: {structure['total_parameters']} parameters, {self.n_lines} lines")
            print(f"FWHM: {self.FWHM}")
        
        # Create data container with all necessary information
        data_container = CompiledModelData(
            atomic_lambda0=self.atomic_lambda0.copy(),
            atomic_gamma=self.atomic_gamma.copy(),
            atomic_f=self.atomic_f.copy(),
            z_factors=self.z_factors.copy(),
            N_indices=self.N_indices.copy(),
            b_indices=self.b_indices.copy(),
            v_indices=self.v_indices.copy(),
            kernel=copy.deepcopy(self.kernel),
            n_lines=self.n_lines,
            total_components=self.total_components
        )
        
        # Return compiled model
        compiled_model = CompiledVoigtModel(data_container)
        self._compiled = True
        
        if verbose:
            print(f"✓ VoigtModel compiled successfully")
        
        return compiled_model    
    
    def evaluate(self, theta: np.ndarray, wavelength: np.ndarray, 
                 return_components: bool = False, return_unconvolved: bool = False,
                 validate_theta: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Evaluate the Voigt model - flexible version for analysis.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter array
        wavelength : np.ndarray
            Wavelength array for evaluation
        return_components : bool, optional
            Return component breakdown (default: False)
        return_unconvolved : bool, optional
            Return unconvolved profile (default: False)
        validate_theta : bool, optional
            Validate parameter array (default: False)
            
        Returns
        -------
        np.ndarray or dict
            Model flux or detailed results if return_components=True
        """
        if validate_theta:
            self.param_manager.validate_theta(theta)
        
        # Direct evaluation using the global function
        data_container = CompiledModelData(
            atomic_lambda0=self.atomic_lambda0,
            atomic_gamma=self.atomic_gamma,
            atomic_f=self.atomic_f,
            z_factors=self.z_factors,
            N_indices=self.N_indices,
            b_indices=self.b_indices,
            v_indices=self.v_indices,
            kernel=None if return_unconvolved else self.kernel,
            n_lines=self.n_lines,
            total_components=self.total_components
        )
        
        flux_total = _evaluate_compiled_model(data_container, theta, wavelength)
        
        if return_components:
            return {
                'total': flux_total,
                'components': [],
                'component_info': []
            }
        else:
            return flux_total
    
    @property
    def is_compiled(self) -> bool:
        """Check if model is compiled for fast evaluation."""
        return self._compiled
    
    def get_info(self) -> str:
        """Get model configuration summary."""
        lines = ["VoigtModel Summary", "=" * 50]
        lines.append(self.config.summary())
        
        if self.FWHM == 'COS':
            lines.append(f"\nLSF: COS {self.grating} LP{self.life_position}")
        else:
            lines.append(f"\nLSF: Gaussian FWHM = {self.FWHM} pixels")
        
        lines.append(f"\nModel state:")
        lines.append(f"  Compiled: {self._compiled}")
        lines.append(f"  Lines: {self.n_lines}")
        lines.append(f"  Components: {self.total_components}")
            
        return "\n".join(lines)

    def print_info(self) -> None:
        """Print model configuration summary."""
        print(self.get_info())

    def show_structure(self) -> None:
        """Display ASCII diagram of model parameter structure."""
        lines = []
        lines.append("Model Structure Diagram")
        lines.append("=" * 50)
        
        # Get parameter structure
        structure = self.param_manager.config_to_theta_structure()
        
        # Header with totals
        lines.append(f"📊 Total: {structure['total_parameters']} parameters, {self.n_lines} lines, {len(self.config.systems)} system(s)")
        lines.append(f"🔧 FWHM: {self.FWHM} pixels")
        lines.append("")
        
        # Track global parameter indices
        global_param_idx = 0
        
        for sys_idx, system in enumerate(self.config.systems):
            # System header
            lines.append(f"🌌 System {sys_idx + 1} (z = {system.redshift:.6f})")
            lines.append("│")
            
            system_params = 0
            system_lines = 0
            
            for ion_idx, ion_group in enumerate(system.ion_groups):
                is_last_ion = (ion_idx == len(system.ion_groups) - 1)
                ion_prefix = "└── " if is_last_ion else "├── "
                
                # Ion group header
                ion_params = ion_group.components * 3  # N, b, v for each component
                ion_lines = len(ion_group.transitions) * ion_group.components
                
                lines.append(f"│{ion_prefix}🧪 {ion_group.ion_name} ({ion_group.components} components)")
                
                # Transitions
                transition_str = ", ".join([f"{w:.1f}Å" for w in ion_group.transitions])
                lines.append(f"│{'    ' if is_last_ion else '│   '}📡 Transitions: [{transition_str}]")
                
                # Parameter structure
                lines.append(f"│{'    ' if is_last_ion else '│   '}📋 Parameters ({ion_params} total):")
                
                # Show parameter indices for each component
                for comp in range(ion_group.components):
                    comp_suffix = "" if is_last_ion else "│   "
                    
                    # Parameter indices
                    N_idx = global_param_idx + comp
                    b_idx = global_param_idx + comp + self.total_components
                    v_idx = global_param_idx + comp + 2 * self.total_components
                    
                    lines.append(f"│{'    ' if is_last_ion else '│   '}  • Component {comp + 1}: "
                                f"θ[{N_idx}]=N, θ[{b_idx}]=b, θ[{v_idx}]=v")
                
                # Ion tying indicator
                if len(ion_group.transitions) > 1:
                    lines.append(f"│{'    ' if is_last_ion else '│   '}🔗 TIED: All {len(ion_group.transitions)} transitions share parameters")
                
                # Line count
                lines.append(f"│{'    ' if is_last_ion else '│   '}📈 Lines: {ion_lines} ({len(ion_group.transitions)} trans × {ion_group.components} comp)")
                
                if not is_last_ion:
                    lines.append("│")
                
                # Update counters
                global_param_idx += ion_group.components
                system_params += ion_params
                system_lines += ion_lines
            
            # System summary
            lines.append("│")
            lines.append(f"└── 📊 System {sys_idx + 1} totals: {system_params} parameters, {system_lines} lines")
            
            if sys_idx < len(self.config.systems) - 1:
                lines.append("")
        
        lines.append("")
        lines.append("Legend:")
        lines.append("🌌 = Absorption system    🧪 = Ion group")
        lines.append("📡 = Transitions          📋 = Parameters")
        lines.append("📈 = Model lines          🔗 = Parameter tying")
        lines.append("θ[i] = Parameter index in theta array")
        
        # Parameter array structure
        lines.append("")
        lines.append("Parameter Array Structure:")
        lines.append("-" * 30)
        lines.append(f"θ[0:{self.total_components}]     = All N values")
        lines.append(f"θ[{self.total_components}:{2*self.total_components}]    = All b values")
        lines.append(f"θ[{2*self.total_components}:{3*self.total_components}]    = All v values")
        
        print("\n".join(lines))


if __name__ == "__main__":
    print("rbvfit 2.0 VoigtModel - Clean Elegant Implementation")
    print("Usage:")
    print("  config = FitConfiguration()")
    print("  model1 = VoigtModel(config, FWHM='2.2')  # Instrument 1")
    print("  model2 = VoigtModel(config, FWHM='4.0')  # Instrument 2")
    print("Happy fitting! 🎉")