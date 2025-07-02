"""
Enhanced results management for rbvfit 2.0 with simplified architecture.

This module provides the FitResults class for managing MCMC fitting results
with HDF5 persistence, analysis capabilities, and enhanced plotting.

Phase I: Core functionality (save/load, parameter summary)
Phase II: Analysis (convergence diagnostics, correlation matrix, corner plots)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import h5py
import json
from pathlib import Path
import warnings
from dataclasses import dataclass

# Core dependencies
import matplotlib.pyplot as plt

from scipy.stats import chi2

# Optional dependencies with fallbacks
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    corner = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


@dataclass
class ParameterSummary:
    """Container for parameter summary statistics."""
    names: List[str]
    best_fit: np.ndarray
    errors: np.ndarray
    percentiles: Dict[str, np.ndarray]  # 16th, 50th, 84th percentiles
    mean: np.ndarray
    std: np.ndarray
    
    @property
    def errors_lower(self) -> np.ndarray:
        """Lower error bars (best_fit - 16th percentile)."""
        return self.best_fit - self.percentiles['16th']
    
    @property
    def errors_upper(self) -> np.ndarray:
        """Upper error bars (84th percentile - best_fit)."""
        return self.percentiles['84th'] - self.best_fit


class FitResults:
    """
    Enhanced container for rbvfit 2.0 MCMC fitting results.
    
    This class provides analysis capabilities, HDF5 persistence, and enhanced
    plotting methods for absorption line fitting results.
    
    Attributes
    ----------
    fitter : vfit
        The MCMC fitter object (contains sampler, data, bounds, etc.)
    model : VoigtModel
        The fitted model with configuration
    """
    
    def __init__(self, fitter, model=None):
        """
        Initialize fit results from MCMC fitter.
        
        Parameters
        ----------
        fitter : vfit
            MCMC fitter object
        model : VoigtModel, optional
            Model object for evaluation. If None, model evaluation will be skipped.
        """
        # Store original objects
        self.fitter = fitter
        self.model = model
        
        # Extract basic info
        self.sampler_name = getattr(fitter, 'sampler_name', 'unknown')
        self.n_walkers = getattr(fitter, 'no_of_Chain', None)
        self.n_steps = getattr(fitter, 'no_of_steps', None)
        
        # Multi-instrument detection
        self.is_multi_instrument = getattr(fitter, 'multi_instrument', False)
        self.instrument_data = getattr(fitter, 'instrument_data', None)
        
        # Extract bounds if available
        self.bounds_lower = getattr(fitter, 'lb', None)
        self.bounds_upper = getattr(fitter, 'ub', None)
        
        # Initialize cache variables
        self._cached_param_summary = None
        self._cached_correlation = None
        self._cached_convergence = None
        self._cached_samples = None
        
        # Get best-fit parameters
        try:
            summary = self.parameter_summary(verbose=False)
            self.best_theta = summary.best_fit
        except:
            self.best_theta = getattr(fitter, 'best_theta', fitter.theta)
        
        # NEW: Pre-compute model evaluations if model is available
        self.model_evaluations = {}
        self.component_evaluations = {}
        self.config_metadata = None
        
        self.model_evaluations = {}
        self.component_evaluations = {}
        self.config_metadata = None

        if model is not None:
            self._compute_model_evaluations()
            self._extract_config_metadata()        
    def _compute_model_evaluations(self):
        """Pre-compute model evaluations for all instruments."""
        if self.model is None:
            return
            
        try:
            if self.is_multi_instrument and self.instrument_data:
                # Multi-instrument: evaluate each instrument's model
                for instrument_name, inst_data in self.instrument_data.items():
                    wave = inst_data['wave']
                    model_func = inst_data['model']
                    
                    try:
                        # Total model flux
                        total_flux = model_func(self.best_theta, wave)
                        
                        # Try to get components (rbvfit 2.0 feature)
                        components_data = None
                        try:
                            if hasattr(self.model, 'evaluate'):
                                # Check if this is a VoigtModel that supports     components
                                result = model_func(self.best_theta, wave,     return_components=True)
                                if isinstance(result, dict) and 'components' in     result:
                                    components_data = {
                                        'components': result['components'],
                                        'component_info':     result.get('component_info', []),
                                        'total': result['total']
                                    }
                        except:
                            # Model doesn't support components, that's fine
                            pass
                        
                        self.model_evaluations[instrument_name] = {
                            'wave': wave.copy(),
                            'total_flux': total_flux,
                            'has_components': components_data is not None
                        }
                        
                        if components_data is not None:
                            self.component_evaluations[instrument_name] =     components_data
                            
                    except Exception as e:
                        print(f"Warning: Could not evaluate model for     {instrument_name}: {e}")
            else:
                # Single instrument
                wave = self.fitter.wave_obs
                
                try:
                    # Total model flux
                    if hasattr(self.model, 'evaluate'):
                        total_flux = self.model.evaluate(self.best_theta, wave)
                        
                        # Try components
                        try:
                            result = self.model.evaluate(self.best_theta, wave,     return_components=True)
                            if isinstance(result, dict) and 'components' in result:
                                components_data = {
                                    'components': result['components'],
                                    'component_info': result.get('component_info',     []),
                                    'total': result['total']
                                }
                                self.component_evaluations['main'] = components_data
                        except:
                            pass
                    else:
                        # Legacy v1.0 model
                        total_flux = self.model(self.best_theta, wave)
                    
                    self.model_evaluations['main'] = {
                        'wave': wave.copy(),
                        'total_flux': total_flux,
                        'has_components': 'main' in self.component_evaluations
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not evaluate model: {e}")
                    
        except Exception as e:
            print(f"Warning: Model evaluation failed: {e}")

    @property
    def instrument_names(self) -> List[str]:
        """Get list of available instrument names."""
        if self.is_multi_instrument and self.instrument_data:
            # Remove 'main' from the list - it's redundant
            names = [name for name in self.instrument_data.keys() if name != 'main']
            return names if names else ['primary']
        else:
            return ['primary']

    
    def _extract_config_metadata(self):
        """Extract configuration metadata for model reconstruction."""
        if self.model is None:
            return
            
        try:
            if hasattr(self.model, 'config'):
                # rbvfit 2.0 VoigtModel
                config = self.model.config
                
                self.config_metadata = {
                    'rbvfit_version': '2.0',
                    'systems': [],
                    'instrumental_params': {
                        'FWHM': getattr(self.model, 'FWHM', None),
                        'grating': getattr(self.model, 'grating', None),
                        'life_position': getattr(self.model, 'life_position', None),
                        'cen_wave': getattr(self.model, 'cen_wave', None)
                    }
                }
                
                # Extract ion systems
                for system in config.systems:
                    system_data = {
                        'redshift': system.redshift,
                        'ion_groups': []
                    }
                    
                    for ion_group in system.ion_groups:
                        ion_data = {
                            'ion': ion_group.ion_name,
                            'transitions': list(ion_group.transitions),
                            'components': ion_group.components,
                            'tied_params': getattr(ion_group, 'tied_params', None)
                        }
                        system_data['ion_groups'].append(ion_data)
                    
                    self.config_metadata['systems'].append(system_data)
                    
            else:
                # rbvfit 1.0 or unknown model
                self.config_metadata = {
                    'rbvfit_version': '1.0',
                    'note': 'Limited metadata for v1.0 model'
                }
                
        except Exception as e:
            print(f"Warning: Could not extract config metadata: {e}")
            self.config_metadata = None    
        
    def _validate_inputs(self):
        """Validate that inputs are compatible with rbvfit 2.0."""
        # Check fitter has required attributes
        required_fitter_attrs = ['sampler', 'theta', 'wave_obs', 'fnorm', 'enorm']
        for attr in required_fitter_attrs:
            if not hasattr(self.fitter, attr):
                raise ValueError(f"Fitter missing required attribute: {attr}")
        
        # Check model is v2.0 VoigtModel
        required_model_attrs = ['config', 'evaluate']
        for attr in required_model_attrs:
            if not hasattr(self.model, attr):
                raise ValueError(f"Model appears to be v1.0, not v2.0. Missing: {attr}")
        
        # Check if MCMC was actually run
        if not hasattr(self.fitter, 'sampler') or self.fitter.sampler is None:
            raise ValueError("MCMC has not been run. No sampler found.")
    
    def _extract_mcmc_info(self):
        """Extract basic MCMC information from fitter."""
        self.n_walkers = getattr(self.fitter, 'no_of_Chain', 0)
        self.n_steps = getattr(self.fitter, 'no_of_steps', 0)
        self.sampler_name = getattr(self.fitter, 'sampler_name', 'emcee')
        
        # Extract bounds
        self.bounds_lower = getattr(self.fitter, 'lb', None)
        self.bounds_upper = getattr(self.fitter, 'ub', None)
        
        # Extract datasets info
        self.is_multi_instrument = getattr(self.fitter, 'multi_instrument', False)
        if self.is_multi_instrument:
            self.instrument_data = getattr(self.fitter, 'instrument_data', {})
        else:
            self.instrument_data = None
    
    def _get_samples(self, burnin_fraction: float = 0.2, thin: int = 1) -> np.ndarray:
        """
        Extract samples from sampler with caching.
        
        Parameters
        ----------
        burnin_fraction : float
            Fraction of chain to discard as burn-in
        thin : int
            Thinning factor for samples
            
        Returns
        -------
        np.ndarray
            MCMC samples array (n_samples, n_params)
        """
        # Initialize cache if not present (for loaded objects)
        if not hasattr(self, '_cached_samples'):
            self._cached_samples = None
            
        if self._cached_samples is not None:
            return self._cached_samples
        
        # Auto-detect burn-in based on autocorrelation if possible
        try:
            burnin = self._estimate_burnin()
        except:
            burnin = int(self.n_steps * burnin_fraction)
        
        # Extract samples using fitter method (borrowed from vfit_mcmc)
        try:
            if hasattr(self.fitter.sampler, 'get_chain'):
                # emcee or zeus with get_chain method
                try:
                    samples = self.fitter.sampler.get_chain(
                        discard=burnin, thin=thin, flat=True
                    )
                except TypeError:
                    # Older versions might not support these parameters
                    chain = self.fitter.sampler.get_chain()
                    samples = chain[burnin::thin].reshape(-1, chain.shape[-1])
            else:
                # Fallback for other samplers
                chain = self.fitter.sampler.chain
                samples = chain[:, burnin::thin, :].reshape(-1, chain.shape[-1])
                
        except Exception as e:
            raise RuntimeError(f"Could not extract samples from sampler: {e}")
        
        self._cached_samples = samples
        return samples
    
    
    def _estimate_burnin(self) -> int:
        """Estimate burn-in length based on autocorrelation time."""
        try:
            if hasattr(self.fitter.sampler, 'get_autocorr_time'):
                tau = self.fitter.sampler.get_autocorr_time()
                mean_tau = np.nanmean(tau)
                
                if np.isfinite(mean_tau) and mean_tau > 0:
                    # Use 3x autocorrelation time, but cap at 40% of chain
                    burnin = min(int(3 * mean_tau), int(0.4 * self.n_steps))
                    return max(burnin, int(0.1 * self.n_steps))  # Minimum 10%
            
        except Exception:
            pass
        
        # Fallback to 20% of chain
        return int(0.2 * self.n_steps)


    def get_model_flux(self, instrument_name: str = None) -> Dict[str, np.ndarray]:
        """
        Get pre-computed model flux for an instrument.
        
        Parameters
        ----------
        instrument_name : str, optional
            Instrument name. If None, returns primary/main instrument.
            
        Returns
        -------
        dict
            Dictionary with 'wave' and 'flux' arrays
        """
        if not self.model_evaluations:
            raise ValueError("No pre-computed model evaluations available")
        
        # Determine instrument
        if instrument_name is None:
            if 'main' in self.model_evaluations:
                instrument_name = 'main'
            else:
                instrument_name = list(self.model_evaluations.keys())[0]
        
        if instrument_name not in self.model_evaluations:
            available = list(self.model_evaluations.keys())
            raise ValueError(f"Instrument '{instrument_name}' not found. Available:     {available}")
        
        eval_data = self.model_evaluations[instrument_name]
        return {
            'wave': eval_data['wave'],
            'flux': eval_data['total_flux']
        }
    
    def get_component_flux(self, instrument_name: str = None) -> Dict[str,     np.ndarray]:
        """
        Get individual component contributions for an instrument.
        
        Parameters
        ----------
        instrument_name : str, optional
            Instrument name. If None, returns primary/main instrument.
            
        Returns
        -------
        dict
            Dictionary with 'wave', 'components', and 'component_info'
        """
        if not self.component_evaluations:
            raise ValueError("No component evaluations available")
        
        # Determine instrument
        if instrument_name is None:
            if 'main' in self.component_evaluations:
                instrument_name = 'main'
            else:
                instrument_name = list(self.component_evaluations.keys())[0]
        
        if instrument_name not in self.component_evaluations:
            available = list(self.component_evaluations.keys())
            raise ValueError(f"Instrument '{instrument_name}' not found. Available:     {available}")
        
        comp_data = self.component_evaluations[instrument_name]
        
        # Get corresponding wavelength
        wave = self.model_evaluations[instrument_name]['wave']
        
        return {
            'wave': wave,
            'components': comp_data.get('components', []),
            'component_info': comp_data.get('component_info', [])
        }
    

    def reconstruct_model(self) -> Dict[str, 'VoigtModel']:
        """
        Reconstruct VoigtModel objects from saved configuration metadata.
        
        Returns
        -------
        Dict[str, VoigtModel]
            Dictionary mapping instrument names to VoigtModel objects.
            For single instrument: {'primary': model}
            For multi-instrument: {'hires': model_hires, 'xshooter': model_xshooter, ...}
            
        Raises
        ------
        ValueError
            If configuration metadata is not available or reconstruction fails
        """
        if self.config_metadata is None:
            raise ValueError("No configuration metadata available for reconstruction")
        
        try:
            # Only works for rbvfit 2.0 models
            if self.config_metadata.get('rbvfit_version') != '2.0':
                raise ValueError("Model reconstruction only supported for rbvfit 2.0")
            
            # Import here to avoid circular imports
            from rbvfit.core.fit_configuration import FitConfiguration
            from rbvfit.core.voigt_model import VoigtModel
            
            models = {}
            
            if self.is_multi_instrument:
                # Multi-instrument: create separate model for each instrument
                instr_params = self.config_metadata.get('instrument_params', {})
                
                for instrument_name in self.instrument_names:
                    # Get instrument-specific FWHM
                    inst_fwhm = instr_params.get(instrument_name, {}).get('FWHM')
                    if inst_fwhm is None:
                        # Fallback to global FWHM
                        inst_fwhm = self.config_metadata.get('instrumental_params', {}).get('FWHM')
                    
                    # Create configuration for this instrument
                    config = FitConfiguration(FWHM=inst_fwhm)
                    
                    for system_data in self.config_metadata['systems']:
                        z = system_data['redshift']
                        
                        for ion_data in system_data['ion_groups']:
                            config.add_system(
                                z=z,
                                ion=ion_data['ion'],
                                transitions=ion_data['transitions'],
                                components=ion_data['components']
                            )
                    
                    models[instrument_name] = VoigtModel(config)
            else:
                # Single instrument
                instr_params = self.config_metadata['instrumental_params']
                config = FitConfiguration(FWHM=instr_params.get('FWHM'))
                
                for system_data in self.config_metadata['systems']:
                    z = system_data['redshift']
                    
                    for ion_data in system_data['ion_groups']:
                        config.add_system(
                            z=z,
                            ion=ion_data['ion'],
                            transitions=ion_data['transitions'],
                            components=ion_data['components']
                        )
                
                models['primary'] = VoigtModel(config)
            
            return models
            
        except Exception as e:
            raise ValueError(f"Model reconstruction failed: {e}")

    def reconstruct_model_flux(self, theta: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Reconstruct model flux evaluations for all instruments.
        
        Parameters
        ----------
        theta : np.ndarray, optional
            Parameter array. If None, uses best-fit parameters.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping instrument names to flux arrays.
            For single instrument: {'primary': flux_array}
            For multi-instrument: {'hires': flux_hires, 'xshooter': flux_xshooter, ...}
            
        Raises
        ------
        ValueError
            If model reconstruction fails
        """
        if theta is None:
            theta = self.best_theta
        
        # Get reconstructed models
        models = self.reconstruct_model()
        
        fluxes = {}
        
        for instrument_name, model in models.items():
            # Get wavelength array for this instrument
            if instrument_name == 'primary':
                # Single instrument case
                wave = self.fitter.wave_obs
            else:
                # Multi-instrument case
                if instrument_name in self.model_evaluations:
                    wave = self.model_evaluations[instrument_name]['wave']
                elif self.instrument_data and instrument_name in self.instrument_data:
                    wave = self.instrument_data[instrument_name]['wave']
                else:
                    raise ValueError(f"Cannot find wavelength data for instrument '{instrument_name}'")
            
            # Evaluate model
            try:
                flux = model.evaluate(theta, wave)
                fluxes[instrument_name] = flux
            except Exception as e:
                raise ValueError(f"Model evaluation failed for {instrument_name}: {e}")
        
        return fluxes
        
    def has_model_evaluations(self) -> bool:
        """Check if pre-computed model evaluations are available."""
        return len(self.model_evaluations) > 0
    
    def has_component_evaluations(self) -> bool:
        """Check if component evaluations are available."""
        return len(self.component_evaluations) > 0
    
    def list_instruments(self) -> List[str]:
        """Get list of available instruments."""
        if self.model_evaluations:
            return list(self.model_evaluations.keys())
        elif self.instrument_data:
            return list(self.instrument_data.keys())
        else:
            return ['main']    

    
    # =============================================================================
    # PHASE I: Core Functionality
    # =============================================================================
    
    def save(self, filename: Union[str, Path]) -> None:
        """
        Save fit results to HDF5 file with pre-computed model evaluations.
        
        Parameters
        ----------
        filename : str or Path
            Output HDF5 filename
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filename, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['rbvfit_version'] = '2.0'
            meta.attrs['sampler'] = self.sampler_name
            meta.attrs['n_walkers'] = self.n_walkers or 0
            meta.attrs['n_steps'] = self.n_steps or 0
            meta.attrs['is_multi_instrument'] = self.is_multi_instrument
            meta.attrs['has_model_evaluations'] = len(self.model_evaluations) > 0
            meta.attrs['has_components'] = len(self.component_evaluations) > 0
            
            # MCMC results
            mcmc = f.create_group('mcmc')
            
            # Save full chain if available
            if hasattr(self.fitter.sampler, 'get_chain'):
                try:
                    chain = self.fitter.sampler.get_chain()
                    mcmc.create_dataset('chain', data=chain)
                except:
                    # Fallback: save flattened samples
                    samples = self._get_samples()
                    if samples is not None:
                        mcmc.create_dataset('samples', data=samples)
            
            # Parameters and bounds
            params = f.create_group('parameters')
            params.create_dataset('initial_guess', data=self.fitter.theta)
            params.create_dataset('best_fit', data=self.best_theta)
            if self.bounds_lower is not None:
                params.create_dataset('bounds_lower', data=self.bounds_lower)
            if self.bounds_upper is not None:
                params.create_dataset('bounds_upper', data=self.bounds_upper)
            
            # Data (primary dataset)
            data = f.create_group('data')
            data.create_dataset('wave_obs', data=self.fitter.wave_obs)
            data.create_dataset('flux_norm', data=self.fitter.fnorm)
            data.create_dataset('error_norm', data=self.fitter.enorm)
            
            # Multi-instrument data if available
            if self.is_multi_instrument and self.instrument_data:
                multi = f.create_group('multi_instrument')
                for name, inst_data in self.instrument_data.items():
                    if name == 'main':  # Skip main dataset (already saved above)
                        continue
                    inst_group = multi.create_group(name)
                    inst_group.create_dataset('wave', data=inst_data['wave'])
                    inst_group.create_dataset('flux', data=inst_data['flux'])
                    inst_group.create_dataset('error', data=inst_data['error'])
            
            # NEW: Save pre-computed model evaluations
            if self.model_evaluations:
                model_group = f.create_group('model_evaluations')
                
                for instrument_name, eval_data in self.model_evaluations.items():
                    inst_model_group = model_group.create_group(instrument_name)
                    inst_model_group.create_dataset('wave', data=eval_data['wave'])
                    inst_model_group.create_dataset('total_flux',     data=eval_data['total_flux'])
                    inst_model_group.attrs['has_components'] =     eval_data['has_components']
            
            # NEW: Save component evaluations
            if self.component_evaluations:
                comp_group = f.create_group('component_evaluations')
                
                for instrument_name, comp_data in self.component_evaluations.items():
                    inst_comp_group = comp_group.create_group(instrument_name)
                    
                    # Save component arrays
                    if 'components' in comp_data and comp_data['components']:
                        components = comp_data['components']
                        comp_array = np.array(components)  # Shape: (n_components,     n_wavelength)
                        inst_comp_group.create_dataset('components', data=comp_array)
                        
                        # Save component info if available
                        if 'component_info' in comp_data:
                            comp_info = comp_data['component_info']
                            # Save as JSON string for complex metadata
                            import json
                            inst_comp_group.attrs['component_info'] = json.dumps(    comp_info)
            
            # NEW: Save configuration metadata
            if self.config_metadata:
                config_group = f.create_group('config_metadata')
                
                # Save as JSON for complex nested structures
                import json
                config_json = json.dumps(self.config_metadata)
                config_group.attrs['config_data'] = config_json        

    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'FitResults':
        """
        Load fit results from HDF5 file.
        
        Parameters
        ----------
        filename : str or Path
            HDF5 filename to load
            
        Returns
        -------
        FitResults
            Loaded fit results object
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Results file not found: {filename}")
        
        # Create a minimal fitter-like object for compatibility
        class MockFitter:
            def __init__(self):
                self.wave_obs = None
                self.fnorm = None  
                self.enorm = None
                self.sampler = None
                
        results = cls.__new__(cls)  # Create without calling __init__
        results.fitter = MockFitter()
        results.model = None  # No model object when loaded
        
        # Initialize cache variables (CRITICAL for loaded objects)
        results._cached_param_summary = None
        results._cached_correlation = None
        results._cached_convergence = None
        results._cached_samples = None
        
        with h5py.File(filename, 'r') as f:
            # Load metadata
            meta = f['metadata']
            results.sampler_name = meta.attrs.get('sampler', 'unknown')
            results.n_walkers = meta.attrs.get('n_walkers', None)
            results.n_steps = meta.attrs.get('n_steps', None)
            results.is_multi_instrument = meta.attrs.get('is_multi_instrument', False)
            
            # Load parameters
            params = f['parameters']
            results.fitter.theta = params['initial_guess'][:]
            results.best_theta = params['best_fit'][:]
            
            results.bounds_lower = params.get('bounds_lower', [None])[:]
            results.bounds_upper = params.get('bounds_upper', [None])[:]
            if results.bounds_lower is not None and len(results.bounds_lower) == 1     and results.bounds_lower[0] is None:
                results.bounds_lower = None
            if results.bounds_upper is not None and len(results.bounds_upper) == 1     and results.bounds_upper[0] is None:
                results.bounds_upper = None
            
            # Load primary data
            data = f['data']
            results.fitter.wave_obs = data['wave_obs'][:]
            results.fitter.fnorm = data['flux_norm'][:]
            results.fitter.enorm = data['error_norm'][:]
            
            # Load multi-instrument data if available
            results.instrument_data = None
            if results.is_multi_instrument and 'multi_instrument' in f:
                results.instrument_data = {
                    'main': {
                        'wave': results.fitter.wave_obs,
                        'flux': results.fitter.fnorm,
                        'error': results.fitter.enorm
                    }
                }
                
                multi = f['multi_instrument']
                for name in multi.keys():
                    inst_data = multi[name]
                    results.instrument_data[name] = {
                        'wave': inst_data['wave'][:],
                        'flux': inst_data['flux'][:],
                        'error': inst_data['error'][:],
                        'model': None  # No model functions when loaded
                    }
            
            # NEW: Load pre-computed model evaluations
            results.model_evaluations = {}
            if 'model_evaluations' in f:
                model_group = f['model_evaluations']
                for instrument_name in model_group.keys():
                    inst_model = model_group[instrument_name]
                    results.model_evaluations[instrument_name] = {
                        'wave': inst_model['wave'][:],
                        'total_flux': inst_model['total_flux'][:],
                        'has_components': inst_model.attrs.get('has_components',     False)
                    }
            
            # NEW: Load component evaluations
            results.component_evaluations = {}
            if 'component_evaluations' in f:
                comp_group = f['component_evaluations']
                for instrument_name in comp_group.keys():
                    inst_comp = comp_group[instrument_name]
                    comp_data = {}
                    
                    if 'components' in inst_comp:
                        comp_data['components'] = inst_comp['components'][:]
                        
                        # Load component info if available
                        if 'component_info' in inst_comp.attrs:
                            import json
                            comp_data['component_info'] = json.loads(    inst_comp.attrs['component_info'])
                        
                    results.component_evaluations[instrument_name] = comp_data
            
            # NEW: Load configuration metadata
            results.config_metadata = None
            if 'config_metadata' in f:
                config_group = f['config_metadata']
                if 'config_data' in config_group.attrs:
                    import json
                    results.config_metadata = json.loads(    config_group.attrs['config_data'])
            
            # Load MCMC samples if available
            if 'mcmc' in f:
                mcmc = f['mcmc']
                if 'chain' in mcmc:
                    # Create a simple sampler-like object
                    class MockSampler:
                        def __init__(self, chain):
                            self._chain = chain
                        def get_chain(self):
                            return self._chain
                    
                    results.fitter.sampler = MockSampler(mcmc['chain'][:])
                elif 'samples' in mcmc:
                    results._cached_samples = mcmc['samples'][:]
        
        return results    
        
    def parameter_summary(self, verbose: bool = True) -> ParameterSummary:
        """
        Generate parameter summary table (borrowed and enhanced from vfit_mcmc).
        
        Parameters
        ----------
        verbose : bool
            Whether to print the summary table
            
        Returns
        -------
        ParameterSummary
            Container with parameter statistics
        """
        # Initialize cache if not present (for loaded objects)
        if not hasattr(self, '_cached_param_summary'):
            self._cached_param_summary = None
            
        if self._cached_param_summary is not None:
            if verbose:
                self._print_parameter_summary(self._cached_param_summary)
            return self._cached_param_summary
        
        # Get samples
        samples = self._get_samples()
        n_params = samples.shape[1]
        
        # Generate parameter names (borrowed from vfit_mcmc logic)
        names = self._generate_parameter_names(n_params)
        
        # Calculate statistics
        percentiles = {
            '16th': np.percentile(samples, 16, axis=0),
            '50th': np.percentile(samples, 50, axis=0),
            '84th': np.percentile(samples, 84, axis=0)
        }
        
        best_fit = percentiles['50th']  # Median as best fit
        lower_err = best_fit - percentiles['16th']
        upper_err = percentiles['84th'] - best_fit
        
        # Combine errors (take larger of lower/upper for symmetric error)
        errors = np.maximum(lower_err, upper_err)
        
        summary = ParameterSummary(
            names=names,
            best_fit=best_fit,
            errors=errors,
            percentiles=percentiles,
            mean=np.mean(samples, axis=0),
            std=np.std(samples, axis=0)
        )
        
        self._cached_param_summary = summary
        
        if verbose:
            self._print_parameter_summary(summary)
        
        return summary    
    def _generate_parameter_names(self, n_params: int) -> List[str]:
        """Generate parameter names based on model structure."""
        # Try to use model configuration if available
        if hasattr(self.model, 'config') and self.model.config is not None:
            try:
                # Use v2.0 parameter manager approach
                names = []
                param_idx = 0
                
                for sys_idx, system in enumerate(self.model.config.systems):
                    z = system.redshift
                    for ion_group in system.ion_groups:
                        ion = ion_group.ion_name
                        for comp in range(ion_group.components):
                            names.append(f"N_{ion}_z{z:.3f}_c{comp}")
                            param_idx += 1
                
                for sys_idx, system in enumerate(self.model.config.systems):
                    z = system.redshift
                    for ion_group in system.ion_groups:
                        ion = ion_group.ion_name
                        for comp in range(ion_group.components):
                            names.append(f"b_{ion}_z{z:.3f}_c{comp}")
                
                for sys_idx, system in enumerate(self.model.config.systems):
                    z = system.redshift
                    for ion_group in system.ion_groups:
                        ion = ion_group.ion_name
                        for comp in range(ion_group.components):
                            names.append(f"v_{ion}_z{z:.3f}_c{comp}")
                
                if len(names) == n_params:
                    return names
                    
            except Exception:
                pass
        
        # Fallback to generic names (borrowed from vfit_mcmc)
        nfit = n_params // 3
        names = []
        
        # Add logN names
        for i in range(nfit):
            names.append(f"logN_{i+1}")
        
        # Add b names
        for i in range(nfit):
            names.append(f"b_{i+1}")
        
        # Add v names
        for i in range(nfit):
            names.append(f"v_{i+1}")
        
        return names
    
    def _print_parameter_summary(self, summary: ParameterSummary) -> None:
        """Print formatted parameter summary (enhanced from vfit_mcmc)."""
        print("\n" + "=" * 70)
        print("PARAMETER SUMMARY")
        print("=" * 70)
        
        print(f"Sampler: {self.sampler_name}")
        print(f"Walkers: {self.n_walkers}")
        print(f"Steps: {self.n_steps}")
        print(f"Parameters: {len(summary.names)}")
        if self.is_multi_instrument:
            n_instruments = len(self.instrument_data) if self.instrument_data else 1
            print(f"Instruments: {n_instruments}")
        
        print(f"\nParameter Values:")
        print("-" * 70)
        print(f"{'Parameter':<20} {'Best Fit':<12} {'Error':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 70)
        
        for i, name in enumerate(summary.names):
            print(f"{name:<20} {summary.best_fit[i]:11.4f} "
                  f"{summary.errors[i]:11.4f} {summary.mean[i]:11.4f} "
                  f"{summary.std[i]:11.4f}")
        
        print("=" * 70)
    
    # =============================================================================
    # PHASE II: Analysis
    # =============================================================================
    
    def convergence_diagnostics(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive convergence diagnostics with recommendations.
        
        Parameters
        ----------
        verbose : bool
            Whether to print diagnostic results
            
        Returns
        -------
        dict
            Dictionary containing all diagnostic metrics and recommendations
        """
        if not hasattr(self, '_cached_convergence'):
            self._cached_convergence = None
            
        if self._cached_convergence is not None:
            if verbose:
                self._print_convergence_diagnostics(self._cached_convergence)
            return self._cached_convergence
        
        diagnostics = {}
        recommendations = []
        
        try:
            # 1. Acceptance fraction analysis
            acceptance_fraction = self._get_acceptance_fraction()
            diagnostics['acceptance_fraction'] = {
                'mean': np.mean(acceptance_fraction) if hasattr(acceptance_fraction, '__len__') else acceptance_fraction,
                'individual': acceptance_fraction
            }
            
            mean_accept = diagnostics['acceptance_fraction']['mean']
            if mean_accept < 0.2:
                recommendations.append("❌ Low acceptance fraction (<0.2). Consider reducing step size or relaxing bounds.")
            elif mean_accept > 0.7:
                recommendations.append("⚠️ High acceptance fraction (>0.7). Consider increasing step size for better mixing.")
            else:
                recommendations.append("✅ Good acceptance fraction (0.2-0.7).")
            
        except Exception as e:
            diagnostics['acceptance_fraction'] = None
            recommendations.append(f"❓ Could not calculate acceptance fraction: {e}")
        
        try:
            # 2. Autocorrelation time analysis
            autocorr_time = self._get_autocorr_time()
            diagnostics['autocorr_time'] = {
                'tau': autocorr_time,
                'mean_tau': np.nanmean(autocorr_time) if autocorr_time is not None else None
            }
            
            if autocorr_time is not None:
                mean_tau = np.nanmean(autocorr_time)
                if np.isfinite(mean_tau):
                    chain_length_ratio = self.n_steps / mean_tau
                    diagnostics['chain_length_ratio'] = chain_length_ratio
                    
                    if chain_length_ratio < 50:
                        recommended_length = int(50 * mean_tau)
                        recommendations.append(
                            f"⏱️ Chain too short. Current: {self.n_steps} steps, "
                            f"Recommended: >{recommended_length} steps (50x autocorr time)"
                        )
                        recommendations.append("📈 **Check trace plots** for visual confirmation of mixing")
                    else:
                        recommendations.append("✅ Chain length adequate (>50x autocorr time).")
                        recommendations.append("📊 Trace plots should show stable, well-mixed chains")
                else:
                    recommendations.append("❓ Could not determine autocorrelation time.")
                    recommendations.append("📈 **Examine trace plots** for mixing assessment")
            else:
                # Autocorr time calculation failed
                recommendations.append("❓ Autocorrelation time could not be calculated - chain likely too short")
                recommended_steps = self.n_steps * 3
                recommendations.append(f"⏱️ Recommend running 2-3x longer (try {recommended_steps} steps)")
                recommendations.append("📈 **CRITICAL: Check trace plots** for visual confirmation of mixing")
                recommendations.append("🔍 Look for: stable mixing, no trends, good between-chain agreement")
            
        except Exception as e:
            diagnostics['autocorr_time'] = None
            recommendations.append(f"❓ Could not calculate autocorrelation time: {e}")
            recommendations.append("📈 **Examine trace plots** and parameter evolution carefully")
        
        try:
            # 3. Effective sample size
            samples = self._get_samples()
            n_eff = self._estimate_effective_sample_size(samples)
            diagnostics['effective_sample_size'] = {
                'n_eff': n_eff,
                'min_n_eff': np.min(n_eff) if n_eff is not None else None
            }
            
            if n_eff is not None:
                min_eff = np.min(n_eff)
                if min_eff < 50:
                    recommendations.append(f"❌ Very low effective sample size (min: {min_eff:.0f}). Results NOT reliable.")
                    recommendations.append("📈 **Check trace plots** - likely show poor mixing or trends")
                elif min_eff < 100:
                    recommendations.append(f"⚠️ Low effective sample size (min: {min_eff:.0f}). Consider longer chains.")
                    recommendations.append("📈 **Check trace plots** for parameter drift or poor mixing")
                else:
                    recommendations.append(f"✅ Good effective sample size (min: {min_eff:.0f}).")
            
        except Exception as e:
            diagnostics['effective_sample_size'] = None
            recommendations.append(f"❓ Could not calculate effective sample size: {e}")
        
        try:
            # 4. Gelman-Rubin R-hat (if zeus sampler)
            if self.sampler_name.lower() == 'zeus':
                r_hat = self._get_gelman_rubin()
                diagnostics['gelman_rubin'] = {
                    'r_hat': r_hat,
                    'max_r_hat': np.max(r_hat) if r_hat is not None else None
                }
                
                if r_hat is not None:
                    max_r_hat = np.max(r_hat)
                    if max_r_hat > 1.2:
                        recommendations.append(f"❌ Poor convergence (max R-hat: {max_r_hat:.3f} > 1.2). Chains have NOT converged.")
                        recommendations.append("📈 **Examine trace plots** - likely show poor mixing or trends")
                    elif max_r_hat > 1.1:
                        recommendations.append(f"⚠️ Marginal convergence (max R-hat: {max_r_hat:.3f} > 1.1). Chains may not have converged.")
                        recommendations.append("📈 **Check trace plots** for parameter drift or poor mixing")
                    else:
                        recommendations.append(f"✅ Good convergence (max R-hat: {max_r_hat:.3f} < 1.1).")
            else:
                diagnostics['gelman_rubin'] = None
                
        except Exception as e:
            diagnostics['gelman_rubin'] = None
            if self.sampler_name.lower() == 'zeus':
                recommendations.append(f"❓ Could not calculate Gelman-Rubin diagnostic: {e}")
        
        # 5. Overall assessment with enhanced recommendations
        overall_status = self._assess_overall_convergence(diagnostics)
        diagnostics['overall_status'] = overall_status
        
        # Add status-specific recommendations
        if overall_status == "GOOD":
            recommendations.append("✅ Excellent convergence - results are reliable")
            recommendations.append("💾 Consider thinning samples if storage is a concern")
        elif overall_status == "MARGINAL":
            recommendations.append("⚠️ Borderline convergence - proceed with caution")
            recommendations.append("🔄 Consider running 1.5-2x longer for better statistics")
            recommendations.append("📊 Results may be reliable but uncertainties could be underestimated")
        elif overall_status == "POOR":
            recommendations.append("❌ Poor convergence - results NOT reliable")
            recommendations.append("🚫 DO NOT use these results for scientific analysis")
            recommendations.append("🔧 Try: longer chains, more walkers, different initial conditions")
            recommendations.append("⚠️ Check bounds - parameters may be hitting limits")
        elif overall_status == "UNKNOWN":
            recommendations.append("❓ Convergence status unclear - requires manual inspection")
            recommendations.append("⚠️ Use results with extreme caution until convergence confirmed")
        
        # Always recommend trace plots for non-GOOD status
        if overall_status != "GOOD":
            recommendations.append("📈 **CRITICAL: Examine trace plots** and parameter evolution carefully")
        
        diagnostics['recommendations'] = recommendations
        
        self._cached_convergence = diagnostics
        
        if verbose:
            self._print_convergence_diagnostics(diagnostics)
        
        return diagnostics
    
    def _get_acceptance_fraction(self):
        """Get acceptance fraction from sampler."""
        if hasattr(self.fitter.sampler, 'acceptance_fraction'):
            return self.fitter.sampler.acceptance_fraction
        elif hasattr(self.fitter.sampler, 'get_chain'):
            # Estimate from chain for zeus
            try:
                chain = self.fitter.sampler.get_chain()
                # Count unique consecutive steps
                n_accepted = 0
                n_total = 0
                for walker_chain in chain.T:
                    for i in range(1, len(walker_chain)):
                        n_total += 1
                        if not np.array_equal(walker_chain[i], walker_chain[i-1]):
                            n_accepted += 1
                return n_accepted / n_total if n_total > 0 else 0.0
            except:
                return None
        return None
    
    def _get_autocorr_time(self):
        if self.sampler_name.lower() == 'emcee':
            # Use emcee autocorr method
            return self.fitter.sampler.get_autocorr_time()
        elif self.sampler_name.lower() == 'zeus':
            # Zeus doesn't have autocorr - return None
            return None

    
    def _get_gelman_rubin(self):
        """Get Gelman-Rubin R-hat statistic for zeus sampler."""
        if self.sampler_name.lower() != 'zeus':
            return None
            
        try:
            # Check zeus version first
            import zeus
            #print(f"Debug: Zeus version: {zeus.__version__}")
            
            # Try the newer diagnostics module first
            try:
                chain = self.fitter.sampler.get_chain().transpose(1, 0, 2)
                # Then compute Gelman-Rubin diagnostic
                r_hat = zeus.diagnostics.gelman_rubin(chain)
                print(f"Using zeus.diagnostics - R-hat: {r_hat}")
                return r_hat
            except AttributeError:
                print("zeus.diagnostics not available, implementing simple R-hat")
                
                # Implement simple R-hat calculation
                chain = self.fitter.sampler.get_chain()  # Shape: (n_steps, n_walkers, n_params)
                
                if len(chain.shape) != 3:
                    print(f"Unexpected chain shape: {chain.shape}")
                    return None
                
                n_steps, n_walkers, n_params = chain.shape
                
                if n_steps < 100:  # Need sufficient samples
                    print(f"Too few samples for R-hat: {n_steps}")
                    return None
                
                # Simple R-hat calculation
                r_hat_values = []
                
                for param_idx in range(n_params):
                    param_chains = chain[:, :, param_idx]  # (n_steps, n_walkers)
                    
                    # Between-chain variance
                    chain_means = np.mean(param_chains, axis=0)  # Mean for each walker
                    grand_mean = np.mean(chain_means)
                    B = n_steps * np.var(chain_means, ddof=1)
                    
                    # Within-chain variance  
                    chain_variances = np.var(param_chains, axis=0, ddof=1)
                    W = np.mean(chain_variances)
                    
                    # R-hat calculation
                    var_plus = ((n_steps - 1) * W + B) / n_steps
                    r_hat = np.sqrt(var_plus / W) if W > 0 else np.inf
                    
                    r_hat_values.append(r_hat)
                
                r_hat_array = np.array(r_hat_values)
                print(f"Calculated R-hat values: {r_hat_array}")
                
                return r_hat_array
                
        except Exception as e:
            print(f"R-hat calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
  
    def _estimate_effective_sample_size(self, samples):
        """Estimate effective sample size for each parameter."""
        try:
            from scipy import signal
            
            n_eff = np.zeros(samples.shape[1])
            
            for i in range(samples.shape[1]):
                # Calculate autocorrelation function
                data = samples[:, i] - np.mean(samples[:, i])
                autocorr = signal.correlate(data, data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]
                
                # Find first negative value
                first_negative = np.where(autocorr < 0)[0]
                if len(first_negative) > 0:
                    cutoff = first_negative[0]
                else:
                    cutoff = len(autocorr) // 4
                
                # Integrated autocorr time
                tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
                n_eff[i] = len(samples) / (2 * tau_int)
            
            return n_eff
            
        except Exception:
            return None
    

    def _assess_overall_convergence(self, diagnostics):
       """Assess overall convergence status with sampler-aware approach."""
       issues = []
       
       # Check acceptance fraction (common to all samplers)
       if diagnostics['acceptance_fraction'] is not None:
           mean_accept = diagnostics['acceptance_fraction']['mean']
           if mean_accept < 0.2 or mean_accept > 0.7:
               issues.append("acceptance_fraction")
       else:
           issues.append("acceptance_fraction_unavailable")
       
       # Sampler-specific convergence logic
       if self.sampler_name.lower() == 'zeus':
           # Zeus: prioritize Gelman-Rubin R-hat diagnostic
           if diagnostics['gelman_rubin'] is not None:
               max_r_hat = diagnostics['gelman_rubin']['max_r_hat']
               if max_r_hat is not None:
                   if max_r_hat > 1.2:
                       issues.append("poor_convergence")
                   elif max_r_hat > 1.1:
                       issues.append("marginal_convergence")
                   # R-hat <= 1.1 is good
                   
                   # Check effective sample size (secondary for zeus)
                   if diagnostics['effective_sample_size'] is not None:
                       min_eff = diagnostics['effective_sample_size']['min_n_eff']
                       if min_eff is not None and min_eff < 50:  # Lower threshold for zeus
                           issues.append("effective_sample_size")
                   
                   # Zeus assessment
                   if "poor_convergence" in issues:
                       return "POOR"
                   elif "marginal_convergence" in issues or len(issues) >= 2:
                       return "MARGINAL"
                   elif len(issues) <= 1:  # Allow minor acceptance issues
                       return "GOOD"
                   else:
                       return "MARGINAL"
               else:
                   return "UNKNOWN"  # R-hat calculation failed
           else:
               return "UNKNOWN"  # No R-hat available for zeus
       
       else:
           # Emcee or unknown sampler: use autocorrelation-based logic
           autocorr_failed = False
           
           # Check autocorrelation time availability
           if diagnostics['autocorr_time'] is None or diagnostics['autocorr_time']['mean_tau'] is None:
               autocorr_failed = True
               issues.append("autocorr_unavailable")
           else:
               # Check chain length ratio
               if diagnostics.get('chain_length_ratio') is not None:
                   if diagnostics['chain_length_ratio'] < 50:
                       issues.append("chain_length")
           
           # Check effective sample size
           if diagnostics['effective_sample_size'] is not None:
               min_eff = diagnostics['effective_sample_size']['min_n_eff']
               if min_eff is not None and min_eff < 100:
                   issues.append("effective_sample_size")
           
           # Check Gelman-Rubin (if available for emcee)
           if diagnostics['gelman_rubin'] is not None:
               max_r_hat = diagnostics['gelman_rubin']['max_r_hat']
               if max_r_hat is not None and max_r_hat > 1.1:
                   issues.append("convergence")
           
           # Conservative assessment logic for emcee
           if autocorr_failed and len([i for i in issues if i != "autocorr_unavailable"]) == 0:
               # Only autocorr failed, other indicators good/unknown
               return "MARGINAL"  # Conservative when missing key diagnostic
           elif autocorr_failed and len(issues) > 2:
               return "UNKNOWN"  # Too many unknowns for reliable assessment
           elif len(issues) == 0:
               return "GOOD"
           elif len(issues) <= 2:
               return "MARGINAL"
           else:
               return "POOR"
    


    def _print_convergence_diagnostics(self, diagnostics):
        """Print formatted convergence diagnostics with sampler-aware recommendations."""
        print("\n" + "=" * 70)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 70)
        
        status = diagnostics['overall_status']
        status_symbols = {
            "GOOD": "✅", 
            "MARGINAL": "⚠️", 
            "POOR": "❌", 
            "UNKNOWN": "❓"
        }
        print(f"Overall Status: {status_symbols.get(status, '?')} {status}")
        print(f"Sampler: {self.sampler_name.upper()}")
        
        print(f"\nDetailed Metrics:")
        print("-" * 50)
        
        # Acceptance fraction
        if diagnostics['acceptance_fraction'] is not None:
            mean_accept = diagnostics['acceptance_fraction']['mean']
            if 0.2 <= mean_accept <= 0.7:
                symbol = "✅"
            elif mean_accept < 0.1 or mean_accept > 0.8:
                symbol = "❌"
            else:
                symbol = "⚠️"
            print(f"Acceptance Fraction: {symbol} {mean_accept:.3f}")
        else:
            print("Acceptance Fraction: ❓ Not available")
        
        # Sampler-specific primary diagnostics
        if self.sampler_name.lower() == 'zeus':
            # Zeus: Show R-hat as primary diagnostic
            if diagnostics['gelman_rubin'] is not None:
                max_r_hat = diagnostics['gelman_rubin']['max_r_hat']
                if max_r_hat is not None:
                    if max_r_hat <= 1.1:
                        symbol = "✅"
                    elif max_r_hat <= 1.2:
                        symbol = "⚠️"
                    else:
                        symbol = "❌"
                    print(f"Max R-hat (primary): {symbol} {max_r_hat:.3f}")
                else:
                    print("R-hat: ❌ Could not calculate")
            else:
                print("R-hat: ❓ Not available")
            
            # Note about autocorr for zeus users
            print("Autocorr Time: ➖ Not available (zeus uses R-hat instead)")
            
        else:
            # Emcee: Show autocorr as primary diagnostic
            if diagnostics['autocorr_time'] is not None:
                mean_tau = diagnostics['autocorr_time']['mean_tau']
                if mean_tau is not None:
                    if 'chain_length_ratio' in diagnostics:
                        ratio = diagnostics['chain_length_ratio']
                        if ratio >= 50:
                            symbol = "✅"
                        elif ratio >= 20:
                            symbol = "⚠️"
                        else:
                            symbol = "❌"
                        print(f"Mean Autocorr Time (primary): {symbol} {mean_tau:.1f} steps")
                        print(f"Chain Length Ratio: {symbol} {ratio:.1f}x autocorr time")
                    else:
                        print(f"Mean Autocorr Time: ⚠️ {mean_tau:.1f} steps")
                else:
                    print("Autocorr Time: ❌ Could not calculate")
            else:
                print("Autocorr Time: ❓ Not available")
            
            # Show R-hat as secondary for emcee
            if diagnostics['gelman_rubin'] is not None:
                max_r_hat = diagnostics['gelman_rubin']['max_r_hat']
                if max_r_hat is not None:
                    if max_r_hat <= 1.1:
                        symbol = "✅"
                    elif max_r_hat <= 1.2:
                        symbol = "⚠️"
                    else:
                        symbol = "❌"
                    print(f"Max R-hat (secondary): {symbol} {max_r_hat:.3f}")
            else:
                print("R-hat: ➖ Not calculated for emcee")
        
        # Effective sample size (common to both)
        if diagnostics['effective_sample_size'] is not None:
            min_eff = diagnostics['effective_sample_size']['min_n_eff']
            if min_eff is not None:
                # Different thresholds for different samplers
                threshold = 50 if self.sampler_name.lower() == 'zeus' else 100
                if min_eff >= threshold:
                    symbol = "✅"
                elif min_eff >= threshold/2:
                    symbol = "⚠️"
                else:
                    symbol = "❌"
                print(f"Min Effective N: {symbol} {min_eff:.0f}")
            else:
                print("Effective Sample Size: ❌ Could not calculate")
        else:
            print("Effective Sample Size: ❓ Not available")
        
        # Sampler-specific recommendations
        print(f"\nSampler-Specific Recommendations:")
        print("-" * 50)
        
        if self.sampler_name.lower() == 'zeus':
            if status == "GOOD":
                print("✅ Excellent! Zeus R-hat < 1.1 indicates good convergence")
                print("💡 Consider thinning samples if storage is a concern")
            elif status == "MARGINAL":
                print("⚠️ Zeus R-hat slightly above 1.1 - borderline convergence")
                print("🔧 To improve: Run 1.5-2x longer or try more walkers")
                print("📊 Current results may be reliable but uncertainties could be underestimated")
            elif status == "POOR":
                print("❌ Zeus R-hat > 1.2 - chains have NOT converged")
                print("🔧 Try: More walkers, longer chains, different initial conditions")
                print("⚠️ Check perturbation parameter (try 1e-4 to 1e-3)")
            else:
                print("❓ Zeus convergence unclear - manual inspection needed")
            
            print("\n💡 Zeus Tips:")
            print("   • R-hat < 1.1 = excellent convergence")
            print("   • Zeus often needs fewer walkers than emcee")
            print("   • Use smaller perturbation (1e-4) if initialization fails")
            
        else:  # emcee
            if status == "GOOD":
                print("✅ Excellent! Emcee autocorr analysis shows good convergence")
                print("💡 Chain length > 50x autocorr time is ideal")
            elif status == "MARGINAL":
                print("⚠️ Emcee shows marginal convergence")
                print("🔧 To improve: Run 2-3x longer chains")
                print("📊 Check trace plots for mixing and stationarity")
            elif status == "POOR":
                print("❌ Emcee shows poor convergence")
                print("🔧 Try: Longer chains, more walkers, better initialization")
                print("⚠️ Check bounds - parameters may be hitting limits")
            else:
                print("❓ Emcee convergence unclear - examine trace plots")
            
            print("\n💡 Emcee Tips:")
            print("   • Chain length should be > 50x autocorr time")
            print("   • Acceptance fraction should be 0.2-0.7")
            print("   • Use optimize=True for better starting positions")
        
        # General recommendations
        print(f"\nGeneral Recommendations:")
        print("-" * 50)
        for i, rec in enumerate(diagnostics['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Bottom line recommendation
        print(f"\n{'🔥 BOTTOM LINE 🔥':^70}")
        if status == "GOOD":
            print("✅ Results are reliable - proceed with analysis")
        elif status == "MARGINAL":
            print("⚠️ Results likely reliable but consider longer chains for publication")
        elif status == "POOR":
            print("❌ DO NOT use these results - re-run with suggested improvements")
        else:
            print("❓ Convergence unclear - examine trace plots manually")
        
        print("=" * 70)    
        
    def chain_trace_plot(self, save_path: Optional[str] = None, 
                        n_cols: int = 3, figsize: Optional[Tuple[float, float]] =     None) -> plt.Figure:
        """
        Create trace plots for visual convergence assessment.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the trace plot
        n_cols : int, optional
            Number of columns in subplot grid
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns
        -------
        plt.Figure
            The trace plot figure
        """
        from rbvfit.core.fit_results_plot import chain_trace_plot
        return chain_trace_plot(self, save_path, n_cols, figsize)    

    def correlation_matrix(self, plot: bool = False, save_path: Optional[str] = None) -> np.ndarray:
        """
        Calculate parameter correlation matrix.
        
        Parameters
        ----------
        plot : bool
            Whether to create a correlation plot
        save_path : str, optional
            Path to save correlation plot
            
        Returns
        -------
        np.ndarray
            Correlation matrix (n_params x n_params)
        """
        # Initialize cache if not present (for loaded objects)
        if not hasattr(self, '_cached_correlation'):
            self._cached_correlation = None
            
        if self._cached_correlation is not None:
            correlation = self._cached_correlation
        else:
            samples = self._get_samples()
            correlation = np.corrcoef(samples.T)
            self._cached_correlation = correlation
        
        if plot:
            from rbvfit.core.fit_results_plot import plot_correlation_matrix
            plot_correlation_matrix(self, save_path)
        
        return correlation  
    

    
    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot parameter correlation matrix heatmap.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save correlation plot
            
        Returns
        -------
        plt.Figure
            Correlation matrix figure
        """
        from rbvfit.core.fit_results_plot import plot_correlation_matrix
        return plot_correlation_matrix(self, save_path) 

    def corner_plot(self, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create corner plot of parameter posterior distributions.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the corner plot
        **kwargs
            Additional arguments passed to corner.corner()
            
        Returns
        -------
        plt.Figure
            The corner plot figure
        """
        from rbvfit.core.fit_results_plot import corner_plot
        return corner_plot(self, save_path, **kwargs)    
    # =============================================================================
    # Additional Utility Methods
    # =============================================================================
    
    def chi_squared(self) -> Dict[str, float]:
        """
        Calculate chi-squared statistics using pre-computed model evaluations when     available.
        
        Returns
        -------
        dict
            Dictionary containing chi-squared metrics
        """
        if self.has_model_evaluations():
            # Use pre-computed model evaluations
            return self._chi_squared_from_evaluations()
        else:
            # Fallback to original method (may fail if model unavailable)
            return self._chi_squared_original()
    
    def _chi_squared_from_evaluations(self) -> Dict[str, float]:
        """Calculate chi-squared using pre-computed model evaluations."""
        chi2_stats = {}
        
        # Primary dataset
        if 'main' in self.model_evaluations:
            eval_data = self.model_evaluations['main']
            model_flux = eval_data['total_flux']
            
            chi2 = np.sum((self.fitter.fnorm - model_flux)**2 / self.fitter.enorm**2)
            n_data = len(self.fitter.fnorm)
            n_params = len(self.best_theta)
            dof = n_data - n_params
            
            chi2_stats.update({
                'chi2': chi2,
                'dof': dof,
                'reduced_chi2': chi2 / dof if dof > 0 else np.inf,
                'n_data_points': n_data,
                'n_parameters': n_params
            })
            
            chi2_total = chi2
            n_total = n_data
        else:
            # No main dataset, start totals at zero
            chi2_total = 0.0
            n_total = 0
            n_params = len(self.best_theta)
        
        # Additional instruments if multi-instrument
        if self.is_multi_instrument and len(self.model_evaluations) > 1:
            for instrument_name, eval_data in self.model_evaluations.items():
                if instrument_name == 'main':
                    continue
                    
                # Get corresponding data
                if self.instrument_data and instrument_name in self.instrument_data:
                    inst_data = self.instrument_data[instrument_name]
                    flux_obs = inst_data['flux']
                    error_obs = inst_data['error']
                    
                    model_flux = eval_data['total_flux']
                    
                    inst_chi2 = np.sum((flux_obs - model_flux)**2 / error_obs**2)
                    chi2_total += inst_chi2
                    n_total += len(flux_obs)
                    
                    chi2_stats[f'chi2_{instrument_name}'] = inst_chi2
            
            # Combined statistics
            if n_total > 0:
                chi2_stats['chi2_total'] = chi2_total
                chi2_stats['n_total_points'] = n_total
                chi2_stats['dof_total'] = n_total - n_params
                chi2_stats['reduced_chi2_total'] = chi2_total / (n_total - n_params)     if (n_total - n_params) > 0 else np.inf
        
        return chi2_stats
    
    def _chi_squared_original(self) -> Dict[str, float]:
        """Original chi-squared calculation (fallback when no pre-computed     evaluations)."""
        summary = self.parameter_summary(verbose=False)
        best_theta = summary.best_fit
        
        # Calculate model for primary dataset
        try:
            if self.model is not None:
                model_flux = self.model.evaluate(best_theta, self.fitter.wave_obs)
            else:
                print("Warning: No model available and no pre-computed evaluations.     Chi-squared calculation skipped.")
                return {'chi2': np.nan, 'reduced_chi2': np.nan, 'dof': np.nan}
        except:
            print("Warning: Could not evaluate model. Chi-squared calculation     skipped.")
            return {'chi2': np.nan, 'reduced_chi2': np.nan, 'dof': np.nan}
        
        # Primary dataset chi-squared
        chi2 = np.sum((self.fitter.fnorm - model_flux)**2 / self.fitter.enorm**2)
        n_data = len(self.fitter.fnorm)
        n_params = len(best_theta)
        dof = n_data - n_params
        
        chi2_stats = {
            'chi2': chi2,
            'dof': dof,
            'reduced_chi2': chi2 / dof if dof > 0 else np.inf,
            'n_data_points': n_data,
            'n_parameters': n_params
        }
        
        # Add multi-instrument contributions if available (requires model)
        if self.is_multi_instrument and self.instrument_data and self.model is not     None:
            chi2_total = chi2
            n_total = n_data
            
            for name, inst_data in self.instrument_data.items():
                if name == 'main':
                    continue
                
                try:
                    # This will fail for loaded results without model functions
                    inst_model = inst_data.get('model')
                    if inst_model is not None:
                        inst_model_flux = inst_model(best_theta, inst_data['wave'])
                        inst_chi2 = np.sum((inst_data['flux'] - inst_model_flux)**2 /     inst_data['error']**2)
                        
                        chi2_total += inst_chi2
                        n_total += len(inst_data['wave'])
                        chi2_stats[f'chi2_{name}'] = inst_chi2
                        
                except:
                    print(f"Warning: Could not evaluate model for instrument {name}")
            
            chi2_stats['chi2_total'] = chi2_total
            chi2_stats['n_total_points'] = n_total
            chi2_stats['dof_total'] = n_total - n_params
            chi2_stats['reduced_chi2_total'] = chi2_total / (n_total - n_params) if (    n_total - n_params) > 0 else np.inf
        
        return chi2_stats        
    def print_fit_summary(self) -> None:
        """Print comprehensive fit summary with new capabilities."""
        print("\n" + "="*60)
        print("FIT RESULTS SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"Sampler: {self.sampler_name}")
        print(f"Walkers: {self.n_walkers}")
        print(f"Steps: {self.n_steps}")
        
        # Multi-instrument info
        if self.is_multi_instrument:
            instruments = self.list_instruments()
            print(f"Multi-instrument: {len(instruments)} datasets")
            print(f"Instruments: {', '.join(instruments)}")
        else:
            print("Single instrument")
        
        # NEW: Model evaluation capabilities
        print(f"\nModel Evaluations:")
        print(f"  Pre-computed flux: {'✓' if self.has_model_evaluations() else '✗'}")
        print(f"  Component analysis: {'✓' if self.has_component_evaluations() else     '✗'}")
        print(f"  Reconstruction: {'✓' if self.config_metadata else '✗'}")
        
        # Parameter summary
        try:
            param_summary = self.parameter_summary(verbose=False)
            print(f"\nParameters: {len(param_summary.best_fit)}")
            print("Best-fit values:")
            for i, (val, err_low, err_high) in enumerate(zip(
                param_summary.best_fit, 
                param_summary.errors_lower, 
                param_summary.errors_upper
            )):
                print(f"  θ[{i:2d}] = {val:8.3f} +{err_high:.3f}/-{err_low:.3f}")
        except Exception as e:
            print(f"Parameter summary unavailable: {e}")
        
        # Chi-squared
        try:
            chi2_stats = self.chi_squared()
            print(f"\nGoodness of Fit:")
            if not np.isnan(chi2_stats.get('chi2', np.nan)):
                print(f"  χ² = {chi2_stats['chi2']:.2f}")
                print(f"  DoF = {chi2_stats['dof']}")
                print(f"  χ²/ν = {chi2_stats['reduced_chi2']:.3f}")
                
                if self.is_multi_instrument and 'chi2_total' in chi2_stats:
                    print(f"  Combined χ² = {chi2_stats['chi2_total']:.2f}")
                    print(f"  Combined χ²/ν = {chi2_stats['reduced_chi2_total']:.3f}")
                    
                    # Per-instrument breakdown
                    for instrument in self.list_instruments():
                        if f'chi2_{instrument}' in chi2_stats:
                            print(f"    {instrument}: χ² = {chi2_stats[    f'chi2_{instrument}']:.2f}")
            else:
                print("  Chi-squared: Not available")
        except Exception as e:
            print(f"  Chi-squared calculation failed: {e}")
        
        # NEW: Configuration info if available
        if self.config_metadata:
            print(f"\nModel Configuration:")
            print(f"  rbvfit version: {self.config_metadata.get('rbvfit_version',     'unknown')}")
            
            if 'systems' in self.config_metadata:
                total_components = sum(
                    ion['components'] 
                    for system in self.config_metadata['systems']
                    for ion in system['ion_groups']
                )
                print(f"  Total components: {total_components}")
                
                for i, system in enumerate(self.config_metadata['systems']):
                    print(f"  System {i+1}: z = {system['redshift']:.6f}")
                    for ion in system['ion_groups']:
                        transitions_str = ', '.join(f"{t:.1f}" for t in     ion['transitions'])
                        print(f"    {ion['ion']}: {ion['components']} comp, λ =     {transitions_str}Å")
        
        print("="*60)        
    def __repr__(self) -> str:
        """String representation of FitResults."""
        summary = self.parameter_summary(verbose=False)
        convergence = self.convergence_diagnostics(verbose=False)
        
        return (f"FitResults(sampler='{self.sampler_name}', "
                f"parameters={len(summary.names)}, "
                f"convergence='{convergence['overall_status']}')")

    # =============================================================================
    # PHASE III: Velocity Space Visualization
    # =============================================================================
    
    def plot_velocity_fits(self, show_components: bool = True, 
                          show_rail_system: bool = True,
                          figsize_per_panel: Tuple[float, float] = (4, 3),
                          save_path: Optional[str] = None,
                          velocity_range: Optional[Tuple[float, float]] = None,
                          **kwargs) -> Dict[str, plt.Figure]:
        """
        Create velocity space plots for each ion group with multi-instrument support.
        
        Parameters
        ----------
        show_components : bool
            Whether to show individual velocity components
        show_rail_system : bool
            Whether to show rail system with component markers
        figsize_per_panel : tuple
            Size of each subplot panel (width, height)
        save_path : str, optional
            Base path for saving figures (will append ion names)
        velocity_range : tuple, optional
            Velocity range (vmin, vmax) in km/s for all plots
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        dict
            Dictionary mapping ion names to their figure objects
        """
        from rbvfit.core.fit_results_plot import plot_velocity_fits
        return plot_velocity_fits(self, show_components, show_rail_system, 
                                figsize_per_panel, save_path, velocity_range,     **kwargs)
        

    def plot_components_breakdown(self, instrument_name: str = None, save_path: str =     None) -> plt.Figure:
        """
        Plot component breakdown for detailed analysis.
        
        Parameters
        ----------
        instrument_name : str, optional
            Instrument to plot. If None, uses first available.
        save_path : str, optional
            Save figure to this path
            
        Returns
        -------
        plt.Figure
            Component breakdown figure
        """
        from rbvfit.core.fit_results_plot import plot_components_breakdown
        return plot_components_breakdown(self, instrument_name, save_path)

    def plot_model_comparison(self, save_path: Optional[str] = None, 
                             show_residuals: bool = True, **kwargs) -> plt.Figure:
        """
        Plot model vs data comparison for all instruments.
        
        Parameters
        ----------
        save_path : str, optional
            Save figure to this path
        show_residuals : bool
            Whether to show residuals subplot
        **kwargs
            Additional plotting parameters
            
        Returns
        -------
        plt.Figure
            Model comparison figure
        """
        from rbvfit.core.fit_results_plot import plot_model_comparison
        return plot_model_comparison(self, save_path, show_residuals, **kwargs)    
    # =============================================================================
    # PHASE IV: Export Functionality
    # =============================================================================
    
    def export_csv(self, filename: Union[str, Path], include_errors: bool = True) -> None:
        """
        Export parameter results to CSV format.
        
        Parameters
        ----------
        filename : str or Path
            Output CSV filename
        include_errors : bool
            Whether to include error columns
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PANDAS:
            # Fallback to basic CSV writing
            self._export_csv_basic(filename, include_errors)
            return
        
        # Use pandas for better formatting
        summary = self.parameter_summary(verbose=False)
        
        # Create DataFrame
        data = {
            'parameter': summary.names,
            'best_fit': summary.best_fit,
            'mean': summary.mean,
            'std': summary.std
        }
        
        if include_errors:
            data.update({
                'lower_error': summary.best_fit - summary.percentiles['16th'],
                'upper_error': summary.percentiles['84th'] - summary.best_fit,
                'percentile_16': summary.percentiles['16th'],
                'percentile_84': summary.percentiles['84th']
            })
        
        df = pd.DataFrame(data)
        
        # Add metadata as comments
        with open(filename, 'w') as f:
            f.write(f"# rbvfit 2.0 Parameter Export\n")
            f.write(f"# Sampler: {self.sampler_name}\n")
            f.write(f"# Walkers: {self.n_walkers}\n")
            f.write(f"# Steps: {self.n_steps}\n")
            
            convergence = self.convergence_diagnostics(verbose=False)
            f.write(f"# Convergence: {convergence['overall_status']}\n")
            
            chi2_stats = self.chi_squared()
            if not np.isnan(chi2_stats['chi2']):
                f.write(f"# Chi-squared: {chi2_stats['chi2']:.2f}\n")
                f.write(f"# Reduced chi-squared: {chi2_stats['reduced_chi2']:.3f}\n")
            
            f.write("#\n")
        
        # Append DataFrame
        df.to_csv(filename, mode='a', index=False, float_format='%.6f')
        
        print(f"✅ Exported parameters to {filename}")
    
    def _export_csv_basic(self, filename: Path, include_errors: bool):
        """Basic CSV export without pandas."""
        summary = self.parameter_summary(verbose=False)
        
        with open(filename, 'w') as f:
            # Header
            f.write("parameter,best_fit,mean,std")
            if include_errors:
                f.write(",lower_error,upper_error,percentile_16,percentile_84")
            f.write("\n")
            
            # Data rows
            for i, name in enumerate(summary.names):
                row = [name, f"{summary.best_fit[i]:.6f}", 
                      f"{summary.mean[i]:.6f}", f"{summary.std[i]:.6f}"]
                
                if include_errors:
                    lower_err = summary.best_fit[i] - summary.percentiles['16th'][i]
                    upper_err = summary.percentiles['84th'][i] - summary.best_fit[i]
                    row.extend([
                        f"{lower_err:.6f}", f"{upper_err:.6f}",
                        f"{summary.percentiles['16th'][i]:.6f}",
                        f"{summary.percentiles['84th'][i]:.6f}"
                    ])
                
                f.write(",".join(row) + "\n")
        
        print(f"✅ Exported parameters to {filename}")
    
    def export_latex(self, filename: Union[str, Path], 
                    table_format: str = 'standard',
                    caption: str = "MCMC fit results for absorption line systems",
                    label: str = "tab:absorption_results") -> None:
        """
        Export parameter results to LaTeX table format.
        
        Parameters
        ----------
        filename : str or Path
            Output LaTeX filename
        table_format : str
            Table format: 'standard', 'compact', or 'publication'
        caption : str
            Table caption
        label : str
            Table label for referencing
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.parameter_summary(verbose=False)
        
        with open(filename, 'w') as f:
            # Table header
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            
            if table_format == 'compact':
                f.write("\\begin{tabular}{lccc}\n")
                f.write("\\hline\n")
                f.write("Parameter & Value & Error & Units \\\\\n")
                f.write("\\hline\n")
                
                for i, name in enumerate(summary.names):
                    # Parse parameter for units
                    if name.startswith('N_'):
                        units = "[cm$^{-2}$]"
                        value_str = f"{summary.best_fit[i]:.2f}"
                    elif name.startswith('b_'):
                        units = "[km s$^{-1}$]"
                        value_str = f"{summary.best_fit[i]:.1f}"
                    elif name.startswith('v_'):
                        units = "[km s$^{-1}$]"
                        value_str = f"{summary.best_fit[i]:.1f}"
                    else:
                        units = ""
                        value_str = f"{summary.best_fit[i]:.3f}"
                    
                    error_str = f"{summary.errors[i]:.2f}"
                    
                    # Clean parameter name for LaTeX
                    param_name = name.replace('_', '\\_')
                    
                    f.write(f"{param_name} & {value_str} & {error_str} & {units} \\\\\n")
                    
            elif table_format == 'publication':
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Ion & $z$ & Component & $\\log N$ & $b$ & $v$ \\\\\n")
                f.write("   &     &           & [cm$^{-2}$] & [km s$^{-1}$] & [km s$^{-1}$] \\\\\n")
                f.write("\\hline\n")
                
                # Try to organize by ions
                self._write_publication_table_rows(f, summary)
                
            else:  # standard
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\hline\n")
                f.write("Parameter & Best Fit & $1\\sigma$ Error \\\\\n")
                f.write("\\hline\n")
                
                for i, name in enumerate(summary.names):
                    param_name = name.replace('_', '\\_')
                    lower_err = summary.best_fit[i] - summary.percentiles['16th'][i]
                    upper_err = summary.percentiles['84th'][i] - summary.best_fit[i]
                    
                    value_str = f"{summary.best_fit[i]:.3f}$_{{-{lower_err:.3f}}}^{{+{upper_err:.3f}}}$"
                    error_str = f"{summary.errors[i]:.3f}"
                    
                    f.write(f"{param_name} & {value_str} & {error_str} \\\\\n")
            
            # Table footer
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")
            f.write("\\end{table}\n")
        
        print(f"✅ Exported LaTeX table to {filename}")
    
    def _write_publication_table_rows(self, f, summary):
        """Write publication-format table rows organized by ions."""
        # Try to extract ion information
        ion_info = self._detect_ions_and_instruments()
        
        if not ion_info:
            # Fallback to simple parameter listing
            for i, name in enumerate(summary.names):
                param_name = name.replace('_', '\\_')
                f.write(f"{param_name} & & & {summary.best_fit[i]:.3f} & & \\\\\n")
            return
        
        for ion_key, ion_data in ion_info.items():
            ion_name = ion_data['ion_name']
            redshift = ion_data['redshift']
            components = ion_data['components']
            
            # Extract parameters for this ion
            ion_params = self._extract_ion_parameters(ion_data, summary)
            
            for comp_idx in range(components):
                if comp_idx < len(ion_params['N']):
                    # Show ion name only for first component
                    ion_col = ion_name if comp_idx == 0 else ""
                    z_col = f"{redshift:.6f}" if comp_idx == 0 else ""
                    
                    N_val = ion_params['N'][comp_idx]
                    N_err = ion_params['N_err'][comp_idx]
                    b_val = ion_params['b'][comp_idx]
                    b_err = ion_params['b_err'][comp_idx]
                    v_val = ion_params['v'][comp_idx]
                    v_err = ion_params['v_err'][comp_idx]
                    
                    f.write(f"{ion_col} & {z_col} & {comp_idx+1} & "
                           f"{N_val:.2f}$\\pm${N_err:.2f} & "
                           f"{b_val:.1f}$\\pm${b_err:.1f} & "
                           f"{v_val:.1f}$\\pm${v_err:.1f} \\\\\n")
    
    def export_summary_report(self, filename: Union[str, Path], 
                            include_plots: bool = True) -> None:
        """
        Export comprehensive summary report with all results.
        
        Parameters
        ----------
        filename : str or Path
            Output text filename
        include_plots : bool
            Whether to save plots alongside the report
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RBVFIT 2.0 COMPREHENSIVE FIT REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write("BASIC INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sampler: {self.sampler_name}\n")
            f.write(f"Walkers: {self.n_walkers}\n")
            f.write(f"Steps: {self.n_steps}\n")
            f.write(f"Multi-instrument: {self.is_multi_instrument}\n")
            
            if self.is_multi_instrument:
                n_instruments = len(self.instrument_data) if self.instrument_data else 1
                f.write(f"Number of instruments: {n_instruments}\n")
            
            f.write("\n")
            
            # Model configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            if hasattr(self.model, 'config') and self.model.config is not None:
                try:
                    for i, system in enumerate(self.model.config.systems):
                        f.write(f"System {i+1}: z = {system.redshift:.6f}\n")
                        for ion_group in system.ion_groups:
                            transitions_str = ", ".join(f"{w:.1f}" for w in ion_group.transitions)
                            f.write(f"  {ion_group.ion_name}: [{transitions_str}] Å, {ion_group.components} components\n")
                except Exception:
                    f.write("Configuration information not available\n")
            else:
                f.write("Model configuration not available\n")
            
            f.write("\n")
            
            # Convergence diagnostics
            f.write("CONVERGENCE DIAGNOSTICS\n")
            f.write("-" * 40 + "\n")
            convergence = self.convergence_diagnostics(verbose=False)
            status = convergence['overall_status']
            f.write(f"Overall Status: {status}\n\n")
            
            for i, rec in enumerate(convergence['recommendations'], 1):
                # Remove emojis for text file
                clean_rec = rec.encode('ascii', 'ignore').decode('ascii')
                f.write(f"{i}. {clean_rec}\n")
            
            f.write("\n")
            
            # Chi-squared statistics
            f.write("GOODNESS OF FIT\n")
            f.write("-" * 40 + "\n")
            chi2_stats = self.chi_squared()
            if not np.isnan(chi2_stats['chi2']):
                f.write(f"Chi-squared: {chi2_stats['chi2']:.2f}\n")
                f.write(f"Degrees of freedom: {chi2_stats['dof']}\n")
                f.write(f"Reduced chi-squared: {chi2_stats['reduced_chi2']:.3f}\n")
                
                if self.is_multi_instrument and 'chi2_total' in chi2_stats:
                    f.write(f"Combined chi-squared: {chi2_stats['chi2_total']:.2f}\n")
                    f.write(f"Combined reduced chi-squared: {chi2_stats['reduced_chi2_total']:.3f}\n")
            else:
                f.write("Chi-squared statistics not available\n")
            
            f.write("\n")
            
            # Parameter summary
            #f.write("PARAMETER SUMMARY\n")
            #f.write("-" * 40 + "\n")
            #summary = self.parameter_summary("""


    def export_model_data(self, filename: str, format: str = 'ascii') -> None:
        """
        Export model evaluations to external format for publication.
        
        Parameters
        ----------
        filename : str
            Output filename
        format : str
            Format: 'ascii', 'fits', or 'csv'
        """
        if not self.has_model_evaluations():
            raise ValueError("No model evaluations available for export")
        
        if format.lower() == 'ascii':
            self._export_ascii(filename)
        elif format.lower() == 'csv':
            self._export_csv(filename)
        elif format.lower() == 'fits':
            self._export_fits(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_ascii(self, filename: str):
        """Export to ASCII format."""
        with open(filename, 'w') as f:
            f.write("# rbvfit model evaluation export\n")
            f.write(f"# Instruments: {', '.join(self.list_instruments())}\n")
            f.write("# Columns: instrument, wavelength, flux_obs, error_obs, model_flux\n")
            
            for instrument in self.list_instruments():
                # Get data
                if instrument == 'main':
                    wave_obs = self.fitter.wave_obs
                    flux_obs = self.fitter.fnorm
                    error_obs = self.fitter.enorm
                elif self.instrument_data and instrument in self.instrument_data:
                    inst_data = self.instrument_data[instrument]
                    wave_obs = inst_data['wave']
                    flux_obs = inst_data['flux']
                    error_obs = inst_data['error']
                else:
                    continue
                
                # Get model
                model_data = self.get_model_flux(instrument)
                model_flux = model_data['flux']
                
                # Write data
                for i in range(len(wave_obs)):
                    f.write(f"{instrument} {wave_obs[i]:.4f} {flux_obs[i]:.6f} "
                           f"{error_obs[i]:.6f} {model_flux[i]:.6f}\n")
    
    def _export_csv(self, filename: str):
        """Export to CSV format."""
        import pandas as pd
        
        data_rows = []
        
        for instrument in self.list_instruments():
            # Get data and model
            if instrument == 'main':
                wave_obs = self.fitter.wave_obs
                flux_obs = self.fitter.fnorm  
                error_obs = self.fitter.enorm
            elif self.instrument_data and instrument in self.instrument_data:
                inst_data = self.instrument_data[instrument]
                wave_obs = inst_data['wave']
                flux_obs = inst_data['flux']
                error_obs = inst_data['error']
            else:
                continue
            
            model_data = self.get_model_flux(instrument)
            model_flux = model_data['flux']
            
            # Add to dataframe
            for i in range(len(wave_obs)):
                data_rows.append({
                    'instrument': instrument,
                    'wavelength': wave_obs[i],
                    'flux_obs': flux_obs[i],
                    'error_obs': error_obs[i],
                    'model_flux': model_flux[i]
                })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
    
    def _export_fits(self, filename: str):
        """Export to FITS format."""
        try:
            from astropy.io import fits
            from astropy.table import Table
        except ImportError:
            raise ImportError("FITS export requires astropy")
        
        # Create HDU list
        hdul = fits.HDUList()
        
        # Primary HDU
        primary = fits.PrimaryHDU()
        primary.header['ORIGIN'] = 'rbvfit'
        primary.header['COMMENT'] = 'Model evaluation export'
        hdul.append(primary)
        
        # One extension per instrument
        for instrument in self.list_instruments():
            # Get data
            if instrument == 'main':
                wave_obs = self.fitter.wave_obs
                flux_obs = self.fitter.fnorm
                error_obs = self.fitter.enorm
            elif self.instrument_data and instrument in self.instrument_data:
                inst_data = self.instrument_data[instrument]
                wave_obs = inst_data['wave']
                flux_obs = inst_data['flux']
                error_obs = inst_data['error']
            else:
                continue
            
            model_data = self.get_model_flux(instrument)
            model_flux = model_data['flux']
            
            # Create table
            table = Table({
                'wavelength': wave_obs,
                'flux_obs': flux_obs,
                'error_obs': error_obs,
                'model_flux': model_flux
            })
            
            hdu = fits.table_to_hdu(table)
            hdu.header['EXTNAME'] = instrument
            hdul.append(hdu)
        
        hdul.writeto(filename, overwrite=True)
    
    def create_quick_summary(self) -> str:
        """
        Create a quick text summary for logging/reports.
        
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append("rbvfit Results Summary")
        lines.append("=" * 30)
        
        # Basic info
        lines.append(f"Sampler: {self.sampler_name}")
        lines.append(f"Parameters: {len(self.best_theta)}")
        
        if self.is_multi_instrument:
            lines.append(f"Instruments: {len(self.list_instruments())}")
        
        # Chi-squared
        try:
            chi2 = self.chi_squared()
            if 'reduced_chi2' in chi2:
                lines.append(f"χ²/ν: {chi2['reduced_chi2']:.3f}")
        except:
            lines.append("χ²/ν: N/A")
        
        # Capabilities
        capabilities = []
        if self.has_model_evaluations():
            capabilities.append("model")
        if self.has_component_evaluations():
            capabilities.append("components")
        if self.config_metadata:
            capabilities.append("reconstruction")
        
        if capabilities:
            lines.append(f"Available: {', '.join(capabilities)}")
        
        return "\n".join(lines)    

# =============================================================================
# Convenience Functions
# =============================================================================

# Add these functions at the module level (outside the class)

def save_fit_results(fitter, model, filename: Union[str, Path]) -> None:
    """
    Convenience function to save fit results with model evaluations.
    
    This is the RECOMMENDED way to save results as it captures
    model evaluations and components immediately.
    
    Parameters
    ----------
    fitter : vfit
        MCMC fitter object
    model : VoigtModel
        rbvfit 2.0 model object
    filename : str or Path
        Output HDF5 filename
    """
    results = FitResults(fitter, model)  # This will compute evaluations
    results.save(filename)
    print(f"✓ Fit results saved to {filename}")
    print(f"  Model evaluations: {'✓' if results.has_model_evaluations() else '✗'}")
    print(f"  Component analysis: {'✓' if results.has_component_evaluations() else '✗'}")


def load_fit_results(filename: Union[str, Path]) -> FitResults:
    """
    Convenience function to load fit results.
    
    Parameters
    ----------
    filename : str or Path
        HDF5 filename to load
        
    Returns
    -------
    FitResults
        Loaded fit results object
    """
    results = FitResults.load(filename)
    print(f"✓ Fit results loaded from {filename}")
    print(f"  Model evaluations: {'✓' if results.has_model_evaluations() else '✗'}")
    print(f"  Component analysis: {'✓' if results.has_component_evaluations() else '✗'}")
    print(f"  Reconstruction: {'✓' if results.config_metadata else '✗'}")
    return results

