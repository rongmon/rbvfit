"""
Unified results management for rbvfit 2.0 - Updated for V2 vfit interface.

This module provides the UnifiedResults class for managing MCMC fitting results
with a self-contained, fitter-independent design.

Key Features:
- Single fitter initialization - extracts everything needed from vfit object
- Self-contained data storage (no external dependencies after creation)
- Clean save/load without backwards compatibility workarounds  
- Unified treatment of single/multi-instrument cases
- Sampler-agnostic diagnostic storage
- Automatic model config extraction from vfit.instrument_configs
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import h5py
import json
import warnings

# Optional dependencies with fallbacks
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
def get_rbvfit_version():
    """
    Get the current rbvfit version.
    
    Returns
    -------
    str
        Version string, defaults to '2.0' if not available
    """
    try:
        import rbvfit
        return getattr(rbvfit, '__version__', '2.0')
    except ImportError:
        return '2.0'
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


class UnifiedResults:
    """
    Self-contained results container for rbvfit 2.0 MCMC fitting results.
    
    This class stores all essential data extracted from fitter objects and provides
    analysis capabilities without dependencies on external objects.
    
    Core Attributes (extracted once, stored permanently):
    - best_fit: Best-fit parameter values
    - samples: Flattened MCMC samples for analysis
    - chain: Full MCMC chain for trace plots
    - instrument_data: Observational data for all instruments
    - config_metadata: Model configuration for reconstruction
    - Diagnostic values: autocorr_time, rhat, acceptance_fraction
    """
    
    def __init__(self, fitter):
        """
        Initialize unified results from V2 vfit object.
        
        Parameters
        ----------
        fitter : vfit
            V2 vfit object after runmcmc() - contains everything needed
        """
        # Extract core data (no external dependencies after this)
        self.best_fit = self._extract_best_fit(fitter)
        self.samples, self.chain = self._extract_samples_and_chain(fitter)
        self.instrument_data = self._extract_instrument_data(fitter)
        self.config_metadata = self._extract_config_metadata(fitter)

        
        # Extract diagnostics (sampler-specific)
        self.autocorr_time, self.rhat, self.acceptance_fraction = self._extract_diagnostics(fitter)
        
        # Extract metadata
        self.n_walkers = getattr(fitter, 'no_of_Chain', 0)
        self.n_steps = getattr(fitter, 'no_of_steps', 0)
        self.burnin_steps = self._estimate_burnin(fitter)
        self.sampler_name = getattr(fitter, 'sampler_name', 'emcee')
        self.is_multi_instrument = getattr(fitter, 'multi_instrument', False)



    
    # =========================================================================
    # Data Extraction Methods
    # =========================================================================
    
    def _extract_best_fit(self, fitter) -> np.ndarray:
        """Extract best-fit parameters."""
        if hasattr(fitter, 'best_theta') and fitter.best_theta is not None:
            return np.array(fitter.best_theta)
        elif hasattr(fitter, 'theta') and fitter.theta is not None:
            return np.array(fitter.theta)
        else:
            raise ValueError("No parameter values found in fitter")
    
    def _extract_samples_and_chain(self, fitter) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both flattened samples and full chain."""
        if not hasattr(fitter, 'sampler') or fitter.sampler is None:
            raise ValueError("No sampler found - MCMC has not been run")
        
        try:
            # Get full chain first
            if hasattr(fitter.sampler, 'get_chain'):
                chain = fitter.sampler.get_chain()  # (n_steps, n_walkers, n_params)
            else:
                raise AttributeError("Sampler has no get_chain method")
            
            # Estimate burn-in
            burnin = self._estimate_burnin_from_chain(chain)
            
            # Extract post-burn-in samples (flattened)
            try:
                samples = fitter.sampler.get_chain(discard=burnin, flat=True)
            except TypeError:
                # Older versions might not support these parameters
                samples = chain[burnin:].reshape(-1, chain.shape[-1])
            
            return samples, chain
            
        except Exception as e:
            raise RuntimeError(f"Could not extract MCMC samples: {e}")
    
    def _extract_instrument_data(self, fitter) -> Dict[str, Dict]:
        """Extract observational data from V2 vfit instrument_data."""
        instrument_data = {}
        
        # V2 vfit always has instrument_data (even for single instrument)
        if hasattr(fitter, 'instrument_data') and fitter.instrument_data:
            for name, data in fitter.instrument_data.items():
                instrument_data[name] = {
                    'wave': np.array(data['wave']),
                    'flux': np.array(data['flux']), 
                    'error': np.array(data['error'])
                }
        else:
            raise ValueError("V2 vfit object missing instrument_data - incompatible fitter")
        
        return instrument_data
    
    def _extract_diagnostics(self, fitter) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Extract sampler-specific diagnostic values."""
        autocorr_time = None
        rhat = None
        acceptance_fraction = None
        
        if not hasattr(fitter, 'sampler') or fitter.sampler is None:
            return autocorr_time, rhat, acceptance_fraction
        
        sampler_name = getattr(fitter, 'sampler_name', 'emcee').lower()
        
        # Extract autocorrelation time (emcee only)
        if sampler_name == 'emcee':
            try:
                autocorr_time = fitter.sampler.get_autocorr_time()
            except Exception:
                autocorr_time = None
        elif sampler_name == 'zeus':
            # Zeus doesn't have autocorr - explicitly set to None
            autocorr_time = None
        
        # Extract R-hat (zeus only)
        if sampler_name == 'zeus':
            try:
                import zeus
                # Try the newer diagnostics module first
                try:
                    chain = fitter.sampler.get_chain().transpose(1, 0, 2)
                    rhat = zeus.diagnostics.gelman_rubin(chain)
                except AttributeError:
                    # Implement simple R-hat calculation (fallback)
                    chain = fitter.sampler.get_chain()  # (n_steps, n_walkers, n_params)
                    
                    if len(chain.shape) == 3 and chain.shape[0] >= 100:
                        n_steps, n_walkers, n_params = chain.shape
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
                        
                        rhat = np.array(r_hat_values)
                    else:
                        rhat = None
                        
            except ImportError:
                rhat = None
            except Exception:
                rhat = None
        else:
            # Not zeus - no R-hat
            rhat = None
        
        # Extract acceptance fraction (both samplers)
        try:
            if hasattr(fitter.sampler, 'acceptance_fraction'):
                # emcee-style
                af = fitter.sampler.acceptance_fraction
                acceptance_fraction = np.mean(af) if hasattr(af, '__len__') else float(af)
            elif hasattr(fitter.sampler, 'get_chain'):
                # zeus-style: estimate manually
                chain = fitter.sampler.get_chain()  # shape: (n_steps, n_walkers, n_params)
                n_steps, n_walkers, n_params = chain.shape
                n_accepted = 0
                n_total = 0
        
                for w in range(n_walkers):
                    walker_chain = chain[:, w, :]  # shape: (n_steps, n_params)
                    for i in range(1, n_steps):
                        n_total += 1
                        if not np.allclose(walker_chain[i], walker_chain[i-1]):
                            n_accepted += 1
        
                acceptance_fraction = n_accepted / n_total if n_total > 0 else 0.0
            else:
                acceptance_fraction = None
        except Exception:
            acceptance_fraction = None
        
        return autocorr_time, rhat, acceptance_fraction
    
    def correlation_matrix(self) -> np.ndarray:
        """Calculate parameter correlation matrix."""
        return np.corrcoef(self.samples.T)
    
    def _extract_config_metadata(self, fitter) -> Optional[Dict]:
        """Extract model configuration metadata from V2 vfit."""
        try:
            config_data = {
                'rbvfit_version': get_rbvfit_version(),
                'systems': [],
                'instrument_params': {}  # Per-instrument FWHM storage
            }
            
            # V2 vfit stores instrument configs in instrument_configs attribute
            if hasattr(fitter, 'instrument_configs') and fitter.instrument_configs:
                # Extract FWHM and other parameters from each instrument config
                for inst_name, inst_config in fitter.instrument_configs.items():
                    # Initialize instrument params for this instrument
                    config_data['instrument_params'][inst_name] = {}
                    
                    # Extract instrumental parameters
                    if hasattr(inst_config, 'instrumental_params'):
                        for param_name, param_value in inst_config.instrumental_params.items():
                            config_data['instrument_params'][inst_name][param_name] = param_value
                    
                    # Extract system configurations from first instrument (they should be identical)
                    if hasattr(inst_config, 'systems') and not config_data['systems']:
                        for system in inst_config.systems:
                            system_data = {
                                'redshift': system.redshift,
                                'ion_groups': []
                            }
                            
                            for ion_group in system.ion_groups:
                                ion_data = {
                                    'ion_name': ion_group.ion_name,
                                    'transitions': list(ion_group.transitions),
                                    'components': ion_group.components
                                }
                                system_data['ion_groups'].append(ion_data)
                            
                            config_data['systems'].append(system_data)
                
            return config_data    
        except Exception as e:
            warnings.warn(f"Could not extract model configuration: {e}")
            return None
    
    

    
    def _estimate_burnin(self, fitter) -> int:
        """Estimate burn-in steps."""
        n_steps = getattr(fitter, 'no_of_steps', 0)
        
        # Try to use autocorrelation time if available
        if hasattr(fitter, 'sampler') and fitter.sampler is not None:
            try:
                if hasattr(fitter.sampler, 'get_autocorr_time'):
                    tau = fitter.sampler.get_autocorr_time()
                    mean_tau = np.nanmean(tau)
                    if np.isfinite(mean_tau) and mean_tau > 0:
                        burnin = min(int(3 * mean_tau), int(0.4 * n_steps))
                        return max(burnin, int(0.1 * n_steps))
            except Exception:
                pass
        
        # Fallback to 20% of chain
        return int(0.2 * n_steps)
    
    def _estimate_burnin_from_chain(self, chain) -> int:
        """Estimate burn-in from chain shape."""
        n_steps = chain.shape[0]
        return int(0.2 * n_steps)  # Conservative 20%
    
    # =========================================================================
    # Properties (Computed on-demand)
    # =========================================================================
    
    @property
    def parameter_names(self) -> List[str]:
        """Generate parameter names from model config or generic names."""
        n_params = len(self.best_fit)
        
        # Try to use model configuration
        if self.config_metadata is not None:
            try:
                names = self._generate_names_from_config(n_params)
                if len(names) == n_params:
                    return names
            except Exception:
                pass
        
        # Fallback to generic names
        return self._generate_generic_names(n_params)
    
    def _generate_names_from_config(self, n_params: int) -> List[str]:
        """Generate parameter names from model configuration."""
        names = []
        
        # N parameters
        for system in self.config_metadata['systems']:
            z = system['redshift']
            for ion_group in system['ion_groups']:
                ion = ion_group['ion_name']
                for comp in range(ion_group['components']):
                    names.append(f"logN_{ion}_z{z:.3f}_c{comp+1}")
        
        # b parameters  
        for system in self.config_metadata['systems']:
            z = system['redshift']
            for ion_group in system['ion_groups']:
                ion = ion_group['ion_name']
                for comp in range(ion_group['components']):
                    names.append(f"b_{ion}_z{z:.3f}_c{comp+1}")
        
        # v parameters
        for system in self.config_metadata['systems']:
            z = system['redshift']
            for ion_group in system['ion_groups']:
                ion = ion_group['ion_name']
                for comp in range(ion_group['components']):
                    names.append(f"v_{ion}_z{z:.3f}_c{comp+1}")
        
        return names
    
    def _generate_generic_names(self, n_params: int) -> List[str]:
        """Generate generic parameter names."""
        if n_params % 3 != 0:
            return [f"θ[{i}]" for i in range(n_params)]
        
        nfit = n_params // 3
        names = []
        
        # N, b, v pattern
        for i in range(nfit):
            names.append(f"logN_{i+1}")
        for i in range(nfit):
            names.append(f"b_{i+1}")
        for i in range(nfit):
            names.append(f"v_{i+1}")
        
        return names
    
    @property
    def bounds_16th(self) -> np.ndarray:
        """16th percentile of parameters (lower error bound)."""
        return np.percentile(self.samples, 16, axis=0)
    
    @property  
    def bounds_84th(self) -> np.ndarray:
        """84th percentile of parameters (upper error bound)."""
        return np.percentile(self.samples, 84, axis=0)
    
    @property
    def instrument_names(self) -> List[str]:
        """List of instrument names."""
        return list(self.instrument_data.keys())
    
    # =========================================================================
    # Save/Load Methods
    # =========================================================================
    
    def save(self, filename: Union[str, Path]) -> None:
        """
        Save unified results to HDF5 file.
        
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
            meta.attrs['results_version'] = 'unified_redesign'
            meta.attrs['sampler_name'] = self.sampler_name
            meta.attrs['n_walkers'] = self.n_walkers
            meta.attrs['n_steps'] = self.n_steps
            meta.attrs['burnin_steps'] = self.burnin_steps
            meta.attrs['is_multi_instrument'] = self.is_multi_instrument
            meta.attrs['n_instruments'] = len(self.instrument_names)
            
            # Core data
            f.create_dataset('best_fit', data=self.best_fit)
            f.create_dataset('samples', data=self.samples)
            f.create_dataset('chain', data=self.chain)
            
            # Instrument data
            instruments = f.create_group('instruments')
            for name, data in self.instrument_data.items():
                inst_group = instruments.create_group(name)
                inst_group.create_dataset('wave', data=data['wave'])
                inst_group.create_dataset('flux', data=data['flux'])
                inst_group.create_dataset('error', data=data['error'])
            
            # Configuration metadata
            if self.config_metadata is not None:
                config_group = f.create_group('config_metadata')
                config_json = json.dumps(self.config_metadata)
                config_group.attrs['config_data'] = config_json
            
            # Diagnostics (with None handling)
            diag = f.create_group('diagnostics')
            if self.autocorr_time is not None:
                diag.create_dataset('autocorr_time', data=self.autocorr_time)
            if self.rhat is not None:
                diag.create_dataset('rhat', data=self.rhat)
            if self.acceptance_fraction is not None:
                diag.attrs['acceptance_fraction'] = self.acceptance_fraction
    
    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'UnifiedResults':
        """
        Load unified results from HDF5 file.
        
        Parameters
        ----------
        filename : str or Path
            HDF5 filename to load
            
        Returns
        -------
        UnifiedResults
            Loaded results object
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Results file not found: {filename}")
        
        # Create object without calling __init__
        results = cls.__new__(cls)
        
        with h5py.File(filename, 'r') as f:
            # Load metadata
            meta = f['metadata']
            results.sampler_name = meta.attrs.get('sampler_name', 'emcee')
            results.n_walkers = meta.attrs.get('n_walkers', 0)
            results.n_steps = meta.attrs.get('n_steps', 0)
            results.burnin_steps = meta.attrs.get('burnin_steps', 0)
            results.is_multi_instrument = meta.attrs.get('is_multi_instrument', False)
            
            # Load core data
            results.best_fit = f['best_fit'][:]
            results.samples = f['samples'][:]
            results.chain = f['chain'][:]
            
            # Load instrument data
            results.instrument_data = {}
            instruments = f['instruments']
            for name in instruments.keys():
                inst_data = instruments[name]
                results.instrument_data[name] = {
                    'wave': inst_data['wave'][:],
                    'flux': inst_data['flux'][:],
                    'error': inst_data['error'][:]
                }
            
            # Load configuration metadata
            results.config_metadata = None
            if 'config_metadata' in f:
                config_group = f['config_metadata']
                if 'config_data' in config_group.attrs:
                    results.config_metadata = json.loads(config_group.attrs['config_data'])
            
            # Load diagnostics (with None defaults)
            diag = f.get('diagnostics', {})
            results.autocorr_time = diag.get('autocorr_time', [None])[:]
            results.rhat = diag.get('rhat', [None])[:]
            results.acceptance_fraction = diag.attrs.get('acceptance_fraction', None)
            
            # Handle None arrays
            if results.autocorr_time is not None and len(results.autocorr_time) == 1 and results.autocorr_time[0] is None:
                results.autocorr_time = None
            if results.rhat is not None and len(results.rhat) == 1 and results.rhat[0] is None:
                results.rhat = None
        
        return results
    
    # =========================================================================
    # Analysis Methods (Foundation)
    # =========================================================================
    
    # =========================================================================
    # Analysis Methods 
    # =========================================================================
    
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
        diagnostics = {}
        recommendations = []
        
        # 1. Acceptance fraction analysis
        diagnostics['acceptance_fraction'] = {
            'mean': self.acceptance_fraction,
            'individual': self.acceptance_fraction  # Single value for our storage
        }
        
        if self.acceptance_fraction is not None:
            mean_accept = self.acceptance_fraction
            if mean_accept < 0.2:
                recommendations.append("❌ Low acceptance fraction (<0.2). Consider reducing step size or relaxing bounds.")
            elif mean_accept > 0.7:
                recommendations.append("⚠️ High acceptance fraction (>0.7). Consider increasing step size for better mixing.")
            else:
                recommendations.append("✅ Good acceptance fraction (0.2-0.7).")
        else:
            recommendations.append("❓ Could not calculate acceptance fraction.")
        
        # 2. Autocorrelation time analysis (emcee only)
        diagnostics['autocorr_time'] = {
            'tau': self.autocorr_time,
            'mean_tau': np.nanmean(self.autocorr_time) if self.autocorr_time is not None else None
        }
        
        if self.autocorr_time is not None:
            mean_tau = np.nanmean(self.autocorr_time)
            if np.isfinite(mean_tau):
                chain_length_ratio = self.n_steps / mean_tau
                diagnostics['chain_length_ratio'] = chain_length_ratio
                
                if chain_length_ratio < 50:
                    recommended_length = int(50 * mean_tau)
                    recommendations.append(
                        f"⏱️ Chain too short. Current: {self.n_steps} steps, "
                        f"Recommended: >{recommended_length} steps (50x autocorr time)"
                    )
                else:
                    recommendations.append("✅ Chain length adequate (>50x autocorr time).")
            else:
                recommendations.append("❓ Could not determine autocorrelation time.")
        elif self.sampler_name.lower() == 'emcee':
            recommendations.append("❓ Autocorrelation time could not be calculated - chain likely too short")
            recommended_steps = self.n_steps * 3
            recommendations.append(f"⏱️ Recommend running 2-3x longer (try {recommended_steps} steps)")
        
        # 3. Gelman-Rubin R-hat analysis (zeus only)
        diagnostics['gelman_rubin'] = {
            'r_hat': self.rhat,
            'max_r_hat': np.max(self.rhat) if self.rhat is not None else None
        }
        
        if self.rhat is not None:
            max_r_hat = np.max(self.rhat)
            if max_r_hat <= 1.1:
                recommendations.append("✅ Excellent convergence (R-hat ≤ 1.1)")
            elif max_r_hat <= 1.2:
                recommendations.append("⚠️ Marginal convergence (1.1 < R-hat ≤ 1.2). Consider longer chains.")
            else:
                recommendations.append("❌ Poor convergence (R-hat > 1.2). Chains have not converged.")
                recommended_steps = self.n_steps * 2
                recommendations.append(f"🔄 Recommend running 2x longer ({recommended_steps} steps)")
        elif self.sampler_name.lower() == 'zeus':
            recommendations.append("❓ R-hat could not be calculated")
        
        # 4. Effective sample size
        try:
            n_eff = self._estimate_effective_sample_size(self.samples)
            diagnostics['effective_sample_size'] = {
                'n_eff': n_eff,
                'min_n_eff': np.min(n_eff) if n_eff is not None else None
            }
            
            if n_eff is not None:
                min_eff = np.min(n_eff)
                threshold = 50 if self.sampler_name.lower() == 'zeus' else 100
                
                if min_eff >= threshold:
                    recommendations.append(f"✅ Good effective sample size (min: {min_eff:.0f})")
                elif min_eff >= threshold/2:
                    recommendations.append(f"⚠️ Marginal effective sample size (min: {min_eff:.0f})")
                else:
                    recommendations.append(f"❌ Low effective sample size (min: {min_eff:.0f})")
            else:
                recommendations.append("❓ Could not calculate effective sample size")
        except Exception:
            diagnostics['effective_sample_size'] = None
            recommendations.append("❓ Could not calculate effective sample size")
        
        # 5. Overall assessment
        diagnostics['overall_status'] = self._assess_overall_convergence(diagnostics)
        diagnostics['recommendations'] = recommendations
        
        if verbose:
            self._print_convergence_diagnostics(diagnostics)
        
        return diagnostics
    
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
            if mean_accept and (mean_accept < 0.2 or mean_accept > 0.7):
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
                    
                    # Check effective sample size (secondary for zeus)
                    if diagnostics['effective_sample_size'] is not None:
                        min_eff = diagnostics['effective_sample_size']['min_n_eff']
                        if min_eff is not None and min_eff < 50:
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
                return "MARGINAL"  # Conservative when missing key diagnostic
            elif autocorr_failed and len(issues) > 2:
                return "UNKNOWN"  # Too many unknowns
            elif len(issues) == 0:
                return "GOOD"
            elif len(issues) <= 2:
                return "MARGINAL"
            else:
                return "POOR"
    
    def _print_convergence_diagnostics(self, diagnostics):
        """Print formatted convergence diagnostics with sampler-aware recommendations."""
        status = diagnostics['overall_status']
        status_symbols = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}
        
        print("\n" + "=" * 70)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 70)
        
        print(f"Overall Status: {status_symbols.get(status, '❓')} {status}")
        print(f"Sampler: {self.sampler_name}")
        print(f"Walkers: {self.n_walkers}, Steps: {self.n_steps}")
        
        print(f"\nDiagnostics Summary:")
        print("-" * 50)
        
        # Sampler-specific primary diagnostics
        if self.sampler_name.lower() == 'zeus':
            # Zeus: R-hat is primary
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
            
            print("Autocorr Time: ➖ Not available (zeus uses R-hat instead)")
            
        else:
            # Emcee: autocorr is primary
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
                        print(f"Autocorr Time: ⚠️ {mean_tau:.1f} steps")
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
        
        # Common diagnostics
        if diagnostics['acceptance_fraction'] is not None:
            mean_accept = diagnostics['acceptance_fraction']['mean']
            if mean_accept is not None:
                if 0.2 <= mean_accept <= 0.7:
                    symbol = "✅"
                elif 0.1 <= mean_accept <= 0.8:
                    symbol = "⚠️"
                else:
                    symbol = "❌"
                print(f"Acceptance Fraction: {symbol} {mean_accept:.3f}")
            else:
                print("Acceptance Fraction: ❓ Not available")
        
        if diagnostics['effective_sample_size'] is not None:
            min_eff = diagnostics['effective_sample_size']['min_n_eff']
            if min_eff is not None:
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
        
        # Recommendations
        print(f"\nRecommendations:")
        print("-" * 50)
        for rec in diagnostics['recommendations']:
            print(f"{rec}")
        
        # Bottom line
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


    def parameter_summary(self):
        """
        Generate comprehensive parameter summary statistics.
        
        """
        # Calculate percentiles from samples
        percentiles = {
            '16th': np.percentile(self.samples, 16, axis=0),
            '50th': np.percentile(self.samples, 50, axis=0),  # median
            '84th': np.percentile(self.samples, 84, axis=0)
        }
        
        # Calculate other statistics
        mean = np.mean(self.samples, axis=0)
        std = np.std(self.samples, axis=0)
        
        # Use best_fit from object (could be median or other estimator)
        best_fit = self.best_fit
        
        # Calculate symmetric errors (max of lower/upper)
        lower_err = best_fit - percentiles['16th']
        upper_err = percentiles['84th'] - best_fit
        errors = np.maximum(lower_err, upper_err)
        
        # Create summary object
        summary = ParameterSummary(
            names=self.parameter_names,
            best_fit=best_fit,
            errors=errors,
            percentiles=percentiles,
            mean=mean,
            std=std
        )
        
        
        return summary
    

    def _print_parameter_summary(self, summary: ParameterSummary) -> None:
        """Print formatted parameter summary."""
        print("\n" + "=" * 70)
        print("PARAMETER SUMMARY")
        print("=" * 70)
        
        print(f"Sampler: {self.sampler_name}")
        print(f"Walkers: {self.n_walkers}")
        print(f"Steps: {self.n_steps}")
        print(f"Parameters: {len(summary.names)}")
        print(f"Instruments: {len(self.instrument_names)}")
        
        print(f"\nParameter Values:")
        print("-" * 70)
        print(f"{'Parameter':<25} {'Best Fit':<12} {'Error':<12} {'16th':<10} {'84th':<10}")
        print("-" * 70)
        
        for i, name in enumerate(summary.names):
            print(f"{name:<25} {summary.best_fit[i]:11.4f} "
                  f"{summary.errors[i]:11.4f} {summary.percentiles['16th'][i]:9.4f} "
                  f"{summary.percentiles['84th'][i]:9.4f}")
        
        print("=" * 70)
    
    def reconstruct_model(self, instrument_name: str = None):
        """
        Reconstruct VoigtModel from configuration metadata.
        
        Parameters
        ----------
        instrument_name : str, optional
            For multi-instrument cases, specify which instrument.
            If None, returns model for primary/first instrument.
        
        Returns
        -------
        VoigtModel
            Reconstructed model object
            
        Raises
        ------
        ValueError
            If configuration metadata is not available
        """
        if self.config_metadata is None:
            raise ValueError("No configuration metadata available for reconstruction")
        
        try:
            # Import here to avoid circular imports
            from rbvfit.core.fit_configuration import FitConfiguration
            from rbvfit.core.voigt_model import VoigtModel
            
            instrument_params = self.config_metadata.get('instrument_params', {})
            default_fwhm = '6.5'
            
            if instrument_name:
                # Use instrument-specific FWHM
                fwhm = instrument_params.get(instrument_name, {}).get('FWHM', default_fwhm)
            else:
                # Use the only entry in the dictionary (assumes one instrument)
                if len(instrument_params) == 1:
                    sole_instrument = next(iter(instrument_params))
                    fwhm = instrument_params[sole_instrument].get('FWHM', default_fwhm)
                else:
                    fwhm = default_fwhm  # Fallback if multiple instruments or none

            
            # Create configuration with FWHM
            config = FitConfiguration()
            
            # Add systems from metadata
            for system_data in self.config_metadata['systems']:
                z = system_data['redshift']
                
                for ion_data in system_data['ion_groups']:
                    config.add_system(
                        z=z,
                        ion=ion_data['ion_name'],
                        transitions=ion_data['transitions'],
                        components=ion_data['components']
                    )
            
            # Create model (FWHM already in config)
            model = VoigtModel(config,FWHM=fwhm)
            
            return model
            
        except Exception as e:
            raise ValueError(f"Model reconstruction failed: {e}")
    
    def reconstruct_all_models(self):
        """
        Reconstruct VoigtModel objects for all instruments.
        
        Returns
        -------
        Dict[str, VoigtModel]
            Dictionary mapping instrument names to VoigtModel objects
        """
        if self.config_metadata is None:
            raise ValueError("No configuration metadata available for reconstruction")
        
        models = {}
        
        if self.is_multi_instrument:
            # Multi-instrument: create model for each instrument
            for instrument_name in self.instrument_names:
                models[instrument_name] = self.reconstruct_model(instrument_name)
        else:
            # Single instrument
            models[self.instrument_names[0]] = self.reconstruct_model()
        
        return models

    
    def print_summary(self) -> None:
        """Print comprehensive results summary."""
        print("\n" + "=" * 60)
        print("UNIFIED FIT RESULTS SUMMARY")
        print("=" * 60)
        
        # Basic info
        print(f"Sampler: {self.sampler_name}")
        print(f"Walkers: {self.n_walkers}")
        print(f"Steps: {self.n_steps}")
        print(f"Burn-in: {self.burnin_steps}")
        
        # Instrument info
        print(f"\nInstruments: {len(self.instrument_names)}")
        for name in self.instrument_names:
            n_points = len(self.instrument_data[name]['wave'])
            wave_range = (self.instrument_data[name]['wave'].min(), 
                         self.instrument_data[name]['wave'].max())
            print(f"  {name}: {n_points} points, {wave_range[0]:.1f}-{wave_range[1]:.1f} Å")
        
        # Model reconstruction capability
        reconstruction_available = self.config_metadata is not None
        print(f"\nModel Reconstruction: {'✓' if reconstruction_available else '✗'}")
        
        # Diagnostics summary
        print(f"\nDiagnostics Available:")
        print(f"  Autocorr Time: {'✓' if self.autocorr_time is not None else '✗'}")
        print(f"  R-hat: {'✓' if self.rhat is not None else '✗'}")
        print(f"  Acceptance: {'✓' if self.acceptance_fraction is not None else '✗'}")
        
        # Basic parameter info
        print(f"\nParameters: {len(self.best_fit)} fitted")
        print(f"Sample size: {len(self.samples)} MCMC samples")
        
        print("=" * 60)

    # =========================================================================
    # Plotting Methods (delegate to results_plot module)
    # =========================================================================
    
    def chain_trace_plot(self, **kwargs):
        """Create chain trace plots. See results_plot.chain_trace_plot for details."""
        from rbvfit.core.results_plot import chain_trace_plot
        return chain_trace_plot(self, **kwargs)
    
    def corner_plot(self, **kwargs):
        """Create corner plot. See results_plot.corner_plot for details."""  
        from rbvfit.core.results_plot import corner_plot
        return corner_plot(self, **kwargs)
    
    def correlation_plot(self, **kwargs):
        """Create correlation matrix plot. See results_plot.correlation_plot for details."""
        from rbvfit.core.results_plot import correlation_plot
        return correlation_plot(self, **kwargs)
    
    def velocity_plot(self, **kwargs):
        """Create velocity space plot. See results_plot.velocity_plot for details."""
        from rbvfit.core.results_plot import velocity_plot
        return velocity_plot(self, **kwargs)
    
    def residuals_plot(self, **kwargs):
        """Create residuals plot. See results_plot.residuals_plot for details."""
        from rbvfit.core.results_plot import residuals_plot
        return residuals_plot(self, **kwargs)
    
    def diagnostic_summary_plot(self, **kwargs):
        """Create diagnostic summary plot. See results_plot.diagnostic_summary_plot for details."""
        from rbvfit.core.results_plot import diagnostic_summary_plot
        return diagnostic_summary_plot(self, **kwargs)

    #---------
    # Help
    #---------


    def help(self, category: str = None) -> None:
        """
        Display dynamic help for UnifiedResults based on your actual data.
        
        Parameters
        ----------
        category : str, optional
            Specific help category. Options: 'status', 'analysis', 'plotting', 
            'models', 'save', 'data', 'examples'
        """
        if category is None:
            self._help_main()
        elif category.lower() == 'status':
            self._help_status()
        elif category.lower() == 'analysis':
            self._help_analysis()
        elif category.lower() == 'plotting':
            self._help_plotting()
        elif category.lower() == 'models':
            self._help_models()
        elif category.lower() == 'save':
            self._help_save()
        elif category.lower() == 'data':
            self._help_data()
        elif category.lower() == 'examples':
            self._help_examples()
        else:
            print(f"Unknown category '{category}'. Available categories:")
            print("'status', 'analysis', 'plotting', 'models', 'save', 'data',     'examples'")
    
    def _help_main(self) -> None:
        """Main help overview."""
        # Get dynamic info about this results object
        n_instruments = len(self.instrument_names)
        n_params = len(self.best_fit)
        
        # Parse system info from config if available
        system_info = "Unknown system"
        if self.config_metadata and 'systems' in self.config_metadata:
            systems = []
            for sys in self.config_metadata['systems']:
                z = sys['redshift']
                ions = [ion['ion_name'] for ion in sys['ion_groups']]
                systems.append(f"{'/'.join(ions)} (z={z:.3f})")
            system_info = ' + '.join(systems)
        
        # Get convergence status
        conv_status = "Unknown"
        try:
            conv_diag = self.convergence_diagnostics(verbose=False)
            conv_status = conv_diag['overall_status']
        except:
            pass
        
        print(f"\nUnifiedResults Help - Your {system_info}")
        print("=" * 60)
        print(f"📊 Your Data: {n_instruments} instrument(s), {n_params} parameters,     {self.sampler_name} sampler, {conv_status} convergence")
        if self.config_metadata:
            print(f"🔬 System: {system_info}")
        
        print(f"\n📚 Help Categories:")
        print(f"  results.help('status')      # Detailed data status")
        print(f"  results.help('analysis')    # Analysis methods") 
        print(f"  results.help('plotting')    # Plotting functions")
        print(f"  results.help('models')      # Model reconstruction")
        print(f"  results.help('save')        # Save/load operations")
        print(f"  results.help('data')        # Data access")
        print(f"  results.help('examples')    # Quick start examples")
        
        print(f"\n🚀 Quick Start:")
        print(f"  results.print_summary()           # Overview")
        print(f"  results.convergence_diagnostics() # Check {self.sampler_name}     convergence")
        print(f"  results.corner_plot()             # {n_params}-parameter posteriors")
        print()
    
    def _help_status(self) -> None:
        """Detailed status information."""
        print(f"\nYour Results Status")
        print("=" * 40)
        
        # File info
        print(f"💾 Save capability: ✓ Full reconstruction available")
        print(f"🔄 Load status: ✓ Self-contained, no fitter dependencies")
        
        # Data completeness
        n_samples = len(self.samples)
        n_steps, n_walkers, n_params = self.chain.shape
        total_points = sum(len(data['wave']) for data in self.instrument_data.values())
        
        print(f"\nData Completeness:")
        print(f"  ✓ MCMC samples: {n_samples:,} samples × {n_params} parameters")
        print(f"  ✓ Full chain: {n_steps} steps × {n_walkers} walkers × {n_params}     parameters")
        print(f"  ✓ Instrument data: {len(self.instrument_names)} instruments,     {total_points} total points")
        print(f"  {'✓' if self.config_metadata else '✗'} Model config: {'Reconstruction     available' if self.config_metadata else 'No config metadata'}")
        
        # Diagnostics status
        diag_status = []
        if self.autocorr_time is not None:
            diag_status.append("autocorr time")
        if self.rhat is not None:
            diag_status.append("R-hat")
        if self.acceptance_fraction is not None:
            diag_status.append("acceptance fraction")
        
        print(f"  ✓ Diagnostics: {', '.join(diag_status) if diag_status else 'None     available'}")
        
        # Instrument details
        print(f"\nInstrument Details:")
        for name in self.instrument_names:
            data = self.instrument_data[name]
            n_points = len(data['wave'])
            wave_range = (data['wave'].min(), data['wave'].max())
            print(f"  {name}: {n_points} points, {wave_range[0]:.1f}-{wave_range[    1]:.1f} Å")
        print()
    
    def _help_analysis(self) -> None:
        """Analysis methods help."""
        # Get convergence info for context
        conv_info = "Unknown"
        try:
            conv_diag = self.convergence_diagnostics(verbose=False)
            status = conv_diag['overall_status']
            if self.sampler_name.lower() == 'zeus' and self.rhat is not None:
                conv_info = f"{status} (R-hat: {np.max(self.rhat):.2f})"
            elif self.acceptance_fraction is not None:
                conv_info = f"{status} (acceptance: {self.acceptance_fraction:.2f})"
            else:
                conv_info = status
        except:
            pass
        
        print(f"\nAnalysis Methods - {self.sampler_name} Sampler Results")
        print("=" * 50)
        print(f"Your results: {self.n_walkers} walkers × {self.n_steps} steps,     {conv_info}")
        
        print(f"\n🔍 Convergence Diagnostics:")
        print(f"  results.convergence_diagnostics()        # {self.sampler_name}     analysis")
        if self.sampler_name.lower() == 'zeus':
            print(f"  # Your status: R-hat analysis, acceptance diagnostics")
        else:
            print(f"  # Your status: Autocorr time analysis, acceptance diagnostics")
        
        print(f"\n📊 Parameter Analysis:")
        print(f"  summary = results.parameter_summary()    # Your {len(self.best_fit)}     parameters")
        print(f"  corr = results.correlation_matrix()      # {len(self.best_fit)}×{len(    self.best_fit)} correlation matrix")
        
        print(f"\n🎯 Goodness of Fit:")
        if self.config_metadata:
            print(f"  chi2 = results.chi_squared()             # Combined χ²/ν for {len(    self.instrument_names)} instrument(s)")
            if len(self.instrument_names) > 1:
                inst_example = self.instrument_names[0]
                print(f"  chi2_{inst_example.lower()} =     results.chi_squared('{inst_example}') # Single instrument")
        else:
            print(f"  # Model reconstruction needed for chi-squared calculation")
        
        print(f"\n📈 Direct Data Access:")
        print(f"  results.best_fit          # {len(self.best_fit)} best-fit parameter     values")
        print(f"  results.bounds_16th       # 16th percentiles (lower errors)")
        print(f"  results.bounds_84th       # 84th percentiles (upper errors)")
        print(f"  results.samples           # Full MCMC samples ({len(    self.samples)}×{len(self.best_fit)})")
        print()
    
    def _help_plotting(self) -> None:
        """Plotting methods help."""
        print(f"\nPlotting Methods - Your {len(self.instrument_names)} Instrument(s)")
        print("=" * 50)
        
        # Show instrument details
        if len(self.instrument_names) > 1:
            print("Your instruments:", end=" ")
            inst_details = []
            for name in self.instrument_names:
                # Try to get FWHM from config if available
                fwhm = "unknown FWHM"
                if self.config_metadata and 'instrument_params' in self.config_metadata:
                    if name in self.config_metadata['instrument_params']:
                        fwhm_val =     self.config_metadata['instrument_params'][name].get('FWHM',     'unknown')
                        fwhm = f"FWHM={fwhm_val}"
                inst_details.append(f"{name} ({fwhm})")
            print(", ".join(inst_details))
        else:
            print(f"Your instrument: {self.instrument_names[0]}")
        
        print(f"\n📈 Available Plots:")
        print(f"  results.corner_plot()                    # {len(    self.best_fit)}-parameter posterior distributions")
        print(f"  results.chain_trace_plot()               # {self.sampler_name} walker     traces ({self.n_walkers} walkers × {self.n_steps} steps)")
        
        if len(self.instrument_names) > 1:
            print(f"  results.velocity_plot()                  # All {len(    self.instrument_names)} instruments")
            # Find highest resolution instrument if possible
            best_inst = self.instrument_names[0]  # Default to first
            print(f"  results.velocity_plot('{best_inst}')           # Single     instrument")
            print(f"  results.residuals_plot()                 # Model vs data     comparison")
        else:
            print(f"  results.velocity_plot()                  # Absorption line fit")
            print(f"  results.residuals_plot()                 # Model vs data     comparison")
        
        print(f"  results.correlation_plot()               # {len(self.best_fit)}×{len(    self.best_fit)} parameter correlation matrix")
        print(f"  results.diagnostic_summary_plot()        # {self.sampler_name}     convergence dashboard")
        
        print(f"\n💡 Plotting Tips:")
        if len(self.instrument_names) > 1:
            print(f"  • Use specific instrument names for detailed plots")
            print(f"  • correlation_plot() shows which of your {len(self.best_fit)}     parameters are coupled")
        print(f"  • All plots work with loaded results (no fitter needed)")
        
        print(f"\n📁 Save Options:")
        print(f"  results.corner_plot(save_path='corner.png')")
        if len(self.instrument_names) > 1:
            inst_example = self.instrument_names[0]
            print(f"  results.velocity_plot('{inst_example}',     save_path='velocity_{inst_example.lower()}.png')")
        else:
            print(f"  results.velocity_plot(save_path='velocity.png')")
        print()
    
    def _help_models(self) -> None:
        """Model reconstruction help."""
        print(f"\nModel Reconstruction - Your {'Multi-Instrument' if len(    self.instrument_names) > 1 else 'Single-Instrument'} Setup")
        print("=" * 60)
        
        if self.config_metadata is None:
            print("❌ Model reconstruction not available (no config metadata)")
            print("   This happens when UnifiedResults was created without a model     object")
            print("   You'll need to manually recreate your model configuration")
            print()
            return
        
        print("✓ Model reconstruction available (config metadata found)")
        
        if len(self.instrument_names) > 1:
            print(f"\n🔧 Single Instrument Models:")
            for name in self.instrument_names:
                # Try to show FWHM if available
                fwhm_info = ""
                if 'instrument_params' in self.config_metadata and name in     self.config_metadata['instrument_params']:
                    fwhm = self.config_metadata['instrument_params'][name].get('FWHM',     'unknown')
                    fwhm_info = f"    # FWHM={fwhm} pixels"
                print(f"  model_{name.lower()} =     results.reconstruct_model('{name}'){fwhm_info}")
            
            print(f"\n🔧 All Models at Once:")
            print(f"  all_models = results.reconstruct_all_models()")
            print(f"  # Returns: {dict((name, f'model_{name.lower()}') for name in     self.instrument_names)}")
        else:
            print(f"\n🔧 Model Reconstruction:")
            print(f"  model = results.reconstruct_model()")
            # Show FWHM if available
            if 'instrumental_params' in self.config_metadata:
                fwhm = self.config_metadata['instrumental_params'].get('FWHM',     'unknown')
                print(f"  # Uses FWHM={fwhm} pixels")
        
        print(f"\n🧪 Model Evaluation:")
        inst_example = self.instrument_names[0]
        if len(self.instrument_names) > 1:
            print(f"  wave = results.instrument_data['{inst_example}']['wave']")
            print(f"  flux = model_{inst_example.lower()}.evaluate(results.best_fit,     wave)")
        else:
            print(f"  wave = results.instrument_data['{inst_example}']['wave']")
            print(f"  flux = model.evaluate(results.best_fit, wave)")
        
        # Show model details from config
        if 'systems' in self.config_metadata:
            print(f"\n📋 Your Model Details:")
            for sys in self.config_metadata['systems']:
                z = sys['redshift']
                for ion in sys['ion_groups']:
                    ion_name = ion['ion_name']
                    n_comp = ion['components']
                    transitions = ion.get('transitions', [])
                    trans_str = f", transitions: {transitions}" if transitions else ""
                    print(f"  • {ion_name}: {n_comp} component(s) at     z={z:.6f}{trans_str}")
            
            total_components = sum(sum(ion['components'] for ion in sys['ion_groups'])     for sys in self.config_metadata['systems'])
            print(f"  • Total: {len(self.best_fit)} parameters ({total_components}     components: 3N + 3b + 3v)")
        print()
    
    def _help_save(self) -> None:
        """Save/load operations help."""
        print(f"\nSave/Load Methods - Persistent Results Storage")
        print("=" * 50)
        print(f"💾 Current status: {'Loaded from file' if hasattr(self,     '_loaded_from_file') else 'Created from fitter'}")
        
        print(f"\n📁 Save Options:")
        print(f"  results.save('my_analysis.h5')           # Save everything (    recommended)")
        print(f"  results.save('/path/to/results.h5')     # Full path")
        print(f"")
        print(f"  # Convenience functions:")
        print(f"  save_unified_results(fitter, model, 'results.h5')  # During fitting")
        
        print(f"\n📂 Load Options:")
        print(f"  results = UnifiedResults.load('my_analysis.h5')")
        print(f"")
        print(f"  # Convenience function:")
        print(f"  results = load_unified_results('my_analysis.h5')")
        
        print(f"\n✅ What Gets Saved:")
        print(f"  • MCMC samples & chain (full reconstruction capability)")
        print(f"  • Best-fit parameters & uncertainties")
        print(f"  • All instrument data (wave, flux, error arrays)")
        print(f"  • Model configuration ({'✓ included' if self.config_metadata else '✗     missing'})")
        
        # Show actual diagnostics being saved
        saved_diags = []
        if self.autocorr_time is not None:
            saved_diags.append("autocorr time")
        if self.rhat is not None:
            saved_diags.append(f"R-hat")
        if self.acceptance_fraction is not None:
            saved_diags.append(f"acceptance fraction")
        
        print(f"  • {self.sampler_name} diagnostics ({', '.join(saved_diags) if     saved_diags else 'none available'})")
        print(f"  • Convergence metadata ({self.n_walkers} walkers, {self.n_steps}     steps)")
        
        print(f"\n🔄 Perfect Roundtrip:")
        print(f"  results.save('backup.h5')               # Save current state")
        print(f"  loaded = UnifiedResults.load('backup.h5') # Load identical copy")
        print(f"  loaded.corner_plot()                    # Works exactly the same")
        
        print(f"\n💡 Pro Tips:")
        print(f"  • Always save with model: UnifiedResults(fitter, model)")
        print(f"  • HDF5 format = fast, compact, cross-platform")
        print(f"  • Loaded results work identically to fresh results")
        print(f"  • No fitter object needed after loading")
        print()
    
    def _help_data(self) -> None:
        """Data access help."""
        print(f"\nData Access - Your {len(self.instrument_names)} Instrument(s)")
        print("=" * 50)
        
        print(f"📊 Core Arrays:")
        print(f"  results.best_fit                         # {len(self.best_fit)}     parameter values")
        print(f"  results.samples                          # {len(self.samples)}×{len(    self.best_fit)} MCMC samples")
        print(f"  results.chain                            # {self.chain.shape[    0]}×{self.chain.shape[1]}×{self.chain.shape[2]} walker traces")
        
        print(f"\n📈 Parameter Info:")
        print(f"  results.parameter_names                  # {len(    self.parameter_names)} parameter labels")
        print(f"  results.bounds_16th                      # Lower error bounds (16th     percentile)")
        print(f"  results.bounds_84th                      # Upper error bounds (84th     percentile)")
        
        print(f"\n🔬 Instrument Data:")
        print(f"  results.instrument_names                 # {self.instrument_names}")
        print(f"  results.instrument_data                  # Dictionary with all data")
        
        for name in self.instrument_names:
            n_points = len(self.instrument_data[name]['wave'])
            print(f"  results.instrument_data['{name}']        # {n_points} data     points")
        
        print(f"\n🔍 Diagnostics:")
        if self.autocorr_time is not None:
            print(f"  results.autocorr_time                    # Emcee autocorrelation     times")
        if self.rhat is not None:
            print(f"  results.rhat                             # Zeus R-hat values")
        if self.acceptance_fraction is not None:
            print(f"  results.acceptance_fraction              # Acceptance fraction:     {self.acceptance_fraction:.3f}")
        
        print(f"\n⚙️ Metadata:")
        print(f"  results.sampler_name                     # '{self.sampler_name}'")
        print(f"  results.n_walkers                        # {self.n_walkers}")
        print(f"  results.n_steps                          # {self.n_steps}")
        print(f"  results.is_multi_instrument              #     {self.is_multi_instrument}")
        
        if self.config_metadata:
            print(f"  results.config_metadata                  # Model reconstruction     data")
        print()
    
    def _help_examples(self) -> None:
        """Quick start examples."""
        # Get system info for context
        system_info = "your system"
        if self.config_metadata and 'systems' in self.config_metadata:
            systems = []
            for sys in self.config_metadata['systems']:
                z = sys['redshift']
                ions = [ion['ion_name'] for ion in sys['ion_groups']]
                systems.append(f"{'/'.join(ions)}")
            system_info = ' + '.join(systems)
        
        print(f"\nQuick Start Examples - Your {system_info}")
        print("=" * 50)
        
        print(f"🚀 After MCMC Fitting:")
        print(f"  # Save your results (do this first!)")
        print(f"  results = UnifiedResults(fitter, model)")
        print(f"  results.save('{system_info.lower().replace('/', '_').replace(' ',     '_')}_analysis.h5')")
        
        print(f"\n🔄 Later Analysis Session:")
        print(f"  # Load and analyze")
        print(f"  results = UnifiedResults.load('{system_info.lower().replace('/',     '_').replace(' ', '_')}_analysis.h5')")
        print(f"  results.convergence_diagnostics()      # Check {self.sampler_name}     convergence")
        print(f"  results.corner_plot()                  # {len(    self.best_fit)}-parameter posteriors")
        
        if len(self.instrument_names) > 1:
            # Find best resolution instrument (just pick first for now)
            best_inst = self.instrument_names[0]
            print(f"  results.velocity_plot('{best_inst}')         # Single instrument     plot")
        else:
            print(f"  results.velocity_plot()                 # Absorption line fit")
        
        print(f"\n📊 Model Work:")
        print(f"  # Reconstruct and evaluate")
        if len(self.instrument_names) > 1:
            inst_example = self.instrument_names[0]
            print(f"  model = results.reconstruct_model('{inst_example}')")
            print(f"  wave = results.instrument_data['{inst_example}']['wave']")
        else:
            inst_example = self.instrument_names[0]
            print(f"  model = results.reconstruct_model()")
            print(f"  wave = results.instrument_data['{inst_example}']['wave']")
        print(f"  flux = model.evaluate(results.best_fit, wave)")
        
        print(f"\n💾 Workflow Integration:")
        print(f"  # In fitting script:")
        print(f"  save_unified_results(fitter, model, 'results.h5')")
        print(f"")
        print(f"  # In analysis script:")
        print(f"  results = load_unified_results('results.h5')")
        print(f"  results.print_summary()")
        print()        

# =============================================================================
# Convenience Functions
# =============================================================================

def save_unified_results(fitter, filename: Union[str, Path]) -> None:
    """
    Convenience function to save unified results from V2 vfit.
    
    Parameters
    ----------
    fitter : vfit
        V2 vfit object after runmcmc() - contains all needed data
    filename : str or Path
        Output HDF5 filename
    """
    results = UnifiedResults(fitter)
    results.save(filename)
    print(f"✓ Unified results saved to {filename}")
    print(f"  Instruments: {len(results.instrument_names)}")
    print(f"  Reconstruction: {'✓' if results.config_metadata else '✗'}")


def load_unified_results(filename: Union[str, Path]) -> UnifiedResults:
    """
    Convenience function to load unified results.
    
    Parameters
    ----------
    filename : str or Path
        HDF5 filename to load
        
    Returns
    -------
    UnifiedResults
        Loaded results object
    """
    results = UnifiedResults.load(filename)
    print(f"✓ Unified results loaded from {filename}")
    print(f"  Instruments: {len(results.instrument_names)}")
    print(f"  Reconstruction: {'✓' if results.config_metadata else '✗'}")
    return results