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
    
    def __init__(self, fitter, model):
        """
        Initialize results container from fitter and model.
        
        Parameters
        ----------
        fitter : vfit
            MCMC fitter object after running fit
        model : VoigtModel
            The v2.0 model object with configuration
        """
        self.fitter = fitter
        self.model = model
        
        # Validate inputs
        self._validate_inputs()
        
        # Extract key information
        self._extract_mcmc_info()
        
        # Cache for expensive operations
        self._cached_samples = None
        self._cached_param_summary = None
        self._cached_correlation = None
        self._cached_convergence = None
        
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
        self.n_walkers = getattr(self.fitter, 'no_of_Chain', 50)
        self.n_steps = getattr(self.fitter, 'no_of_steps', 1000)
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
    
    # =============================================================================
    # PHASE I: Core Functionality
    # =============================================================================
    
    def save(self, filename: Union[str, Path]) -> None:
        """
        Save complete fit results to HDF5 file for full reproducibility.
        
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
            meta.attrs['n_walkers'] = self.n_walkers
            meta.attrs['n_steps'] = self.n_steps
            meta.attrs['is_multi_instrument'] = self.is_multi_instrument
            
            # MCMC results
            mcmc = f.create_group('mcmc')
            
            # Save full chain
            if hasattr(self.fitter.sampler, 'get_chain'):
                try:
                    chain = self.fitter.sampler.get_chain()
                    mcmc.create_dataset('chain', data=chain)
                except:
                    # Fallback: save flattened samples
                    samples = self._get_samples()
                    mcmc.create_dataset('samples', data=samples)
            
            # Save sampler state for reproducibility
            if hasattr(self.fitter.sampler, 'random_state'):
                # Save random state if available
                try:
                    state = self.fitter.sampler.random_state
                    if state is not None:
                        mcmc.attrs['random_state_available'] = True
                except:
                    pass
            
            # Parameters and bounds
            params = f.create_group('parameters')
            params.create_dataset('initial_guess', data=self.fitter.theta)
            if self.bounds_lower is not None:
                params.create_dataset('bounds_lower', data=self.bounds_lower)
            if self.bounds_upper is not None:
                params.create_dataset('bounds_upper', data=self.bounds_upper)
            
            # Data
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
            
            # Model configuration
            model_group = f.create_group('model')
            try:
                # Save model configuration as JSON string
                config_dict = {
                    'systems': []
                }
                
                for system in self.model.config.systems:
                    sys_dict = {
                        'redshift': system.redshift,
                        'ion_groups': []
                    }
                    for ion_group in system.ion_groups:
                        ion_dict = {
                            'ion_name': ion_group.ion_name,
                            'transitions': ion_group.transitions,
                            'components': ion_group.components
                        }
                        sys_dict['ion_groups'].append(ion_dict)
                    config_dict['systems'].append(sys_dict)
                
                config_json = json.dumps(config_dict)
                model_group.attrs['configuration'] = config_json
                
            except Exception as e:
                warnings.warn(f"Could not save model configuration: {e}")
        
        print(f"✓ Saved fit results to {filename}")
    
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
            
        Note
        ----
        This creates a minimal fitter and model object for analysis.
        Full MCMC functionality may not be available.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Create minimal objects for analysis
        class MinimalFitter:
            """Minimal fitter object for loaded results."""
            pass
        
        class MinimalModel:
            """Minimal model object for loaded results."""
            pass
        
        with h5py.File(filename, 'r') as f:
            # Load metadata
            meta = f['metadata']
            
            # Create minimal fitter
            fitter = MinimalFitter()
            fitter.sampler_name = meta.attrs.get('sampler', 'emcee')
            fitter.no_of_Chain = meta.attrs.get('n_walkers', 50)
            fitter.no_of_steps = meta.attrs.get('n_steps', 1000)
            fitter.multi_instrument = meta.attrs.get('is_multi_instrument', False)
            
            # Load MCMC data
            mcmc = f['mcmc']
            
            # Create a minimal sampler object with samples
            class MinimalSampler:
                def __init__(self, samples_or_chain):
                    if samples_or_chain.ndim == 2:
                        # Flattened samples
                        self._samples = samples_or_chain
                        self._chain = None
                    else:
                        # Full chain
                        self._chain = samples_or_chain
                        self._samples = None
                
                def get_chain(self, discard=0, thin=1, flat=False):
                    if self._chain is not None:
                        if flat:
                            return self._chain[discard::thin].reshape(-1, self._chain.shape[-1])
                        else:
                            return self._chain[discard::thin]
                    else:
                        # Return samples as flattened chain
                        return self._samples[discard::thin]
            
            # Load samples or chain
            if 'chain' in mcmc:
                sampler_data = mcmc['chain'][...]
            elif 'samples' in mcmc:
                sampler_data = mcmc['samples'][...]
            else:
                raise ValueError("No MCMC samples found in file")
            
            fitter.sampler = MinimalSampler(sampler_data)
            
            # Load parameters and data
            params = f['parameters']
            fitter.theta = params['initial_guess'][...]
            fitter.lb = params['bounds_lower'][...] if 'bounds_lower' in params else None
            fitter.ub = params['bounds_upper'][...] if 'bounds_upper' in params else None
            
            data = f['data']
            fitter.wave_obs = data['wave_obs'][...]
            fitter.fnorm = data['flux_norm'][...]
            fitter.enorm = data['error_norm'][...]
            
            # Load multi-instrument data if present
            if 'multi_instrument' in f:
                fitter.instrument_data = {'main': {
                    'wave': fitter.wave_obs,
                    'flux': fitter.fnorm,
                    'error': fitter.enorm
                }}
                
                multi = f['multi_instrument']
                for name in multi.keys():
                    inst_data = multi[name]
                    fitter.instrument_data[name] = {
                        'wave': inst_data['wave'][...],
                        'flux': inst_data['flux'][...],
                        'error': inst_data['error'][...]
                    }
            
            # Create minimal model
            model = MinimalModel()
            
            # Try to load model configuration
            if 'model' in f and 'configuration' in f['model'].attrs:
                try:
                    config_json = f['model'].attrs['configuration']
                    config_dict = json.loads(config_json)
                    
                    # Create minimal config object
                    class MinimalConfig:
                        def __init__(self, config_dict):
                            self.systems = []
                            for sys_dict in config_dict['systems']:
                                sys_obj = type('System', (), {
                                    'redshift': sys_dict['redshift'],
                                    'ion_groups': []
                                })
                                
                                for ion_dict in sys_dict['ion_groups']:
                                    ion_obj = type('IonGroup', (), ion_dict)
                                    sys_obj.ion_groups.append(ion_obj)
                                
                                self.systems.append(sys_obj)
                    
                    model.config = MinimalConfig(config_dict)
                    
                except Exception as e:
                    warnings.warn(f"Could not load model configuration: {e}")
                    model.config = None
            else:
                model.config = None
        
        # Create and return FitResults object
        results = cls.__new__(cls)  # Create without calling __init__
        results.fitter = fitter
        results.model = model
        results.n_walkers = fitter.no_of_Chain
        results.n_steps = fitter.no_of_steps
        results.sampler_name = fitter.sampler_name
        results.bounds_lower = fitter.lb
        results.bounds_upper = fitter.ub
        results.is_multi_instrument = fitter.multi_instrument
        results.instrument_data = getattr(fitter, 'instrument_data', None)
        
        # Initialize caches
        results._cached_samples = None
        results._cached_param_summary = None
        results._cached_correlation = None
        results._cached_convergence = None
        
        print(f"✓ Loaded fit results from {filename}")
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
        """Get autocorrelation time from sampler."""
        try:
            if hasattr(self.fitter.sampler, 'get_autocorr_time'):
                return self.fitter.sampler.get_autocorr_time()
        except:
            pass
        return None
    
    def _get_gelman_rubin(self):
        """Get Gelman-Rubin R-hat statistic for zeus sampler."""
        try:
            if self.sampler_name.lower() == 'zeus':
                import zeus
                chain = self.fitter.sampler.get_chain()
                return zeus.diagnostics.gelman_rubin(chain)
        except:
            pass
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
        """Assess overall convergence status with conservative approach."""
        issues = []
        autocorr_failed = False
        
        # Check acceptance fraction
        if diagnostics['acceptance_fraction'] is not None:
            mean_accept = diagnostics['acceptance_fraction']['mean']
            if mean_accept < 0.2 or mean_accept > 0.7:
                issues.append("acceptance_fraction")
        else:
            issues.append("acceptance_fraction_unavailable")
        
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
        
        # Check Gelman-Rubin
        if diagnostics['gelman_rubin'] is not None:
            max_r_hat = diagnostics['gelman_rubin']['max_r_hat']
            if max_r_hat is not None and max_r_hat > 1.1:
                issues.append("convergence")
        
        # Conservative assessment logic
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
        """Print formatted convergence diagnostics with emojis."""
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
        
        # Autocorrelation time
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
                    print(f"Mean Autocorr Time: {symbol} {mean_tau:.1f} steps")
                    print(f"Chain Length Ratio: {symbol} {ratio:.1f}x autocorr time")
                else:
                    print(f"Mean Autocorr Time: ⚠️ {mean_tau:.1f} steps")
            else:
                print("Autocorr Time: ❌ Could not calculate")
        else:
            print("Autocorr Time: ❓ Not available")
        
        # Effective sample size
        if diagnostics['effective_sample_size'] is not None:
            min_eff = diagnostics['effective_sample_size']['min_n_eff']
            if min_eff is not None:
                if min_eff >= 100:
                    symbol = "✅"
                elif min_eff >= 50:
                    symbol = "⚠️"
                else:
                    symbol = "❌"
                print(f"Min Effective N: {symbol} {min_eff:.0f}")
            else:
                print("Effective Sample Size: ❌ Could not calculate")
        else:
            print("Effective Sample Size: ❓ Not available")
        
        # Gelman-Rubin
        if diagnostics['gelman_rubin'] is not None:
            max_r_hat = diagnostics['gelman_rubin']['max_r_hat']
            if max_r_hat is not None:
                if max_r_hat <= 1.1:
                    symbol = "✅"
                elif max_r_hat <= 1.2:
                    symbol = "⚠️"
                else:
                    symbol = "❌"
                print(f"Max R-hat: {symbol} {max_r_hat:.3f}")
            else:
                print("Gelman-Rubin: ❌ Could not calculate")
        elif self.sampler_name.lower() == 'zeus':
            print("Gelman-Rubin: ❓ Not available (expected for zeus)")
        
        print(f"\nRecommendations:")
        print("-" * 50)
        for i, rec in enumerate(diagnostics['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("=" * 70)
    
    def chain_trace_plot(self, save_path: Optional[str] = None, 
                        n_cols: int = 3, figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
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
        # Get samples and chain
        try:
            if hasattr(self.fitter.sampler, 'get_chain'):
                chain = self.fitter.sampler.get_chain()
            else:
                # Fallback: create artificial chain from samples
                samples = self._get_samples()
                chain = samples.reshape(self.n_walkers, -1, samples.shape[1])
        except Exception as e:
            print(f"Could not extract chain for trace plots: {e}")
            return None
        
        # Get parameter names
        summary = self.parameter_summary(verbose=False)
        param_names = summary.names
        n_params = len(param_names)
        
        # Calculate subplot layout
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Set figure size
        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Normalize axes to a flat list
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]        
        # Plot each parameter
        for i in range(n_params):
            ax = axes[i]
            
            # Plot all walkers for this parameter
            for walker in range(self.n_walkers):
                ax.plot(chain[walker, :, i], alpha=0.7, linewidth=0.5)
            
            # Add best-fit line
            ax.axhline(summary.best_fit[i], color='red', linestyle='--', 
                      linewidth=2, alpha=0.8, label='Best fit')
            
            # Format subplot
            ax.set_title(param_names[i])
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add convergence assessment text
            convergence = self.convergence_diagnostics(verbose=False)
            status = convergence['overall_status']
            status_colors = {
                "GOOD": "green", 
                "MARGINAL": "orange", 
                "POOR": "red", 
                "UNKNOWN": "purple"
            }
            
            ax.text(0.02, 0.98, status, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10, weight='bold',
                   color=status_colors.get(status, 'black'),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        # Overall title
        convergence = self.convergence_diagnostics(verbose=False)
        status = convergence['overall_status']
        #status_symbol = {"GOOD": "✅", "MARGINAL": "⚠️", "POOR": "❌", "UNKNOWN": "❓"}
        status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗","UNKNOWN": "?"}


        
        fig.suptitle(f'Chain Trace Plots - {status_symbol.get(status, "?")} {status} Convergence\n'
                    f'{self.sampler_name} sampler: {self.n_walkers} walkers × {self.n_steps} steps', 
                    fontsize=14, y=0.98)
        
        fig.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room at top for suptitle

        
        # Add interpretation guide
        fig.text(0.02, 0.02, 
                'Good traces: stable mixing around best-fit, no trends or jumps\n'
                'Poor traces: trending, stuck walkers, large jumps, non-stationary behavior',
                fontsize=10, style='italic', alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved trace plot to {save_path}")
        else:
            plt.show()
        
        return fig
    
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
        if self._cached_correlation is not None:
            correlation = self._cached_correlation
        else:
            samples = self._get_samples()
            correlation = np.corrcoef(samples.T)
            self._cached_correlation = correlation
        
        if plot:
            self._plot_correlation_matrix(correlation, save_path)
        
        return correlation
    
    def _plot_correlation_matrix(self, correlation, save_path=None):
        """Plot correlation matrix heatmap."""
        summary = self.parameter_summary(verbose=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        n_params = len(summary.names)
        ax.set_xticks(range(n_params))
        ax.set_yticks(range(n_params))
        ax.set_xticklabels(summary.names, rotation=45, ha='right')
        ax.set_yticklabels(summary.names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # Add correlation values as text
        for i in range(n_params):
            for j in range(n_params):
                if abs(correlation[i, j]) > 0.3:  # Only show significant correlations
                    text = ax.text(j, i, f'{correlation[i, j]:.2f}', 
                                 ha="center", va="center", 
                                 color="white" if abs(correlation[i, j]) > 0.7 else "black",
                                 fontsize=8)
        
        ax.set_title('Parameter Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved correlation plot to {save_path}")
        else:
            plt.show()
    
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
        if not HAS_CORNER:
            raise ImportError(
                "Corner plots require the 'corner' package. "
                "Install with: pip install corner"
            )
        
        # Get samples and parameter info
        samples = self._get_samples()
        summary = self.parameter_summary(verbose=False)
        
        # Default corner plot arguments
        corner_kwargs = {
            'labels': summary.names,
            'truths': summary.best_fit,
            'show_titles': True,
            'title_fmt': '.3f',
            'quantiles': [0.16, 0.5, 0.84],
            'levels': (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)),
            'plot_density': False,
            'plot_datapoints': True,
            'fill_contours': True,
            'max_n_ticks': 3
        }
        
        # Update with user-provided kwargs
        corner_kwargs.update(kwargs)
        
        # Create corner plot
        fig = corner.corner(samples, **corner_kwargs)
        
        # Add title with convergence status
        convergence = self.convergence_diagnostics(verbose=False)
        status = convergence['overall_status']
        status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗"}
        
        fig.suptitle(f'{status_symbol.get(status, "?")} MCMC Results - {status} Convergence\n'
                     f'{self.sampler_name} sampler, {self.n_walkers} walkers, {self.n_steps} steps', 
                     fontsize=14, y=0.96)
        
        fig.tight_layout(rect=[0, 0, 1, 0.98])  # Leave room at top for suptitle
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved corner plot to {save_path}")
        else:
            plt.show()
        
        return fig
    
    # =============================================================================
    # Additional Utility Methods
    # =============================================================================
    
    def chi_squared(self) -> Dict[str, float]:
        """
        Calculate chi-squared statistics for the fit.
        
        Returns
        -------
        dict
            Dictionary containing chi-squared metrics
        """
        summary = self.parameter_summary(verbose=False)
        best_theta = summary.best_fit
        
        # Calculate model for primary dataset
        try:
            model_flux = self.model.evaluate(best_theta, self.fitter.wave_obs)
        except:
            # Fallback for loaded results
            print("Warning: Could not evaluate model. Chi-squared calculation skipped.")
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
        
        # Add multi-instrument contributions if available
        if self.is_multi_instrument and self.instrument_data:
            chi2_total = chi2
            n_total = n_data
            
            for name, inst_data in self.instrument_data.items():
                if name == 'main':
                    continue
                
                try:
                    # Try to evaluate model for this instrument
                    # This will fail for loaded results without full model
                    inst_model = self.model.evaluate(best_theta, inst_data['wave'])
                    inst_chi2 = np.sum((inst_data['flux'] - inst_model)**2 / inst_data['error']**2)
                    
                    chi2_total += inst_chi2
                    n_total += len(inst_data['wave'])
                    chi2_stats[f'chi2_{name}'] = inst_chi2
                    
                except:
                    print(f"Warning: Could not evaluate model for instrument {name}")
            
            chi2_stats['chi2_total'] = chi2_total
            chi2_stats['n_total_points'] = n_total
            chi2_stats['dof_total'] = n_total - n_params
            chi2_stats['reduced_chi2_total'] = chi2_total / (n_total - n_params) if (n_total - n_params) > 0 else np.inf
        
        return chi2_stats
    
    def print_fit_summary(self) -> None:
        """Print comprehensive fit summary."""
        print("\n" + "=" * 80)
        print("RBVFIT 2.0 FIT SUMMARY")
        print("=" * 80)
        
        # Basic info
        print(f"Model: rbvfit 2.0 VoigtModel")
        print(f"Sampler: {self.sampler_name}")
        print(f"Configuration: {self.n_walkers} walkers × {self.n_steps} steps")
        
        if self.is_multi_instrument:
            n_instruments = len(self.instrument_data) if self.instrument_data else 1
            print(f"Multi-instrument fit: {n_instruments} datasets")
        
        # Model configuration
        if hasattr(self.model, 'config') and self.model.config is not None:
            try:
                n_systems = len(self.model.config.systems)
                total_ions = sum(len(sys.ion_groups) for sys in self.model.config.systems)
                total_components = sum(ig.components for sys in self.model.config.systems 
                                     for ig in sys.ion_groups)
                
                print(f"Physical model: {n_systems} system(s), {total_ions} ion group(s), {total_components} component(s)")
                
                for i, system in enumerate(self.model.config.systems):
                    ions = [ig.ion_name for ig in system.ion_groups]
                    print(f"  System {i+1}: z={system.redshift:.6f}, ions={ions}")
                    
            except Exception:
                print("Physical model: Configuration not available")
        
        # Chi-squared statistics
        chi2_stats = self.chi_squared()
        if not np.isnan(chi2_stats['chi2']):
            print(f"\nGoodness of fit:")
            if self.is_multi_instrument and 'chi2_total' in chi2_stats:
                print(f"  Combined χ² = {chi2_stats['chi2_total']:.2f}")
                print(f"  Combined χ²/ν = {chi2_stats['reduced_chi2_total']:.3f}")
                print(f"  DOF = {chi2_stats['dof_total']}")
            else:
                print(f"  χ² = {chi2_stats['chi2']:.2f}")
                print(f"  χ²/ν = {chi2_stats['reduced_chi2']:.3f}")
                print(f"  DOF = {chi2_stats['dof']}")
        
        # Convergence status
        convergence = self.convergence_diagnostics(verbose=False)
        status = convergence['overall_status']
        status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗"}
        print(f"\nConvergence: {status_symbol.get(status, '?')} {status}")
        
        # Quick parameter summary
        summary = self.parameter_summary(verbose=False)
        print(f"\nParameters: {len(summary.names)} fitted")
        
        if len(summary.names) <= 12:  # Show details for small parameter sets
            print("  Best-fit values:")
            for name, value, error in zip(summary.names, summary.best_fit, summary.errors):
                print(f"    {name}: {value:.3f} ± {error:.3f}")
        else:
            print("  (Use .parameter_summary() for detailed values)")
        
        print("=" * 80)
    
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
        
        This is the main plotting function that creates separate figures for each ion,
        with transitions as rows and instruments as columns.
        
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
        # Detect ions and instruments from model and data
        ion_info = self._detect_ions_and_instruments()
        
        if not ion_info:
            print("❌ No ion information could be extracted from model")
            return {}
        
        print(f"📊 Creating velocity plots for {len(ion_info)} ion group(s)")
        
        figures = {}
        
        for ion_key, ion_data in ion_info.items():
            print(f"  📈 Plotting {ion_data['ion_name']} at z={ion_data['redshift']:.6f}")
            
            # Create figure for this ion
            fig = self._create_ion_velocity_figure(
                ion_data, show_components, show_rail_system, 
                figsize_per_panel, velocity_range, **kwargs
            )
            
            figures[ion_key] = fig
            
            # Save individual figure if requested
            if save_path:
                ion_filename = f"{save_path}_{ion_data['ion_name']}_z{ion_data['redshift']:.3f}.pdf"
                fig.savefig(ion_filename, dpi=300, bbox_inches='tight')
                print(f"  ✅ Saved {ion_data['ion_name']} plot to {ion_filename}")
        
        if not save_path:
            plt.show()
        
        return figures
    
    def _detect_ions_and_instruments(self) -> Dict[str, Dict]:
        """
        Detect ion groups and instruments from model configuration and data.
        
        Returns
        -------
        dict
            Dictionary mapping ion keys to ion information
        """
        ion_info = {}
        
        # Try to get ion information from model configuration
        if hasattr(self.model, 'config') and self.model.config is not None:
            try:
                for sys_idx, system in enumerate(self.model.config.systems):
                    for ion_group in system.ion_groups:
                        ion_key = f"{ion_group.ion_name}_z{system.redshift:.6f}"
                        
                        ion_info[ion_key] = {
                            'ion_name': ion_group.ion_name,
                            'redshift': system.redshift,
                            'transitions': ion_group.transitions,
                            'components': ion_group.components,
                            'system_idx': sys_idx
                        }
                        
            except Exception as e:
                print(f"⚠️ Could not extract ion info from model config: {e}")
        
        # Detect instruments
        instruments = ['Primary']
        if self.is_multi_instrument and self.instrument_data:
            instruments = [name for name in self.instrument_data.keys() if name != 'main']
            if 'main' in self.instrument_data:
                instruments = ['Primary'] + [name for name in instruments if name != 'Primary']
        
        # Add instrument info to each ion
        for ion_key in ion_info:
            ion_info[ion_key]['instruments'] = instruments
        
        return ion_info
    
    def _create_ion_velocity_figure(self, ion_data: Dict, show_components: bool,
                                   show_rail_system: bool, figsize_per_panel: Tuple[float, float],
                                   velocity_range: Optional[Tuple[float, float]], **kwargs) -> plt.Figure:
        """
        Create velocity space figure for a single ion group.
        
        Layout: transitions (rows) × instruments (columns)
        """
        transitions = ion_data['transitions']
        instruments = ion_data['instruments']
        ion_name = ion_data['ion_name']
        redshift = ion_data['redshift']
        
        n_transitions = len(transitions)
        n_instruments = len(instruments)
        
        # Calculate figure size
        fig_width = figsize_per_panel[0] * n_instruments
        fig_height = figsize_per_panel[1] * n_transitions
        
        # Create subplot grid
        fig, axes = plt.subplots(n_transitions, n_instruments, 
                                figsize=(fig_width, fig_height))
        
        # Handle single subplot cases
        if n_transitions == 1 and n_instruments == 1:
            axes = [[axes]]
        elif n_transitions == 1:
            axes = [axes]
        elif n_instruments == 1:
            axes = [[ax] for ax in axes]
        
        # Get model parameters for this ion
        summary = self.parameter_summary(verbose=False)
        ion_params = self._extract_ion_parameters(ion_data, summary)
        
        # Plot each transition × instrument combination
        for i, transition in enumerate(transitions):
            for j, instrument in enumerate(instruments):
                ax = axes[i][j]
                
                # Get data for this instrument
                if instrument == 'Primary':
                    wave_data = self.fitter.wave_obs
                    flux_data = self.fitter.fnorm
                    error_data = self.fitter.enorm
                else:
                    inst_data = self.instrument_data[instrument]
                    wave_data = inst_data['wave']
                    flux_data = inst_data['flux']
                    error_data = inst_data['error']
                
                # Convert to velocity space for this transition
                velocity = self._wavelength_to_velocity(wave_data, transition, redshift)
                
                # Plot data and model for this panel
                self._plot_velocity_panel(
                    ax, velocity, flux_data, error_data,
                    ion_data, transition, instrument, ion_params,
                    show_components, show_rail_system and (i == 0),  # Rail only on top row
                    velocity_range, **kwargs
                )
                
                # Panel labeling
                if i == 0:  # Top row
                    ax.set_title(f'{instrument}', fontsize=12, weight='bold')
                if j == 0:  # Left column
                    ax.set_ylabel(f'{transition:.1f} Å\nNormalized Flux', fontsize=10)
                if i == n_transitions - 1:  # Bottom row
                    ax.set_xlabel('Velocity (km/s)', fontsize=10)
        
        # Overall figure title
        convergence = self.convergence_diagnostics(verbose=False)
        status = convergence['overall_status']
        #status_symbol = {"GOOD": "✅", "MARGINAL": "⚠️", "POOR": "❌", "UNKNOWN": "❓"}
        status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗","UNKNOWN": "?"}

        
        fig.suptitle(f'{status_symbol.get(status, "?")} {ion_name} at z = {redshift:.6f}\n'
                    f'rbvfit 2.0: {ion_data["components"]} component(s), {status} convergence',
                    fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room at top for suptitle

        
        #plt.tight_layout()
        return fig
    
    def _extract_ion_parameters(self, ion_data: Dict, summary) -> Dict:
        """Extract parameters for specific ion group."""
        ion_name = ion_data['ion_name']
        redshift = ion_data['redshift']
        components = ion_data['components']
        
        # Find parameters matching this ion
        ion_params = {
            'N': [], 'b': [], 'v': [],
            'N_err': [], 'b_err': [], 'v_err': []
        }
        
        for i, name in enumerate(summary.names):
            # Check if parameter belongs to this ion
            if (ion_name in name and 
                f"z{redshift:.3f}" in name and
                any(f"c{c}" in name for c in range(components))):
                
                if name.startswith('N_'):
                    ion_params['N'].append(summary.best_fit[i])
                    ion_params['N_err'].append(summary.errors[i])
                elif name.startswith('b_'):
                    ion_params['b'].append(summary.best_fit[i])
                    ion_params['b_err'].append(summary.errors[i])
                elif name.startswith('v_'):
                    ion_params['v'].append(summary.best_fit[i])
                    ion_params['v_err'].append(summary.errors[i])
        
        # Convert to arrays and sort by component index
        for key in ion_params:
            ion_params[key] = np.array(ion_params[key])
        
        return ion_params
    
    def _wavelength_to_velocity(self, wavelength: np.ndarray, rest_wavelength: float, 
                               redshift: float) -> np.ndarray:
        """Convert wavelength to velocity space relative to transition."""
        c_kms = 299792.458  # km/s
        
        # Expected observed wavelength at systemic redshift
        lambda_sys = rest_wavelength * (1 + redshift)
        
        # Convert to velocity relative to systemic
        velocity = c_kms * (wavelength / lambda_sys - 1)
        
        return velocity
    
    def _plot_velocity_panel(self, ax, velocity: np.ndarray, flux: np.ndarray, 
                           error: np.ndarray, ion_data: Dict, transition: float,
                           instrument: str, ion_params: Dict, show_components: bool,
                           show_rail: bool, velocity_range: Optional[Tuple[float, float]], **kwargs):
        """Plot data and model for a single velocity panel."""
        
        # Plot data
        ax.step(velocity, flux, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax.step(velocity, error, 'gray', where='mid', alpha=0.3, linewidth=0.5)
        
        # Plot model if possible
        try:
            # Get corresponding wavelength array
            rest_wavelength = transition
            redshift = ion_data['redshift']
            c_kms = 299792.458
            lambda_sys = rest_wavelength * (1 + redshift)
            wavelength = lambda_sys * (1 + velocity / c_kms)
            
            # Evaluate model
            summary = self.parameter_summary(verbose=False)
            model_flux = self.model.evaluate(summary.best_fit, wavelength)
            
            ax.plot(velocity, model_flux, 'r-', linewidth=2, label='Best Fit')
            
            # Plot individual components if requested
            if show_components and len(ion_params['v']) > 0:
                self._add_component_profiles(ax, velocity, wavelength, ion_data, 
                                           transition, ion_params)
            
        except Exception as e:
            print(f"⚠️ Could not evaluate model for {instrument} {transition:.1f}Å: {e}")
        
        # Add rail system for component positions
        if show_rail and len(ion_params['v']) > 0:
            self._add_rail_system(ax, ion_params, velocity_range)
        
        # Format panel
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
        
        if velocity_range:
            ax.set_xlim(velocity_range)
        else:
            # Auto-range around components
            if len(ion_params['v']) > 0:
                v_center = np.mean(ion_params['v'])
                v_range = max(200, np.ptp(ion_params['v']) * 2)
                ax.set_xlim(v_center - v_range, v_center + v_range)
        
        # Add zero velocity reference
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    def _add_component_profiles(self, ax, velocity: np.ndarray, wavelength: np.ndarray,
                              ion_data: Dict, transition: float, ion_params: Dict):
        """Add individual component Voigt profiles to plot."""
        try:
            # This would require access to individual component evaluation
            # For now, just mark component positions
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            
            for i, (v_comp, N_comp, b_comp) in enumerate(zip(
                ion_params['v'], ion_params['N'], ion_params['b']
            )):
                color = colors[i % len(colors)]
                
                # Add vertical line at component velocity
                ax.axvline(v_comp, color=color, linestyle='--', alpha=0.7, 
                          linewidth=2, label=f'Comp {i+1}')
                
                # Add component info text
                if i < 3:  # Only label first 3 components to avoid clutter
                    y_pos = 0.9 - i * 0.15
                    ax.text(0.02, y_pos, f'C{i+1}: N={N_comp:.2f}, b={b_comp:.0f}', 
                           transform=ax.transAxes, fontsize=8, color=color,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                           
        except Exception as e:
            print(f"⚠️ Could not add component profiles: {e}")
    
    def _add_rail_system(self, ax, ion_params: Dict, velocity_range: Optional[Tuple[float, float]]):
        """Add rail system showing component velocity positions."""
        if len(ion_params['v']) == 0:
            return
            
        # Rail positioning
        y_rail = 1.05
        rail_height = 0.03
        tick_height = 0.02
        
        # Component colors
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Determine rail extent
        if velocity_range:
            rail_start, rail_end = velocity_range
        else:
            v_min, v_max = np.min(ion_params['v']), np.max(ion_params['v'])
            v_range = max(100, v_max - v_min)
            rail_start = v_min - v_range * 0.2
            rail_end = v_max + v_range * 0.2
        
        # Draw horizontal rail
        ax.plot([rail_start, rail_end], [y_rail, y_rail], 
               color='gray', linewidth=3, alpha=0.7)
        
        # Add component ticks and labels
        for i, (v_comp, v_err) in enumerate(zip(ion_params['v'], ion_params['v_err'])):
            color = colors[i % len(colors)]
            
            # Vertical tick at component position
            ax.plot([v_comp, v_comp], [y_rail - tick_height, y_rail + tick_height],
                   color=color, linewidth=3, alpha=0.8)
            
            # Error bar if available
            if v_err > 0:
                ax.plot([v_comp - v_err, v_comp + v_err], [y_rail, y_rail],
                       color=color, linewidth=2, alpha=0.5)
            
            # Component label
            ax.text(v_comp, y_rail + tick_height + 0.01, f'C{i+1}', 
                   ha='center', va='bottom', fontsize=9, color=color, weight='bold')
        
        # Adjust y-limits to accommodate rail
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1], y_rail + 0.08))
    
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



# =============================================================================
# Convenience Functions
# =============================================================================

def save_fit_results(fitter, model, filename: Union[str, Path]) -> None:
    """
    Convenience function to save fit results.
    
    Parameters
    ----------
    fitter : vfit
        MCMC fitter object
    model : VoigtModel
        rbvfit 2.0 model object
    filename : str or Path
        Output HDF5 filename
    """
    results = FitResults(fitter, model)
    results.save(filename)


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
    return FitResults.load(filename)