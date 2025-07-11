"""
Clean MCMC fitter for rbvfit 2.0 - V2 unified interface only.

This module provides a clean implementation of MCMC fitting for VoigtModel objects
with automatic model compilation and unified multi-instrument support.

Key Features:
- Unified interface for single and multi-instrument fitting
- Automatic VoigtModel compilation with per-instrument FWHM
- Sampler-agnostic (emcee and zeus support)
- GUI-friendly dictionary interface
"""

from __future__ import print_function
import numpy as np
import sys
import scipy.optimize as op
import warnings
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union
import corner
import matplotlib.pyplot as plt

# Import samplers
import emcee
try:
    import zeus
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False
    zeus = None

from rbvfit import rb_setline as rb

# Detect OS to set multiprocessing context
if sys.platform.startswith('win'):
    # Windows requires 'spawn' context
    MP_CONTEXT = 'spawn'
elif sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    # Use 'fork' for efficiency in Unix-like systems
    MP_CONTEXT = 'fork'

# Set up optimized multiprocessing context
try:
    OptimizedPool = mp.get_context(MP_CONTEXT).Pool
except (AttributeError, RuntimeError):
    # Fallback to default if fork is not available
    OptimizedPool = mp.Pool
    MP_CONTEXT = 'default'



def rb_veldiff(lam_cen,lam_offset):    
    z=(lam_offset/lam_cen) -1.
    C = 299792.458;  #% speed of light [km/sec]
    Beta =((z+1.)**2 - 1.)/(1. + (z+1.)**2.)
    return Beta*C



def vel2shift(Vel):
#%----------------------------------------------------------------
#% vel2shift function    calculate the red/blue shift (Z)
#%                     from velocity.
#% Input  : - vector of Velocities in km/sec.
#% Output : - red/blue shift (Z).
#% Tested : Matlab 2012
#%     By : Rongmon Bordoloi             Dec 2012
#%----------------------------------------------------------------
    C = 299792.458;  #% speed of light [km/sec]
    Beta  = Vel/C;
    Z = np.sqrt((1.+Beta)/(1.-Beta)) - 1.;
    return Z



# Ion-specific bounds lookup table
ION_BOUNDS_TABLE = {
    'HI': {'N': (12.0, 22.0), 'b': (5.0, 100.0), 'v': (-500.0, 500.0)},
    'CIV': {'N': (12.0, 16.0), 'b': (5.0, 80.0), 'v': (-200.0, 200.0)},
    'OVI': {'N': (13.0, 16.0), 'b': (10.0, 100.0), 'v': (-300.0, 300.0)},
    'SiIV': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-150.0, 150.0)},
    'MgII': {'N': (11.0, 16.0), 'b': (5.0, 80.0), 'v': (-100.0, 100.0)},
    'FeII': {'N': (11.0, 16.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'AlIII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'NV': {'N': (12.0, 15.0), 'b': (10.0, 80.0), 'v': (-200.0, 200.0)},
    'OI': {'N': (13.0, 16.0), 'b': (5.0, 50.0), 'v': (-100.0, 100.0)},
    'SiII': {'N': (11.0, 16.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'AlII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'CII': {'N': (13.0, 17.0), 'b': (5.0, 50.0), 'v': (-100.0, 100.0)},
    'NII': {'N': (13.0, 16.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'SiIII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'CIII': {'N': (13.0, 16.0), 'b': (5.0, 80.0), 'v': (-150.0, 150.0)},
    'NiII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'MnII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'CrII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'TiII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
    'ZnII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)}
}


class vfit:
    """
    Clean MCMC fitter for rbvfit 2.0 VoigtModel objects.
    
    This class provides a unified interface for single and multi-instrument 
    MCMC fitting with automatic model compilation.
    
    Examples
    --------
    Single instrument:
    >>> instrument_data = {
    ...     'HIRES': {'model': voigt_model, 'wave': wave, 'flux': flux, 'error': error}
    ... }
    >>> fitter = vfit(instrument_data, theta, lb, ub)
    >>> fitter.runmcmc()
    
    Multi-instrument:
    >>> instrument_data = {
    ...     'HIRES': {'model': voigt_model_a, 'wave': wave_a, 'flux': flux_a, 'error': error_a},
    ...     'FIRE':  {'model': voigt_model_b, 'wave': wave_b, 'flux': flux_b, 'error': error_b}
    ... }
    >>> fitter = vfit(instrument_data, theta, lb, ub)
    >>> fitter.runmcmc()
    """
    
    def __init__(self, instrument_data: Dict, theta, lb, ub,
                 no_of_Chain=50, no_of_steps=1000, perturbation=1e-4,
                 sampler='emcee', skip_initial_state_check=False):
        """
        Initialize V2 MCMC fitter.
        
        Parameters
        ----------
        instrument_data : dict
            Dictionary with instrument data and VoigtModel objects
            Format: {'instrument_name': {'model': VoigtModel, 'wave': array, 'flux': array, 'error': array}}
        theta : array_like
            Initial parameter guess
        lb, ub : array_like
            Lower and upper parameter bounds
        no_of_Chain : int, optional
            Number of MCMC walkers (default: 50)
        no_of_steps : int, optional
            Number of MCMC steps (default: 1000)
        perturbation : float, optional
            Initial walker perturbation (default: 1e-4)
        sampler : str, optional
            MCMC sampler: 'emcee' or 'zeus' (default: 'emcee')
        skip_initial_state_check : bool, optional
            Skip initial state validation (default: False)
        """
        
        # Validate input
        self._validate_unified_instrument_data(instrument_data)
        
        # Compile models and extract configurations
        self.instrument_data = self._compile_models(instrument_data)
        self.instrument_configs = self._extract_configs(instrument_data)
        
        # Set compatibility flag
        self.multi_instrument = len(instrument_data) > 1
        
        # Set MCMC parameters
        self.theta = np.asarray(theta)
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.no_of_Chain = no_of_Chain
        self.no_of_steps = no_of_steps
        self.perturbation = perturbation
        self.skip_initial_state_check = skip_initial_state_check
        
        # Sampler selection
        self.sampler_name = sampler.lower()
        if self.sampler_name not in ['emcee', 'zeus']:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'emcee' or 'zeus'.")
        
        if self.sampler_name == 'zeus' and not HAS_ZEUS:
            raise ImportError("Zeus sampler requested but not installed. Install with: pip install zeus-mcmc")
        
        # Status flags
        self.mcmc_flag = False
        self.sampler = None
        self.best_theta = None
        self.samples = None
        self.ndim = len(self.theta)
        self.nwalkers = no_of_Chain
        
        # Print setup confirmation
        if self.multi_instrument:
            instrument_names = ', '.join(self.instrument_data.keys())
            print(f"✓ V2 Interface: {len(self.instrument_data)} instruments configured ({instrument_names})")
        else:
            print(f"✓ V2 Interface: Single instrument configured")
    
    def _validate_unified_instrument_data(self, instrument_data):
        """Validate unified interface data."""
        if not isinstance(instrument_data, dict):
            raise TypeError("instrument_data must be a dictionary")
        
        if len(instrument_data) == 0:
            raise ValueError("instrument_data cannot be empty")
        
        required_keys = {'model', 'wave', 'flux', 'error'}
        for name, data in instrument_data.items():
            if not isinstance(data, dict):
                raise TypeError(f"instrument_data['{name}'] must be a dictionary")
            
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                raise ValueError(f"instrument_data['{name}'] missing keys: {missing_keys}")
            
            # Check that arrays have same length
            wave_len = len(data['wave'])
            if len(data['flux']) != wave_len or len(data['error']) != wave_len:
                raise ValueError(f"instrument_data['{name}']: wave, flux, and error must have same length")
    
    def _compile_models(self, instrument_data):
        """Compile VoigtModel objects into evaluation functions."""
        compiled_data = {}
        
        for name, data in instrument_data.items():
            model = data['model']
            
            # Check if it's a VoigtModel object
            if hasattr(model, 'config') and hasattr(model, 'compile'):
                # Compile VoigtModel
                compiled_model = model.compile(verbose=False)
                model_func = compiled_model.model_flux
            else:
                # Already compiled function
                model_func = model
            
            compiled_data[name] = {
                'model': model_func,
                'wave': np.asarray(data['wave']),
                'flux': np.asarray(data['flux']),
                'error': np.asarray(data['error']),
                'inv_sigma2': 1.0 / (np.asarray(data['error']) ** 2)
            }
        
        return compiled_data
    
    def _extract_configs(self, instrument_data):
        """Extract model configurations for UnifiedResults with FWHM information."""
        import copy
        
        configs = {}
        for name, data in instrument_data.items():
            model = data['model']
            if hasattr(model, 'config'):
                # Create a deep copy to avoid modifying original
                config_copy = copy.deepcopy(model.config)
                
                # Ensure instrumental_params exists
                if not hasattr(config_copy, 'instrumental_params'):
                    config_copy.instrumental_params = {}
                
                # Extract FWHM and other LSF parameters from model
                if hasattr(model, 'FWHM') and model.FWHM is not None:
                    config_copy.instrumental_params['FWHM'] = model.FWHM
                
                # Extract other LSF parameters
                lsf_params = ['grating', 'life_position', 'cen_wave']
                for param in lsf_params:
                    if hasattr(model, param) and getattr(model, param) is not None:
                        config_copy.instrumental_params[param] = getattr(model, param)
                
                configs[name] = config_copy
        
        return configs

    
    def lnprior(self, theta):
        """Log prior probability - uniform within bounds."""
        if np.any(theta < self.lb) or np.any(theta > self.ub):
            return -np.inf
        return 0.0
    
    def lnlike(self, theta):
        """Log likelihood calculation - unified across all instruments."""
        try:
            lnlike_total = 0.0
            
            for instrument_name, data in self.instrument_data.items():
                # Evaluate model for this instrument
                model_dat = data['model'](theta, data['wave'])
                
                # Calculate likelihood contribution
                inv_sigma2 = data['inv_sigma2']#1.0 / (data['error'] ** 2)
                lnlike_instrument = -0.5 * (
                    np.sum((data['flux'] - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2))
                )
                
                lnlike_total += lnlike_instrument
            
            return lnlike_total
            
        except Exception:
            # Return -inf for any evaluation errors
            return -np.inf

    #def lnlike(self, theta):
    #    """Log likelihood calculation - unified across all instruments.
    #        More accurate but slower. no need to use this for mcmc 
    #     """
    #    try:
    #        lnlike_total = 0.0
    #
    #        for instrument_name, data in self.instrument_data.items():
    #            flux = data['flux']
    #            error = data['error']
    #            wave = data['wave']
    #            model_flux = data['model'](theta, wave)
    #
    #            inv_sigma2 = 1.0 / (error ** 2)
    #            residual = flux - model_flux
    #
    #            # Vectorized log-likelihood for this instrument
    #            lnlike = -0.5 * np.sum(residual**2 * inv_sigma2 + np.log(2 * np.pi * error**2))
    #            lnlike_total += lnlike
    #
    #        return lnlike_total
    #
    #    except Exception:
    #        return -np.inf
    

    
    def lnprob(self, theta):
        """Log posterior probability."""
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)
    
    def optimize_guess(self, theta):
        """Optimize starting guess using scipy."""
        bounds = list(zip(self.lb, self.ub))
        nll = lambda theta: -self.lnprob(theta)
        result = op.minimize(nll, theta, method='L-BFGS-B', bounds=bounds)
        return result.x
    
    def fit_quick(self, verbose=True):
        """
        Quick deterministic fitting using scipy optimization.
        
        Parameters
        ----------
        verbose : bool, optional
            Print fitting progress
            
        Returns
        -------
        theta_best : np.ndarray
            Best-fit parameters
        theta_best_error : np.ndarray
            Parameter uncertainties (1-sigma)
        """
        self.mcmc_flag = False
        try:
            from rbvfit.core.quick_fit_interface import quick_fit_vfit
        except ImportError as e:
            raise ImportError("Quick fitting requires scipy") from e
        
        # Multi-instrument info
        if self.multi_instrument and verbose:
            print(f"🚀 Multi-instrument quick fit: {len(self.instrument_data)} instruments")
        elif verbose:
            print("🚀 Starting quick fit...")
        
        try:
            theta_best, theta_best_error = quick_fit_vfit(self)
            
            if verbose:
                print("✅ Quick fit completed")
                if np.any(np.isnan(theta_best_error)):
                    print("⚠️  Some uncertainties are NaN")
            
            self.theta_best = theta_best
            self.theta_best_error = theta_best_error
            
            return theta_best, theta_best_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Quick fit failed: {str(e)}")
            raise RuntimeError("Quick fitting failed") from e
    
    def _setup_emcee_sampler(self, use_pool=True):
        """Set up emcee sampler with optimized multiprocessing."""
        if use_pool:
            try:
                # Use optimized pool with proper context
                pool = OptimizedPool()
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, pool=pool)
                return sampler, pool
            except Exception as e:
                warnings.warn(f"Failed to create multiprocessing pool: {e}")
                # Fall back to no pool
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)
                return sampler, None
        else:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)
            return sampler, None
    
    def _setup_zeus_sampler(self, use_pool=True):
        """Set up zeus sampler with optimized multiprocessing."""
        if use_pool:
            try:
                # Use optimized pool with proper context
                pool = OptimizedPool()
                sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, pool=pool)
                return sampler, pool
            except Exception as e:
                warnings.warn(f"Failed to create multiprocessing pool: {e}")
                # Fall back to no pool
                sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)
                return sampler, None
        else:
            sampler = zeus.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)
            return sampler, None
    
    def _initialize_walkers(self, popt):
        """Initialize walker positions with proper bounds checking."""
        guesses = []
        max_attempts = 1000
        
        for i in range(self.nwalkers):
            attempts = 0
            while attempts < max_attempts:
                # Create perturbed guess
                guess = popt + self.perturbation * np.random.randn(self.ndim)
                
                # Ensure within bounds
                guess = np.clip(guess, self.lb + 1e-10, self.ub - 1e-10)
                
                # Check if valid
                if np.isfinite(self.lnprob(guess)):
                    guesses.append(guess)
                    break
                    
                attempts += 1
            
            if attempts >= max_attempts:
                raise RuntimeError(f"Could not initialize walker {i} after {max_attempts} attempts")
        
        return np.array(guesses)
    
    def _get_acceptance_fraction(self, sampler):
        """Get acceptance fraction in a sampler-agnostic way."""
        if hasattr(sampler, 'acceptance_fraction'):
            # emcee
            return sampler.acceptance_fraction
        elif hasattr(sampler, 'get_chain') and self.sampler_name == 'zeus':
            try:
                chain = sampler.get_chain()  # shape: (nsteps, nwalkers, ndim)
                n_accepted = 0
                n_total = 0
                nsteps, nwalkers, _ = chain.shape
                for w in range(nwalkers):
                    walker_chain = chain[:, w, :]
                    for i in range(1, nsteps):
                        n_total += 1
                        if np.any(walker_chain[i] != walker_chain[i-1]):
                            n_accepted += 1
                return n_accepted / n_total if n_total > 0 else 0.0
            except Exception:
                print("Could not calculate acceptance fraction for Zeus")
                return None
        else:
            raise NotImplementedError("Acceptance fraction not implemented for this sampler.")
    
    def runmcmc(self, optimize=True, verbose=True, use_pool=True, progress=True):
        """
        Run MCMC sampling.
        
        Parameters
        ----------
        optimize : bool, optional
            Optimize starting guess before MCMC (default: True)
        verbose : bool, optional
            Print progress and diagnostics (default: True)
        use_pool : bool, optional
            Use multiprocessing pool (default: True)
        progress : bool, optional
            Show MCMC progress bar (default: True)
        """
        if optimize:
            if verbose:
                print("🔧 Optimizing starting guess...")
            self.theta = self.optimize_guess(self.theta)
            if verbose:
                print("✓ Starting guess optimized")
        
        # Create initial walker positions
        guesses = self._initialize_walkers(self.theta)
        
        # Set up sampler
        if self.sampler_name == 'emcee':
            sampler, pool = self._setup_emcee_sampler(use_pool)
        elif self.sampler_name == 'zeus':
            sampler, pool = self._setup_zeus_sampler(use_pool)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_name}")
        
        if verbose:
            print(f"🚀 Starting {self.sampler_name} MCMC...")
            print(f"   Walkers: {self.nwalkers}")
            print(f"   Steps: {self.no_of_steps}")
            print(f"   Instruments: {len(self.instrument_data)}")
            if pool:
                print(f"   Multiprocessing: {MP_CONTEXT}")
        
        # Run MCMC
        try:
            if self.sampler_name == 'emcee':
                sampler.run_mcmc(guesses, self.no_of_steps, 
                                progress=progress,
                                skip_initial_state_check=self.skip_initial_state_check)
            elif self.sampler_name == 'zeus':
                sampler.run_mcmc(guesses, self.no_of_steps, progress=progress)
        except Exception as e:
            if pool:
                pool.close()
                pool.join()
            raise RuntimeError(f"MCMC sampling failed: {e}")
        
        # Clean up pool
        if pool:
            pool.close()
            pool.join()
        
        # Store results
        self.sampler = sampler
        self.mcmc_flag = True
        
        # Extract best fit and samples
        self.compute_best_theta()
        
        if verbose:
            print("✅ MCMC completed")
            self._print_diagnostics(verbose=verbose)
    
    def compute_best_theta(self, burntime=100):
        """Compute best-fit parameters and their uncertainties from MCMC samples."""
        if not hasattr(self, 'samples') or self.samples is None:
            self.samples = self._extract_samples(self.sampler, burntime)
    
        self.best_theta = np.percentile(self.samples, 50, axis=0)
        self.low_theta = np.percentile(self.samples, 16, axis=0)
        self.high_theta = np.percentile(self.samples, 84, axis=0)
    
    def _extract_samples(self, sampler, burntime):
        """Extract samples from sampler with burn-in."""
        try:
            if self.sampler_name == 'emcee':
                samples = sampler.get_chain(discard=burntime, flat=True)
            elif self.sampler_name == 'zeus':
                samples = sampler.get_chain(discard=burntime, flat=True)
            else:
                # Fallback
                chain = sampler.get_chain()
                samples = chain[burntime:, :, :].reshape((-1, self.ndim))
            
            return samples
        except Exception as e:
            warnings.warn(f"Could not extract samples: {e}")
            return np.array([])
    
    def _print_diagnostics(self, verbose=True):
        """Print MCMC diagnostics."""
        if not self.mcmc_flag or not verbose:
            return
        
        print("\n" + "="*60)
        print("MCMC DIAGNOSTICS")
        print("="*60)
        
        # Acceptance fraction
        try:
            acceptance_fraction = self._get_acceptance_fraction(self.sampler)
            if acceptance_fraction is not None:
                if hasattr(acceptance_fraction, '__len__'):
                    mean_acceptance_fraction = np.mean(acceptance_fraction)
                    print(f"Mean acceptance fraction: {mean_acceptance_fraction:.3f}")
                    
                    if mean_acceptance_fraction < 0.2:
                        print("⚠️  Low acceptance fraction (<0.2). Consider reducing step size.")
                    elif mean_acceptance_fraction > 0.7:
                        print("⚠️  High acceptance fraction (>0.7). Consider increasing step size.")
                    else:
                        print("✅ Good acceptance fraction (0.2-0.7)")
                else:
                    print(f"Acceptance fraction: {acceptance_fraction:.3f}")
        except Exception:
            print("Acceptance fraction not available")
        
        # Autocorrelation time (emcee)
        if self.sampler_name == 'emcee':
            try:
                autocorr_time = self.sampler.get_autocorr_time()
                mean_autocorr_time = np.nanmean(autocorr_time)
                print(f"Mean auto-correlation time: {mean_autocorr_time:.3f} steps")
                
                # Check if chain is long enough
                if self.no_of_steps < 50 * mean_autocorr_time:
                    print("⚠️  Warning: Chain may be too short for reliable results")
                    print(f"   Recommended: >{50 * mean_autocorr_time:.0f} steps")
                else:
                    print("✅ Chain length adequate")
            except Exception:
                print("⚠️  Warning: Could not calculate auto-correlation time")
        
        # Gelman-Rubin R-hat (zeus)
        if self.sampler_name == 'zeus':
            try:
                import zeus.diagnostics
                chain = self.sampler.get_chain().transpose(1, 0, 2)
                rhat = zeus.diagnostics.gelman_rubin(chain)
                max_rhat = np.max(rhat)
                print(f"Gelman-Rubin R-hat: {max_rhat:.3f}")
                if max_rhat > 1.1:
                    print("⚠️  Warning: R-hat > 1.1, chains may not have converged")
                else:
                    print("✅ Good convergence (R-hat < 1.1)")
            except Exception:
                print("Could not calculate Gelman-Rubin diagnostic")
        
        # Parameter summary
        if hasattr(self, 'best_theta') and self.best_theta is not None:
            print(f"\nParameter Summary:")
            print(f"Best-fit parameters: {len(self.best_theta)} values")
            if hasattr(self, 'samples') and self.samples is not None:
                print(f"Effective samples: {len(self.samples)}")
        
        print("="*60)
    
    def get_samples(self, flat=True, burn_in=0.5):
        """Get MCMC samples."""
        if not self.mcmc_flag:
            raise RuntimeError("MCMC has not been run")
        
        discard = int(burn_in * self.no_of_steps)
        return self.sampler.get_chain(discard=discard, flat=flat)
    
    def plot_corner(self, outfile=False, burntime=100, **kwargs):
        """Plot corner plot with sampler-agnostic sample extraction."""
        
        if self.mcmc_flag is not True:
            print("⚠️  MCMC Fit not done! Can't do corner plot. Exiting gracefully.")
            return
        
        # Extract samples
        samples = self._extract_samples(self.sampler, burntime)
        
        if len(samples) == 0:
            print("⚠️  No samples available for corner plot")
            return
        
        # Create corner plot
        try:
            fig = corner.corner(samples, **kwargs)
            
            if outfile:
                fig.savefig(outfile, dpi=300, bbox_inches='tight')
                print(f"Corner plot saved to {outfile}")
            else:
                plt.show()
                
            return fig
        except Exception as e:
            print(f"Error creating corner plot: {e}")
            return None


def set_bounds(nguess, bguess, vguess, **kwargs):
    """
    Set parameter bounds for Voigt profile fitting.
    
    Parameters
    ----------
    nguess : list
        Column density guesses [log10(N)]
    bguess : list
        Doppler parameter guesses [km/s]
    vguess : list
        Velocity guesses [km/s]
    ions : list, optional
        Ion names for ion-specific bounds
    **kwargs
        Additional bound overrides
    
    Returns
    -------
    bounds : list
        List of (lower, upper) bounds for each parameter
    lb : np.ndarray
        Lower bounds array
    ub : np.ndarray
        Upper bounds array
    """
    nguess = np.asarray(nguess)
    bguess = np.asarray(bguess)
    vguess = np.asarray(vguess)
    
    # Check if ion-aware bounds are requested
    ions = kwargs.get('ions', None)
    custom_ion_bounds = kwargs.get('ion_bounds', {})
    
    if ions is not None:
        # Use ion-aware bounds
        if len(ions) != len(nguess):
            raise ValueError(f"Length of ions list ({len(ions)}) must match number of components ({len(nguess)})")
        
        # Initialize bounds arrays
        Nlow = np.zeros_like(nguess)
        NHI = np.zeros_like(nguess)
        blow = np.zeros_like(bguess)
        bHI = np.zeros_like(bguess)
        vlow = np.zeros_like(vguess)
        vHI = np.zeros_like(vguess)
        
        # Apply ion-specific bounds
        for i, ion in enumerate(ions):
            # Use custom ion bounds if provided, otherwise use lookup table
            if ion in custom_ion_bounds:
                ion_data = custom_ion_bounds[ion]
            elif ion in ION_BOUNDS_TABLE:
                ion_data = ION_BOUNDS_TABLE[ion]
            else:
                print(f"Warning: Ion '{ion}' not found in bounds table, using defaults")
                # Fall back to traditional bounds for this component
                Nlow[i] = nguess[i] - 2.0
                NHI[i] = nguess[i] + 2.0
                blow[i] = max(2.0, bguess[i] - 40.0)
                bHI[i] = min(150.0, bguess[i] + 40.0)
                vlow[i] = ion_data['v'][0]
            vHI[i] = ion_data['v'][1]
    
    else:
        # Traditional bounds (original behavior)
        Nlow = nguess - 2.0
        blow = np.clip(bguess - 40.0, 2.0, None)
        vlow = vguess - 50.0
        NHI = nguess + 2.0
        bHI = np.clip(bguess + 40.0, None, 150.0)
        vHI = vguess + 50.0
    
    # Apply custom overrides (highest priority)
    if 'Nlow' in kwargs:
        Nlow = np.asarray(kwargs['Nlow'])
    if 'blow' in kwargs:
        blow = np.asarray(kwargs['blow'])
    if 'vlow' in kwargs:
        vlow = np.asarray(kwargs['vlow'])
    if 'Nhi' in kwargs:
        NHI = np.asarray(kwargs['Nhi'])
    if 'bhi' in kwargs:
        bHI = np.asarray(kwargs['bhi'])
    if 'vhi' in kwargs:
        vHI = np.asarray(kwargs['vhi'])
    
    # Concatenate bounds
    lb = np.concatenate([Nlow, blow, vlow])
    ub = np.concatenate([NHI, bHI, vHI])
    bounds = [lb, ub]
    
    return bounds, lb, ub


def add_ion_to_bounds_table(ion_name, N_range, b_range, v_range):
    """
    Add a new ion to the bounds lookup table.
    
    Parameters
    ----------
    ion_name : str
        Ion name (e.g., 'MgII')
    N_range : tuple
        (min, max) log column density
    b_range : tuple
        (min, max) Doppler parameter in km/s
    v_range : tuple
        (min, max) velocity in km/s
    
    Example
    -------
    >>> add_ion_to_bounds_table('CaII', (11.0, 14.0), (3.0, 80.0), (-200.0, 200.0))
    """
    ION_BOUNDS_TABLE[ion_name] = {
        'N': N_range,
        'b': b_range,
        'v': v_range
    }
    print(f"Added {ion_name} to bounds table: N={N_range}, b={b_range}, v={v_range}")


def list_available_ions():
    """List all ions available in the bounds table."""
    print("Available ions in bounds table:")
    print("-" * 40)
    for ion, bounds in ION_BOUNDS_TABLE.items():
        print(f"{ion:6s}: N={bounds['N']}, b={bounds['b']}, v={bounds['v']}")


def get_ion_bounds(ion_name):
    """
    Get bounds for a specific ion.
    
    Parameters
    ----------
    ion_name : str
        Ion name
    
    Returns
    -------
    dict or None
        Dictionary with 'N', 'b', 'v' bounds or None if not found
    """
    return ION_BOUNDS_TABLE.get(ion_name, None)


def get_available_samplers():
    """Get list of available MCMC samplers."""
    samplers = ['emcee']
    if HAS_ZEUS:
        samplers.append('zeus')
    return samplers


def print_sampler_info():
    """Print information about available samplers."""
    print("Available MCMC Samplers:")
    print("=" * 30)
    print("✓ emcee: Affine-invariant ensemble sampler (default)")
    
    if HAS_ZEUS:
        print("✓ zeus: Slice sampling ensemble sampler (high performance)")
        print("\nRecommendations:")
        print("- Use 'emcee' for most cases (well-tested, robust)")
        print("- Use 'zeus' for high-dimensional problems or difficult posteriors")
        print("- Zeus often needs fewer walkers and may converge faster")
    else:
        print("✗ zeus: Not installed (pip install zeus-mcmc)")
        print("\nTo enable zeus sampler:")
        print("  pip install zeus-mcmc")
    
    print(f"\nCurrent backend: emcee {emcee.__version__}")
    if HAS_ZEUS:
        print(f"Zeus version: {zeus.__version__}")
    
    print(f"\nMultiprocessing context: {MP_CONTEXT}")
    if MP_CONTEXT == 'fork':
        print("✓ Optimal for Unix systems (Mac/Linux)")
    else:
        print("⚠ Consider using Unix system for better MP performance")


def print_performance_tips():
    """Print performance optimization tips."""
    print("\nPerformance Tips:")
    print("=" * 20)
    print("🚀 Speed Optimization:")
    print("• Use compiled models (VoigtModel.compile())")
    print("• Enable multiprocessing with use_pool=True")
    print("• On Mac/Linux: automatic 'fork' context provides best MP performance")
    print("• Zeus often converges faster than emcee for difficult posteriors")
    
    print("\n⚙️ Memory Optimization:")
    print("• Fork context shares memory efficiently")
    print("• Use fewer walkers with Zeus (often 2/3 of emcee requirements)")
    print("• Consider thinning for very long chains")
    
    print("\n🎯 Convergence Tips:")
    print("• Check acceptance fractions (0.2-0.7 ideal)")
    print("• Monitor R-hat < 1.1 for Zeus")
    print("• Ensure chain length > 50x autocorrelation time")
    print("• Use optimize=True for better starting positions")
    
    print("\n🔧 Multi-Instrument Tips:")
    print("• Use consistent parameter bounds across instruments")
    print("• Monitor per-instrument likelihood contributions")
    print("• Consider instrument-specific noise models")
    print("• Joint fitting provides better parameter constraints")


def print_usage_examples():
    """Print usage examples for the V2 unified interface."""
    print("\n" + "="*60)
    print("RBVFIT 2.0 MCMC FITTING - V2 UNIFIED INTERFACE")
    print("="*60)
    
    print("\n🆕 Unified Interface Examples:")
    print("="*35)
    
    print("\n📖 Single Instrument:")
    print("```python")
    print("import rbvfit.vfit_mcmc as mc")
    print("")
    print("# Define instrument data")
    print("instrument_data = {")
    print("    'HIRES': {")
    print("        'model': voigt_model,")
    print("        'wave': wave_array,")
    print("        'flux': flux_array,")
    print("        'error': error_array")
    print("    }")
    print("}")
    print("")
    print("# Create fitter")
    print("fitter = mc.vfit(instrument_data, theta, lb, ub)")
    print("fitter.runmcmc()")
    print("```")
    
    print("\n📖 Multi-Instrument:")
    print("```python")
    print("# Define multiple instruments")
    print("instrument_data = {")
    print("    'HIRES': {")
    print("        'model': model_hires,")
    print("        'wave': wave_hires,")
    print("        'flux': flux_hires,")
    print("        'error': error_hires")
    print("    },")
    print("    'FIRE': {")
    print("        'model': model_fire,")
    print("        'wave': wave_fire,")
    print("        'flux': flux_fire,")
    print("        'error': error_fire")
    print("    },")
    print("    'UVES': {")
    print("        'model': model_uves,")
    print("        'wave': wave_uves,")
    print("        'flux': flux_uves,")
    print("        'error': error_uves")
    print("    }")
    print("}")
    print("")
    print("# Same interface - automatically detects multi-instrument!")
    print("fitter = mc.vfit(instrument_data, theta, lb, ub)")
    print("fitter.runmcmc()")
    print("```")
    
    print("\n📖 Ion-Aware Bounds:")
    print("```python")
    print("# Use ion-specific parameter bounds")
    print("nguess = [14.2, 13.8]  # MgII, FeII")
    print("bguess = [15.0, 12.0]")
    print("vguess = [25.0, 30.0]")
    print("")
    print("bounds, lb, ub = mc.set_bounds(")
    print("    nguess, bguess, vguess,")
    print("    ions=['MgII', 'FeII']")
    print(")")
    print("```")
    
    print("\n📖 Quick Fitting:")
    print("```python")
    print("# Fast deterministic fitting")
    print("theta_best, theta_err = fitter.fit_quick()")
    print("```")
    
    print("\n📖 Advanced Sampler Options:")
    print("```python")
    print("# Use Zeus sampler for high-dimensional problems")
    print("fitter = mc.vfit(")
    print("    instrument_data, theta, lb, ub,")
    print("    sampler='zeus',")
    print("    no_of_Chain=40,    # Fewer walkers with Zeus")
    print("    no_of_steps=2000")
    print(")")
    print("```")
    
    print("\n✨ Key Features:")
    print("• Clean, symmetric treatment of all instruments")
    print("• Automatic single/multi-instrument detection")
    print("• No legacy interface complexity")
    print("• Ion-aware parameter bounds")
    print("• Support for both emcee and zeus samplers")
    print("• Built-in quick fitting capability")
    print("• Optimized multiprocessing")


def print_multi_instrument_help():
    """Print detailed multi-instrument usage help."""
    print("\nMulti-Instrument Joint Fitting:")
    print("=" * 35)
    
    print("\n🔗 Benefits:")
    print("• Shared physical parameters across all datasets")
    print("• Improved parameter constraints")
    print("• Consistent ion physics between instruments")
    print("• Automatic handling of different instrumental resolutions")
    
    print("\n⚙️ How it Works:")
    print("• Each instrument evaluates the model with its own FWHM")
    print("• Likelihood contributions are summed across all instruments")
    print("• Single theta parameter vector describes all physics")
    print("• Different wavelength grids handled automatically")
    
    print("\n📝 Best Practices:")
    print("• Ensure consistent redshift/velocity zero points")
    print("• Use realistic error estimates for each instrument")
    print("• Consider instrument-specific systematic uncertainties")
    print("• Monitor convergence diagnostics carefully")
    
    print("\n🔧 Troubleshooting:")
    print("• Check that model evaluates without errors")
    print("• Verify wavelength/flux/error array lengths match")
    print("• Ensure reasonable parameter bounds")
    print("• Use optimize=True for better starting positions")




# Module-level help function
def help():
    """Print comprehensive help for the vfit_mcmc module."""
    print_usage_examples()
    print_sampler_info()
    print_performance_tips()
    print_multi_instrument_help()


if __name__ == "__main__":
    # Print comprehensive help when module is run directly
    help()