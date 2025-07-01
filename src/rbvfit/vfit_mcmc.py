from __future__ import print_function
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
import sys
import scipy.optimize as op
from rbvfit.rb_vfit import rb_veldiff as rb_veldiff
from rbvfit import rb_setline as rb
import warnings
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union

# Try to import zeus sampler
try:
    import zeus
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False
    zeus = None


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




class vfit:
    """
    MCMC fitter for Voigt profile absorption lines with unified interface.
    
    This class supports both single and multi-instrument fitting through a
    clean, unified interface that automatically detects the number of instruments.
    
    Examples
    --------
    Single instrument (new interface):
    >>> fitter = vfit({
    ...     'HIRES': {
    ...         'model': model_func,
    ...         'wave': wave_array,
    ...         'flux': flux_array, 
    ...         'error': error_array
    ...     }
    ... }, theta, lb, ub)
    
    Multi-instrument (new interface):
    >>> fitter = vfit({
    ...     'HIRES': {'model': model_a, 'wave': wave_a, 'flux': flux_a, 'error': error_a},
    ...     'FIRE': {'model': model_b, 'wave': wave_b, 'flux': flux_b, 'error': error_b}
    ... }, theta, lb, ub)
    
    Legacy interface (still supported):
    >>> fitter = vfit(model, theta, lb, ub, wave, flux, error)
    """
    
    def __init__(self, model_or_instrument_data, theta, lb, ub, 
                 wave_obs=None, fnorm=None, enorm=None,
                 no_of_Chain=50, no_of_steps=1000, perturbation=1e-4,
                 sampler='emcee', skip_initial_state_check=False,
                 # Legacy parameters (for backward compatibility)
                 multi_instrument=None, instrument_data=None,
                 second_spec=False, second_spec_dict=None, model2=None,
                 **kwargs):
        """
        Initialize MCMC fitter with unified interface.
        
        Parameters
        ----------
        model_or_instrument_data : callable or dict
            Either:
            - Dict: New unified interface with instrument data
              Format: {'instrument_name': {'model': func, 'wave': array, 
                                          'flux': array, 'error': array}}
            - Callable: Legacy interface with single model function
        theta : array_like
            Initial parameter guess
        lb, ub : array_like
            Lower and upper bounds for parameters
        wave_obs, fnorm, enorm : array_like, optional
            Legacy interface data arrays (required if using legacy interface)
        no_of_Chain : int, optional
            Number of MCMC chains/walkers (default: 50)
        no_of_steps : int, optional
            Number of MCMC steps (default: 1000)
        perturbation : float, optional
            Initial walker perturbation (default: 1e-4)
        sampler : str, optional
            MCMC sampler: 'emcee' or 'zeus' (default: 'emcee')
        skip_initial_state_check : bool, optional
            Skip initial state validation (default: False)
        
        Legacy Parameters (deprecated but supported)
        -------------------------------------------
        multi_instrument : bool, optional
            Deprecated - automatically detected
        instrument_data : dict, optional
            Legacy secondary instrument data
        second_spec : bool, optional
            Legacy two-instrument mode
        second_spec_dict : dict, optional
            Legacy second instrument data
        model2 : callable, optional
            Legacy second instrument model
        """

        if isinstance(model_or_instrument_data, dict):
            # New unified interface
            self._setup_unified_interface(model_or_instrument_data, theta, lb, ub, 
                                    no_of_Chain, no_of_steps, perturbation, 
                                    sampler, skip_initial_state_check)
            return

        # Legacy interface - existing code
        model = model_or_instrument_data

        # Store main dataset
        self.wave_obs = wave_obs
        self.fnorm = fnorm
        self.enorm = enorm
        self.model = model
        self.lb = lb
        self.ub = ub
        self.theta = theta
        self.no_of_Chain = no_of_Chain
        self.no_of_steps = no_of_steps
        self.perturbation = perturbation
        self.skip_initial_state_check = skip_initial_state_check
        
        # Sampler selection
        self.sampler_name = sampler.lower()
        if self.sampler_name not in ['emcee', 'zeus']:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'emcee' or 'zeus'.")
        
        if self.sampler_name == 'zeus' and not HAS_ZEUS:
            raise ImportError(
                "Zeus sampler requested but not installed. "
                "Install with: pip install zeus-mcmc"
            )
        
        # Handle multi-instrument setup
        self._setup_multi_instrument(multi_instrument, instrument_data, 
                                    second_spec, second_spec_dict, model2)


    def _setup_unified_interface(self, instrument_data_dict, theta, lb, ub, 
                               no_of_Chain, no_of_steps, perturbation, 
                               sampler, skip_initial_state_check):
        """Set up fitter using new unified interface."""
        # Validate
        self._validate_unified_instrument_data(instrument_data_dict)
        
        # Store instrument data
        self.instrument_data = instrument_data_dict
        
        # CRITICAL FIX: Set multi_instrument flag based on number of instruments
        self.multi_instrument = len(instrument_data_dict) > 1
        
        # Set primary (first) for backward compatibility  
        primary_name = list(instrument_data_dict.keys())[0]
        primary_data = instrument_data_dict[primary_name]
        
        self.model = primary_data['model']
        self.wave_obs = primary_data['wave']
        self.fnorm = primary_data['flux'] 
        self.enorm = primary_data['error']
        
        # Set MCMC parameters
        self.theta = np.asarray(theta)
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.no_of_Chain = no_of_Chain
        self.no_of_steps = no_of_steps
        self.perturbation = perturbation
        self.skip_initial_state_check = skip_initial_state_check
        
        # Sampler selection (copy from legacy path)
        self.sampler_name = sampler.lower()
        if self.sampler_name not in ['emcee', 'zeus']:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'emcee' or 'zeus'.")
        
        if self.sampler_name == 'zeus' and not HAS_ZEUS:
            raise ImportError(
                "Zeus sampler requested but not installed. "
                "Install with: pip install zeus-mcmc"
            )
        
        # Print setup confirmation
        if self.multi_instrument:
            instrument_names = ', '.join(self.instrument_data.keys())
            print(f"✓ Unified interface: {len(self.instrument_data)} instruments configured ({instrument_names})")
        else:
            print(f"✓ Unified interface: Single instrument configured")


    def _setup_multi_instrument(self, multi_instrument, instrument_data,
                               second_spec, second_spec_dict, model2):
        """Set up multi-instrument configuration"""
        
        # Handle legacy interface first
        if second_spec:
            if multi_instrument:
                raise ValueError(
                    "Cannot use both 'second_spec' (legacy) and 'multi_instrument' "
                    "parameters simultaneously. Use 'multi_instrument' for new code."
                )
            
            # Convert legacy interface to new multi-instrument format
            if second_spec_dict is None or model2 is None:
                raise ValueError(
                    "Legacy 'second_spec=True' requires both 'second_spec_dict' and 'model2'"
                )
            
            print("Using legacy second_spec interface. Consider upgrading to multi_instrument.")
            
            # Convert to new format internally
            self.multi_instrument = True
            self.instrument_data = {
                'main': {
                    'model': self.model,
                    'wave': self.wave_obs,
                    'flux': self.fnorm,
                    'error': self.enorm
                },
                'second': {
                    'model': model2,
                    'wave': second_spec_dict['wave'],
                    'flux': second_spec_dict['flux'],
                    'error': second_spec_dict['error']
                }
            }
            
        elif multi_instrument:
            # New multi-instrument interface
            if instrument_data is None:
                raise ValueError(
                    "multi_instrument=True requires 'instrument_data' dictionary"
                )
            
            # Validate instrument_data format
            self._validate_instrument_data(instrument_data)
            
            # Set up multi-instrument data
            self.multi_instrument = True
            self.instrument_data = {
                'main': {
                    'model': self.model,
                    'wave': self.wave_obs,
                    'flux': self.fnorm,
                    'error': self.enorm
                }
            }
            
            # Add additional instruments
            for name, data in instrument_data.items():
                self.instrument_data[name] = data
            
        else:
            # Single instrument mode
            self.multi_instrument = False
            self.instrument_data = None

    def _validate_unified_instrument_data(self, instrument_data):
        """Validate unified interface data."""
        required_keys = {'model', 'wave', 'flux', 'error'}
        for name, data in instrument_data.items():
            if not required_keys.issubset(data.keys()):
                raise ValueError(f"Instrument '{name}' missing keys: {required_keys - set(data.keys())}")
    
    
    def _validate_instrument_data(self, instrument_data):
        """Validate instrument_data format"""
        if not isinstance(instrument_data, dict):
            raise ValueError("instrument_data must be a dictionary")
        
        required_keys = {'model', 'wave', 'flux', 'error'}
        
        for name, data in instrument_data.items():
            if not isinstance(data, dict):
                raise ValueError(f"Instrument '{name}' data must be a dictionary")
            
            if not required_keys.issubset(data.keys()):
                missing = required_keys - set(data.keys())
                raise ValueError(
                    f"Instrument '{name}' missing required keys: {missing}"
                )
            
            # Check that arrays have consistent lengths
            wave_len = len(data['wave'])
            if len(data['flux']) != wave_len or len(data['error']) != wave_len:
                raise ValueError(
                    f"Instrument '{name}': wave, flux, and error arrays must have same length"
                )

    def lnprior(self, theta):
        #theta = np.asarray(theta)
        if np.any((theta < self.lb) | (theta > self.ub)):
            return -np.inf
        return 0.0
    
    def lnlike(self, theta):
        """
        Calculate log-likelihood for single or multi-instrument data.
        Uses unified instrument_data when available, falls back to legacy mode.
        """
        try:
            # NEW UNIFIED PATH: Check for instrument_data first
            if hasattr(self, 'instrument_data') and self.instrument_data is not None:
                # Unified likelihood calculation (works for 1 or N instruments)
                lnlike_total = 0.0
                
                for instrument_name, data in self.instrument_data.items():
                    # Evaluate model for this instrument
                    model_dat = data['model'](theta, data['wave'])
                    
                    # Calculate likelihood contribution
                    inv_sigma2 = 1.0 / (data['error'] ** 2)
                    lnlike_instrument = -0.5 * (
                        np.sum((data['flux'] - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2))
                    )
                    
                    lnlike_total += lnlike_instrument
                
                return lnlike_total
                
            # LEGACY PATH: Original logic for backward compatibility
            elif hasattr(self, 'multi_instrument') and self.multi_instrument:
                # Legacy multi-instrument likelihood calculation (should not happen with new interface)
                lnlike_total = 0.0
                
                for instrument_name, data in self.instrument_data.items():
                    # Evaluate model for this instrument
                    model_dat = data['model'](theta, data['wave'])
                    
                    # Calculate likelihood contribution
                    inv_sigma2 = 1.0 / (data['error'] ** 2)
                    lnlike_instrument = -0.5 * (
                        np.sum((data['flux'] - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2))
                    )
                    
                    lnlike_total += lnlike_instrument
                
                return lnlike_total
                
            else:
                # Legacy single instrument likelihood
                model_dat = self.model(theta, self.wave_obs)
                inv_sigma2 = 1.0 / (self.enorm ** 2)
                lnlike_total = -0.5 * (
                    np.sum((self.fnorm - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2))
                )
                
                return lnlike_total
        
        except Exception:
            # Return -inf for any evaluation errors
            return -np.inf
    
    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)
    
    def optimize_guess(self, theta):
        bounds = list(zip(self.lb, self.ub))  # assuming self.lb and self.ub are arrays
        nll = lambda theta: -self.lnprob(theta)
        result = op.minimize(nll, theta, method='L-BFGS-B',bounds=bounds)
        return result["x"]

    def _setup_emcee_sampler(self, guesses, use_pool=True):
        """Set up emcee sampler with optimized multiprocessing."""
        ndim = len(self.lb)
        nwalkers = self.no_of_Chain
        
        if use_pool:
            pool = OptimizedPool()
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.lnprob, pool=pool
            )
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.lnprob
            )
        
        return sampler, pool if use_pool else None

    def _setup_zeus_sampler(self, guesses, use_pool=True):
        """Set up zeus sampler with optimized multiprocessing."""
        ndim = len(self.lb)
        nwalkers = self.no_of_Chain
        
        if use_pool:
            pool = OptimizedPool()
            sampler = zeus.EnsembleSampler(
                nwalkers, ndim, self.lnprob, pool=pool
            )
        else:
            sampler = zeus.EnsembleSampler(
                nwalkers, ndim, self.lnprob
            )
        
        return sampler, pool if use_pool else None

    def _initialize_walkers(self, popt):
        """Initialize walker positions with proper bounds checking."""
        ndim = len(self.lb)
        nwalkers = self.no_of_Chain
        
        guesses = []
        max_attempts = 1000
        
        for i in range(nwalkers):
            attempts = 0
            while attempts < max_attempts:
                # Create perturbed guess
                guess = popt + self.perturbation * np.random.randn(ndim)
                
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
            # Zeus doesn't have acceptance_fraction, estimate from chain
            try:
                chain = sampler.get_chain()
                # Simple estimate: count unique consecutive steps
                n_accepted = 0
                n_total = 0
                for walker_chain in chain.T:  # Iterate over walkers
                    for i in range(1, len(walker_chain)):
                        n_total += 1
                        if not np.array_equal(walker_chain[i], walker_chain[i-1]):
                            n_accepted += 1
                return n_accepted / n_total if n_total > 0 else 0.0
            except Exception:
                print("Could not calculate Gelman-Rubin diagnostic")

        if verbose == True:
            self._print_parameter_summary(sampler, burntime)

        self.sampler = sampler
        self.ndim = len(self.lb)
        self.nwalkers = no_of_Chain


    def _print_parameter_summary(self, sampler, burntime):
        """Print detailed parameter summary."""
        try:
            # Detect environment
            import IPython
            ipython = IPython.get_ipython()
            
            # Check if we're in a Jupyter notebook (rich frontend)
            if ipython is not None:
                frontend = getattr(ipython, '__class__', None)
                is_jupyter = 'zmq' in str(frontend).lower() or 'notebook' in str(frontend).lower()
            else:
                is_jupyter = False
                
            if is_jupyter:
                # Jupyter notebook - use Math display
                from IPython.display import display, Math
                use_math_display = True
            else:
                # Terminal (IPython or regular Python) - use text
                use_math_display = False
                
        except ImportError:
            use_math_display = False
        
        # Get samples
        samples = self._extract_samples(sampler, int(burntime))
        
        ndim = samples.shape[1]
        nfit = int(ndim / 3)
        N_tile = np.tile("logN", nfit)
        b_tile = np.tile("b", nfit)
        v_tile = np.tile("v", nfit)
        
        tmp = np.append(N_tile, b_tile)
        text_label = np.append(tmp, v_tile)
        
        if use_math_display:
            # Rich display for Jupyter only
            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
                display(Math(txt))
        else:
            # Text display for all terminals (IPython and regular Python)
            print("\nParameter Summary:")
            print("-" * 40)
            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                print(f"{text_label[i]} = {mcmc[1]:.2f} +{q[1]:.2f} -{q[0]:.2f}")


    def _extract_samples(self, sampler, burnin=200, thin=15):
        """Extract samples in a sampler-agnostic way."""
        if hasattr(sampler, 'get_chain'):
            if self.sampler_name == 'emcee':
                # emcee
                return sampler.get_chain(discard=burnin, thin=thin, flat=True)
            elif self.sampler_name == 'zeus':
                # Zeus
                try:
                    return sampler.get_chain(discard=burnin, thin=thin, flat=True)
                except TypeError:
                    # Older zeus version might not support these parameters
                    chain = sampler.get_chain()
                    return chain[burnin::thin].reshape(-1, chain.shape[-1])
        else:
            # Fallback
            chain = sampler.get_chain()
            return chain[burnin::thin].reshape(-1, chain.shape[-1])

    def runmcmc(self, optimize=True, verbose=False, use_pool=True, progress=True):
        """
        Run MCMC with selected sampler.
        
        Parameters
        ----------
        optimize : bool, optional
            Whether to optimize initial guess first (default: True)
        verbose : bool, optional
            Whether to print detailed progress messages (default: False)
        use_pool : bool, optional
            Whether to use multiprocessing (default: True)
        progress : bool, optional
            Whether to show MCMC progress bar (default: True)
        """
        model = self.model
        theta = self.theta
        lb = self.lb
        ub = self.ub
        wave_obs = self.wave_obs
        fnorm = self.fnorm
        enorm = self.enorm
        no_of_Chain = self.no_of_Chain
        no_of_steps = self.no_of_steps
        perturbation = self.perturbation
    
        if verbose:
            print(f"Using {self.sampler_name.upper()} sampler")
            if use_pool:
                print(f"Multiprocessing context: {MP_CONTEXT}")
            print(f"Skip initial state check: {self.skip_initial_state_check}")
    
        if optimize == True:
            if verbose:
                print('Optimizing Guess ***********')
            # Now make a better guess
            try:
                popt = self.optimize_guess(theta)
                if verbose:
                    print('Optimization completed ***********')
            except Exception as e:
                if verbose:
                    print(f'Optimization failed: {e}. Using original guess.')
                popt = theta
        else:
            if verbose:
                print('Skipping Optimizing Guess ***********')
                print('Using input guess for mcmc ***********')
            popt = theta
    
        if verbose:
            print('Preparing MCMC ***********')
        
        # Initialize walkers
        guesses = self._initialize_walkers(popt)
        
        # Set up sampler
        pool = None
        if self.sampler_name == 'emcee':
            sampler, pool = self._setup_emcee_sampler(guesses, use_pool)
        elif self.sampler_name == 'zeus':
            sampler, pool = self._setup_zeus_sampler(guesses, use_pool)
        
        if verbose:
            print(f"Starting {self.sampler_name.upper()} with {no_of_Chain} walkers ***********")
        
        # Calculate burn-in
        burntime = np.round(no_of_steps * 0.2)
        
        try:
            # Run MCMC
            if self.sampler_name == 'emcee':
                pos, prob, state = sampler.run_mcmc(
                    guesses, no_of_steps, 
                    progress=progress,
                    skip_initial_state_check=self.skip_initial_state_check
                )
            elif self.sampler_name == 'zeus':
                # Zeus has slightly different interface
                sampler.run_mcmc(
                    guesses, no_of_steps,
                    progress=progress
                )
                
        finally:
            # Always close pool if it was created
            if pool is not None:
                pool.close()
                pool.join()
    
        if verbose:
            print("Done!")
            print("*****************")
        
        # Calculate diagnostics - ALWAYS show these (critical for assessment)
        acceptance_fraction = self._get_acceptance_fraction(sampler)
        if acceptance_fraction is not None:
            if isinstance(acceptance_fraction, (list, np.ndarray)):
                mean_acceptance_fraction = np.mean(acceptance_fraction)
            else:
                mean_acceptance_fraction = acceptance_fraction
            
            print("Mean acceptance fraction: {0:.3f}".format(mean_acceptance_fraction))
            
            # Acceptance fraction interpretation
            if mean_acceptance_fraction < 0.2:
                print("⚠ Warning: Low acceptance fraction (<0.2). Consider reducing step size.")
            elif mean_acceptance_fraction > 0.7:
                print("⚠ Warning: High acceptance fraction (>0.7). Consider increasing step size.")
            else:
                print("✓ Good acceptance fraction (0.2-0.7)")
        else:
            print("Acceptance fraction not available for this sampler")
        
        # Autocorrelation time (if available) - ALWAYS show
        try:
            try:
                autocorr_time = sampler.get_autocorr_time()
                mean_autocorr_time = np.nanmean(autocorr_time)
                print("Mean auto-correlation time: {0:.3f} steps".format(mean_autocorr_time))
    
                
                # Check if chain is long enough
                if no_of_steps < 50 * mean_autocorr_time:
                    print("⚠ Warning: Chain may be too short for reliable results")
                    print(f"  Recommended: >{50 * mean_autocorr_time:.0f} steps")
            except:
                mean_autocorr_time = 0
                print("⚠ Warning: Chain Length is less than 50 Autocorrelation times")
                print(f"  Recommended: >{2 * no_of_steps:.0f} steps")
        except Exception:
            print("⚠ Warning: Could not calculate auto-correlation time")
    
        # Sampler-specific diagnostics - ALWAYS show
        if self.sampler_name == 'zeus':
            # Zeus-specific diagnostics
            try:
                chain = sampler.get_chain().transpose(1, 0, 2)
                # Then compute Gelman-Rubin diagnostic
                rhat = zeus.diagnostics.gelman_rubin(chain)
                print(f"Gelman-Rubin R-hat: {np.max(r_hat):.3f}")
                if np.max(r_hat) > 1.1:
                    print("⚠ Warning: R-hat > 1.1, chains may not have converged")
                else:
                    print("✓ Good convergence (R-hat < 1.1)")
            except Exception:
                print("Could not calculate Gelman-Rubin diagnostic")
    
        # Detailed parameter summary only if verbose
        if verbose:
            self._print_parameter_summary(sampler, burntime)
    
        self.sampler = sampler
        self.ndim = len(self.lb)
        self.nwalkers = no_of_Chain
        self.mcmc_flag=True
        self.compute_best_theta()
    
    
    def fit_quick(self, verbose=True):
        """
        Quick deterministic fitting using scipy optimization.
        
        Supports both single and multi-instrument datasets automatically.
        
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
        
        # Print instrument info
        if self.multi_instrument and verbose:
            n_instruments = len(self.instrument_data)
            instrument_names = ', '.join(self.instrument_data.keys())
            print(f"🔗 Multi-instrument mode: {n_instruments} instruments ({instrument_names})")
        elif verbose:
            print("📊 Single instrument mode")
        
        # Validate required attributes for legacy interface
        if not hasattr(self, 'instrument_data') or self.instrument_data is None:
            required_attrs = ['model', 'theta', 'wave_obs', 'fnorm', 'enorm']
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing: {attr}")
        
        if verbose:
            print("🚀 Starting quick fit...")
        
        try:
            theta_best, theta_best_error = quick_fit_vfit(self)
            
            if verbose:
                print("✅ Quick fit completed")
                if np.any(np.isnan(theta_best_error)):
                    print("⚠️  Some parameter uncertainties are NaN")
            
            self.theta_best = theta_best
            self.theta_best_error = theta_best_error
            
            return theta_best, theta_best_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Quick fit failed: {str(e)}")
            raise RuntimeError("Quick fitting failed") from e
    
    def compute_best_theta(self, burntime=100):
        """Compute best-fit parameters and their uncertainties from MCMC samples."""
        if not hasattr(self, 'samples') or self.samples is None:
            self.samples = self._extract_samples(self.sampler, burntime)
    
        self.best_theta = np.percentile(self.samples, 50, axis=0)
        self.low_theta = np.percentile(self.samples, 10, axis=0)
        self.high_theta = np.percentile(self.samples, 90, axis=0)

        

    def plot_corner(self, outfile=False, burntime=100, **kwargs):
        """Plot corner plot with sampler-agnostic sample extraction."""
    
        if self.mcmc_flag is not True:
            print("⚠️  MCMC Fit not done! Can't do corner plot. Exiting gracefully.")
            return
    
        # Only extract samples if not already done
        if not hasattr(self, 'samples') or self.samples is None:
            self.samples = self._extract_samples(self.sampler, burntime)
    
        # Only compute theta estimates if not already done
        if not hasattr(self, 'best_theta') or self.best_theta is None:
            self.compute_best_theta(burntime)
    
        ndim = self.ndim
        samples = self.samples
        st = self.best_theta
    
        nfit = int(ndim / 3)
        N_tile = np.tile("logN", nfit)
        b_tile = np.tile("b", nfit)
        v_tile = np.tile("v", nfit)
        text_label = np.append(np.append(N_tile, b_tile), v_tile)
    
        # Plot corner
        truths = kwargs.get('True_values', st)
        figure = corner.corner(samples, labels=text_label, truths=truths)
    
        # Optional figure title
        if outfile:
            plt.title(outfile, y=1.05 * ndim, loc='right')
    
        # Add vertical lines at error bounds
        value1 = self.low_theta
        value2 = self.high_theta
        axes = np.array(figure.axes).reshape((ndim, ndim))
    
        for i in range(ndim):
            axes[i, i].axvline(value1[i], color="aqua")
            axes[i, i].axvline(value2[i], color="aqua")
    
        for yi in range(ndim):
            for xi in range(yi):
                axes[yi, xi].axvline(value1[xi], color="aqua")
                axes[yi, xi].axvline(value2[xi], color="aqua")
    
        # Save or show plot
        if outfile:
            figure.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
    
    
        def get_sampler_info(self):
            """Get information about the sampler used."""
            info = {
                'sampler': self.sampler_name,
                'walkers': self.no_of_Chain,
                'steps': self.no_of_steps,
                'multi_instrument': self.multi_instrument,
                'has_zeus': HAS_ZEUS,
                'mp_context': MP_CONTEXT
            }
            
            if self.multi_instrument:
                info['n_instruments'] = len(self.instrument_data)
                info['instruments'] = list(self.instrument_data.keys())
            
            if hasattr(self, 'sampler'):
                # Get acceptance fraction in sampler-agnostic way
                acceptance_fraction = self._get_acceptance_fraction(self.sampler)
                if acceptance_fraction is not None:
                    if isinstance(acceptance_fraction, (list, np.ndarray)):
                        info['acceptance_fraction'] = np.mean(acceptance_fraction)
                    else:
                        info['acceptance_fraction'] = acceptance_fraction
                
                # Add sampler-specific info
                if self.sampler_name == 'zeus' and hasattr(self.sampler, 'get_chain'):
                    try:
                        r_hat = zeus.diagnostics.gelman_rubin(self.sampler.get_chain())
                        info['r_hat_max'] = np.max(r_hat)
                    except:
                        pass
            
            return info
    
    
def plot_model(model, fitter, outfile=False, xlim=None, ylim=None, show_residuals=True, 
               show_components=True, velocity_marks=True, burntime=None, 
               burn_fraction=0.2, n_posterior_samples=300, verbose=True,
               datasets=None, **kwargs):
    """
    Enhanced plot_model function supporting both rbvfit 1.0 and 2.0 models with optional dataset override.
    
    Parameters
    ----------
    model : object
        Model object (v1.0 create_voigt object or v2.0 VoigtModel)
    fitter : vfit
        MCMC fitter object after running fit
    outfile : str or False, optional
        Save figure to file (default: False)
    xlim : list, optional
        X-axis limits [start, end] - wavelength limits for v2.0, velocity limits for v1.0
    ylim : list, optional
        Y-axis limits [start, end] for flux plots (default: auto for v2.0, [0, 1.6] for v1.0)
    show_residuals : bool, optional
        Include residual plots (default: True)
    show_components : bool, optional
        Show individual velocity components (default: True)
    velocity_marks : bool, optional
        Mark component velocities (default: True)
    burntime : int, optional
        Explicit burn-in steps (default: auto-detect)
    burn_fraction : float, optional
        Fraction of chain to discard as burn-in (default: 0.2)
    n_posterior_samples : int, optional
        Number of posterior samples for uncertainty clouds (default: 300)
    verbose : bool, optional
        Print parameter summary (default: True)
    datasets : dict, optional
        Override datasets for plotting. Format:
        - Single instrument: {'wave': array, 'flux': array, 'error': array}
        - Multi-instrument: {'instr1': {'wave': array, 'flux': array, 'error': array}, ...}
        If None, uses data from fitter object.
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        The plot figure
    """
    
    # Detect result type first
    if fitter.mcmc_flag==True:
        print('IDENTIFIED MCMC ')
        # MCMC results available
        if _is_v2_model(model):
            print("Detected rbvfit 2.0 MCMC results - using enhanced plotting")
            return _plot_model_v2(model, fitter, outfile=outfile, xlim=xlim, ylim=ylim, 
                                 show_residuals=show_residuals, show_components=show_components, 
                                 velocity_marks=velocity_marks, burntime=burntime, 
                                 burn_fraction=burn_fraction, n_posterior_samples=n_posterior_samples, 
                                 verbose=verbose, datasets=datasets, **kwargs)
        else:
            print("Detected rbvfit 1.0 MCMC results - using original plotting")
            return _plot_model_v1(model, fitter, outfile=outfile, xlim=xlim, ylim=ylim, 
                                 show_residuals=show_residuals, show_components=show_components, 
                                 velocity_marks=velocity_marks, verbose=verbose, datasets=datasets, **kwargs)
    
    elif  fitter.mcmc_flag==False:
        # Quick fit results or deterministic results
        print("Detected quick fit results - using deterministic plotting")
        return _plot_quick_fit(model, fitter, outfile=outfile, xlim=xlim, ylim=ylim,
                              show_residuals=show_residuals, show_components=show_components,
                              velocity_marks=velocity_marks, verbose=verbose, 
                              datasets=datasets, **kwargs)
    
    else:
        raise ValueError("Cannot determine result type. Fitter must have either MCMC samples or quick fit results.")

def _setup_datasets(fitter, datasets=None, verbose=True):
    """
    Set up datasets for plotting, with optional override.
    """
    plot_datasets = {}
    
    # NEW UNIFIED INTERFACE: Use instrument_data if available
    if hasattr(fitter, 'instrument_data') and fitter.instrument_data is not None:
        for name, data in fitter.instrument_data.items():
            plot_datasets[name] = (data['wave'], data['flux'], data['error'])
    else:
        # LEGACY FALLBACK: Use primary attributes
        plot_datasets['Primary'] = (fitter.wave_obs, fitter.fnorm, fitter.enorm)
    
    # Apply dataset overrides if provided
    if datasets is not None:
        if _is_single_dataset_format(datasets):
            # Single dataset override - replace first instrument
            first_name = list(plot_datasets.keys())[0]
            _validate_dataset(datasets, first_name)
            plot_datasets[first_name] = (datasets['wave'], datasets['flux'], datasets['error'])
            if verbose:
                print(f"Using provided dataset for {first_name}")
        else:
            # Multi-instrument format
            for name, data in datasets.items():
                _validate_dataset(data, name)
                plot_datasets[name] = (data['wave'], data['flux'], data['error'])
                if verbose:
                    print(f"Using provided dataset for {name}")
    
    return plot_datasets

def _is_single_dataset_format(datasets):
    """Check if datasets is in single instrument format."""
    required_keys = {'wave', 'flux', 'error'}
    return isinstance(datasets, dict) and required_keys.issubset(datasets.keys())


def _validate_dataset(dataset, name):
    """Validate dataset format."""
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset '{name}' must be a dictionary")
    
    required_keys = {'wave', 'flux', 'error'}
    if not required_keys.issubset(dataset.keys()):
        missing = required_keys - set(dataset.keys())
        raise ValueError(f"Dataset '{name}' missing required keys: {missing}")
    
    # Check array lengths match
    wave_len = len(dataset['wave'])
    if len(dataset['flux']) != wave_len or len(dataset['error']) != wave_len:
        raise ValueError(f"Dataset '{name}': wave, flux, and error arrays must have same length")
    
    # Check for valid arrays
    for key in required_keys:
        if not hasattr(dataset[key], '__len__'):
            raise ValueError(f"Dataset '{name}': {key} must be array-like")

def _is_v2_model(model):
    """
    Detect whether model is rbvfit 2.0 VoigtModel.
    
    Parameters
    ----------
    model : object
        Model object to check
        
    Returns
    -------
    bool
        True if v2.0 VoigtModel, False for v1.0
    """
    # Check for v2.0 indicators
    v2_indicators = [
        hasattr(model, 'config'),                # FitConfiguration
        hasattr(model, 'param_manager'),         # ParameterManager
        hasattr(model, 'compile'),               # Compilable model
        hasattr(model, 'evaluate'),              # v2.0 evaluation method
    ]
    
    return any(v2_indicators)


def _plot_model_v1(model, fitter, outfile=False, xlim=[-600., 600.], ylim=None,
                   show_residuals=True, show_components=True, 
                   velocity_marks=True, verbose=False, datasets=None, **kwargs):
    """
    Original rbvfit 1.0 plotting logic with dataset override support.
    """
    # Set up datasets
    plot_datasets = _setup_datasets(fitter, datasets)
    
    # For v1.0, we only use the primary dataset (single instrument)
    wave_obs, fnorm, enorm = plot_datasets['Primary']
    
    if len(plot_datasets) > 1 and verbose:
        print("Warning: v1.0 model plotting only supports single dataset. Using Primary dataset.")

    if ylim is None:
        ylim = [0, 1.6]
    
    # Extract MCMC results from fitter
    samples = fitter.samples
    best_theta = fitter.best_theta
    
    # Get model parameters
    n_clump = model.nclump 
    n_clump_total = int(len(best_theta) / 3)
    ntransition = model.ntransition
    zabs = model.zabs

    # Get rest wavelengths for plotting
    wave_list = np.zeros(len(model.lambda_rest_original))
    for i in range(len(wave_list)):
        s = rb.rb_setline(model.lambda_rest_original[i], 'closest')
        wave_list[i] = s['wave']

    wave_rest = wave_obs / (1 + zabs[0])
    
    # Extract best-fit parameters
    best_N = best_theta[0:n_clump_total]
    best_b = best_theta[n_clump_total:2 * n_clump_total]
    best_v = best_theta[2 * n_clump_total:3 * n_clump_total]
    
    # Calculate uncertainties
    low_theta = np.percentile(samples, 16, axis=0)
    high_theta = np.percentile(samples, 84, axis=0)
    
    low_N = low_theta[0:n_clump_total]
    low_b = low_theta[n_clump_total:2 * n_clump_total]
    low_v = low_theta[2 * n_clump_total:3 * n_clump_total]
    
    high_N = high_theta[0:n_clump_total]
    high_b = high_theta[n_clump_total:2 * n_clump_total]
    high_v = high_theta[2 * n_clump_total:3 * n_clump_total]

    # Generate best-fit model on provided wavelength grid
    best_fit, f1 = model.model_fit(best_theta, wave_obs)

    # Create plots
    if show_residuals:
        fig, axes = plt.subplots(ntransition + 1, 1, figsize=(12, 6 * (ntransition + 1)), 
                               gridspec_kw={'height_ratios': [3] * ntransition + [1]})
        if ntransition == 1:
            axes = [axes[0], axes[1]]
    else:
        fig, axes = plt.subplots(ntransition, 1, figsize=(12, 6 * ntransition))
        if ntransition == 1:
            axes = [axes]

    # Set up larger font sizes
    BIGGER_SIZE = 18
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('legend', fontsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Random sample for uncertainty cloud
    n_samples = min(100, len(samples))
    index = np.random.randint(0, high=len(samples), size=n_samples)
    
    # Plot each transition
    if ntransition == 1:
        ax = axes[0] if show_residuals else axes
        vel = rb_veldiff(wave_list[0], wave_rest)
        
        # Plot data
        ax.step(vel, fnorm, 'k-', linewidth=1., label=f'{wave_list[0]:.1f}')
        ax.legend()
        ax.step(vel, enorm, color='r', linewidth=1.)
        
        # Plot uncertainty cloud
        for ind in range(len(index)):
            model_sample = fitter.model(samples[index[ind], :], wave_obs)
            ax.plot(vel, model_sample, color="k", alpha=0.1)
        
        # Plot best fit and components
        ax.plot(vel, best_fit, color='b', linewidth=3, label='Best Fit')
        ax.plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
        
        if show_components:
            for dex in range(np.shape(f1)[1]):
                ax.plot(vel, f1[:, dex], 'g:', linewidth=3)

        # Mark velocity components
        if velocity_marks:
            for iclump in range(n_clump):
                ax.plot([best_v[iclump], best_v[iclump]], [1.05, 1.15], 'k--', lw=4)
                
                # Parameter annotations
                text1 = r'$\log N = ' + f'{best_N[iclump]:.2f}' + '^{+' + f'{high_N[iclump]-best_N[iclump]:.2f}' + '}_{-' + f'{best_N[iclump]-low_N[iclump]:.2f}' + '}$'
                ax.text(best_v[iclump], 1.2, text1, fontsize=14, rotation=90, rotation_mode='anchor')
                
                text2 = r'$b = ' + f'{best_b[iclump]:.0f}' + '^{+' + f'{high_b[iclump]-best_b[iclump]:.0f}' + '}_{-' + f'{best_b[iclump]-low_b[iclump]:.0f}' + '}$'
                ax.text(best_v[iclump] + 30, 1.2, text2, fontsize=14, rotation=90, rotation_mode='anchor')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Normalized Flux')
        
    else:
        # Multiple transitions
        for i in range(ntransition):
            ax = axes[i]
            vel = rb_veldiff(wave_list[i], wave_rest)
            
            # Plot data
            ax.step(vel, fnorm, 'k-', linewidth=1., label=f'{wave_list[i]:.1f}')
            ax.legend()
            ax.step(vel, enorm, color='r', linewidth=1.)
            
            # Plot uncertainty cloud
            for ind in range(len(index)):
                model_sample = fitter.model(samples[index[ind], :], wave_obs)
                ax.plot(vel, model_sample, color="k", alpha=0.1)
            
            # Plot best fit and components
            ax.plot(vel, best_fit, color='b', linewidth=3)
            ax.plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
            
            if show_components:
                for dex in range(np.shape(f1)[1]):
                    ax.plot(vel, f1[:, dex], 'g:', linewidth=3)
            
            # Mark velocity components (only on first panel)
            if velocity_marks and i == 0:
                for iclump in range(n_clump):
                    ax.plot([best_v[iclump], best_v[iclump]], [1.05, 1.15], 'k--', lw=4)
                    
                    text1 = r'$\log N = ' + f'{best_N[iclump]:.2f}' + '^{+' + f'{high_N[iclump]-best_N[iclump]:.2f}' + '}_{-' + f'{best_N[iclump]-low_N[iclump]:.2f}' + '}$'
                    ax.text(best_v[iclump], 1.2, text1, fontsize=14, rotation=90, rotation_mode='anchor')
                    
                    text2 = r'$b = ' + f'{best_b[iclump]:.0f}' + '^{+' + f'{high_b[iclump]-best_b[iclump]:.0f}' + '}_{-' + f'{best_b[iclump]-low_b[iclump]:.0f}' + '}$'
                    ax.text(best_v[iclump] + 30, 1.2, text2, fontsize=14, rotation=90, rotation_mode='anchor')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_ylabel('Normalized Flux')
            
            if i == ntransition - 1:
                ax.set_xlabel('Velocity (km/s)')

    # Add residuals if requested
    if show_residuals:
        ax_resid = axes[-1]
        residuals = (fnorm - best_fit) / enorm
        
        if ntransition == 1:
            vel_resid = rb_veldiff(wave_list[0], wave_rest)
        else:
            # Use first transition for velocity scale
            vel_resid = rb_veldiff(wave_list[0], wave_rest)
        
        ax_resid.step(vel_resid, residuals, 'k-', linewidth=1.)
        ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
        ax_resid.axhline(1, color='gray', linestyle=':', alpha=0.5)
        ax_resid.axhline(-1, color='gray', linestyle=':', alpha=0.5)
        
        ax_resid.set_xlabel('Velocity (km/s)')
        ax_resid.set_ylabel('Residuals (σ)')
        ax_resid.set_xlim(xlim)
        ax_resid.grid(True, alpha=0.3)
        
        # RMS
        rms = np.sqrt(np.mean(residuals**2))
        ax_resid.text(0.02, 0.95, f'RMS = {rms:.2f}', 
                     transform=ax_resid.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    # Title with dataset info
    title_parts = []
    if outfile != False:
        title_parts.append(outfile)
    if datasets is not None:
        title_parts.append("(Custom Dataset)")
    
    if title_parts:
        plt.suptitle(" ".join(title_parts), y=0.98, fontsize=12)

    # Verbose parameter output
    if verbose:
        _print_v1_parameter_summary(samples, n_clump_total)

    # Save or show
    if outfile != False:
        fig.savefig(outfile, bbox_inches='tight')
        print(f"Saved plot to {outfile}")
    else:
        plt.show()
        
    return fig


def _plot_model_v2(model, fitter, outfile=False, xlim=None, ylim=None, show_residuals=True,
                   show_components=True, velocity_marks=True, burntime=None,
                   burn_fraction=0.2, n_posterior_samples=300, verbose=True, 
                   datasets=None, **kwargs):
    """
    Enhanced rbvfit 2.0 plotting with rail system and dataset override support.
    """
    print("Creating rbvfit 2.0 enhanced visualization...")
    
    # Set up datasets
    plot_datasets = _setup_datasets(fitter, datasets, verbose=verbose)

    
    # Determine burn-in and extract samples
    effective_burntime = _determine_burntime_v2(fitter, burntime, burn_fraction)
    samples = _extract_samples_smart_v2(fitter.sampler, effective_burntime)
    best_theta = np.percentile(samples, 50, axis=0)  # Median as best-fit
    
    print(f"Extracted {len(samples)} post-burn-in samples")
    
    # Check if multi-instrument
    is_multi_instrument = len(plot_datasets) > 1
    
    if is_multi_instrument:
        return _plot_multi_instrument_v2_with_datasets(
            model, plot_datasets, best_theta, samples, show_residuals,
            show_components, velocity_marks, xlim, ylim, n_posterior_samples, outfile, 
            verbose=True)
    else:
        # FIX: Use first available instrument instead of hard-coded 'Primary'
        first_instrument_name = list(plot_datasets.keys())[0]
        first_dataset = plot_datasets[first_instrument_name]
        
        return _plot_single_instrument_v2_with_datasets(
            model, first_dataset, best_theta, samples, show_residuals,
            show_components, velocity_marks, xlim, ylim, n_posterior_samples, 
            outfile, datasets is not None, verbose=True)

def _plot_single_instrument_v2_with_datasets(model, dataset, best_theta, samples, 
                                           show_residuals, show_components, velocity_marks,
                                           xlim, ylim, n_posterior_samples, outfile,
                                           is_custom_dataset, verbose=True):
    """Plot single instrument v2.0 with dataset support."""
    wave_data, flux_data, error_data = dataset

    if ylim is None:
        ylim = [0, 1.2]
    
    # Create plots
    if show_residuals:
        fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(12, 10), 
                                               gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot data
    data_label = 'Custom Data' if is_custom_dataset else 'Observed Data'
    ax_main.step(wave_data, flux_data, 'k-', where='mid', linewidth=1, 
                alpha=0.8, label=data_label)
    ax_main.step(wave_data, error_data, 'gray', where='mid', alpha=0.3, 
                linewidth=0.5, label='1σ Error')
    
    # Plot uncertainty cloud
    n_samples = min(n_posterior_samples, len(samples))
    sample_indices = np.random.choice(len(samples), size=n_samples, replace=False)
    
    print(f"Adding uncertainty cloud from {n_samples} posterior samples...")
    for idx in sample_indices:
        try:
            model_sample = model.evaluate(samples[idx], wave_data)
            ax_main.plot(wave_data, model_sample, 'blue', alpha=0.01, linewidth=0.5)
        except:
            continue
    
    # Plot best fit
    try:
        best_model = model.evaluate(best_theta, wave_data)
        ax_main.plot(wave_data, best_model, 'red', linewidth=2, label='Best Fit')
    except Exception as e:
        print(f"Error: Could not generate best-fit model: {e}")
        return None
    
    # Add rail system for component visualization
    if velocity_marks:
        print("Adding rail system component visualization...")
        _add_rail_system_v2(ax_main, model, best_theta, wave_data)
    
    # Format main plot
    ax_main.set_ylabel('Normalized Flux')
    title = 'rbvfit 2.0: Enhanced Absorption Line Fit with Component Rails'
    if is_custom_dataset:
        title += ' (Custom Dataset)'
    ax_main.set_title(title)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(ylim)
    
    # Print limit info if custom
    if verbose:
        if xlim is not None:
            print(f"Applied wavelength limits: {xlim[0]:.1f} - {xlim[1]:.1f} Å")
        if ylim != [0, 1.2]:
            print(f"Applied flux limits: {ylim[0]:.2f} - {ylim[1]:.2f}")
    
    # Residuals if requested
    if show_residuals:
        residuals = (flux_data - best_model) / error_data
        
        ax_resid.step(wave_data, residuals, 'k-', where='mid', alpha=0.7, linewidth=1)
        ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
        ax_resid.axhline(1, color='gray', linestyle=':', alpha=0.5)
        ax_resid.axhline(-1, color='gray', linestyle=':', alpha=0.5)
        
        ax_resid.set_xlabel('Observed Wavelength (Å)')
        ax_resid.set_ylabel('Residuals (σ)')
        ax_resid.grid(True, alpha=0.3)
        
        if xlim is not None:
            ax_resid.set_xlim(xlim)
        
        # RMS
        rms = np.sqrt(np.mean(residuals**2))
        ax_resid.text(0.02, 0.95, f'RMS = {rms:.2f}', 
                     transform=ax_resid.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax_main.set_xlabel('Observed Wavelength (Å)')
    
    plt.tight_layout()
    
    # Print parameter summary
    if verbose:
        _print_parameter_summary_v2(model, best_theta, samples)
    
    # Save or show
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced plot to {outfile}")
    else:
        plt.show()
    
    return fig


def _plot_multi_instrument_v2_with_datasets(model, plot_datasets, best_theta, samples,
                                          show_residuals, show_components, velocity_marks,
                                          xlim, ylim, n_posterior_samples,outfile, verbose=True):
    """Plot multi-instrument v2.0 with dataset support."""
    
    instrument_names = list(plot_datasets.keys())
    n_instruments = len(instrument_names)
    
    print(f"Multi-instrument plotting: {n_instruments} datasets ({', '.join(instrument_names)})")
    
    # Create subplot layout
    if show_residuals:
        fig, axes = plt.subplots(n_instruments * 2, 1, figsize=(15, 6 * n_instruments),
                               gridspec_kw={'height_ratios': [3, 1] * n_instruments})
    else:
        fig, axes = plt.subplots(n_instruments, 1, figsize=(15, 5 * n_instruments))
    
    if n_instruments == 1:
        axes = [axes] if not show_residuals else axes
    
    if ylim is None:
        ylim = [0, 1.2]


    # Random sample indices for uncertainty clouds
    n_samples = min(n_posterior_samples, len(samples))
    sample_indices = np.random.choice(len(samples), size=n_samples, replace=False)
    
    # Plot each instrument
    for i, (instrument_name, dataset) in enumerate(plot_datasets.items()):
        wave_data, flux_data, error_data = dataset
        
        # Main plot
        ax_main = axes[i * 2] if show_residuals else axes[i]
        
        # Plot data
        ax_main.step(wave_data, flux_data, 'k-', where='mid', linewidth=1, 
                    alpha=0.8, label=f'{instrument_name} Data')
        ax_main.step(wave_data, error_data, 'gray', where='mid', alpha=0.3, 
                    linewidth=0.5, label='1σ Error')
        
        # Plot uncertainty cloud
        for idx in sample_indices:
            try:
                model_sample = model.evaluate(samples[idx], wave_data)
                ax_main.plot(wave_data, model_sample, 'blue', alpha=0.01, linewidth=0.5)
            except:
                continue
        
        # Plot best fit
        try:
            best_model = model.evaluate(best_theta, wave_data)
            ax_main.plot(wave_data, best_model, 'red', linewidth=2, label='Best Fit')
        except Exception as e:
            print(f"Warning: Could not generate model for {instrument_name}: {e}")
            continue
        
        # Add rail system (only on first panel to avoid clutter)
        if i == 0 and velocity_marks:
            _add_rail_system_v2(ax_main, model, best_theta, wave_data)
        
        # Format main plot
        ax_main.set_ylabel('Normalized Flux')
        ax_main.set_title(f'{instrument_name}: rbvfit 2.0 Multi-Instrument Joint Fit')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_ylim(ylim)
        
        if xlim is not None:
            ax_main.set_xlim(xlim)
        
        # Residuals plot
        if show_residuals:
            ax_resid = axes[i * 2 + 1]
            residuals = (flux_data - best_model) / error_data
            
            ax_resid.step(wave_data, residuals, 'k-', where='mid', alpha=0.7, linewidth=1)
            ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
            ax_resid.axhline(1, color='gray', linestyle=':', alpha=0.5)
            ax_resid.axhline(-1, color='gray', linestyle=':', alpha=0.5)
            
            ax_resid.set_ylabel('Residuals (σ)')
            ax_resid.grid(True, alpha=0.3)
            
            if xlim is not None:
                ax_resid.set_xlim(xlim)
            
            # Calculate and display RMS
            rms = np.sqrt(np.mean(residuals**2))
            ax_resid.text(0.02, 0.95, f'RMS = {rms:.2f}', 
                         transform=ax_resid.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Final formatting
    if show_residuals:
        axes[-1].set_xlabel('Observed Wavelength (Å)')
    else:
        axes[-1].set_xlabel('Observed Wavelength (Å)')
    
    # Add overall title
    plt.suptitle(f'rbvfit 2.0 Multi-Instrument Joint Fit\n', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Verbose output
    if verbose:
        _print_parameter_summary_v2(model, best_theta, samples)
    
    # Save or show
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved multi-instrument plot to {outfile}")
    else:
        plt.show()
    
    return fig
def _add_rail_system_v2(ax, model, best_theta, wave_data):
    """
    Add the rail system visualization for v2.0 ion groups.
    """
    try:
        # Get configuration and parameter manager
        config = model.config
        param_manager = model.param_manager
        
        # Convert theta to parameters
        parameters = param_manager.theta_to_parameters(best_theta)
        
        # Speed of light
        c_kms = 299792.458
        
        # Component colors
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Rail positioning
        y_rail_base = 1.05
        rail_spacing = 0.06
        tick_len = 0.02
        
        # Process each system
        rail_idx = 0
        for system in config.systems:
            z = system.redshift
            
            for ion_group in system.ion_groups:
                ion_name = ion_group.ion_name
                transitions = ion_group.transitions
                n_components = ion_group.components
                
                # Current rail height
                y_rail = y_rail_base + rail_idx * rail_spacing
                
                # Find velocity components for this ion
                component_velocities = []
                for comp_idx in range(n_components):
                    for param_key, param_set in parameters.items():
                        sys_idx, ion, comp = param_key
                        if ion == ion_name and comp == comp_idx:
                            component_velocities.append(param_set.v)
                            break
                
                if not component_velocities:
                    continue
                
                # Calculate wavelength range for rail
                all_wavelengths = []
                for rest_wave in transitions:
                    for v_comp in component_velocities:
                        obs_wave = rest_wave * (1 + z) * (1 + v_comp / c_kms)
                        if wave_data.min() <= obs_wave <= wave_data.max():
                            all_wavelengths.append(obs_wave)
                
                if not all_wavelengths:
                    continue
                
                rail_start = min(all_wavelengths)
                rail_end = max(all_wavelengths)
                
                # Draw horizontal rail
                ax.plot([rail_start, rail_end], [y_rail, y_rail], 
                       color='gray', linewidth=2, alpha=0.7)
                
                # Ion label
                rail_center = (rail_start + rail_end) / 2
                ax.text(rail_center, y_rail + 0.015, f'{ion_name} z={z:.3f}', 
                       ha='center', va='bottom', fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
                
                # Component ticks
                for comp_idx, v_comp in enumerate(component_velocities):
                    color = colors[comp_idx % len(colors)]
                    
                    for rest_wave in transitions:
                        obs_wave = rest_wave * (1 + z) * (1 + v_comp / c_kms)
                        
                        if wave_data.min() <= obs_wave <= wave_data.max():
                            # Vertical tick
                            ax.plot([obs_wave, obs_wave], [y_rail, y_rail - tick_len],
                                   color=color, linewidth=3, alpha=0.8)
                            
                            # Velocity label (only on first transition)
                            if rest_wave == transitions[0]:
                                ax.text(obs_wave, y_rail - tick_len - 0.01, 
                                       f'c{comp_idx+1}\n{v_comp:.0f}', 
                                       ha='center', va='top', fontsize=9, 
                                       color=color, weight='bold')
                
                rail_idx += 1
        
        # Adjust plot limits
        current_ylim = ax.get_ylim()
        new_top = max(current_ylim[1], y_rail_base + rail_idx * rail_spacing + 0.08)
        ax.set_ylim(current_ylim[0], new_top)
        
        print(f"Added rail system for {rail_idx} ion groups")
        
    except Exception as e:
        print(f"Warning: Could not add rail system: {e}")


def _determine_burntime_v2(fitter, burntime=None, burn_fraction=0.2):
    """Determine burn-in time for v2.0 plotting."""
    if burntime is not None:
        return int(burntime)
    
    total_steps = fitter.no_of_steps
    
    # Try autocorrelation-based detection
    try:
        if hasattr(fitter.sampler, 'get_autocorr_time'):
            tau = fitter.sampler.get_autocorr_time()
            mean_tau = np.nanmean(tau)
            
            if np.isfinite(mean_tau) and mean_tau > 0:
                auto_burntime = int(3 * mean_tau)
                min_burn = int(0.1 * total_steps)
                max_burn = int(0.4 * total_steps)
                auto_burntime = np.clip(auto_burntime, min_burn, max_burn)
                return auto_burntime
    except:
        pass
    
    # Fallback to fraction
    return int(total_steps * burn_fraction)


def _extract_samples_smart_v2(sampler, burnin=200, thin=15):
    """Extract samples for v2.0 plotting."""
    try:
        if hasattr(sampler, 'get_chain'):
            try:
                return sampler.get_chain(discard=burnin, thin=thin, flat=True)
            except TypeError:
                chain = sampler.get_chain()
                return chain[burnin::thin].reshape(-1, chain.shape[-1])
        elif hasattr(sampler, 'chain'):
            chain = sampler.chain
            return chain[:, burnin::thin, :].reshape(-1, chain.shape[-1])
        else:
            raise AttributeError("Sampler has no recognized chain interface")
    except Exception as e:
        print(f"Error extracting samples: {e}")
        if hasattr(sampler, 'flatchain'):
            return sampler.flatchain[burnin::thin]
        else:
            raise RuntimeError("Could not extract samples from sampler")


def _print_v1_parameter_summary(samples, n_params):
    """Print v1.0 parameter summary."""
    try:
        from IPython.display import display, Math
        
        nfit = int(n_params / 3)
        N_tile = np.tile("logN", nfit)
        b_tile = np.tile("b", nfit)
        v_tile = np.tile("v", nfit)

        tmp = np.append(N_tile, b_tile)
        text_label = np.append(tmp, v_tile)

        for i in range(len(text_label)):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
            display(Math(txt))
    except ImportError:
        print("IPython not available for detailed parameter display")
    except Exception as e:
        print(f"Could not display parameter summary: {e}")


def _print_parameter_summary_v2(model, best_theta, samples):
    """Print v2.0 parameter summary with ion organization."""
    print("\n" + "=" * 60)
    print("rbvfit 2.0 PARAMETER SUMMARY")
    print("=" * 60)
    
    try:
        param_manager = model.param_manager
        parameters = param_manager.theta_to_parameters(best_theta)
        
        # Calculate uncertainties
        uncertainties = np.std(samples, axis=0)
        param_names = param_manager.get_parameter_names()
        
        # Organize by systems and ions
        current_system = None
        for (sys_idx, ion_name, comp_idx), param_set in parameters.items():
            
            # System header
            if current_system != sys_idx:
                if hasattr(model.config, 'systems') and sys_idx < len(model.config.systems):
                    system = model.config.systems[sys_idx]
                    print(f"\nSystem {sys_idx + 1} (z = {system.redshift:.6f}):")
                else:
                    print(f"\nSystem {sys_idx + 1}:")
                current_system = sys_idx
            
            # Find parameter indices for uncertainties
            N_idx = next((i for i, name in enumerate(param_names) 
                         if f"N_{ion_name}" in name and f"c{comp_idx}" in name), None)
            b_idx = next((i for i, name in enumerate(param_names) 
                         if f"b_{ion_name}" in name and f"c{comp_idx}" in name), None)
            v_idx = next((i for i, name in enumerate(param_names) 
                         if f"v_{ion_name}" in name and f"c{comp_idx}" in name), None)
            
            # Get uncertainties
            N_err = uncertainties[N_idx] if N_idx is not None else 0.0
            b_err = uncertainties[b_idx] if b_idx is not None else 0.0
            v_err = uncertainties[v_idx] if v_idx is not None else 0.0
            
            print(f"  {ion_name} Component {comp_idx + 1}:")
            print(f"    logN = {param_set.N:.3f} ± {N_err:.3f} [log cm⁻²]")
            print(f"    b    = {param_set.b:.1f} ± {b_err:.1f} km/s")
            print(f"    v    = {param_set.v:.1f} ± {v_err:.1f} km/s")
        
        print(f"\nPosterior samples: {len(samples)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Warning: Could not generate parameter summary: {e}")
        print(f"Parameters: {len(best_theta)} fitted")
        print(f"Samples: {len(samples)}")
        print("=" * 60)


def _plot_quick_fit(model, fitter, outfile=False, xlim=None, ylim=None,
                   show_residuals=True, show_components=False, velocity_marks=True,
                   verbose=True, datasets=None, **kwargs):
    """
    Plot quick fit (deterministic) results for both rbvfit 1.0 and 2.0.
    
    Parameters
    ----------
    model : object
        Model object (v1.0 create_voigt object or v2.0 VoigtModel)
    fitter : vfit
        Fitter object with quick fit results or best_theta
    outfile : str or False, optional
        Save figure to file (default: False)
    xlim : list, optional
        X-axis limits [start, end]
    ylim : list, optional
        Y-axis limits [start, end] for flux plots
    show_residuals : bool, optional
        Include residual plots (default: True)
    show_components : bool, optional
        Show individual velocity components (default: False for quick fits)
    velocity_marks : bool, optional
        Mark component velocities (default: True)
    verbose : bool, optional
        Print parameter summary (default: True)
    datasets : dict, optional
        Override datasets for plotting
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        The plot figure
    """
    
    print("Creating quick fit visualization...")
    
    # Extract best-fit parameters
    if hasattr(fitter, 'theta_best'):
        # From QuickFitResults object
        best_theta = fitter.theta_best
        theta_errors = fitter.theta_best_error
        fit_info = getattr(fitter, 'fit_info', {})
        method = getattr(fitter, 'method', 'quick_fit')
        success = getattr(fitter, 'success', True)
    else:
        # Fallback to current theta (assumes already fitted)
        best_theta = fitter.theta
        theta_errors = None
        fit_info = {}
        method = 'quick_fit'
        success = True
    
    # Set up datasets
    plot_datasets = _setup_datasets(fitter, datasets, verbose=verbose)
    
    # Detect rbvfit version for model evaluation
    is_v2_model = _is_v2_model(model)
    
    
    # Set default y-limits based on version
    if ylim is None:
        ylim = [0, 1.2] if is_v2_model else [0, 1.6]
    
    if show_residuals:
        fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(12, 10),
                                               gridspec_kw={'height_ratios': [3, 1]})
        axes = [ax_main, ax_resid]
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax_main]
    
    # Set up larger font sizes for publication quality
    BIGGER_SIZE = 14
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('legend', fontsize=BIGGER_SIZE)
    
    
    # Calculate model spectrum
    try:
        if is_v2_model:
            # rbvfit 2.0
            model_flux = model.evaluate(best_theta, fitter.wave_obs)
        else:
            # rbvfit 1.0
            model_flux = model.model_flux(best_theta, fitter.wave_obs)
        
        ax_main = axes[0]
        
        # Plot data
        data_label = f'Observed Data'
        ax_main.step(fitter.wave_obs, fitter.fnorm, 'k-', where='mid', linewidth=1,
                    alpha=0.8, label=data_label)
        ax_main.step(fitter.wave_obs, fitter.enorm, 'gray', where='mid', alpha=0.3,
                    linewidth=0.5, label='1σ Error')
        
        # Plot best-fit model
        ax_main.plot(fitter.wave_obs, model_flux, 'red', linewidth=2, label='Best Fit')
        
        # Add parameter uncertainty band if available
        if theta_errors is not None and np.any(np.isfinite(theta_errors)):
            try:
                # Calculate model uncertainty using simple parameter perturbation
                n_samples = 50
                model_samples = []
                
                for _ in range(n_samples):
                    # Perturb parameters within 1-sigma
                    perturbed_theta = best_theta + np.random.normal(0, theta_errors)
                    
                    try:
                        if is_v2_model:
                            perturbed_flux = model.evaluate(perturbed_theta, fitter.wave_obs)
                        else:
                            perturbed_flux = model.model_flux(perturbed_theta, fitter.wave_obs)
                        
                        model_samples.append(perturbed_flux)
                    except:
                        continue
                
                if model_samples:
                    model_samples = np.array(model_samples)
                    model_std = np.std(model_samples, axis=0)
                    ax_main.fill_between(fitter.wave_obs, model_flux - model_std, model_flux + model_std,
                                       color='red', alpha=0.2, label='1σ Model Uncertainty')
            except Exception as e:
                if verbose:
                    print(f"Could not calculate model uncertainty: {e}")
        
        # Add velocity marks for rbvfit 2.0
        if is_v2_model and velocity_marks:
            try:
                _add_rail_system_v2(ax_main, model, best_theta, fitter.wave_obs)
            except Exception as e:
                if verbose:
                    print(f"Could not add velocity marks: {e}")
        
        # Format main plot
        ax_main.set_ylabel('Normalized Flux')        
        title = f'Quick Fit Results ({method})'
        
        if not success:
            title += ' [FAILED]'
        
        ax_main.set_title(title)
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_ylim(ylim)
        
        if xlim is not None:
            ax_main.set_xlim(xlim)
        
        # Residuals plot
        if show_residuals:
            ax_resid = axes[1]
            
            residuals = (fitter.fnorm - model_flux) / fitter.enorm
            
            ax_resid.step(fitter.wave_obs, residuals, 'k-', where='mid', alpha=0.7, linewidth=1)
            ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
            ax_resid.axhline(1, color='gray', linestyle=':', alpha=0.5)
            ax_resid.axhline(-1, color='gray', linestyle=':', alpha=0.5)
            
            ax_resid.set_ylabel('Residuals (σ)')
            ax_resid.grid(True, alpha=0.3)
            
            if xlim is not None:
                ax_resid.set_xlim(xlim)
            
            # Calculate and display statistics
            rms = np.sqrt(np.mean(residuals**2))
            chi2_reduced = np.mean(residuals**2)
            
            stats_text = f'RMS = {rms:.2f}\nχ²/ν = {chi2_reduced:.2f}'
            
            # Add method-specific info
            if 'nfev' in fit_info:
                stats_text += f'\nFunc evals: {fit_info["nfev"]}'
            
            ax_resid.text(0.02, 0.95, stats_text,
                         transform=ax_resid.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontsize=10)
        
    except Exception as e:
        if verbose:
            print(f"Error creating plot: {e}")
        return None
    
    # Final formatting
    if show_residuals:
        axes[-1].set_xlabel('Observed Wavelength (Å)')
    else:
        axes[-1].set_xlabel('Observed Wavelength (Å)')
    
    # Add overall title with fit information
    title_parts = []
    if success:
        title_parts.append("✅ Quick Fit Results")
    else:
        title_parts.append("❌ Quick Fit Failed")
    
    if len(plot_datasets) > 1:
        title_parts.append(f"({len(plot_datasets)} datasets)")
    
    if theta_errors is not None:
        n_finite_errors = np.sum(np.isfinite(theta_errors))
        title_parts.append(f"• {n_finite_errors}/{len(best_theta)} parameter uncertainties")
    
    plt.suptitle(" ".join(title_parts), fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Print parameter summary if verbose
    if verbose:
        _print_quick_fit_summary(best_theta, theta_errors, fit_info, method, fitter)
    
    # Save or show
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved quick fit plot to {outfile}")
    else:
        plt.show()
    
    return fig


def _print_quick_fit_summary(best_theta, theta_errors, fit_info, method, fitter):
    """Print summary of quick fit results."""
    print("\n" + "=" * 60)
    print("QUICK FIT PARAMETER SUMMARY")
    print("=" * 60)
    
    print(f"Method: {method}")
    print(f"Parameters: {len(best_theta)}")
    
    
    # Parameter values
    print(f"\nParameter Values:")
    print("-" * 40)
    
    for i, val in enumerate(best_theta):
        if theta_errors is not None and np.isfinite(theta_errors[i]):
            print(f"  theta[{i:2d}] = {val:8.3f} ± {theta_errors[i]:.3f}")
        else:
            print(f"  theta[{i:2d}] = {val:8.3f}")
    
    
    print("=" * 60)


# Convenience function to create fitter with sampler selection
def create_fitter(model, theta, lb, ub, wave_obs, fnorm, enorm, 
                  sampler='emcee', **kwargs):
    """
    Convenience function to create vfit object with sampler selection.
    
    Parameters
    ----------
    sampler : str
        Sampler to use: 'emcee' or 'zeus'
    **kwargs
        Additional arguments passed to vfit constructor
    """
    return vfit(model, theta, lb, ub, wave_obs, fnorm, enorm, 
                sampler=sampler, **kwargs)


# Utility function to check available samplers
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


def print_multi_instrument_help():
    """Print help for multi-instrument usage with updated examples."""
    print("\nMulti-Instrument Usage:")
    print("=" * 25)
    
    print("\n🆕 NEW UNIFIED INTERFACE (Recommended):")
    print("="*45)
    
    print("\n📖 Single Instrument:")
    print("```python")
    print("# Clean, symmetric interface")
    print("fitter = vfit({")
    print("    'HIRES': {")
    print("        'model': model_func,")
    print("        'wave': wave_array,")
    print("        'flux': flux_array,")
    print("        'error': error_array")
    print("    }")
    print("}, theta, lb, ub)")
    print("```")
    
    print("\n📖 Multi-Instrument:")
    print("```python")
    print("# Same pattern for any number of instruments")
    print("fitter = vfit({")
    print("    'HIRES': {'model': model_a, 'wave': wave_a, 'flux': flux_a, 'error': error_a},")
    print("    'FIRE':  {'model': model_b, 'wave': wave_b, 'flux': flux_b, 'error': error_b},")
    print("    'UVES':  {'model': model_c, 'wave': wave_c, 'flux': flux_c, 'error': error_c}")
    print("}, theta, lb, ub)")
    print("")
    print("# Automatic detection: no flags needed!")
    print("# Multi-instrument mode detected automatically")
    print("```")
    
    print("\n📜 LEGACY INTERFACE (Still Supported):")
    print("="*40)
    
    print("\n📖 Legacy Single Instrument:")
    print("```python")
    print("fitter = vfit(model, theta, lb, ub, wave, flux, error)")
    print("```")
    
    print("\n📖 Legacy Multi-Instrument:")
    print("```python")
    print("fitter = vfit(model_a, theta, lb, ub, wave_a, flux_a, error_a,")
    print("              multi_instrument=True,")
    print("              instrument_data={")
    print("                  'FIRE': {'model': model_b, 'wave': wave_b, 'flux': flux_b, 'error': error_b}")
    print("              })")
    print("```")
    
    print("\n✨ BENEFITS OF NEW UNIFIED INTERFACE:")
    print("• 🎯 Symmetric treatment of all instruments")
    print("• 🚫 No more primary/secondary confusion") 
    print("• 🤖 Automatic single/multi-instrument detection")
    print("• 🧹 Cleaner, more intuitive API")
    print("• 🔄 Full backward compatibility")
    print("• 📈 Same performance and analysis capabilities")
    
    print("\n🔧 MIGRATION GUIDE:")
    print("```python")
    print("# OLD WAY (still works)")
    print("fitter = vfit(model_main, theta, lb, ub, wave_main, flux_main, error_main,")
    print("              multi_instrument=True,")
    print("              instrument_data={'secondary': {...}})")
    print("")
    print("# NEW WAY (recommended)")
    print("fitter = vfit({")
    print("    'main': {'model': model_main, 'wave': wave_main, 'flux': flux_main, 'error': error_main},")
    print("    'secondary': {'model': model_sec, 'wave': wave_sec, 'flux': flux_sec, 'error': error_sec}")
    print("}, theta, lb, ub)")
    print("```")
    
    print("\n💡 COMPATIBILITY NOTES:")
    print("• All plotting functions work identically")
    print("• fit_results analysis unchanged") 
    print("• MCMC diagnostics identical")
    print("• Parameter extraction same")
    print("• New interface produces same fitter object after runmcmc()")

# Ion-specific bounds lookup table
ION_BOUNDS_TABLE = {
    'HI': {
        'N': (12.0, 22.0),     # DLA range
        'b': (10.0, 200.0),   # Thermal + turbulent
        'v': (-500.0, 500.0)
    },
    'MgII': {
        'N': (11.0, 16.0),    # Typical MgII range
        'b': (3.0, 100.0),    # Lower thermal + turbulent
        'v': (-300.0, 300.0)
    },
    'FeII': {
        'N': (11.0, 15.5),    # Associated with MgII
        'b': (3.0, 100.0),
        'v': (-300.0, 300.0)
    },
    'CIV': {
        'N': (12.0, 16.0),    # High-ion tracer
        'b': (5.0, 150.0),    # Higher velocities
        'v': (-400.0, 400.0)
    },
    'OVI': {
        'N': (12.5, 16.0),    # Hot gas tracer
        'b': (10.0, 150.0),   # High temperature
        'v': (-400.0, 400.0)
    },
    'SiII': {
        'N': (11.0, 15.5),    # Metal-line system
        'b': (3.0, 100.0),
        'v': (-300.0, 300.0)
    },
    'SiIV': {
        'N': (12.0, 15.5),    # Intermediate-ion
        'b': (5.0, 150.0),
        'v': (-350.0, 350.0)
    },
    'CII': {
        'N': (12.0, 17.0),    # Associated with HI
        'b': (5.0, 120.0),
        'v': (-400.0, 400.0)
    },
    'NV': {
        'N': (12.5, 15.5),    # High-ion tracer
        'b': (10.0, 150.0),
        'v': (-400.0, 400.0)
    },
    'AlII': {
        'N': (10.5, 14.5),    # Metal-line system
        'b': (3.0, 80.0),
        'v': (-250.0, 250.0)
    },
    'AlIII': {
        'N': (11.0, 15.0),    # Metal-line system
        'b': (5.0, 100.0),
        'v': (-300.0, 300.0)
    },
    'OI': {
        'N': (12.0, 16.5),    # Neutral oxygen
        'b': (5.0, 100.0),
        'v': (-300.0, 300.0)
    }
}



def set_bounds(nguess, bguess, vguess, **kwargs):
    """
    Set bounds for MCMC parameters with optional custom overrides and ion-aware defaults.

    Parameters:
    - nguess, bguess, vguess: arrays of initial guesses for logN, b, and v.
    - Optional keyword arguments:
        - Nlow, blow, vlow: custom lower bounds
        - Nhi, bhi, vhi: custom upper bounds
        - ions: list of ion names for smart bounds (e.g., ['MgII', 'FeII'])
        - ion_bounds: dict with custom ion bounds to override defaults

    Returns:
    - bounds: list containing [lower_bounds, upper_bounds]
    - lb: concatenated lower bounds
    - ub: concatenated upper bounds

    Examples:
        # Traditional usage (unchanged)
        bounds, lb, ub = set_bounds(nguess, bguess, vguess)

        # With ion-aware bounds
        bounds, lb, ub = set_bounds(nguess, bguess, vguess, ions=['MgII', 'FeII'])

        # Mix of ion-aware and custom bounds
        bounds, lb, ub = set_bounds(nguess, bguess, vguess, 
                                  ions=['MgII', 'FeII'], 
                                  Nlow=[12.0, 11.5])  # Custom N lower bounds

        # Custom ion bounds table
        custom_ions = {'MgII': {'N': (11.5, 15.0), 'b': (5.0, 50.0)}}
        bounds, lb, ub = set_bounds(nguess, bguess, vguess,
                                  ions=['MgII'], 
                                  ion_bounds=custom_ions)
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
            raise ValueError(f"Length of ions list ({len(ions)}) must match "
                           f"number of components ({len(nguess)})")
        
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
                vlow[i] = vguess[i] - 50.0
                vHI[i] = vguess[i] + 50.0
                continue
            
            # Apply ion-specific bounds
            Nlow[i] = ion_data['N'][0]
            NHI[i] = ion_data['N'][1]
            blow[i] = ion_data['b'][0]
            bHI[i] = ion_data['b'][1]
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
    
    Parameters:
    - ion_name: string name of ion (e.g., 'MgII')
    - N_range: tuple of (min, max) log column density
    - b_range: tuple of (min, max) Doppler parameter in km/s
    - v_range: tuple of (min, max) velocity in km/s
    
    Example:
        add_ion_to_bounds_table('CaII', (11.0, 14.0), (3.0, 80.0), (-200.0, 200.0))
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
    
    Parameters:
    - ion_name: string name of ion
    
    Returns:
    - dict with 'N', 'b', 'v' bounds or None if not found
    """
    return ION_BOUNDS_TABLE.get(ion_name, None)

# Legacy compatibility
vfit_mcmc = vfit  # Alias for backward compatibility

def print_usage_examples():
    """Print usage examples for the refactored interface."""
    print("\n" + "="*60)
    print("RBVFIT 2.0 MCMC FITTING - USAGE EXAMPLES")
    print("="*60)
    
    print("\n🆕 New Unified Interface (Recommended):")
    print("="*40)
    
    print("\n📖 Single Instrument:")
    print("```python")
    print("fitter = vfit({")
    print("    'HIRES': {")
    print("        'model': model_func,")
    print("        'wave': wave_array,")
    print("        'flux': flux_array,")
    print("        'error': error_array")
    print("    }")
    print("}, theta, lb, ub)")
    print("```")
    
    print("\n📖 Multi-Instrument:")
    print("```python")
    print("fitter = vfit({")
    print("    'HIRES': {'model': model_a, 'wave': wave_a, 'flux': flux_a, 'error': error_a},")
    print("    'FIRE': {'model': model_b, 'wave': wave_b, 'flux': flux_b, 'error': error_b},")
    print("    'UVES': {'model': model_c, 'wave': wave_c, 'flux': flux_c, 'error': error_c}")
    print("}, theta, lb, ub)")
    print("```")
    
    print("\n📜 Legacy Interface (Still Supported):")
    print("="*40)
    
    print("\n📖 Single Instrument (Legacy):")
    print("```python")
    print("fitter = vfit(model, theta, lb, ub, wave, flux, error)")
    print("```")
    
    print("\n📖 Multi-Instrument (Legacy):")
    print("```python")
    print("fitter = vfit(model_a, theta, lb, ub, wave_a, flux_a, error_a,")
    print("              multi_instrument=True,")
    print("              instrument_data={")
    print("                  'FIRE': {'model': model_b, 'wave': wave_b, 'flux': flux_b, 'error': error_b}")
    print("              })")
    print("```")
    
    print("\n✨ Benefits of New Interface:")
    print("• Symmetric treatment of all instruments")
    print("• No more primary/secondary confusion")
    print("• Automatic single/multi-instrument detection")
    print("• Cleaner, more intuitive API")
    print("• Full backward compatibility")

if __name__ == "__main__":
    # Print comprehensive help when module is run directly
    print_sampler_info()
    print_performance_tips()
    print_multi_instrument_help()
    print_usage_examples()

