from __future__ import print_function
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
import sys
import scipy.optimize as op
from rbvfit.rb_vfit import rb_veldiff as rb_veldiff
from rbvfit import rb_setline as rb
import pdb
import warnings
import multiprocessing as mp

# Try to import zeus sampler
try:
    import zeus
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False
    zeus = None

# Set up optimized multiprocessing context
try:
    OptimizedPool = mp.get_context('fork').Pool
    MP_CONTEXT = 'fork'
except (AttributeError, RuntimeError):
    # Fallback to default if fork is not available
    OptimizedPool = mp.Pool
    MP_CONTEXT = 'default'


def plot_model(wave_obs,fnorm,enorm,fit,model,outfile= False,xlim=[-600.,600.],verbose=False):
        #This model only works if there are no nuissance paramteres
        

        theta_prime=fit.best_theta
        value1=fit.low_theta
        value2=fit.high_theta
        n_clump=model.nclump 
        n_clump_total=int(len(theta_prime)/3)

        ntransition=model.ntransition
        zabs=model.zabs

        samples=fit.samples
        model_mcmc=fit.model

        wave_list=np.zeros( len(model.lambda_rest_original),)
        # Use the input lambda rest list to plot correctly
        for i in range(0,len(wave_list)):
            s=rb.rb_setline(model.lambda_rest_original[i],'closest')
            wave_list[i]=s['wave']


        wave_rest=wave_obs/(1+zabs[0])
        
        best_N = theta_prime[0:n_clump_total]
        best_b = theta_prime[n_clump_total:2 * n_clump_total]
        best_v = theta_prime[2 * n_clump_total:3 * n_clump_total]
        
        low_N = value1[0:n_clump_total]
        low_b = value1[n_clump_total:2 * n_clump_total]
        low_v = value1[2 * n_clump_total:3 * n_clump_total]
        
        high_N = value2[0:n_clump_total]
        high_b = value2[n_clump_total:2 * n_clump_total]
        high_v = value2[2 * n_clump_total:3 * n_clump_total]
            


        #Now extracting individual fitted components
        best_fit, f1 = model.model_fit(theta_prime, wave_obs)

        fig, axs = plt.subplots(ntransition, sharex=True, sharey=False,figsize=(12,18 ),gridspec_kw={'hspace': 0})
        # Sets title to outfile is not False
        if outfile==False:
            pass
        else:
            plt.title(outfile, y=1.07*ntransition, loc='right', size=12)            
        
        BIGGER_SIZE = 18
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        index = np.random.randint(0, high=len(samples), size=100)
        
        
        if ntransition == 1:
            #When there are no nuissance parameter
            #Now loop through each transition and plot them in velocity space
            vel=rb_veldiff(wave_list[0],wave_rest)
            axs.step(vel, fnorm, 'k-', linewidth=1., label=wave_list[0])
            axs.legend()
            axs.step(vel, enorm, color='r', linewidth=1.)
            # Plotting a random sample of outputs extracted from posterior dis
            for ind in range(len(index)):
                axs.plot(vel, model_mcmc(samples[index[ind], :], wave_obs), color="k", alpha=0.1)
            axs.set_ylim([0, 1.6])
            axs.set_xlim(xlim)
            axs.plot(vel, best_fit, color='b', linewidth=3)
            axs.plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
            # plot individual components
            for dex in range(0,np.shape(f1)[1]):
                axs.plot(vel, f1[:, dex], 'g:', linewidth=3)
    
            for iclump in range(0,n_clump):
                axs.plot([best_v[iclump],best_v[iclump]],[1.05,1.15],'k--',lw=4)
                text1=r'$logN \;= '+ str('%.2f' % best_N[iclump]) +'^{ + ' + str('%.2f' % (high_N[iclump]-best_N[iclump]))+'}'+ '_{ -' +  str('%.2f' % (best_N[iclump]-low_N[iclump]))+'}$'
                axs.text(best_v[iclump],1.2,text1,
                     fontsize=14,rotation=90, rotation_mode='anchor')
                text2=r'$b ='+str('%.0f' % best_b[iclump]) +'^{ + ' + str('%.0f' % (high_b[iclump]-best_b[iclump]))+'}'+ '_{ -' +  str('%.0f' % (best_b[iclump]-low_b[iclump]))+'}$'
    
                axs.text(best_v[iclump]+30,1.2, text2,fontsize=14,rotation=90, rotation_mode='anchor')
  
        
        
        
        
        else:
     
            
            #Now loop through each transition and plot them in velocity space
            for i in range(0,ntransition):
                print(wave_list[i])
                vel=rb_veldiff(wave_list[i],wave_rest)
                axs[i].step(vel, fnorm, 'k-', linewidth=1., label=wave_list[i])
                axs[i].legend()
                axs[i].step(vel, enorm, color='r', linewidth=1.)
                #pdb.set_trace()
                # Plotting a random sample of outputs extracted from posterior distribution
                for ind in range(len(index)):
                    axs[i].plot(vel, model_mcmc(samples[index[ind], :], wave_obs), color="k", alpha=0.1)
                axs[i].set_ylim([0, 1.6])
                axs[i].set_xlim(xlim)
                
                
            
                axs[i].plot(vel, best_fit, color='b', linewidth=3)
                axs[i].plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
    
                # plot individual components
                for dex in range(0,np.shape(f1)[1]):
                    axs[i].plot(vel, f1[:, dex], 'g:', linewidth=3)
                for iclump in range(0,n_clump):
                    axs[i].plot([best_v[iclump],best_v[iclump]],[1.05,1.15],'k--',lw=4)
                    if i ==0:
                        text1=r'$logN \;= '+ str('%.2f' % best_N[iclump]) +'^{ + ' + str('%.2f' % (high_N[iclump]-best_N[iclump]))+'}'+ '_{ -' +  str('%.2f' % (best_N[iclump]-low_N[iclump]))+'}$'
                        axs[i].text(best_v[iclump],1.2,text1,
                                 fontsize=14,rotation=90, rotation_mode='anchor')
                        text2=r'$b ='+str('%.0f' % best_b[iclump]) +'^{ + ' + str('%.0f' % (high_b[iclump]-best_b[iclump]))+'}'+ '_{ -' +  str('%.0f' % (best_b[iclump]-low_b[iclump]))+'}$'
                
                        axs[i].text(best_v[iclump]+30,1.2, text2,
                                 fontsize=14,rotation=90, rotation_mode='anchor')
        
        if verbose==True:
            from IPython.display import display, Math
    
            samples = fit.sampler.get_chain(discard=100, thin=15, flat=True)
            nfit = int(fit.ndim / 3)
            N_tile = np.tile("logN", nfit)
            b_tile = np.tile("b", nfit)
            v_tile = np.tile("v", nfit)
            tmp = np.append(N_tile, b_tile)
            text_label = np.append(tmp, v_tile)
            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
    
            
                display(Math(txt))

      



        if outfile==False:
            plt.show()
        else:
            outfile_fig =outfile
            fig.savefig(outfile_fig, bbox_inches='tight')








def set_bounds(nguess, bguess, vguess, **kwargs):
    """
    Set bounds for MCMC parameters with optional custom overrides.

    Parameters:
    - nguess, bguess, vguess: arrays of initial guesses for logN, b, and v.
    - Optional keyword arguments:
        - Nlow, blow, vlow: custom lower bounds
        - Nhi, bhi, vhi: custom upper bounds

    Returns:
    - bounds: list containing [lower_bounds, upper_bounds]
    - lb: concatenated lower bounds
    - ub: concatenated upper bounds


        example :
            This command sets default bounds
             > bounds,lb,ub=mc.set_bounds(nguess,bguess,vguess)

            Customize bounds
            lets say nguess=[12.2,12.3]
                     bguess=[10,12]
                     vguess=[0,199]

                     We want to set custom lower bound for logN

                     Nlow=[12.1,11.9]
                     >bounds,lb,ub=mc.set_bounds(nguess,bguess,vguess,Nlow=Nlow)



    """

    nguess = np.asarray(nguess)
    bguess = np.asarray(bguess)
    vguess = np.asarray(vguess)

    Nlow = np.asarray(kwargs.get('Nlow', nguess - 2.0))
    blow = np.asarray(kwargs.get('blow', np.clip(bguess - 40.0, 2.0, None)))
    vlow = np.asarray(kwargs.get('vlow', vguess - 50.0))

    NHI  = np.asarray(kwargs.get('Nhi', nguess + 2.0))
    bHI  = np.asarray(kwargs.get('bhi', np.clip(bguess + 40.0, None, 150.0)))
    vHI  = np.asarray(kwargs.get('vhi', vguess + 50.0))

    lb = np.concatenate([Nlow, blow, vlow])
    ub = np.concatenate([NHI,  bHI,  vHI])
    bounds = [lb, ub]

    return bounds, lb, ub


class vfit(object):
    def __init__(self, model, theta, lb, ub, wave_obs, fnorm, enorm, 
                 no_of_Chain=50, no_of_steps=1000, perturbation=1e-6,
                 skip_initial_state_check=False, sampler='emcee',
                 # New multi-instrument interface
                 multi_instrument=False, instrument_data=None,
                 # Legacy interface for backward compatibility
                 second_spec=False, second_spec_dict=None, model2=None,
                 **kwargs):
        """
        Enhanced vfit class with multi-instrument support and multiple MCMC samplers.
        
        Parameters
        ----------
        model : callable
            Model function that takes (theta, wave) and returns flux
        theta : array_like
            Initial parameter guess
        lb : array_like
            Lower bounds for parameters
        ub : array_like
            Upper bounds for parameters
        wave_obs : array_like
            Observed wavelength array
        fnorm : array_like
            Normalized flux array
        enorm : array_like
            Error array
        no_of_Chain : int, optional
            Number of MCMC walkers (default: 50)
        no_of_steps : int, optional
            Number of MCMC steps (default: 1000)
        perturbation : float, optional
            Initial walker perturbation scale (default: 1e-6)
        skip_initial_state_check : bool, optional
            Skip initial state check in sampler (default: False)
        sampler : str, optional
            MCMC sampler to use: 'emcee' or 'zeus' (default: 'emcee')
        multi_instrument : bool, optional
            Whether to use multi-instrument fitting (default: False)
        instrument_data : dict, optional
            Dictionary of additional instruments. Format:
            {'instrument_name': {'model': model_func, 'wave': wave_array, 
                               'flux': flux_array, 'error': error_array}}
        second_spec : bool, optional
            Legacy parameter for two-instrument fitting (default: False)
        second_spec_dict : dict, optional
            Legacy parameter for second instrument data
        model2 : callable, optional
            Legacy parameter for second instrument model
        **kwargs
            Additional keyword arguments
        """
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
        """
        try:
            if self.multi_instrument:
                # Multi-instrument likelihood calculation
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
                # Single instrument likelihood (original logic)
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
        nll = lambda *args: -self.lnprob(*args)
        result = op.minimize(nll, [theta])
        p = result["x"]
        return p

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
            from IPython.display import display, Math
            
            # Get samples - handle different sampler interfaces
            samples = self._extract_samples(sampler, int(burntime))
            
            ndim = samples.shape[1]
            nfit = int(ndim / 3)
            N_tile = np.tile("logN", nfit)
            b_tile = np.tile("b", nfit)
            v_tile = np.tile("v", nfit)

            tmp = np.append(N_tile, b_tile)
            text_label = np.append(tmp, v_tile)

            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
                display(Math(txt))
        except ImportError:
            print("IPython not available for detailed parameter display")
        except Exception as e:
            print(f"Could not display parameter summary: {e}")

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

    def runmcmc(self, optimize=True, verbose=False, use_pool=True):
        """
        Run MCMC with selected sampler.
        
        Parameters
        ----------
        optimize : bool, optional
            Whether to optimize initial guess first (default: True)
        verbose : bool, optional
            Whether to print detailed results (default: False)
        use_pool : bool, optional
            Whether to use multiprocessing (default: True)
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

        print(f"Using {self.sampler_name.upper()} sampler")
        if use_pool:
            print(f"Multiprocessing context: {MP_CONTEXT}")
        print(f"Skip initial state check: {self.skip_initial_state_check}")

        if optimize == True:
            print('Optimizing Guess ***********')
            # Now make a better guess
            try:
                popt = self.optimize_guess(theta)
                print('Optimization completed ***********')
            except Exception as e:
                print(f'Optimization failed: {e}. Using original guess.')
                popt = theta
        else:
            print('Skipping Optimizing Guess ***********')
            print('Using input guess for mcmc ***********')
            popt = theta

        print('Preparing MCMC ***********')
        
        # Initialize walkers
        guesses = self._initialize_walkers(popt)
        
        # Set up sampler
        pool = None
        if self.sampler_name == 'emcee':
            sampler, pool = self._setup_emcee_sampler(guesses, use_pool)
        elif self.sampler_name == 'zeus':
            sampler, pool = self._setup_zeus_sampler(guesses, use_pool)
        
        print(f"Starting {self.sampler_name.upper()} with {no_of_Chain} walkers ***********")
        
        # Calculate burn-in
        burntime = np.round(no_of_steps * 0.2)
        
        try:
            # Run MCMC
            if self.sampler_name == 'emcee':
                pos, prob, state = sampler.run_mcmc(
                    guesses, no_of_steps, 
                    progress=True,
                    skip_initial_state_check=self.skip_initial_state_check
                )
            elif self.sampler_name == 'zeus':
                # Zeus has slightly different interface
                sampler.run_mcmc(
                    guesses, no_of_steps,
                    progress=True
                )
                
        finally:
            # Always close pool if it was created
            if pool is not None:
                pool.close()
                pool.join()

        print("Done!")
        print("*****************")
        
        # Calculate diagnostics - handle sampler differences
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
        
        # Autocorrelation time (if available)
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

        # Sampler-specific diagnostics
        if self.sampler_name == 'zeus':
            # Zeus-specific diagnostics
            try:
                r_hat = zeus.diagnostics.gelman_rubin(sampler.get_chain())
                print(f"Gelman-Rubin R-hat: {np.max(r_hat):.3f}")
                if np.max(r_hat) > 1.1:
                    print("⚠ Warning: R-hat > 1.1, chains may not have converged")
                else:
                    print("✓ Good convergence (R-hat < 1.1)")
            except Exception:
                print("Could not calculate Gelman-Rubin diagnostic")

        if verbose == True:
            self._print_parameter_summary(sampler, burntime)

        self.sampler = sampler
        self.ndim = len(self.lb)
        self.nwalkers = no_of_Chain


    def plot_corner(self, outfile=False, burntime=100, **kwargs):
        """Plot corner plot with sampler-agnostic sample extraction."""
        ndim = self.ndim
        
        # Extract samples based on sampler type
        samples = self._extract_samples(self.sampler, burntime)

        st = np.percentile(samples, 50, axis=0)

        nfit = int(ndim / 3)
        N_tile = np.tile("logN", nfit)
        b_tile = np.tile("b", nfit)
        v_tile = np.tile("v", nfit)

        tmp = np.append(N_tile, b_tile)
        text_label = np.append(tmp, v_tile)

        if 'True_values' in kwargs:
            figure = corner.corner(samples, labels=text_label, truths=kwargs['True_values'])
        else:
            figure = corner.corner(samples, labels=text_label, truths=st)

        # Sets title to outfile is not False
        if outfile == False:
            pass
        else:
            plt.title(outfile, y=1.05*ndim, loc='right') 

        theta_prime = st
        value1 = np.percentile(samples, 10, axis=0)
        value2 = np.percentile(samples, 90, axis=0)
        
        # Extract the axes
        axes = np.array(figure.axes).reshape((ndim, ndim))

        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(value1[i], color="aqua")
            ax.axvline(value2[i], color="aqua")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value1[xi], color="aqua")
                ax.axvline(value2[xi], color="aqua")

        self.best_theta = theta_prime
        self.low_theta = value1
        self.high_theta = value2
        self.samples = samples

        if outfile == False:
            plt.show()
        else:
            outfile_fig = outfile
            figure.savefig(outfile_fig, bbox_inches='tight')

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
    """Print help for multi-instrument usage."""
    print("\nMulti-Instrument Usage:")
    print("=" * 25)
    
    print("\n📖 New Interface (Recommended):")
    print("```python")
    print("# Set up models for each instrument")
    print("model_A = lambda theta, wave: compiled.model_flux(theta, wave, instrument='A')")
    print("model_B = lambda theta, wave: compiled.model_flux(theta, wave, instrument='B')")
    print("")
    print("# Create fitter with multi-instrument data")
    print("fitter = vfit(model_A, theta, lb, ub, wave_A, flux_A, error_A,")
    print("              multi_instrument=True,")
    print("              instrument_data={")
    print("                  'B': {'model': model_B, 'wave': wave_B, 'flux': flux_B, 'error': error_B},")
    print("                  'C': {'model': model_C, 'wave': wave_C, 'flux': flux_C, 'error': error_C}")
    print("              })")
    print("```")
    
    print("\n📜 Legacy Interface (Still Supported):")
    print("```python")
    print("# Two-instrument fitting (legacy)")
    print("fitter = vfit(model_A, theta, lb, ub, wave_A, flux_A, error_A,")
    print("              second_spec=True,")
    print("              second_spec_dict={'wave': wave_B, 'flux': flux_B, 'error': error_B},")
    print("              model2=model_B)")
    print("```")
    
    print("\n✨ Benefits of Multi-Instrument Fitting:")
    print("• Shared physical parameters across instruments")
    print("• Better parameter constraints from joint data")
    print("• Automatic handling of instrument-specific wavelength coverage")
    print("• Consistent error propagation across all datasets")


# Legacy compatibility
vfit_mcmc = vfit  # Alias for backward compatibility


if __name__ == "__main__":
    # Print comprehensive help when module is run directly
    print_sampler_info()
    print_performance_tips()
    print_multi_instrument_help()

