from __future__ import print_function
import emcee
from multiprocessing import Pool
import numpy as np
import corner
import matplotlib.pyplot as plt
import sys
import scipy.optimize as op
from rbvfit.rb_vfit import rb_veldiff as rb_veldiff
from rbvfit import rb_setline as rb
import pdb
import warnings

# Try to import zeus sampler
try:
    import zeus
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False
    zeus = None

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







def set_bounds(nguess,bguess,vguess,**kwargs):
    """
        Setting up bounds and giving option to manually update bounds.
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


    if 'Nlow' in kwargs:
        Nlow=kwargs['Nlow']
    else:
        Nlow=np.zeros((len(nguess,)))

    if 'blow' in kwargs:
        blow=kwargs['blow']
    else:
        blow=np.zeros((len(nguess,)))
    
    if 'vlow' in kwargs:
        vlow=kwargs['vlow']
    else:
        vlow=np.zeros((len(nguess,)))

    if 'Nhi' in kwargs:
        NHI=kwargs['Nhi']
    else:
        NHI=np.zeros((len(nguess,)))

    if 'bhi' in kwargs:
        bHI=kwargs['bhi']
    else:
        bHI=np.zeros((len(nguess,)))
    
    if 'vhi' in kwargs:
        vHI=kwargs['vhi']
    else:
        vHI=np.zeros((len(nguess,)))


    for i in range(0,len(nguess)):

        if 'Nlow' not in kwargs:
            Nlow[i]=nguess[i]-2.

        if 'blow' not in kwargs:
            blow[i]=bguess[i]-40.
            if blow[i] < 2.:
                blow[i] = 2.

        if 'vlow' not in kwargs:
            vlow[i]=vguess[i]-50.

        if 'Nhi' not in kwargs:
            NHI[i]=nguess[i]+2.

        if 'bhi' not in kwargs:
            bHI[i]=bguess[i]+40.
            if bHI[i] > 200.:
                bHI[i] = 150.
        if 'vhi' not in kwargs:
            vHI[i]=vguess[i]+50.
    lb=np.concatenate((Nlow,blow,vlow))
    ub=np.concatenate((NHI,bHI,vHI))
    bounds=[lb,ub]
    return bounds, lb, ub

class vfit(object):
    def __init__(self, model, theta, lb, ub, wave_obs, fnorm, enorm, 
                 no_of_Chain=50, no_of_steps=1000, perturbation=1e-6,
                 skip_initial_state_check=False, sampler='emcee', **kwargs):
        """
        Enhanced vfit class with support for multiple MCMC samplers.
        
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
        **kwargs
            Additional keyword arguments for multi-dataset fitting
        """
        # Main class that performs all the fitting
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
        
        # Create a flag to check if a second model is used. 
        # Add the second model
        # Add second set of wave,flux,error
        # Then update the loglikelihood function
        if 'second_spec' in kwargs:
            self.second_spec_flag = True
        else:
            self.second_spec_flag = False

        if self.second_spec_flag == True:
            if 'second_spec_dict' in kwargs:
                second_spec_dict = kwargs['second_spec_dict']
                self.wave2 = second_spec_dict['wave']
                self.fnorm2 = second_spec_dict['flux']
                self.enorm2 = second_spec_dict['error']

            if 'model2' in kwargs:
                self.model2 = kwargs['model2']



    def lnprior(self, theta):
            for index in range(0, len(self.lb)):
                if (self.lb[index] > theta[index]) or (self.ub[index] < theta[index]):
                    return -np.inf
                    break
            return 0.0
    
    def lnlike(self, theta):
        # Update this function to enable joint likelihood of two models
        try:
            model_dat = self.model(theta, self.wave_obs)
            inv_sigma2 = 1.0 / (self.enorm ** 2)
            lnlike_total1 = -0.5 * (np.sum((self.fnorm - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    
            if self.second_spec_flag == True:
                model_dat2 = self.model2(theta, self.wave2)
                inv_sigma2_2 = 1.0 / (self.enorm2 ** 2)
                lnlike_total2 = -0.5 * (np.sum((self.fnorm2 - model_dat2) ** 2 * inv_sigma2_2 - np.log(inv_sigma2_2)))
                lnlike_total = lnlike_total1 + lnlike_total2
            else:
                lnlike_total = lnlike_total1
    
            # Check for invalid values
            if not np.isfinite(lnlike_total):
                return -np.inf
                
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
        """Set up emcee sampler."""
        ndim = len(self.lb)
        nwalkers = self.no_of_Chain
        
        if use_pool:
            pool = Pool()
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.lnprob, pool=pool
            )
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.lnprob
            )
        
        return sampler, pool if use_pool else None

    def _setup_zeus_sampler(self, guesses, use_pool=True):
        """Set up zeus sampler."""
        ndim = len(self.lb)
        nwalkers = self.no_of_Chain
        
        if use_pool:
            pool = Pool()
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
        
        # Calculate diagnostics
        mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print("Mean acceptance fraction: {0:.3f}".format(mean_acceptance_fraction))
        
        # Acceptance fraction interpretation
        if mean_acceptance_fraction < 0.2:
            print("⚠ Warning: Low acceptance fraction (<0.2). Consider reducing step size.")
        elif mean_acceptance_fraction > 0.7:
            print("⚠ Warning: High acceptance fraction (>0.7). Consider increasing step size.")
        else:
            print("✓ Good acceptance fraction (0.2-0.7)")
        
        # Autocorrelation time (if available)
        try:
            if hasattr(sampler, 'get_autocorr_time'):
                autocorr_time = sampler.get_autocorr_time()
                mean_autocorr_time = np.nanmean(autocorr_time)
                print("Mean auto-correlation time: {0:.3f} steps".format(mean_autocorr_time))
                
                # Check if chain is long enough
                if no_of_steps < 50 * mean_autocorr_time:
                    print("⚠ Warning: Chain may be too short for reliable results")
                    print(f"  Recommended: >{50 * mean_autocorr_time:.0f} steps")
            else:
                print("Auto-correlation time not available for this sampler")
        except Exception:
            print("Could not calculate auto-correlation time")

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

    def _print_parameter_summary(self, sampler, burntime):
        """Print detailed parameter summary."""
        try:
            from IPython.display import display, Math
            
            # Get samples
            if hasattr(sampler, 'get_chain'):  # emcee
                samples = sampler.get_chain(discard=int(burntime), thin=15, flat=True)
            else:  # zeus or other
                chain = sampler.get_chain()
                samples = chain[int(burntime)::15].reshape(-1, chain.shape[-1])
            
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

    def plot_corner(self, outfile=False, burntime=100, **kwargs):
        """Plot corner plot with sampler-agnostic sample extraction."""
        ndim = self.ndim
        
        # Extract samples based on sampler type
        if hasattr(self.sampler, 'get_chain'):  # emcee
            samples = self.sampler.get_chain(discard=burntime, thin=15, flat=True)
        else:  # zeus or other
            chain = self.sampler.get_chain()
            samples = chain[burntime::15].reshape(-1, chain.shape[-1])

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
            'has_zeus': HAS_ZEUS
        }
        
        if hasattr(self, 'sampler'):
            info['acceptance_fraction'] = np.mean(self.sampler.acceptance_fraction)
            
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


if __name__ == "__main__":
    # Print sampler information when module is run directly
    print_sampler_info()
