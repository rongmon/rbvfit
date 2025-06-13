import numpy as np
import time
import emcee

def simple_v1_v2_test():
    """Simple test: v1 vs v2 MCMC with same synthetic data."""
    
    print("Simple v1 vs v2 MCMC Test")
    print("=" * 40)
    
    # 1. Create synthetic data with v1
    from rbvfit import model as v1_model
    from rbvfit import vfit_mcmc as v1_mcmc
    
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    theta_true = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    wave = np.linspace(3760, 3800, 10000)
    
    v1_m = v1_model.create_voigt(
        np.array([zabs]), lambda_rest, nclump=2, ntransition=2,
        FWHM='6.5', verbose=False
    )
    
    true_flux = v1_m.model_flux(theta_true, wave)
    noise = np.random.normal(0, 0.02, len(wave))
    observed_flux = true_flux + noise
    error = np.full_like(wave, 0.02)
    
    print(f"Created synthetic data: {len(wave)} wavelength points")
    
    # Set up bounds
    nguess = theta_true[:2]
    bguess = theta_true[2:4]
    vguess = theta_true[4:6]
    bounds, lb, ub = v1_mcmc.set_bounds(nguess, bguess, vguess)
    
    # 2. Fit with v1 MCMC
    print("\n2. Running v1 MCMC...")
    start = time.time()
    
    v1_fitter = v1_mcmc.vfit(
        v1_m.model_flux, theta_true, lb, ub, wave, observed_flux, error,
        no_of_Chain=20, no_of_steps=1000, perturbation=1e-6
    )
    v1_fitter.runmcmc(optimize=False, verbose=False)
    
    v1_time = time.time() - start
    print(f"v1 MCMC time: {v1_time:.3f} s")
    
    # 3. Set up v2 model
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    
    config = FitConfiguration()
    config.add_system(z=zabs, ion='MgII', transitions=lambda_rest, components=2)
    v2_m = VoigtModel(config, FWHM='6.5')
    
    # 4. Write v1-style functions for v2 model
    def lnprior_v2(theta):
        """v1-style prior for v2."""
        for index in range(0, len(lb)):
            if (lb[index] > theta[index]) or (ub[index] < theta[index]):
                return -np.inf
        return 0.0
    
    def lnlike_v2(theta):
        """v1-style likelihood for v2."""
        model_dat = v2_m.evaluate(theta, wave, validate_theta=False)
        inv_sigma2 = 1.0 / (error ** 2)
        return -0.5 * (np.sum((observed_flux - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    
    def lnprob_v2(theta):
        """v1-style probability for v2."""
        lp = lnprior_v2(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_v2(theta)
    
    # 5. Run v2 MCMC with v1-style setup
    print("\n5. Running v2 MCMC with v1-style functions...")
    start = time.time()
    
    # Initialize walkers same as v1
    nwalkers = 20
    ndim = len(theta_true)
    pos = theta_true + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # Run emcee directly
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v2)
    sampler.run_mcmc(pos, 1000, progress=True)
    
    v2_time = time.time() - start
    print(f"v2 MCMC time: {v2_time:.3f} s")
    
    # 6. Results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"v1 MCMC: {v1_time:.3f} s")
    print(f"v2 MCMC: {v2_time:.3f} s")
    print(f"Speedup: v1 is {v2_time/v1_time:.2f}x faster")
    
    if v2_time < v1_time:
        print("✅ v2 is faster!")
    else:
        print("❌ v1 is faster - v2 has overhead")

if __name__ == "__main__":
    simple_v1_v2_test()