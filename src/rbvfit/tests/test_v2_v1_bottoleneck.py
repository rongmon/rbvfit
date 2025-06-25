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
    nguess = [13.8, 13.3,13.2,13.7]
    bguess = [20.0, 30.0,15,8]
    vguess = [-40.0,0,20, 40.0]
    

    theta_true  = np.concatenate([nguess, bguess, vguess])
    n_component=len(nguess)
    

    wave = np.linspace(3760, 3800, 500)
    
    v1_m = v1_model.create_voigt(
        np.array([zabs]), lambda_rest, nclump=n_component, ntransition=2,
        FWHM='6.5', verbose=False
    )
    
    true_flux = v1_m.model_flux(theta_true, wave)
    noise = np.random.normal(0, 0.02, len(wave))
    observed_flux = true_flux + noise
    error = np.full_like(wave, 0.02)
    
    print(f"Created synthetic data: {len(wave)} wavelength points")
    
    # Set up bounds
    bounds, lb, ub = v1_mcmc.set_bounds(nguess, bguess, vguess)
    
    # 2. Fit with v1 MCMC
    print("\n2. Running v1 MCMC...")
    start = time.time()
    
    v1_fitter = v1_mcmc.vfit(
        v1_m.model_flux, theta_true, lb, ub, wave, observed_flux, error,
        no_of_Chain=50, no_of_steps=500, perturbation=1e-6
    )
    v1_fitter.runmcmc(optimize=True, verbose=False)
    
    v1_time = time.time() - start
    print(f"v1 MCMC time: {v1_time:.3f} s")
    
    # 3. Set up v2 model
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    
    config = FitConfiguration()
    config.add_system(z=zabs, ion='MgII', transitions=lambda_rest, components=n_component)
    v2_m = VoigtModel(config, FWHM='6.5')
    v2_m_compile=v2_m.compile()

    
    
    # 5. Run v2 MCMC
    print("\n5. Running v2 MCMC ...")
    start = time.time()
    
    
    v2_fitter = v1_mcmc.vfit(
        v2_m_compile.model_flux, theta_true, lb, ub, wave, observed_flux, error,
        no_of_Chain=50, no_of_steps=500, perturbation=1e-6
    )
    v2_fitter.runmcmc(optimize=True, verbose=False)
    
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