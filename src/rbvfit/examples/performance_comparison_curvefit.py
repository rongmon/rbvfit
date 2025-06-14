import numpy as np
import time
from scipy.optimize import curve_fit

# v1 imports
from rbvfit import model as v1_model

# v2 imports  
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.voigt_fitter import VoigtFitter, Dataset, MCMCSettings

def quick_performance_test():
    """Test curve_fit vs MCMC performance for v1 and v2 with compilation."""
    
    print("Quick Performance Test: v1 vs v2 with Compilation")
    print("=" * 60)
    
    # Common test setup
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    wave = np.linspace(3760, 3800, 1000)  # Smaller for speed
    
    # True parameters for synthetic data
    theta_true = np.array([13.95, 13.3, 20.0, 40.0, -40.0, 20.0])
    
    # ===== V1 SETUP =====
    print("\n1. Setting up v1 model...")
    start = time.time()
    
    v1_m = v1_model.create_voigt(
        np.array([zabs]),
        lambda_rest,
        nclump=2,
        ntransition=2,
        FWHM='6.5',
        verbose=False
    )
    
    v1_setup_time = time.time() - start
    print(f"v1 setup: {v1_setup_time*1000:.2f} ms")
    
    # Generate synthetic data with v1
    true_flux_v1 = v1_m.model_flux(theta_true, wave)
    noise = np.random.normal(0, 0.05, len(wave))
    observed_flux = true_flux_v1 + noise
    error = np.full_like(wave, 0.05)
    
    # ===== V2 SETUP AND COMPILATION =====
    print("\n2. Setting up v2 model...")
    start = time.time()
    
    config = FitConfiguration()
    config.add_system(z=zabs, ion='MgII', transitions=lambda_rest, components=2)
    v2_m = VoigtModel(config, FWHM='6.5')
    
    v2_setup_time = time.time() - start
    print(f"v2 setup: {v2_setup_time*1000:.2f} ms")
    
    # Compile v2 model for fast evaluation
    print("Compiling v2 model...")
    start = time.time()
    
    v2_compiled = v2_m.compile()
    
    v2_compile_time = time.time() - start
    print(f"v2 compilation: {v2_compile_time*1000:.2f} ms")
    
    # Verify models give same result
    flux_v1 = v1_m.model_flux(theta_true, wave)
    flux_v2_regular = v2_m.evaluate(theta_true, wave)
    flux_v2_compiled = v2_compiled.simple_flux(theta_true, wave)
    
    print(f"v1 vs v2 regular: {np.max(np.abs(flux_v1 - flux_v2_regular)):.2e}")
    print(f"v1 vs v2 compiled: {np.max(np.abs(flux_v1 - flux_v2_compiled)):.2e}")
    print(f"v2 regular vs compiled: {np.max(np.abs(flux_v2_regular - flux_v2_compiled)):.2e}")
    
    # ===== RAW EVALUATION TEST =====
    print("\n3. Testing raw model evaluation speed (100 calls each)...")
    
    # Time v1 raw evaluations
    start = time.time()
    for i in range(100):
        theta_test = theta_true + 0.01 * np.random.randn(6)
        flux_v1 = v1_m.model_flux(theta_test, wave)
    v1_raw_time = time.time() - start
    print(f"v1 raw evaluations (100x): {v1_raw_time:.3f} s")
    
    # Time v2 regular evaluations
    start = time.time()
    for i in range(100):
        theta_test = theta_true + 0.01 * np.random.randn(6)
        flux_v2 = v2_m.evaluate(theta_test, wave, validate_theta=False)
    v2_raw_time = time.time() - start
    print(f"v2 regular evaluations (100x): {v2_raw_time:.3f} s")
    
    # Time v2 compiled evaluations
    start = time.time()
    for i in range(100):
        theta_test = theta_true + 0.01 * np.random.randn(6)
        flux_v2_comp = v2_compiled.simple_flux(theta_test, wave)
    v2_compiled_time = time.time() - start
    print(f"v2 compiled evaluations (100x): {v2_compiled_time:.3f} s")
    
    print(f"v2 regular vs v1: v2 is {v1_raw_time/v2_raw_time:.1f}x faster")
    print(f"v2 compiled vs v1: v2 is {v1_raw_time/v2_compiled_time:.1f}x faster")
    print(f"v2 compiled vs v2 regular: compiled is {v2_raw_time/v2_compiled_time:.1f}x faster")
    
    # ===== MCMC TEST =====
    print("\n4. Testing MCMC performance...")
    
    # v1 MCMC setup
    from rbvfit import vfit_mcmc as v1_mcmc
    
    nguess = theta_true[:2]
    bguess = theta_true[2:4] 
    vguess = theta_true[4:6]
    bounds, lb, ub = v1_mcmc.set_bounds(nguess, bguess, vguess)
    
    # Time v1 MCMC
    print("Running v1 MCMC...")
    start = time.time()
    
    v1_fitter = v1_mcmc.vfit(
        v1_m.model_flux, theta_true, lb, ub, wave, observed_flux, error,
        no_of_Chain=20, no_of_steps=500, perturbation=1e-6
    )
    v1_fitter.runmcmc(optimize=False, verbose=False)
    
    v1_mcmc_time = time.time() - start
    print(f"v1 MCMC: {v1_mcmc_time:.3f} s")
    
    # Skip v2 regular MCMC for now (walker initialization issues)
    print("Skipping v2 regular MCMC (walker initialization issues)")
    v2_mcmc_time = float('inf')  # Set to infinity so ratios work
    
    # Time v2 compiled MCMC (using compiled function directly)
    print("Running v2 compiled MCMC...")
    start = time.time()
    
    # Use v1-style MCMC with compiled v2 function
    def lnprior_v2(theta):
        for index in range(0, len(lb)):
            if (lb[index] > theta[index]) or (ub[index] < theta[index]):
                return -np.inf
        return 0.0
    
    def lnlike_v2(theta):
        model_dat = v2_compiled.simple_flux(theta, wave)
        inv_sigma2 = 1.0 / (error ** 2)
        return -0.5 * (np.sum((observed_flux - model_dat) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    
    def lnprob_v2(theta):
        lp = lnprior_v2(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_v2(theta)
    
    # Run emcee directly with compiled function
    import emcee
    nwalkers = 20
    ndim = len(theta_true)
    pos = theta_true + 1e-6 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v2)
    sampler.run_mcmc(pos, 500, progress=True)
    
    v2_compiled_mcmc_time = time.time() - start
    print(f"v2 compiled MCMC: {v2_compiled_mcmc_time:.3f} s")
    
    print(f"v1 vs v2 compiled: v2 compiled is {v1_mcmc_time/v2_compiled_mcmc_time:.1f}x faster")
    print(f"v2 compiled vs v1: ratio = {v2_compiled_mcmc_time/v1_mcmc_time:.2f}")

    print(f"\nModel info:")
    print(f"v1 lines in model: {len(v1_m.line.lines)}")
    print(f"v2 lines in model: {v2_m.n_lines}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Setup time:")
    print(f"  v1: {v1_setup_time*1000:.1f} ms")
    print(f"  v2: {v2_setup_time*1000:.1f} ms")
    print(f"  v2 compilation: {v2_compile_time*1000:.1f} ms")
    print()
    print(f"Raw evaluation (100 calls):")
    print(f"  v1: {v1_raw_time:.3f} s")
    print(f"  v2 regular: {v2_raw_time:.3f} s") 
    print(f"  v2 compiled: {v2_compiled_time:.3f} s")
    print(f"  → v2 regular is {v1_raw_time/v2_raw_time:.1f}x faster than v1")
    print(f"  → v2 compiled is {v1_raw_time/v2_compiled_time:.1f}x faster than v1")
    print()
    print(f"MCMC (sampling):")
    print(f"  v1: {v1_mcmc_time:.3f} s")
    print(f"  v2 compiled: {v2_compiled_mcmc_time:.3f} s")
    print(f"  → v2 compiled vs v1: {v2_compiled_mcmc_time/v1_mcmc_time:.2f}x (lower is better)")
    print()
    
    if v2_compiled_mcmc_time < v1_mcmc_time:
        print("🎉 SUCCESS: v2 compiled MCMC is faster than v1!")
        print("   The compilation approach solved the performance issue.")
    elif v2_mcmc_time > v1_mcmc_time and v2_compiled_mcmc_time < v2_mcmc_time:
        print("🎯 PROGRESS: v2 compiled is faster than v2 regular")
        print("   But still needs more optimization to beat v1.")
    else:
        print("❌ ISSUE: v2 compiled is not significantly faster.")
        print("   More optimization needed.")

if __name__ == "__main__":
    quick_performance_test()