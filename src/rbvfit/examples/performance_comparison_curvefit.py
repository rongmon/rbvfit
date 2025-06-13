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
    """Test curve_fit vs MCMC performance for v1 and v2."""
    
    print("Quick Performance Test: v1 vs v2")
    print("=" * 50)
    
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
    
    # ===== V2 SETUP =====
    print("\n2. Setting up v2 model...")
    start = time.time()
    
    config = FitConfiguration()
    config.add_system(z=zabs, ion='MgII', transitions=lambda_rest, components=2)
    v2_m = VoigtModel(config, FWHM='6.5')
    
    v2_setup_time = time.time() - start
    print(f"v2 setup: {v2_setup_time*1000:.2f} ms")
    
    # Verify models give same result
    flux_v1 = v1_m.model_flux(theta_true, wave)
    flux_v2 = v2_m.evaluate(theta_true, wave)
    print(f"Model difference: {np.max(np.abs(flux_v1 - flux_v2)):.2e}")
    
    # ===== CURVE_FIT TEST =====
    print("\n3. Testing curve_fit performance...")
    
    # v1 curve_fit wrapper
    def v1_func(wave, *params):
        theta = np.array(params)
        return v1_m.model_flux(theta, wave)
    
    # v2 curve_fit wrapper  
    def v2_func(wave, *params):
        theta = np.array(params)
        return v2_m.evaluate(theta, wave, validate_theta=False)
    
    # Initial guess
    p0 = theta_true + 0.1 * np.random.randn(6)
    
    # Test raw model evaluation speed instead of curve_fit
    print("Testing raw model evaluation speed (100 calls each)...")
    
    # Time v1 raw evaluations
    start = time.time()
    for i in range(100):
        theta_test = theta_true + 0.01 * np.random.randn(6)
        flux_v1 = v1_m.model_flux(theta_test, wave)
    v1_raw_time = time.time() - start
    print(f"v1 raw evaluations (100x): {v1_raw_time:.3f} s")
    
    # Time v2 raw evaluations
    start = time.time()
    for i in range(100):
        theta_test = theta_true + 0.01 * np.random.randn(6)
        flux_v2 = v2_m.evaluate(theta_test, wave, validate_theta=False)
    v2_raw_time = time.time() - start
    print(f"v2 raw evaluations (100x): {v2_raw_time:.3f} s")
    
    print(f"Raw model speedup: v2 is {v1_raw_time/v2_raw_time:.1f}x faster")
    
    # ===== MCMC TEST =====
    print("\n4. Testing MCMC performance...")
    
    # v1 MCMC setup
    from rbvfit import vfit_mcmc as v1_mcmc
    
    nguess = theta_true[:2]
    bguess = theta_true[2:4] 
    vguess = theta_true[4:6]
    bounds, lb, ub = v1_mcmc.set_bounds(nguess, bguess, vguess)
    
    # Time v1 MCMC (short run)
    print("Running v1 MCMC...")
    start = time.time()
    
    v1_fitter = v1_mcmc.vfit(
        v1_m.model_flux, theta_true, lb, ub, wave, observed_flux, error,
        no_of_Chain=20, no_of_steps=5000, perturbation=1e-6
    )
    v1_fitter.runmcmc(optimize=False, verbose=False)
    
    v1_mcmc_time = time.time() - start
    print(f"v1 MCMC: {v1_mcmc_time:.3f} s")
    
    # Time v2 MCMC (short run)
    print("Running v2 MCMC...")
    start = time.time()
    
    dataset = Dataset(wave, observed_flux, error, name="test")
    mcmc_settings = MCMCSettings(n_walkers=20, n_steps=5000, progress=False)
    v2_fitter = VoigtFitter(v2_m, dataset, mcmc_settings)
    
    result = v2_fitter.fit(theta_true, optimize_first=False)
    
    v2_mcmc_time = time.time() - start
    print(f"v2 MCMC: {v2_mcmc_time:.3f} s")
    
    print(f"MCMC speedup: v1 is {v2_mcmc_time/v1_mcmc_time:.1f}x faster")

    print(f"v1 lines in model: {len(v1_m.line.lines)}")
    print(f"v2 lines in model: {v2_m.n_lines}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Setup time:")
    print(f"  v1: {v1_setup_time*1000:.1f} ms")
    print(f"  v2: {v2_setup_time*1000:.1f} ms")
    print()
    print(f"Raw evaluation (100 calls):")
    print(f"  v1: {v1_raw_time:.3f} s")
    print(f"  v2: {v2_raw_time:.3f} s") 
    print(f"  → v2 is {v1_raw_time/v2_raw_time:.1f}x faster")
    print()
    print(f"MCMC (sampling):")
    print(f"  v1: {v1_mcmc_time:.3f} s")
    print(f"  v2: {v2_mcmc_time:.3f} s")
    print(f"  → v1 is {v2_mcmc_time/v1_mcmc_time:.1f}x faster")
    print()
    
    if v1_raw_time/v2_raw_time > 1.1 and v2_mcmc_time/v1_mcmc_time > 1.1:
        print("🔍 CONCLUSION: v2 model is faster, but v2 MCMC has overhead!")
        print("   The bottleneck is in the MCMC infrastructure, not the model.")
    elif v1_raw_time/v2_raw_time > 1.1:
        print("✅ CONCLUSION: v2 optimizations working - model is faster!")
    else:
        print("❌ CONCLUSION: v2 model optimizations not working as expected.")

if __name__ == "__main__":
    quick_performance_test()