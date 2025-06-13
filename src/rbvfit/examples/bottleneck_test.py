import numpy as np
import time

def bottleneck_test():
    """Test to isolate where v2 is slower than v1."""
    
    # Setup same as before
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    theta_true = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    
    # Test with different wavelength sizes
    wave_sizes = [1000, 3000, 5000, 10000]
    
    print("Bottleneck Analysis: Model Evaluation Scaling")
    print("=" * 60)
    
    for wave_size in wave_sizes:
        wave = np.linspace(3760, 3800, wave_size)
        
        print(f"\nWavelength array size: {wave_size}")
        print("-" * 40)
        
        # === V1 TIMING ===
        from rbvfit import model as v1_model
        v1_m = v1_model.create_voigt(
            np.array([zabs]), lambda_rest, nclump=2, ntransition=2, 
            FWHM='6.5', verbose=False
        )
        
        # Time 10 v1 evaluations
        start = time.time()
        for _ in range(10):
            flux_v1 = v1_m.model_flux(theta_true, wave)
        v1_time = time.time() - start
        
        # === V2 TIMING ===
        from rbvfit.core.fit_configuration import FitConfiguration
        from rbvfit.core.voigt_model import VoigtModel
        
        config = FitConfiguration()
        config.add_system(z=zabs, ion='MgII', transitions=lambda_rest, components=2)
        v2_m = VoigtModel(config, FWHM='6.5')
        
        # Time 10 v2 evaluations
        start = time.time()
        for _ in range(10):
            flux_v2 = v2_m.evaluate(theta_true, wave, validate_theta=False)
        v2_time = time.time() - start
        
        # === BREAKDOWN V2 TIMING ===
        # Time just the core loop without convolution
        start = time.time()
        for _ in range(10):
            flux_v2_raw = v2_m.evaluate(theta_true, wave, 
                                       validate_theta=False, return_unconvolved=True)
        v2_raw_time = time.time() - start
        
        # Results
        slowdown = v2_time / v1_time
        print(f"v1 (10 evals):     {v1_time:.4f} s")
        print(f"v2 (10 evals):     {v2_time:.4f} s") 
        print(f"v2 raw (no conv):  {v2_raw_time:.4f} s")
        print(f"Slowdown factor:   {slowdown:.2f}x")
        
        # Check if convolution is the issue
        conv_overhead = (v2_time - v2_raw_time) / v1_time
        print(f"Convolution overhead: {conv_overhead:.2f}x of v1 total")
        
        # Verify results are equivalent
        diff = np.max(np.abs(flux_v1 - flux_v2))
        print(f"Max difference:    {diff:.2e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("- If slowdown increases with array size → scaling issue in v2")
    print("- If convolution overhead is large → convolution difference") 
    print("- If raw time is still slow → issue in v2's core loop")

if __name__ == "__main__":
    bottleneck_test()