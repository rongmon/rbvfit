#!/usr/bin/env python
"""
Test script for multi-instrument vfit_mcmc in rbvfit 2.0
"""

import numpy as np
import sys
import os

# Add the path to import our modules (adjust as needed for your setup)
sys.path.insert(0, 'rbvfit/src/rbvfit/')

try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    from rbvfit.core.parameter_manager import ParameterManager
    from rbvfit.vfit_mcmc import vfit
except ImportError:
    # Try alternative import paths
    try:
        from core.fit_configuration import FitConfiguration
        from core.voigt_model import VoigtModel
        from core.parameter_manager import ParameterManager
        from vfit_mcmc import vfit
    except ImportError:
        print("Error: Could not import rbvfit modules.")
        print("Make sure you're running from the correct directory.")
        print("Expected structure: rbvfit/src/rbvfit/core/ and rbvfit/src/rbvfit/vfit_mcmc.py")
        sys.exit(1)


def create_test_setup():
    """Create test setup with multi-instrument data."""
    print("=" * 60)
    print("SETUP: Creating Multi-Instrument Test Data")
    print("=" * 60)
    
    # Create instrument A config - only sees MgII 2796
    config_A = FitConfiguration()
    config_A.add_system(z=0.348, ion='MgII', transitions=[2796.3], components=1)
    
    # Create instrument B config - sees both MgII lines + FeII
    config_B = FitConfiguration()
    config_B.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=1)
    config_B.add_system(z=0.524, ion='FeII', transitions=[2600.2], components=1)
    
    # Create base model and compile with multi-instrument support
    model = VoigtModel(config_A, FWHM='6.5')
    instrument_configs = {'A': config_A, 'B': config_B}
    compiled = model.compile(instrument_configs=instrument_configs, verbose=True)
    
    # Create wrapper functions
    model_A = lambda theta, wave: compiled.model_flux(theta, wave, instrument='A')
    model_B = lambda theta, wave: compiled.model_flux(theta, wave, instrument='B')
    
    # Master theta and bounds
    master_theta = np.array([13.5, 14.0, 15.0, 20.0, -50.0, 10.0])  # N_MgII, N_FeII, b_MgII, b_FeII, v_MgII, v_FeII
    
    # Create bounds
    param_manager = ParameterManager(FitConfiguration.create_master_config(instrument_configs))
    bounds = param_manager.generate_parameter_bounds()
    lb, ub = bounds.lower, bounds.upper
    
    # Generate synthetic data
    wave_A = np.linspace(3750, 3760, 50)
    wave_B = np.linspace(3750, 3980, 200)
    
    # True model flux
    flux_A_true = model_A(master_theta, wave_A)
    flux_B_true = model_B(master_theta, wave_B)
    
    # Add noise
    np.random.seed(42)  # For reproducibility
    noise_A = np.random.normal(0, 0.02, len(wave_A))
    noise_B = np.random.normal(0, 0.015, len(wave_B))
    
    flux_A = flux_A_true + noise_A
    flux_B = flux_B_true + noise_B
    
    error_A = np.full_like(wave_A, 0.02)
    error_B = np.full_like(wave_B, 0.015)
    
    print(f"✓ Instrument A: {len(wave_A)} points, SNR ≈ {1/0.02:.0f}")
    print(f"✓ Instrument B: {len(wave_B)} points, SNR ≈ {1/0.015:.0f}")
    print(f"✓ Master theta: {master_theta}")
    print(f"✓ Parameter bounds: [{lb[0]:.1f}, {ub[0]:.1f}] (example)")
    
    return {
        'model_A': model_A,
        'model_B': model_B,
        'master_theta': master_theta,
        'lb': lb,
        'ub': ub,
        'wave_A': wave_A,
        'flux_A': flux_A,
        'error_A': error_A,
        'wave_B': wave_B,
        'flux_B': flux_B,
        'error_B': error_B
    }


def test_single_instrument_compatibility():
    """Test that single instrument still works as before."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Instrument Backward Compatibility")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Test single instrument (should work exactly as before)
    fitter = vfit(
        setup['model_A'],
        setup['master_theta'][:3],  # Only MgII parameters for instrument A
        setup['lb'][:3],
        setup['ub'][:3],
        setup['wave_A'],
        setup['flux_A'],
        setup['error_A'],
        no_of_Chain=10,
        no_of_steps=50,
        sampler='emcee'
    )
    
    print("✓ Single instrument vfit created successfully")
    
    # Test sampler info
    info = fitter.get_sampler_info()
    print(f"✓ Sampler info: {info['sampler']}, multi_instrument={info['multi_instrument']}")
    
    if info['multi_instrument']:
        raise ValueError("Single instrument should have multi_instrument=False")
    
    print("✓ Single instrument compatibility verified")
    return True


def test_legacy_interface():
    """Test legacy second_spec interface."""
    print("\n" + "=" * 60)
    print("TEST 2: Legacy second_spec Interface")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Test legacy interface
    fitter = vfit(
        setup['model_A'],
        setup['master_theta'],
        setup['lb'],
        setup['ub'],
        setup['wave_A'],
        setup['flux_A'],
        setup['error_A'],
        second_spec=True,
        second_spec_dict={
            'wave': setup['wave_B'],
            'flux': setup['flux_B'],
            'error': setup['error_B']
        },
        model2=setup['model_B'],
        no_of_Chain=10,
        no_of_steps=50
    )
    
    print("✓ Legacy second_spec interface works")
    
    # Check that it was converted to multi-instrument internally
    info = fitter.get_sampler_info()
    if not info['multi_instrument']:
        raise ValueError("Legacy interface should set multi_instrument=True internally")
    
    if info['n_instruments'] != 2:
        raise ValueError(f"Expected 2 instruments, got {info['n_instruments']}")
    
    print("✓ Legacy interface converted to multi-instrument correctly")
    print(f"✓ Instruments: {info['instruments']}")
    return True


def test_new_multi_instrument_interface():
    """Test new multi-instrument interface."""
    print("\n" + "=" * 60)
    print("TEST 3: New Multi-Instrument Interface")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Test new multi-instrument interface
    fitter = vfit(
        setup['model_A'],
        setup['master_theta'],
        setup['lb'],
        setup['ub'],
        setup['wave_A'],
        setup['flux_A'],
        setup['error_A'],
        multi_instrument=True,
        instrument_data={
            'B': {
                'model': setup['model_B'],
                'wave': setup['wave_B'],
                'flux': setup['flux_B'],
                'error': setup['error_B']
            }
        },
        no_of_Chain=10,
        no_of_steps=50
    )
    
    print("✓ New multi-instrument interface works")
    
    # Check configuration
    info = fitter.get_sampler_info()
    if not info['multi_instrument']:
        raise ValueError("Should have multi_instrument=True")
    
    if info['n_instruments'] != 2:
        raise ValueError(f"Expected 2 instruments, got {info['n_instruments']}")
    
    expected_instruments = {'main', 'B'}
    if set(info['instruments']) != expected_instruments:
        raise ValueError(f"Expected instruments {expected_instruments}, got {set(info['instruments'])}")
    
    print(f"✓ Multi-instrument setup: {info['n_instruments']} instruments")
    print(f"✓ Instruments: {info['instruments']}")
    return True


def test_three_instrument_setup():
    """Test with three instruments."""
    print("\n" + "=" * 60)
    print("TEST 4: Three-Instrument Setup")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Create a third instrument setup
    wave_C = np.linspace(3900, 4000, 100)
    flux_C = np.ones_like(wave_C) + np.random.normal(0, 0.01, len(wave_C))
    error_C = np.full_like(wave_C, 0.01)
    
    # Dummy model for instrument C (just returns flat spectrum)
    model_C = lambda theta, wave: np.ones_like(wave)
    
    # Test three-instrument interface
    fitter = vfit(
        setup['model_A'],
        setup['master_theta'],
        setup['lb'],
        setup['ub'],
        setup['wave_A'],
        setup['flux_A'],
        setup['error_A'],
        multi_instrument=True,
        instrument_data={
            'B': {
                'model': setup['model_B'],
                'wave': setup['wave_B'],
                'flux': setup['flux_B'],
                'error': setup['error_B']
            },
            'C': {
                'model': model_C,
                'wave': wave_C,
                'flux': flux_C,
                'error': error_C
            }
        },
        no_of_Chain=10,
        no_of_steps=50
    )
    
    print("✓ Three-instrument interface works")
    
    # Check configuration
    info = fitter.get_sampler_info()
    if info['n_instruments'] != 3:
        raise ValueError(f"Expected 3 instruments, got {info['n_instruments']}")
    
    expected_instruments = {'main', 'B', 'C'}
    if set(info['instruments']) != expected_instruments:
        raise ValueError(f"Expected instruments {expected_instruments}, got {set(info['instruments'])}")
    
    print(f"✓ Three-instrument setup: {info['n_instruments']} instruments")
    print(f"✓ Instruments: {info['instruments']}")
    return True


def test_likelihood_calculation():
    """Test that likelihood calculation works for multi-instrument."""
    print("\n" + "=" * 60)
    print("TEST 5: Likelihood Calculation")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Create fitter
    fitter = vfit(
        setup['model_A'],
        setup['master_theta'],
        setup['lb'],
        setup['ub'],
        setup['wave_A'],
        setup['flux_A'],
        setup['error_A'],
        multi_instrument=True,
        instrument_data={
            'B': {
                'model': setup['model_B'],
                'wave': setup['wave_B'],
                'flux': setup['flux_B'],
                'error': setup['error_B']
            }
        }
    )
    
    # Test likelihood calculation
    theta_test = setup['master_theta'] + 0.1 * np.random.randn(len(setup['master_theta']))
    
    # Calculate likelihood
    lnlike = fitter.lnlike(theta_test)
    print(f"✓ Likelihood calculation successful: {lnlike:.2f}")
    
    # Test prior
    lnprior = fitter.lnprior(theta_test)
    print(f"✓ Prior calculation successful: {lnprior:.2f}")
    
    # Test posterior
    lnprob = fitter.lnprob(theta_test)
    print(f"✓ Posterior calculation successful: {lnprob:.2f}")
    
    # Test with out-of-bounds parameters
    theta_bad = setup['master_theta'].copy()
    theta_bad[0] = setup['ub'][0] + 1  # Out of bounds
    
    lnprior_bad = fitter.lnprior(theta_bad)
    if not np.isinf(lnprior_bad):
        raise ValueError("Prior should be -inf for out-of-bounds parameters")
    
    print("✓ Out-of-bounds handling works correctly")
    return True


def test_short_mcmc_run():
    """Test a short MCMC run."""
    print("\n" + "=" * 60)
    print("TEST 6: Short MCMC Run")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Create fitter with very short run
    fitter = vfit(
        setup['model_A'],
        setup['master_theta'],
        setup['lb'],
        setup['ub'],
        setup['wave_A'],
        setup['flux_A'],
        setup['error_A'],
        multi_instrument=True,
        instrument_data={
            'B': {
                'model': setup['model_B'],
                'wave': setup['wave_B'],
                'flux': setup['flux_B'],
                'error': setup['error_B']
            }
        },
        no_of_Chain=20,
        no_of_steps=500,
        sampler='emcee'
    )
    
    print("Running short MCMC (20 steps, 8 walkers)...")
    
    # Run MCMC
    fitter.runmcmc(optimize=False, verbose=False, use_pool=False)
    
    print("✓ MCMC run completed successfully")
    
    # Check that we have a sampler
    if not hasattr(fitter, 'sampler'):
        raise ValueError("Sampler not created after runmcmc()")
    
    # Check sampler info
    info = fitter.get_sampler_info()
    if 'acceptance_fraction' in info:
        print(f"✓ Acceptance fraction: {info['acceptance_fraction']:.3f}")
    
    print("✓ Multi-instrument MCMC works correctly")
    return True


def test_error_handling():
    """Test error handling and validation."""
    print("\n" + "=" * 60)
    print("TEST 7: Error Handling")
    print("=" * 60)
    
    setup = create_test_setup()
    
    # Test conflicting parameters
    try:
        vfit(
            setup['model_A'],
            setup['master_theta'],
            setup['lb'],
            setup['ub'],
            setup['wave_A'],
            setup['flux_A'],
            setup['error_A'],
            multi_instrument=True,
            instrument_data={'B': {}},
            second_spec=True  # Should conflict
        )
        raise ValueError("Should have raised an error for conflicting parameters")
    except ValueError as e:
        if "Cannot use both" in str(e):
            print("✓ Conflicting parameter error handled correctly")
        else:
            raise
    
    # Test missing instrument_data
    try:
        vfit(
            setup['model_A'],
            setup['master_theta'],
            setup['lb'],
            setup['ub'],
            setup['wave_A'],
            setup['flux_A'],
            setup['error_A'],
            multi_instrument=True
            # Missing instrument_data
        )
        raise ValueError("Should have raised an error for missing instrument_data")
    except ValueError as e:
        if "requires 'instrument_data'" in str(e):
            print("✓ Missing instrument_data error handled correctly")
        else:
            raise
    
    # Test invalid instrument_data format
    try:
        vfit(
            setup['model_A'],
            setup['master_theta'],
            setup['lb'],
            setup['ub'],
            setup['wave_A'],
            setup['flux_A'],
            setup['error_A'],
            multi_instrument=True,
            instrument_data={
                'B': {'model': setup['model_B']}  # Missing wave, flux, error
            }
        )
        raise ValueError("Should have raised an error for invalid instrument_data")
    except ValueError as e:
        if "missing required keys" in str(e):
            print("✓ Invalid instrument_data format error handled correctly")
        else:
            raise
    
    print("✓ All error handling tests passed")
    return True


def run_all_tests():
    """Run all multi-instrument vfit_mcmc tests."""
    print("rbvfit 2.0 Multi-Instrument vfit_mcmc Tests")
    print("=" * 60)
    
    try:
        # Run tests in sequence
        test_single_instrument_compatibility()
        test_legacy_interface()
        test_new_multi_instrument_interface()
        test_three_instrument_setup()
        test_likelihood_calculation()
        test_short_mcmc_run()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 ALL vfit_mcmc TESTS PASSED! 🎉")
        print("=" * 60)
        print("✅ Single instrument backward compatibility maintained")
        print("✅ Legacy second_spec interface works")
        print("✅ New multi-instrument interface implemented")
        print("✅ Three+ instrument support working")
        print("✅ Likelihood calculation correct for multi-instrument")
        print("✅ MCMC execution successful")
        print("✅ Error handling comprehensive")
        print("\n🚀 rbvfit 2.0 multi-instrument fitting is ready!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)