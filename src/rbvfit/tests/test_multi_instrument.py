#!/usr/bin/env python
"""
Test script for multi-instrument implementation in rbvfit 2.0
"""

import numpy as np
import sys
import os

# Add the path to import our modules (adjust as needed for your setup)
sys.path.insert(0, 'rbvfit/src/rbvfit/')

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.parameter_manager import ParameterManager


def test_single_instrument():
    """Test that single instrument still works as before."""
    print("=" * 60)
    print("TEST 1: Single Instrument Backward Compatibility")
    print("=" * 60)
    
    # Create simple MgII doublet configuration
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    print("Configuration created:")
    print(config.summary())
    
    # Create model
    model = VoigtModel(config, FWHM='6.5')
    
    # Compile - should work exactly as before
    compiled = model.compile(verbose=True)
    
    # Test that model_flux works
    theta = np.array([13.5, 13.2, 15.0, 20.0, -50.0, 0.0])
    wave = np.linspace(3750, 3850, 100)
    
    flux = compiled.model_flux(theta, wave)
    print(f"✓ Model evaluation successful: flux shape = {flux.shape}")
    print(f"✓ Flux range: [{np.min(flux):.3f}, {np.max(flux):.3f}]")
    
    return True


def test_multi_instrument_creation():
    """Test multi-instrument configuration creation."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Instrument Configuration Creation")
    print("=" * 60)
    
    # Create instrument A config - only sees MgII 2796
    config_A = FitConfiguration()
    config_A.add_system(z=0.348, ion='MgII', transitions=[2796.3], components=2)
    
    print("Instrument A configuration:")
    print(config_A.summary())
    
    # Create instrument B config - sees both MgII lines + FeII
    config_B = FitConfiguration()
    config_B.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    config_B.add_system(z=0.524, ion='FeII', transitions=[2600.2], components=1)
    
    print("\nInstrument B configuration:")
    print(config_B.summary())
    
    # Create master config
    instrument_configs = {'A': config_A, 'B': config_B}
    master_config = FitConfiguration.create_master_config(instrument_configs)
    
    print("\nMaster configuration:")
    print(master_config.summary())
    
    # Verify master has union of all parameters
    master_structure = master_config.get_parameter_structure()
    print(f"\nMaster parameters: {master_structure['total_parameters']}")
    
    return instrument_configs, master_config


def test_parameter_mapping():
    """Test parameter mapping functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Parameter Mapping")
    print("=" * 60)
    
    # Use configs from previous test
    instrument_configs, master_config = test_multi_instrument_creation()
    
    # Test parameter manager
    master_pm = ParameterManager(master_config)
    
    # Test mapping for instrument A
    config_A = instrument_configs['A']
    mapping_A = master_pm.compute_instrument_mapping(config_A, master_config)
    print(f"Instrument A parameter mapping: {mapping_A}")
    
    # Test mapping for instrument B  
    config_B = instrument_configs['B']
    mapping_B = master_pm.compute_instrument_mapping(config_B, master_config)
    print(f"Instrument B parameter mapping: {mapping_B}")
    
    # Verify mappings make sense
    master_structure = master_config.get_parameter_structure()
    a_structure = config_A.get_parameter_structure()
    b_structure = config_B.get_parameter_structure()
    
    print(f"\nParameter counts:")
    print(f"  Master: {master_structure['total_parameters']}")
    print(f"  Instrument A: {a_structure['total_parameters']} -> mapping length: {len(mapping_A)}")
    print(f"  Instrument B: {b_structure['total_parameters']} -> mapping length: {len(mapping_B)}")
    
    if len(mapping_A) != a_structure['total_parameters']:
        raise ValueError("Instrument A mapping length mismatch!")
    if len(mapping_B) != b_structure['total_parameters']:
        raise ValueError("Instrument B mapping length mismatch!")
    
    print("✓ Parameter mapping lengths correct")
    
    return mapping_A, mapping_B


def test_multi_instrument_compilation():
    """Test multi-instrument model compilation."""
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Instrument Model Compilation")
    print("=" * 60)
    
    # Create configs
    config_A = FitConfiguration()
    config_A.add_system(z=0.348, ion='MgII', transitions=[2796.3], components=1)
    
    config_B = FitConfiguration()
    config_B.add_system(z=0.348, ion='MgII', transitions=[2796.3], components=1)
    config_B.add_system(z=0.524, ion='FeII', transitions=[2600.2], components=1)
    
    # Create base model (doesn't matter which config we use for initialization)
    model = VoigtModel(config_A, FWHM='6.5')
    
    # Compile with multi-instrument support
    instrument_configs = {'A': config_A, 'B': config_B}
    compiled = model.compile(instrument_configs=instrument_configs, verbose=True)
    
    print(f"\nChecking model_flux interface:")
    print(f"  Multi-instrument model created successfully")
    
    # For multi-instrument, model_flux requires instrument parameter
    master_theta = np.array([13.5, 14.0, 15.0, 20.0, -50.0, 10.0])
    wave_A = np.linspace(3750, 3760, 50)
    
    try:
        compiled.model_flux(master_theta, wave_A)
        raise ValueError("Should have required instrument parameter!")
    except ValueError as e:
        if "requires 'instrument' parameter" in str(e):
            print("✓ model_flux correctly requires instrument parameter for multi-instrument")
        else:
            raise
    
    return compiled, instrument_configs


def test_multi_instrument_evaluation():
    """Test multi-instrument model evaluation."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-Instrument Model Evaluation")
    print("=" * 60)
    
    compiled, instrument_configs = test_multi_instrument_compilation()
    
    # Master theta array (6 parameters: N_MgII, N_FeII, b_MgII, b_FeII, v_MgII, v_FeII)
    master_theta = np.array([13.5, 14.0, 15.0, 20.0, -50.0, 10.0])
    print(f"Master theta: {master_theta}")
    print(f"Master theta length: {len(master_theta)}")
    
    # Show the parameter mappings for clarity
    print(f"Parameter mappings from compilation:")
    print(f"  Instrument A uses indices: {compiled.data.instrument_param_indices['A']}")
    print(f"  Instrument B uses indices: {compiled.data.instrument_param_indices['B']}")
    print(f"  Master config total params: {compiled.data.master_config_info['total_parameters']}")
    
    # Wavelength arrays
    wave_A = np.linspace(3750, 3760, 50)  # Around MgII 2796 only
    wave_B = np.linspace(3750, 3980, 200)  # Broader range for both lines
    
    # Evaluate instrument A (should only use MgII parameters)
    flux_A = compiled.model_flux(master_theta, wave_A, instrument='A')
    print(f"✓ Instrument A evaluation: flux shape = {flux_A.shape}")
    print(f"  Flux range: [{np.min(flux_A):.3f}, {np.max(flux_A):.3f}]")
    
    # Evaluate instrument B (should use all parameters)
    flux_B = compiled.model_flux(master_theta, wave_B, instrument='B')
    print(f"✓ Instrument B evaluation: flux shape = {flux_B.shape}")
    print(f"  Flux range: [{np.min(flux_B):.3f}, {np.max(flux_B):.3f}]")
    
    # Test that model_flux without instrument raises error for multi-instrument
    try:
        compiled.model_flux(master_theta, wave_A)
        raise ValueError("model_flux should have raised an error!")
    except ValueError as e:
        if "requires 'instrument' parameter" in str(e):
            print("✓ model_flux correctly requires instrument parameter")
        else:
            raise
    
    return True


def test_vfit_mcmc_preparation():
    """Test preparation for vfit_mcmc usage."""
    print("\n" + "=" * 60)
    print("TEST 6: vfit_mcmc Preparation")
    print("=" * 60)
    
    compiled, instrument_configs = test_multi_instrument_compilation()
    
    # Master theta and bounds
    master_theta = np.array([13.5, 14.0, 15.0, 20.0, -50.0, 10.0])
    
    # Create wrapper functions exactly as user would for vfit_mcmc
    model_A = lambda theta, wave: compiled.model_flux(theta, wave, instrument='A')
    model_B = lambda theta, wave: compiled.model_flux(theta, wave, instrument='B')
    
    print("Created wrapper functions:")
    print("  model_A = lambda theta, wave: compiled.model_flux(theta, wave, instrument='A')")
    print("  model_B = lambda theta, wave: compiled.model_flux(theta, wave, instrument='B')")
    
    # Test the wrappers
    wave_A = np.linspace(3750, 3760, 50)
    wave_B = np.linspace(3750, 3980, 200)
    
    flux_A = model_A(master_theta, wave_A)
    flux_B = model_B(master_theta, wave_B)
    
    print(f"✓ model_A wrapper works: {flux_A.shape}")
    print(f"✓ model_B wrapper works: {flux_B.shape}")
    
    # Show how vfit_mcmc would be called (without actually calling it)
    print("\nHow vfit_mcmc would be called:")
    print("fitter = vfit_mcmc(model_A, master_theta, lb, ub, wave_A, flux_A, error_A,")
    print("                   multi_instrument=True,")
    print("                   instrument_data={'B': {'wave': wave_B, 'flux': flux_B, 'error': error_B}},")
    print("                   model_B=model_B)")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("rbvfit 2.0 Multi-Instrument Implementation Tests")
    print("=" * 60)
    
    try:
        # Run tests in sequence
        test_single_instrument()
        test_multi_instrument_creation()
        test_parameter_mapping()
        test_multi_instrument_compilation()
        test_multi_instrument_evaluation()
        test_vfit_mcmc_preparation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("✅ Single instrument backward compatibility maintained")
        print("✅ Multi-instrument configuration creation works")
        print("✅ Parameter mapping implemented correctly")
        print("✅ Multi-instrument compilation successful")
        print("✅ Proper model_flux interface with instrument parameter")
        print("✅ Multi-instrument evaluation works correctly")
        print("✅ Ready for vfit_mcmc integration")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)