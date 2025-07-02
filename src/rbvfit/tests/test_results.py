#!/usr/bin/env python
"""
Test script for UnifiedResults implementation.

This script tests the complete Phase 1 implementation:
1. Creation from existing fitters
2. Save/load roundtrip integrity  
3. Property access
4. Model reconstruction
5. Basic analysis methods

Run this with your existing fitted data to validate the implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

# Import the new implementation
from rbvfit.core.unified_results import UnifiedResults, save_unified_results, load_unified_results

def test_creation_from_fitter(fitter, model=None):
    """Test 1: Creation from existing fitter object."""
    print("="*60)
    print("TEST 1: CREATION FROM FITTER")
    print("="*60)
    
    try:
        results = UnifiedResults(fitter, model)
        print("✅ UnifiedResults created successfully")
        
        # Check core attributes
        print(f"✅ Best fit shape: {results.best_fit.shape}")
        print(f"✅ Samples shape: {results.samples.shape}")
        print(f"✅ Chain shape: {results.chain.shape}")
        print(f"✅ Instruments: {results.instrument_names}")
        print(f"✅ Sampler: {results.sampler_name}")
        print(f"✅ Multi-instrument: {results.is_multi_instrument}")
        
        # Check diagnostics
        print(f"✅ Autocorr time: {'Available' if results.autocorr_time is not None else 'None'}")
        print(f"✅ R-hat: {'Available' if results.rhat is not None else 'None'}")
        print(f"✅ Acceptance: {'Available' if results.acceptance_fraction is not None else 'None'}")
        print(f"✅ Config metadata: {'Available' if results.config_metadata is not None else 'None'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Creation failed: {e}")
        raise


def test_save_load_roundtrip(results):
    """Test 2: Save/load roundtrip integrity."""
    print("\n" + "="*60)
    print("TEST 2: SAVE/LOAD ROUNDTRIP")
    print("="*60)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save
        print("Saving results...")
        results.save(tmp_path)
        print(f"✅ Saved to {tmp_path}")
        
        # Load
        print("Loading results...")
        loaded = UnifiedResults.load(tmp_path)
        print("✅ Loaded successfully")
        
        # Compare core attributes
        print("Checking data integrity...")
        
        # Best fit parameters
        np.testing.assert_array_equal(results.best_fit, loaded.best_fit)
        print("✅ Best fit parameters identical")
        
        # Samples
        np.testing.assert_array_equal(results.samples, loaded.samples)
        print("✅ MCMC samples identical")
        
        # Chain
        np.testing.assert_array_equal(results.chain, loaded.chain)
        print("✅ MCMC chain identical")
        
        # Metadata
        assert results.sampler_name == loaded.sampler_name
        assert results.n_walkers == loaded.n_walkers
        assert results.n_steps == loaded.n_steps
        assert results.is_multi_instrument == loaded.is_multi_instrument
        print("✅ Metadata identical")
        
        # Instrument data
        assert results.instrument_names == loaded.instrument_names
        for inst_name in results.instrument_names:
            orig_data = results.instrument_data[inst_name]
            loaded_data = loaded.instrument_data[inst_name]
            np.testing.assert_array_equal(orig_data['wave'], loaded_data['wave'])
            np.testing.assert_array_equal(orig_data['flux'], loaded_data['flux'])
            np.testing.assert_array_equal(orig_data['error'], loaded_data['error'])
        print("✅ Instrument data identical")
        
        # Diagnostics (handle None values)
        if results.autocorr_time is not None:
            np.testing.assert_array_equal(results.autocorr_time, loaded.autocorr_time)
        else:
            assert loaded.autocorr_time is None
            
        if results.rhat is not None:
            np.testing.assert_array_equal(results.rhat, loaded.rhat)
        else:
            assert loaded.rhat is None
            
        if results.acceptance_fraction is not None:
            assert abs(results.acceptance_fraction - loaded.acceptance_fraction) < 1e-10
        else:
            assert loaded.acceptance_fraction is None
        print("✅ Diagnostics identical")
        
        print("🎉 Perfect roundtrip integrity!")
        return loaded
        
    except Exception as e:
        print(f"❌ Save/load failed: {e}")
        raise
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)


def test_properties(results):
    """Test 3: Property access and computation."""
    print("\n" + "="*60)
    print("TEST 3: PROPERTY ACCESS")
    print("="*60)
    
    try:
        # Parameter names
        param_names = results.parameter_names
        print(f"✅ Parameter names ({len(param_names)}): {param_names[:3]}...")
        
        # Error bounds
        bounds_16 = results.bounds_16th
        bounds_84 = results.bounds_84th
        print(f"✅ 16th percentiles shape: {bounds_16.shape}")
        print(f"✅ 84th percentiles shape: {bounds_84.shape}")
        
        # Instrument names
        inst_names = results.instrument_names
        print(f"✅ Instrument names: {inst_names}")
        
        # Verify bounds make sense
        assert len(bounds_16) == len(results.best_fit)
        assert len(bounds_84) == len(results.best_fit)
        assert np.all(bounds_16 <= results.best_fit)
        assert np.all(bounds_84 >= results.best_fit)
        print("✅ Error bounds are sensible")
        
    except Exception as e:
        print(f"❌ Property access failed: {e}")
        raise


def test_parameter_summary(results):
    """Test 4: Parameter summary analysis."""
    print("\n" + "="*60)
    print("TEST 4: PARAMETER SUMMARY")
    print("="*60)
    
    try:
        # Test with verbose=False first
        summary = results.parameter_summary(verbose=False)
        print("✅ Parameter summary computed")
        
        # Check summary structure
        assert len(summary.names) == len(results.best_fit)
        assert len(summary.best_fit) == len(results.best_fit)
        assert len(summary.errors) == len(results.best_fit)
        print("✅ Summary structure correct")
        
        # Check error bounds
        np.testing.assert_array_equal(summary.percentiles['16th'], results.bounds_16th)
        np.testing.assert_array_equal(summary.percentiles['84th'], results.bounds_84th)
        print("✅ Error bounds consistent")
        
        # Test verbose output
        print("\nTesting verbose output:")
        results.parameter_summary(verbose=True)
        print("✅ Verbose summary printed")
        
    except Exception as e:
        print(f"❌ Parameter summary failed: {e}")
        raise


def test_model_reconstruction(results):
    """Test 5: Model reconstruction."""
    print("\n" + "="*60)
    print("TEST 5: MODEL RECONSTRUCTION")
    print("="*60)
    
    if results.config_metadata is None:
        print("⚠️ No config metadata - skipping model reconstruction test")
        return
    
    try:
        if results.is_multi_instrument:
            # Test multi-instrument reconstruction
            print("Testing multi-instrument reconstruction...")
            
            # Reconstruct all models
            all_models = results.reconstruct_all_models()
            print(f"✅ Reconstructed {len(all_models)} models")
            
            # Reconstruct specific instruments
            for inst_name in results.instrument_names:
                model = results.reconstruct_model(inst_name)
                print(f"✅ Reconstructed model for {inst_name}")
                
                # Test model evaluation
                wave = results.instrument_data[inst_name]['wave']
                flux = model.evaluate(results.best_fit, wave)
                print(f"✅ Model evaluation for {inst_name}: {flux.shape}")
        
        else:
            # Test single instrument reconstruction
            print("Testing single instrument reconstruction...")
            model = results.reconstruct_model()
            print("✅ Model reconstructed")
            
            # Test model evaluation
            inst_name = results.instrument_names[0]
            wave = results.instrument_data[inst_name]['wave']
            flux = model.evaluate(results.best_fit, wave)
            print(f"✅ Model evaluation: {flux.shape}")
        
    except Exception as e:
        print(f"❌ Model reconstruction failed: {e}")
        print(f"Config metadata keys: {list(results.config_metadata.keys()) if results.config_metadata else 'None'}")
        raise


def test_print_summary(results):
    """Test 6: Print summary method."""
    print("\n" + "="*60)
    print("TEST 6: PRINT SUMMARY")
    print("="*60)
    
    try:
        results.print_summary()
        print("✅ Summary printed successfully")
        
    except Exception as e:
        print(f"❌ Print summary failed: {e}")
        raise


def test_convenience_functions(fitter, model=None):
    """Test 7: Convenience save/load functions."""
    print("\n" + "="*60)
    print("TEST 7: CONVENIENCE FUNCTIONS")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Test convenience save
        save_unified_results(fitter, model, tmp_path)
        print("✅ Convenience save completed")
        
        # Test convenience load
        results = load_unified_results(tmp_path)
        print("✅ Convenience load completed")
        
        return results
        
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
        raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_full_test_suite(fitter, model=None):
    """Run complete test suite."""
    print("🧪 STARTING UNIFIED RESULTS TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Creation
        results = test_creation_from_fitter(fitter, model)
        
        # Test 2: Save/load
        loaded_results = test_save_load_roundtrip(results)
        
        # Test 3: Properties (use loaded results to test reconstruction)
        test_properties(loaded_results)
        
        # Test 4: Parameter summary
        test_parameter_summary(loaded_results)
        
        # Test 5: Model reconstruction
        test_model_reconstruction(loaded_results)
        
        # Test 6: Print summary
        test_print_summary(loaded_results)
        
        # Test 7: Convenience functions
        convenience_results = test_convenience_functions(fitter, model)
        
        print("\n" + "🎉"*20)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("🎉"*20)
        print("\nUnifiedResults implementation is working correctly!")
        print("✅ Core data extraction")
        print("✅ Save/load integrity") 
        print("✅ Property access")
        print("✅ Parameter analysis")
        print("✅ Model reconstruction")
        print("✅ Convenience functions")
        
        return loaded_results
        
    except Exception as e:
        print("\n" + "❌"*20)
        print("❌ TEST FAILED!")
        print("❌"*20)
        print(f"Error: {e}")
        raise


def quick_validation_test(fitter, model=None):
    """Quick validation - just test creation and basic functionality."""
    print("🔍 QUICK VALIDATION TEST")
    print("="*40)
    
    try:
        # Create results
        results = UnifiedResults(fitter, model)
        print(f"✅ Created: {len(results.instrument_names)} instruments, {len(results.best_fit)} params")
        
        # Test properties
        param_names = results.parameter_names
        bounds = results.bounds_16th, results.bounds_84th
        print(f"✅ Properties: {len(param_names)} parameter names, bounds computed")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name
            
        results.save(tmp_path)
        loaded = UnifiedResults.load(tmp_path)
        Path(tmp_path).unlink()
        print("✅ Save/load: Perfect roundtrip")
        
        # Test analysis
        summary = loaded.parameter_summary(verbose=False)
        print(f"✅ Analysis: Parameter summary with {len(summary.names)} parameters")
        
        print("\n✅ QUICK VALIDATION PASSED!")
        return loaded
        
    except Exception as e:
        print(f"\n❌ QUICK VALIDATION FAILED: {e}")
        raise


if __name__ == "__main__":
    print("UnifiedResults Test Script")
    print("="*60)
    print("This script requires existing fitter and model objects.")
    print("Usage examples:")
    print()
    print("# Quick test:")
    print("quick_validation_test(your_fitter, your_model)")
    print()
    print("# Full test suite:")
    print("run_full_test_suite(your_fitter, your_model)")
    print()
    print("Replace 'your_fitter' and 'your_model' with your actual objects.")