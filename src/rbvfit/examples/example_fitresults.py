#!/usr/bin/env python
"""
Example: Using the Simplified Enhanced FitResults Class in rbvfit 2.0

This example demonstrates how to use the simplified ion-aware FitResults class
for analyzing MCMC fitting results with automatic parameter organization,
physical interpretation, and publication-ready outputs.
"""

import numpy as np
import matplotlib.pyplot as plt

# rbvfit 2.0 imports
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.voigt_fitter import Dataset
from rbvfit import vfit_mcmc as mc

def run_example_fit():
    """Run an example multi-ion fit and demonstrate simplified FitResults usage."""
    
    print("=" * 60)
    print("SIMPLIFIED ENHANCED FitResults EXAMPLE")
    print("=" * 60)
    
    # ========================================================================
    # STEP 1: Set up a multi-ion system (MgII + FeII)
    # ========================================================================
    print("Setting up multi-ion absorption system...")
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    config.add_system(z=0.348, ion='FeII', transitions=[2600.2], components=1)
    
    print("System configuration:")
    print("  MgII doublet at z=0.348: 2 components")
    print("  FeII 2600 at z=0.348: 1 component")
    print("  Total: 9 parameters (3 ions × 3 params each)")
    
    # ========================================================================
    # STEP 2: Generate synthetic data and run MCMC fit
    # ========================================================================
    print("\nGenerating synthetic data and running MCMC...")
    
    # Create model and generate synthetic spectrum
    model = VoigtModel(config, FWHM='6.5')
    compiled_model=model.compile()
    
    # True parameters: [N1_MgII, N2_MgII, N1_FeII, b1_MgII, b2_MgII, b1_FeII, v1_MgII, v2_MgII, v1_FeII]
    true_theta = np.array([13.5, 13.2, 14.0,  # Column densities
                          15.0, 25.0, 20.0,  # Doppler parameters
                          -50.0, 20.0, 0.0]) # Velocities
    
    # Wavelength grid and synthetic data
    wave = np.linspace(3400, 3600, 1000)
    true_flux = compiled_model.model_flux(true_theta, wave)
    noise = np.random.normal(0, 0.02, len(wave))
    observed_flux = true_flux + noise
    error = np.full_like(wave, 0.02)
    
    # Set up MCMC parameters
    bounds, lb, ub = mc.set_bounds([13.5, 13.2, 14.0], [15.0, 25.0, 20.0], [-50.0, 20.0, 0.0])
    
    # Run MCMC fitting (shortened for example)
    fitter = mc.vfit(
        compiled_model.model_flux,
        true_theta, lb, ub,
        wave, observed_flux, error,
        no_of_Chain=30,
        no_of_steps=1000
    )
    fitter.runmcmc(optimize=True)
    
    print("✓ MCMC fitting completed")
    
    # ========================================================================
    # STEP 3: Create simplified FitResults object
    # ========================================================================
    print("\nCreating simplified FitResults object...")
    
    # SIMPLIFIED USAGE - Just pass fitter and model!
    from rbvfit.core.fit_results import FitResults
    
    results = FitResults(fitter, model)
    
    print("✓ FitResults object created with automatic:")
    print("  - Parameter manager from model.config")
    print("  - Dataset extraction from fitter")
    print("  - MCMC settings from fitter attributes")
    print("  - Bounds from fitter.lb, fitter.ub")
    print("  - Ion-aware analysis and organization")
    
    return results, fitter, model


def demonstrate_dataset_override(fitter, model):
    """Demonstrate dataset override functionality."""
    
    print("\n" + "=" * 60)
    print("DATASET OVERRIDE EXAMPLES")
    print("=" * 60)
    
    # ========================================================================
    # Scenario 1: Use fitted data (default)
    # ========================================================================
    print("1. DEFAULT BEHAVIOR (fitted data)")
    print("-" * 40)
    
    results_default = FitResults(fitter, model)
    print(f"   Datasets: {len(results_default.datasets)}")
    print(f"   Dataset name: {results_default.datasets[0].name}")
    print(f"   Wavelength range: {results_default.datasets[0].wavelength.min():.1f} - {results_default.datasets[0].wavelength.max():.1f} Å")
    
    # ========================================================================
    # Scenario 2: Override with extended wavelength range
    # ========================================================================
    print("\n2. EXTENDED WAVELENGTH RANGE")
    print("-" * 40)
    
    # Create extended dataset for visualization
    extended_wave = np.linspace(3300, 3700, 1500)  # Wider range
    extended_flux = model.evaluate(fitter.best_theta, extended_wave)
    extended_error = np.full_like(extended_wave, 0.02)
    
    extended_dataset = Dataset(extended_wave, extended_flux, extended_error, name="Extended Range")
    
    results_extended = FitResults(fitter, model, dataset=extended_dataset)
    print(f"   Dataset name: {results_extended.datasets[0].name}")
    print(f"   Wavelength range: {results_extended.datasets[0].wavelength.min():.1f} - {results_extended.datasets[0].wavelength.max():.1f} Å")
    print("   → Model will be plotted on extended range")
    print("   → Chi-squared still calculated on fitted data")
    
    # Show that statistics are the same (based on fitted data)
    print(f"\n   Chi-squared comparison:")
    print(f"   Default:  χ² = {results_default.chi2_best:.2f}")
    print(f"   Extended: χ² = {results_extended.chi2_best:.2f}")
    print("   → Same because chi-squared uses fitted data, not visualization data")
    
    return results_default, results_extended


def demonstrate_multi_instrument():
    """Demonstrate multi-instrument FitResults."""
    
    print("\n" + "=" * 60)
    print("MULTI-INSTRUMENT EXAMPLE")
    print("=" * 60)
    
    # ========================================================================
    # Set up multi-instrument system
    # ========================================================================
    print("Setting up multi-instrument system...")
    
    # Same physical system observed with two instruments
    config_xshooter = FitConfiguration()
    config_xshooter.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    config_fire = FitConfiguration()
    config_fire.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    # Different instrumental resolutions
    model_xshooter = VoigtModel(config_xshooter, FWHM='2.2')  # Higher resolution
    model_fire = VoigtModel(config_fire, FWHM='4.0')  # Lower resolution
    
    # Compile multi-instrument model
    instrument_configs = {'XShooter': config_xshooter, 'FIRE': config_fire}
    compiled = model_xshooter.compile(instrument_configs=instrument_configs)
    
    # Generate synthetic data for both instruments
    theta = np.array([13.5, 13.2, 15.0, 25.0, -50.0, 20.0])
    
    wave_xs = np.linspace(3760, 3820, 800)
    wave_fire = np.linspace(3750, 3830, 600)
    
    flux_xs = compiled.model_flux(theta, wave_xs, instrument='XShooter')
    flux_fire = compiled.model_flux(theta, wave_fire, instrument='FIRE')
    
    # Add noise
    flux_xs += np.random.normal(0, 0.01, len(flux_xs))
    flux_fire += np.random.normal(0, 0.015, len(flux_fire))
    
    error_xs = np.full_like(wave_xs, 0.01)
    error_fire = np.full_like(wave_fire, 0.015)
    
    # Set up multi-instrument fitter
    bounds, lb, ub = mc.set_bounds([13.5, 13.2], [15.0, 25.0], [-50.0, 20.0])
    
    fitter_multi = mc.vfit(
        lambda th, w: compiled.model_flux(th, w, instrument='XShooter'),
        theta, lb, ub,
        wave_xs, flux_xs, error_xs,
        no_of_Chain=20,
        no_of_steps=500,
        multi_instrument=True,
        instrument_data={
            'FIRE': {
                'model': lambda th, w: compiled.model_flux(th, w, instrument='FIRE'),
                'wave': wave_fire,
                'flux': flux_fire,
                'error': error_fire
            }
        }
    )
    
    print("Running multi-instrument MCMC...")
    fitter_multi.runmcmc(optimize=True)
    
    # ========================================================================
    # Create MultiInstrumentFitResults
    # ========================================================================
    from rbvfit.core.fit_results import MultiInstrumentFitResults
    
    # Default: uses fitted data
    results_multi = MultiInstrumentFitResults(fitter_multi, model_xshooter)
    
    print(f"\n✓ MultiInstrumentFitResults created:")
    print(f"   Instruments: {len(results_multi.instrument_datasets)}")
    print(f"   Per-instrument chi-squared:")
    for instrument, chi2 in results_multi.chi2_by_instrument.items():
        print(f"     {instrument}: χ² = {chi2:.2f}")
    print(f"   Combined χ² = {results_multi.chi2_best:.2f}")
    
    # Override with extended datasets
    extended_xs = Dataset(
        np.linspace(3700, 3900, 1000), 
        compiled.model_flux(theta, np.linspace(3700, 3900, 1000), instrument='XShooter'),
        np.full(1000, 0.01),
        name="Extended XShooter"
    )
    
    results_multi_extended = MultiInstrumentFitResults(
        fitter_multi, model_xshooter, 
        datasets={'XShooter': extended_xs, 'FIRE': None}  # Override XShooter only
    )
    
    print(f"\n✓ With dataset overrides:")
    print(f"   XShooter wavelength range: {extended_xs.wavelength.min():.0f} - {extended_xs.wavelength.max():.0f} Å")
    print(f"   FIRE uses original fitted data")
    print(f"   Chi-squared unchanged: {results_multi_extended.chi2_best:.2f}")
    
    return results_multi, results_multi_extended


def demonstrate_usage_patterns(results):
    """Demonstrate common usage patterns."""
    
    print("\n" + "=" * 60)
    print("COMMON USAGE PATTERNS")
    print("=" * 60)
    
    # ========================================================================
    # 1. Quick summary
    # ========================================================================
    print("1. QUICK SUMMARY")
    print("-" * 30)
    print(results.summary())
    
    # ========================================================================
    # 2. Ion-specific analysis
    # ========================================================================
    print("\n\n2. ION-SPECIFIC ANALYSIS")
    print("-" * 30)
    print(results.get_ion_summary_table())
    
    # ========================================================================
    # 3. Publication exports
    # ========================================================================
    print("\n\n3. PUBLICATION EXPORTS")
    print("-" * 30)
    
    # LaTeX table (first few lines)
    latex_table = results.export_ion_table(format='latex')
    print("LaTeX table (excerpt):")
    print("\n".join(latex_table.split('\n')[:8]))
    print("   ... (table continues)")
    
    # ========================================================================
    # 4. Plotting
    # ========================================================================
    print("\n\n4. PLOTTING")
    print("-" * 30)
    print("Creating plots...")
    
    try:
        # Corner plot
        fig_corner = results.plot_corner(group_by_ion=True, figsize=(10, 8))
        print("   ✓ Ion-grouped corner plots created")
        
        # Model comparison
        fig_model = results.plot_model_comparison(show_residuals=True, figsize=(12, 8))
        print("   ✓ Model comparison plot created")
        
        plt.show()
        
    except ImportError as e:
        print(f"   ⚠ Plotting requires additional packages: {e}")


def main():
    """Main example function with simplified workflow."""
    print("rbvfit 2.0 Simplified Enhanced FitResults Example")
    print("This example shows the streamlined workflow for ion-aware analysis.\n")
    
    try:
        # ====================================================================
        # BASIC SINGLE-INSTRUMENT WORKFLOW
        # ====================================================================
        print("🚀 SINGLE-INSTRUMENT WORKFLOW")
        
        # Step 1: Run fit (complex setup)
        results, fitter, model = run_example_fit()
        
        # Step 2: Analyze results (simple usage!)
        demonstrate_usage_patterns(results)
        
        # Step 3: Dataset override examples
        demonstrate_dataset_override(fitter, model)
        
        # ====================================================================
        # MULTI-INSTRUMENT WORKFLOW  
        # ====================================================================
        print("\n\n🚀 MULTI-INSTRUMENT WORKFLOW")
        
        # Step 1: Multi-instrument setup and analysis
        results_multi, results_multi_extended = demonstrate_multi_instrument()
        
        # Step 2: Multi-instrument summary
        print("\nMulti-instrument summary:")
        print(results_multi.summary())
        
        print("\n" + "=" * 60)
        print("🎉 SIMPLIFIED EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📝 KEY SIMPLIFICATIONS:")
        print("✓ Single line: results = FitResults(fitter, model)")
        print("✓ Automatic extraction of all information from fitter")
        print("✓ Optional dataset override for visualization")
        print("✓ Separate class for multi-instrument: MultiInstrumentFitResults")
        print("✓ All ion-aware analysis capabilities preserved")
        
        print("\n🔧 PRACTICAL WORKFLOW:")
        print("1. Set up configuration and model")
        print("2. Run MCMC: fitter.runmcmc()")
        print("3. Analyze: results = FitResults(fitter, model)")
        print("4. Visualize: results.plot_corner(), results.plot_model_comparison()")
        print("5. Export: results.export_ion_table(format='latex')")
        
        print("\n💡 DATASET OVERRIDE USE CASES:")
        print("• Plot model on extended wavelength range")
        print("• Visualize on higher resolution data")
        print("• Show model on unmasked data")
        print("• Multi-instrument: override per-instrument")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("This might be due to missing dependencies or import issues.")
        print("Make sure rbvfit 2.0 is properly installed.")


if __name__ == "__main__":
    main()


# ============================================================================
# SIMPLIFIED QUICK START GUIDE
# ============================================================================
"""
SIMPLIFIED QUICK START GUIDE for Enhanced FitResults

1. BASIC USAGE (Single Instrument):
   ```python
   # After MCMC fitting
   results = FitResults(fitter, model)
   
   # That's it! Everything extracted automatically:
   # - Parameter manager from model.config
   # - Datasets from fitter data
   # - MCMC settings from fitter attributes
   # - Bounds from fitter.lb, fitter.ub
   ```

2. DATASET OVERRIDE (Visualization):
   ```python
   # Use fitted data (default)
   results = FitResults(fitter, model)
   
   # Plot on extended wavelength range
   results = FitResults(fitter, model, extended_dataset)
   
   # Note: Chi-squared always calculated on fitted data
   #       Visualization uses override dataset
   ```

3. MULTI-INSTRUMENT:
   ```python
   # Default behavior
   results = MultiInstrumentFitResults(fitter, master_model)
   
   # Override specific instruments
   results = MultiInstrumentFitResults(fitter, master_model, datasets={
       'XShooter': extended_dataset,
       'FIRE': None  # Use original
   })
   
   # Per-instrument statistics available
   print(results.chi2_by_instrument)
   ```

4. ANALYSIS AND EXPORT:
   ```python
   # Quick summary
   print(results.summary())
   
   # Ion-specific table
   print(results.get_ion_summary_table())
   
   # Publication-ready LaTeX
   latex_table = results.export_ion_table(format='latex')
   
   # Visualization
   results.plot_corner(group_by_ion=True)
   results.plot_model_comparison(show_residuals=True)
   ```

5. CONVENIENCE METHODS (Future Enhancement):
   ```python
   # Could add to vfit class:
   class vfit:
       def get_results(self, model, dataset=None):
           return FitResults(self, model, dataset)
   
   # Then just:
   results = fitter.get_results(model)
   ```

KEY PRINCIPLES:
• Fitter contains the MCMC results and fitted data
• Model contains the physics and configuration
• FitResults combines them for ion-aware analysis
• Dataset override enables flexible visualization
• Multi-instrument handled by separate class
• Everything else extracted automatically
"""

