#!/usr/bin/env python
"""
Examples demonstrating the enhanced fitter capabilities in rbvfit 2.0.

This script shows:
1. Single dataset fitting with emcee
2. Multi-dataset joint fitting 
3. Different MCMC backends (emcee vs zeus)
4. Custom MCMC settings
5. Dataset-specific LSF parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.voigt_fitter import VoigtFitter, JointFitter, Dataset, MCMCSettings
from rbvfit.core.parameter_manager import ParameterManager


def generate_synthetic_data(wave, theta, model, noise_level=0.02, lsf_fwhm='6.5'):
    """Generate synthetic data for testing."""
    # Create model with specified LSF
    if lsf_fwhm != model.FWHM:
        flux_true = model.evaluate(theta, wave, FWHM=lsf_fwhm)
    else:
        flux_true = model.evaluate(theta, wave)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(wave))
    flux_obs = flux_true + noise
    error = np.full_like(wave, noise_level)
    
    return flux_obs, error, flux_true


def example_single_dataset_fitting():
    """Example 1: Single dataset fitting with MgII doublet."""
    print("=" * 60)
    print("Example 1: Single Dataset Fitting")
    print("=" * 60)
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    # Create model
    model = VoigtModel(config, FWHM='6.5')
    print(model.get_info())
    
    # True parameters for synthetic data
    theta_true = np.array([
        13.5, 13.2,     # log N values
        15.0, 25.0,     # b values (km/s)
        -50.0, 20.0     # v values (km/s)
    ])
    
    # Generate synthetic data
    wave = np.linspace(3760, 3820, 2000)
    flux_obs, error, flux_true = generate_synthetic_data(wave, theta_true, model, noise_level=0.03)
    
    # Create dataset
    dataset = Dataset(wave, flux_obs, error, name="COS_G130M")
    
    # Create fitter with default MCMC settings
    fitter = VoigtFitter(model, dataset)
    
    # Initial guess (slightly perturbed from truth)
    initial_guess = theta_true + 0.1 * np.random.randn(len(theta_true))
    
    # Generate parameter bounds
    param_manager = ParameterManager(config)
    bounds = param_manager.generate_parameter_bounds()
    
    print("\nParameter bounds:")
    param_names = param_manager.get_parameter_names()
    for i, name in enumerate(param_names):
        print(f"  {name}: [{bounds.lower[i]:.1f}, {bounds.upper[i]:.1f}]")
    
    print(f"\nTrue parameters: {theta_true}")
    print(f"Initial guess: {initial_guess}")
    
    # Run fit (short run for example)
    mcmc_settings = MCMCSettings(n_walkers=30, n_steps=500, n_burn=100)
    fitter.mcmc_settings = mcmc_settings
    
    print("\nRunning MCMC fit...")
    result = fitter.fit(initial_guess, bounds, optimize_first=True)
    
    # Show results
    print("\n" + result.summary())
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot data and best fit
    plt.subplot(2, 1, 1)
    plt.step(wave, flux_obs, 'k-', where='mid', label='Observed', alpha=0.7)
    plt.plot(wave, flux_true, 'r-', label='True model', linewidth=2)
    
    best_model = model.evaluate(result.best_fit, wave)
    plt.plot(wave, best_model, 'b--', label='Best fit', linewidth=2)
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title('Single Dataset Fit: MgII Doublet')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = (flux_obs - best_model) / error
    plt.step(wave, residuals, 'k-', where='mid', alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Residuals (σ)')
    plt.title('Fit Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return result, model, dataset


def example_joint_fitting():
    """Example 2: Joint fitting of multiple datasets."""
    print("\n" + "=" * 60)
    print("Example 2: Joint Fitting of Multiple Datasets")
    print("=" * 60)
    
    # Create configuration for multi-ion system
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    config.add_system(z=0.348, ion='FeII', transitions=[2600.2], components=1)
    
    # Create model
    model = VoigtModel(config, FWHM='6.5')
    print(model.get_info())
    
    # True parameters
    theta_true = np.array([
        13.5, 13.2, 14.8,    # N values (MgII, MgII, FeII)
        15.0, 25.0, 20.0,    # b values
        -50.0, 20.0, 0.0     # v values
    ])
    
    # Generate datasets with different instrumental setups
    # Dataset 1: COS G130M (higher resolution, MgII region)
    wave1 = np.linspace(3760, 3820, 1500)
    flux1, error1, _ = generate_synthetic_data(
        wave1, theta_true, model, noise_level=0.025, lsf_fwhm='6.5'
    )
    dataset1 = Dataset(
        wave1, flux1, error1, 
        name="COS_G130M_MgII",
        lsf_params={'FWHM': '6.5'}
    )
    
    # Dataset 2: STIS (lower resolution, FeII region)  
    wave2 = np.linspace(3500, 3550, 800)
    flux2, error2, _ = generate_synthetic_data(
        wave2, theta_true, model, noise_level=0.035, lsf_fwhm='12.0'
    )
    dataset2 = Dataset(
        wave2, flux2, error2,
        name="STIS_G430L_FeII", 
        lsf_params={'FWHM': '12.0'}
    )
    
    # Dataset 3: COS G160M (different grating, MgII region)
    wave3 = np.linspace(3770, 3810, 1000)
    flux3, error3, _ = generate_synthetic_data(
        wave3, theta_true, model, noise_level=0.030, lsf_fwhm='8.5'
    )
    dataset3 = Dataset(
        wave3, flux3, error3,
        name="COS_G160M_MgII",
        lsf_params={'FWHM': '8.5'}
    )
    
    datasets = [dataset1, dataset2, dataset3]
    
    # Create joint fitter
    mcmc_settings = MCMCSettings(
        sampler='emcee',  # Try zeus if available
        n_walkers=40,
        n_steps=600,
        n_burn=120,
        parallel=True
    )
    
    fitter = JointFitter(model, datasets, mcmc_settings)
    
    # Initial guess
    initial_guess = theta_true + 0.1 * np.random.randn(len(theta_true))
    
    # Custom bounds for this system
    param_manager = ParameterManager(config)
    custom_bounds = {
        'MgII': {'N': (12.0, 15.0), 'b': (5.0, 50.0), 'v': (-100.0, 100.0)},
        'FeII': {'N': (13.0, 16.0), 'b': (5.0, 40.0), 'v': (-50.0, 50.0)}
    }
    bounds = param_manager.generate_parameter_bounds(custom_bounds=custom_bounds)
    
    print(f"\nFitting {len(datasets)} datasets jointly:")
    for dataset in datasets:
        print(f"  - {dataset.name}: {len(dataset.wavelength)} points, "
              f"FWHM={dataset.lsf_params['FWHM']}")
    
    print(f"\nTrue parameters: {theta_true}")
    print(f"Initial guess: {initial_guess}")
    
    # Run joint fit
    print("\nRunning joint MCMC fit...")
    result = fitter.fit(initial_guess, bounds, optimize_first=True)
    
    # Show results
    print("\n" + result.summary())
    
    # Plot joint fitting results
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    
    for i, (dataset, ax) in enumerate(zip(datasets, axes)):
        # Generate model for this dataset
        if dataset.lsf_params:
            best_model = model.evaluate(result.best_fit, dataset.wavelength, 
                                       FWHM=dataset.lsf_params['FWHM'])
            true_model = model.evaluate(theta_true, dataset.wavelength,
                                       FWHM=dataset.lsf_params['FWHM'])
        else:
            best_model = model.evaluate(result.best_fit, dataset.wavelength)
            true_model = model.evaluate(theta_true, dataset.wavelength)
        
        # Plot
        ax.step(dataset.wavelength, dataset.flux, 'k-', where='mid', 
                label='Observed', alpha=0.7)
        ax.plot(dataset.wavelength, true_model, 'r-', label='True model', linewidth=2)
        ax.plot(dataset.wavelength, best_model, 'b--', label='Best fit', linewidth=2)
        
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Dataset {i+1}: {dataset.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return result, model, datasets


def example_mcmc_backend_comparison():
    """Example 3: Compare emcee vs zeus backends."""
    print("\n" + "=" * 60)
    print("Example 3: MCMC Backend Comparison")
    print("=" * 60)
    
    # Simple MgII doublet for speed
    config = FitConfiguration()
    config.add_system(z=0.1, ion='MgII', transitions=[2796.3, 2803.5], components=1)
    model = VoigtModel(config)
    
    # Generate data
    theta_true = np.array([13.8, 20.0, 0.0])  # N, b, v
    wave = np.linspace(3070, 3090, 800)
    flux_obs, error, _ = generate_synthetic_data(wave, theta_true, model, noise_level=0.02)
    dataset = Dataset(wave, flux_obs, error)
    
    # Test different backends
    backends = ['emcee']
    if HAS_ZEUS:
        backends.append('zeus')
        print("Testing both emcee and zeus backends")
    else:
        print("Zeus not available, testing emcee only")
    
    results = {}
    
    for backend in backends:
        print(f"\n--- Testing {backend} backend ---")
        
        mcmc_settings = MCMCSettings(
            sampler=backend,
            n_walkers=24,
            n_steps=300,
            n_burn=60,
            progress=True
        )
        
        fitter = VoigtFitter(model, dataset, mcmc_settings)
        initial_guess = theta_true + 0.05 * np.random.randn(len(theta_true))
        
        import time
        start_time = time.time()
        result = fitter.fit(initial_guess, optimize_first=False)
        end_time = time.time()
        
        results[backend] = {
            'result': result,
            'time': end_time - start_time,
            'best_fit': result.best_fit,
            'uncertainties': result.uncertainties
        }
        
        print(f"{backend} completed in {end_time - start_time:.1f} seconds")
        print(f"Best fit: {result.best_fit}")
        print(f"True values: {theta_true}")
        print(f"Difference: {np.abs(result.best_fit - theta_true)}")
    
    # Compare results if multiple backends tested
    if len(results) > 1:
        print("\n--- Backend Comparison ---")
        param_names = ['N', 'b', 'v']
        
        for i, param in enumerate(param_names):
            print(f"\n{param} parameter:")
            print(f"  True value: {theta_true[i]:.3f}")
            for backend in backends:
                res = results[backend]
                best = res['best_fit'][i]
                err = res['uncertainties'][i]
                print(f"  {backend}: {best:.3f} ± {err:.3f}")
        
        print("\nTiming comparison:")
        for backend in backends:
            print(f"  {backend}: {results[backend]['time']:.1f} seconds")
    
    return results


def example_custom_mcmc_settings():
    """Example 4: Custom MCMC settings and diagnostics."""
    print("\n" + "=" * 60)
    print("Example 4: Custom MCMC Settings and Diagnostics")
    print("=" * 60)
    
    # Create a more complex system to test MCMC performance
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=3)
    model = VoigtModel(config)
    
    # True parameters for 3-component system
    theta_true = np.array([
        13.8, 13.5, 13.2,    # N values
        25.0, 15.0, 35.0,    # b values
        -80.0, -20.0, 40.0   # v values
    ])
    
    # Generate data
    wave = np.linspace(3760, 3820, 2500)
    flux_obs, error, _ = generate_synthetic_data(wave, theta_true, model, noise_level=0.02)
    dataset = Dataset(wave, flux_obs, error, name="Complex_MgII_System")
    
    # Custom MCMC settings for complex system
    mcmc_settings = MCMCSettings(
        sampler='emcee',
        n_walkers=60,  # More walkers for complex parameter space
        n_steps=1000,
        n_burn=200,    # Longer burn-in
        thin=2,        # Thin samples to reduce correlation
        parallel=True,
        progress=True
    )
    
    fitter = VoigtFitter(model, dataset, mcmc_settings)
    
    # Generate reasonable initial guess
    param_manager = ParameterManager(config)
    bounds = param_manager.generate_parameter_bounds()
    
    # Start from a perturbed version of truth
    initial_guess = theta_true + 0.2 * np.random.randn(len(theta_true))
    
    # Ensure within bounds
    initial_guess = np.clip(initial_guess, bounds.lower, bounds.upper)
    
    print(f"Complex system with {len(theta_true)} parameters")
    print(f"MCMC settings: {mcmc_settings.n_walkers} walkers, {mcmc_settings.n_steps} steps")
    print(f"True parameters: {theta_true}")
    print(f"Initial guess: {initial_guess}")
    
    # Run fit
    result = fitter.fit(initial_guess, bounds, optimize_first=True)
    
    # Detailed analysis
    print("\n" + result.summary())
    
    # Calculate acceptance fraction and autocorrelation
    if hasattr(result.sampler, 'acceptance_fraction'):
        acc_frac = np.mean(result.sampler.acceptance_fraction)
        print(f"\nMean acceptance fraction: {acc_frac:.3f}")
        
        if acc_frac < 0.2:
            print("  Warning: Low acceptance fraction (<0.2)")
        elif acc_frac > 0.5:
            print("  Warning: High acceptance fraction (>0.5)")
        else:
            print("  Good acceptance fraction (0.2-0.5)")
    
    # Check autocorrelation time if available
    try:
        if hasattr(result.sampler, 'get_autocorr_time'):
            autocorr_time = result.sampler.get_autocorr_time()
            max_autocorr = np.max(autocorr_time)
            effective_samples = len(result.samples) / (2 * max_autocorr)
            
            print(f"Max autocorrelation time: {max_autocorr:.1f}")
            print(f"Effective samples: {effective_samples:.0f}")
            
            if effective_samples < 100:
                print("  Warning: Few effective samples (<100)")
            else:
                print("  Good number of effective samples")
                
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")
    
    # Plot chain evolution for first few parameters
    if hasattr(result.sampler, 'get_chain'):
        chain = result.sampler.get_chain()
        
        fig, axes = plt.subplots(min(3, len(theta_true)), 1, figsize=(10, 8))
        if len(theta_true) == 1:
            axes = [axes]
        elif len(theta_true) == 2:
            axes = axes[:2]
        
        param_names = param_manager.get_parameter_names()
        
        for i in range(min(3, len(theta_true))):
            ax = axes[i] if len(theta_true) > 1 else axes
            
            # Plot chains for all walkers
            for j in range(min(10, mcmc_settings.n_walkers)):  # Plot first 10 walkers
                ax.plot(chain[:, j, i], alpha=0.3, color='blue', linewidth=0.5)
            
            # Mark burn-in
            ax.axvline(mcmc_settings.n_burn, color='red', linestyle='--', 
                      label='Burn-in end' if i == 0 else '')
            
            # Mark true value
            ax.axhline(theta_true[i], color='green', linestyle='-', 
                      label='True value' if i == 0 else '')
            
            ax.set_ylabel(param_names[i])
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
            if i == len(axes) - 1:
                ax.set_xlabel('Step')
        
        plt.suptitle('MCMC Chain Evolution')
        plt.tight_layout()
        plt.show()
    
    return result, model


def example_dataset_specific_lsf():
    """Example 5: Different LSF parameters for different datasets."""
    print("\n" + "=" * 60)
    print("Example 5: Dataset-Specific LSF Parameters")
    print("=" * 60)
    
    # Simple system to focus on LSF effects
    config = FitConfiguration()
    config.add_system(z=0.1, ion='MgII', transitions=[2796.3], components=1)
    model = VoigtModel(config, FWHM='6.5')  # Default LSF
    
    # True parameters
    theta_true = np.array([13.5, 15.0, 0.0])  # N, b, v
    
    # Generate datasets with different LSF
    wave_base = np.linspace(3070, 3085, 500)
    
    lsf_configs = [
        {'name': 'COS_G130M', 'FWHM': '6.5', 'noise': 0.02},
        {'name': 'COS_G160M', 'FWHM': '8.5', 'noise': 0.025}, 
        {'name': 'STIS_G430L', 'FWHM': '15.0', 'noise': 0.03},
        {'name': 'High_Res', 'FWHM': '3.0', 'noise': 0.015}
    ]
    
    datasets = []
    
    for lsf_config in lsf_configs:
        flux_obs, error, _ = generate_synthetic_data(
            wave_base, theta_true, model, 
            noise_level=lsf_config['noise'],
            lsf_fwhm=lsf_config['FWHM']
        )
        
        dataset = Dataset(
            wave_base, flux_obs, error,
            name=lsf_config['name'],
            lsf_params={'FWHM': lsf_config['FWHM']}
        )
        datasets.append(dataset)
    
    print(f"Created {len(datasets)} datasets with different LSF:")
    for dataset in datasets:
        print(f"  {dataset.name}: FWHM = {dataset.lsf_params['FWHM']} pixels")
    
    # Fit each dataset individually to show LSF effects
    individual_results = {}
    
    for dataset in datasets:
        print(f"\nFitting {dataset.name}...")
        
        mcmc_settings = MCMCSettings(n_walkers=20, n_steps=300, n_burn=60)
        fitter = VoigtFitter(model, dataset, mcmc_settings)
        
        initial_guess = theta_true + 0.05 * np.random.randn(len(theta_true))
        result = fitter.fit(initial_guess, optimize_first=False)
        
        individual_results[dataset.name] = result
        
        print(f"  Best fit: {result.best_fit}")
        print(f"  Uncertainties: {result.uncertainties}")
    
    # Joint fit of all datasets
    print(f"\nJoint fitting all {len(datasets)} datasets...")
    
    mcmc_settings = MCMCSettings(n_walkers=30, n_steps=400, n_burn=80)
    joint_fitter = JointFitter(model, datasets, mcmc_settings)
    
    initial_guess = theta_true + 0.05 * np.random.randn(len(theta_true))
    joint_result = joint_fitter.fit(initial_guess, optimize_first=True)
    
    print("Joint fit results:")
    print(joint_result.summary())
    
    # Compare results
    print("\n--- Results Comparison ---")
    param_names = ['log N', 'b (km/s)', 'v (km/s)']
    
    print(f"{'Parameter':<12} {'True':<8} {'Joint Fit':<15} ", end="")
    for dataset in datasets:
        print(f"{dataset.name:<12}", end=" ")
    print()
    
    for i, param_name in enumerate(param_names):
        print(f"{param_name:<12} {theta_true[i]:<8.2f} ", end="")
        
        # Joint fit result
        joint_val = joint_result.best_fit[i]
        joint_err = joint_result.uncertainties[i]
        print(f"{joint_val:.2f}±{joint_err:.2f}    ", end="")
        
        # Individual fit results
        for dataset in datasets:
            ind_result = individual_results[dataset.name]
            ind_val = ind_result.best_fit[i]
            ind_err = ind_result.uncertainties[i]
            print(f"{ind_val:.2f}±{ind_err:.2f}  ", end="")
        
        print()
    
    # Visualize all fits
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (dataset, ax) in enumerate(zip(datasets, axes)):
        # Individual fit
        ind_model = model.evaluate(
            individual_results[dataset.name].best_fit, 
            dataset.wavelength,
            FWHM=dataset.lsf_params['FWHM']
        )
        
        # Joint fit
        joint_model = model.evaluate(
            joint_result.best_fit,
            dataset.wavelength, 
            FWHM=dataset.lsf_params['FWHM']
        )
        
        # True model
        true_model = model.evaluate(
            theta_true,
            dataset.wavelength,
            FWHM=dataset.lsf_params['FWHM']
        )
        
        # Plot
        ax.step(dataset.wavelength, dataset.flux, 'k-', where='mid', 
                label='Data', alpha=0.7)
        ax.plot(dataset.wavelength, true_model, 'g-', label='True', linewidth=2)
        ax.plot(dataset.wavelength, ind_model, 'r--', label='Individual fit', linewidth=2)
        ax.plot(dataset.wavelength, joint_model, 'b:', label='Joint fit', linewidth=2)
        
        ax.set_title(f'{dataset.name} (FWHM={dataset.lsf_params["FWHM"]})')
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Normalized Flux')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return joint_result, individual_results


if __name__ == "__main__":
    """Run all examples demonstrating the enhanced fitter capabilities."""
    
    # Import check
    try:
        # These would be the actual imports in the real implementation
        print("rbvfit 2.0 Enhanced Fitter Examples")
        print("=" * 60)
        
        # Note: In actual implementation, need to handle missing zeus gracefully
        from rbvfit.core.voigt_fitter import HAS_ZEUS
        if not HAS_ZEUS:
            print("Note: Zeus sampler not available. Only emcee will be tested.")
        
        print("Running enhanced fitter examples...")
        
        # Run examples
        #print("\nExample 1: Single dataset fitting...")
        #result1, model1, dataset1 = example_single_dataset_fitting()
        
        #print("\nExample 2: Joint fitting...")
        #result2, model2, datasets2 = example_joint_fitting()
        
        #print("\nExample 3: MCMC backend comparison...")
        #results3 = example_mcmc_backend_comparison()
        
        #print("\nExample 4: Custom MCMC settings...")
        #result4, model4 = example_custom_mcmc_settings()
        
        print("\nExample 5: Dataset-specific LSF...")
        result5, individual_results5 = example_dataset_specific_lsf()
        
        print("\n" + "=" * 60)
        print("All enhanced fitter examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("This example requires the full rbvfit 2.0 implementation.")
        print("Run this after implementing the core modules.")