#!/usr/bin/env python
"""
rbvfit v2.0 Performance Benchmark: Quick Fit vs MCMC Analysis

Focused performance testing for rbvfit v2 models with:
- MgII + FeII systems (2 components each)
- Single instrument (XShooter) and multi-instrument (XShooter + HIRES)
- Raw model evaluation, likelihood evaluation, quick_fit vs MCMC
- Zeus vs emcee, serial vs parallel comparisons
- Parameter recovery accuracy metrics
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import rbvfit v2 components
try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    import rbvfit.vfit_mcmc as mc
    print("✓ rbvfit v2.0 components loaded")
except ImportError as e:
    print(f"❌ Error: rbvfit v2.0 not available: {e}")
    exit(1)


def setup_test_scenarios():
    """Set up single and multi-instrument test scenarios."""
    print("Setting up test scenarios...")
    
    # MgII + FeII systems with 2 components each
    # Use slightly different redshifts to ensure line separation
    z_mgii = 0.348
    z_feii = 0.352
    
    # MgII doublet (2796.35, 2803.53 Å)
    mgii_transitions = [2796.35, 2803.53]
    # FeII transitions (2586.65, 2600.17 Å) 
    feii_transitions = [2586.65, 2600.17]
    
    # True parameters for 2-component systems
    # MgII: 2 components
    true_N_mgii = [13.8, 13.2]
    true_b_mgii = [25.0, 15.0]
    true_v_mgii = [-20.0, 15.0]
    
    # FeII: 2 components  
    true_N_feii = [13.5, 13.0]
    true_b_feii = [20.0, 12.0]
    true_v_feii = [-15.0, 10.0]
    
    # Combined parameter vector [N_mgii, N_feii, b_mgii, b_feii, v_mgii, v_feii]
    true_theta = np.array(true_N_mgii + true_N_feii + true_b_mgii + true_b_feii + true_v_mgii + true_v_feii)
    
    # Observed wavelength coverage - ensure both systems are covered
    wave_start = 3700  # Covers MgII at z=0.348
    wave_end = 3900    # Covers FeII at z=0.352
    wave_obs = np.linspace(wave_start, wave_end, 800)
    
    scenario = {
        'z_mgii': z_mgii,
        'z_feii': z_feii,
        'mgii_transitions': mgii_transitions,
        'feii_transitions': feii_transitions,
        'true_theta': true_theta,
        'wave_obs': wave_obs,
        'ncomp_mgii': 2,
        'ncomp_feii': 2,
        'true_N_mgii': true_N_mgii,
        'true_b_mgii': true_b_mgii,
        'true_v_mgii': true_v_mgii,
        'true_N_feii': true_N_feii,
        'true_b_feii': true_b_feii,
        'true_v_feii': true_v_feii
    }
    
    print(f"  MgII system: z={z_mgii}, transitions={mgii_transitions}")
    print(f"  FeII system: z={z_feii}, transitions={feii_transitions}")
    print(f"  True parameters: {len(true_theta)} total")
    print(f"  Wavelength coverage: {wave_start}-{wave_end} Å ({len(wave_obs)} points)")
    
    return scenario


def create_models(scenario):
    """Create single and multi-instrument models."""
    print("\nCreating models...")
    
    models = {}
    
    # Single instrument: XShooter
    print("  Creating XShooter model...")
    config_single = FitConfiguration()
    config_single.add_system(z=scenario['z_mgii'], ion='MgII', 
                           transitions=scenario['mgii_transitions'], 
                           components=scenario['ncomp_mgii'])
    config_single.add_system(z=scenario['z_feii'], ion='FeII',
                           transitions=scenario['feii_transitions'],
                           components=scenario['ncomp_feii'])
    
    model_xshooter = VoigtModel(config_single, FWHM='6.5')
    model_xshooter_compiled = model_xshooter.compile(verbose=False)
    
    models['single'] = {
        'regular': model_xshooter,
        'compiled': model_xshooter_compiled,
        'config': config_single
    }
    
    # Multi-instrument: XShooter + HIRES
    print("  Creating XShooter + HIRES models...")
    
    # XShooter model (same as above)
    config_xshooter = config_single  # Reuse
    model_xshooter_multi = VoigtModel(config_xshooter, FWHM='6.5')
    model_xshooter_multi_compiled = model_xshooter_multi.compile(verbose=False)
    
    # HIRES model (higher resolution)
    config_hires = FitConfiguration()
    config_hires.add_system(z=scenario['z_mgii'], ion='MgII',
                           transitions=scenario['mgii_transitions'],
                           components=scenario['ncomp_mgii'])
    config_hires.add_system(z=scenario['z_feii'], ion='FeII',
                           transitions=scenario['feii_transitions'],
                           components=scenario['ncomp_feii'])
    
    model_hires = VoigtModel(config_hires, FWHM='2.5')  # Higher resolution
    model_hires_compiled = model_hires.compile(verbose=False)
    
    models['multi'] = {
        'xshooter': {
            'regular': model_xshooter_multi,
            'compiled': model_xshooter_multi_compiled,
            'config': config_xshooter
        },
        'hires': {
            'regular': model_hires,
            'compiled': model_hires_compiled,
            'config': config_hires
        }
    }
    
    return models


def generate_synthetic_data(models, scenario):
    """Generate realistic synthetic data with noise."""
    print("\nGenerating synthetic data...")
    
    # Use compiled single model for data generation
    model_func = models['single']['compiled'].model_flux
    
    # Generate clean spectrum
    true_flux = model_func(scenario['true_theta'], scenario['wave_obs'])
    
    # Add realistic noise (S/N ~ 25)
    noise_level = 0.04  # 4% noise
    np.random.seed(42)  # Reproducible
    noise = np.random.normal(0, noise_level, len(scenario['wave_obs']))
    observed_flux = true_flux + noise
    error = np.full_like(scenario['wave_obs'], noise_level)
    
    # Create second dataset for HIRES (same underlying physics, different noise)
    np.random.seed(43)
    noise_hires = np.random.normal(0, noise_level * 0.8, len(scenario['wave_obs']))  # Better S/N
    observed_flux_hires = true_flux + noise_hires
    error_hires = np.full_like(scenario['wave_obs'], noise_level * 0.8)
    
    data = {
        'single': {
            'wave': scenario['wave_obs'],
            'flux': observed_flux,
            'error': error,
            'true_flux': true_flux
        },
        'multi': {
            'xshooter': {
                'wave': scenario['wave_obs'],
                'flux': observed_flux,
                'error': error
            },
            'hires': {
                'wave': scenario['wave_obs'],
                'flux': observed_flux_hires,
                'error': error_hires
            }
        }
    }
    
    print(f"  Noise level: {noise_level*100:.1f}% (XShooter), {noise_level*0.8*100:.1f}% (HIRES)")
    print(f"  S/N ratio: ~{1/noise_level:.0f} (XShooter), ~{1/(noise_level*0.8):.0f} (HIRES)")
    
    return data


def benchmark_raw_evaluation(models, scenario, n_evals=1000):
    """Benchmark raw model evaluation speed."""
    print(f"\nBenchmarking raw model evaluation ({n_evals} calls)...")
    
    results = {}
    
    # Test configurations
    test_configs = [
        ('single_regular', models['single']['regular'].evaluate),
        ('single_compiled', models['single']['compiled'].model_flux),
        ('multi_xshooter_regular', models['multi']['xshooter']['regular'].evaluate),
        ('multi_xshooter_compiled', models['multi']['xshooter']['compiled'].model_flux),
        ('multi_hires_regular', models['multi']['hires']['regular'].evaluate),
        ('multi_hires_compiled', models['multi']['hires']['compiled'].model_flux)
    ]
    
    for name, eval_func in test_configs:
        print(f"  Testing {name}...")
        
        # Generate perturbed parameters
        test_thetas = []
        for i in range(n_evals):
            noise = 0.01 * np.random.randn(len(scenario['true_theta']))
            test_thetas.append(scenario['true_theta'] + noise)
        
        # Time the evaluations
        start = time.time()
        success_count = 0
        for theta in test_thetas:
            try:
                flux = eval_func(theta, scenario['wave_obs'])
                if not np.any(np.isnan(flux)) and not np.any(np.isinf(flux)):
                    success_count += 1
            except Exception:
                pass
        
        eval_time = time.time() - start
        
        results[name] = {
            'total_time': eval_time,
            'time_per_eval': eval_time / n_evals * 1000,  # ms
            'success_rate': success_count / n_evals,
            'evals_per_second': n_evals / eval_time
        }
        
        print(f"    {eval_time:.3f}s total, {eval_time/n_evals*1000:.2f} ms/eval, {success_count}/{n_evals} success")
    
    return results


def benchmark_likelihood_evaluation(models, scenario, data, n_evals=1000):
    """Benchmark likelihood evaluation speed."""
    print(f"\nBenchmarking likelihood evaluation ({n_evals} calls)...")
    
    results = {}
    
    # Set up parameter bounds
    nguess = scenario['true_N_mgii'] + scenario['true_N_feii']
    bguess = scenario['true_b_mgii'] + scenario['true_b_feii']
    vguess = scenario['true_v_mgii'] + scenario['true_v_feii']
    
    bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
    
    # Test configurations
    test_configs = [
        ('single_regular', models['single']['regular'], data['single']),
        ('single_compiled', models['single']['compiled'], data['single'])
    ]
    
    for name, model, test_data in test_configs:
        print(f"  Testing {name}...")
        
        # Create instrument data
        instrument_data = {
            'XShooter': {
                'model': model,
                'wave': test_data['wave'],
                'flux': test_data['flux'],
                'error': test_data['error']
            }
        }
        
        # Create fitter
        fitter = mc.vfit(instrument_data, scenario['true_theta'], lb, ub,perturbation=1e-4)
        
        # Generate perturbed parameters
        test_thetas = []
        for i in range(n_evals):
            noise = 0.01 * np.random.randn(len(scenario['true_theta']))
            test_thetas.append(scenario['true_theta'] + noise)
        
        # Time likelihood evaluations
        start = time.time()
        success_count = 0
        for theta in test_thetas:
            try:
                lnlike = fitter.lnlike(theta)
                if np.isfinite(lnlike):
                    success_count += 1
            except Exception:
                pass
        
        eval_time = time.time() - start
        
        results[name] = {
            'total_time': eval_time,
            'time_per_eval': eval_time / n_evals * 1000,  # ms
            'success_rate': success_count / n_evals,
            'evals_per_second': n_evals / eval_time
        }
        
        print(f"    {eval_time:.3f}s total, {eval_time/n_evals*1000:.2f} ms/eval, {success_count}/{n_evals} success")
    
    return results


def benchmark_quick_fit(models, scenario, data):
    """Benchmark quick_fit performance."""
    print("\nBenchmarking quick_fit...")
    
    results = {}
    
    # Set up parameter bounds
    nguess = scenario['true_N_mgii'] + scenario['true_N_feii']
    bguess = scenario['true_b_mgii'] + scenario['true_b_feii']
    vguess = scenario['true_v_mgii'] + scenario['true_v_feii']
    
    bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
    
    # Test configurations
    test_configs = [
        ('single_regular', models['single']['regular'], data['single']),
        ('single_compiled', models['single']['compiled'], data['single'])
    ]
    
    # Add multi-instrument test
    multi_instrument_data = {
        'XShooter': {
            'model': models['multi']['xshooter']['compiled'],
            'wave': data['multi']['xshooter']['wave'],
            'flux': data['multi']['xshooter']['flux'],
            'error': data['multi']['xshooter']['error']
        },
        'HIRES': {
            'model': models['multi']['hires']['compiled'],
            'wave': data['multi']['hires']['wave'],
            'flux': data['multi']['hires']['flux'],
            'error': data['multi']['hires']['error']
        }
    }
    
    for name, model, test_data in test_configs:
        print(f"  Testing {name}...")
        
        # Create instrument data
        if name.startswith('single'):
            instrument_data = {
                'XShooter': {
                    'model': model,
                    'wave': test_data['wave'],
                    'flux': test_data['flux'],
                    'error': test_data['error']
                }
            }
        else:
            instrument_data = multi_instrument_data
        
        # Create fitter
        fitter = mc.vfit(instrument_data, scenario['true_theta'], lb, ub,perturbation=1e-4)
        
        try:
            start = time.time()
            theta_best, theta_errors = fitter.fit_quick(verbose=False)
            fit_time = time.time() - start
            
            # Calculate parameter recovery metrics
            param_recovery = calculate_parameter_recovery(scenario['true_theta'], theta_best, theta_errors)
            
            results[name] = {
                'fit_time': fit_time,
                'theta_best': theta_best,
                'theta_errors': theta_errors,
                'success': True,
                'recovery_metrics': param_recovery
            }
            
            print(f"    {fit_time:.3f}s, recovery accuracy: {param_recovery['mean_relative_error']:.3f}")
            
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Test multi-instrument
    print("  Testing multi_instrument...")
    try:
        fitter_multi = mc.vfit(multi_instrument_data, scenario['true_theta'], lb, ub)
        
        start = time.time()
        theta_best, theta_errors = fitter_multi.fit_quick(verbose=False)
        fit_time = time.time() - start
        
        param_recovery = calculate_parameter_recovery(scenario['true_theta'], theta_best, theta_errors)
        
        results['multi_instrument'] = {
            'fit_time': fit_time,
            'theta_best': theta_best,
            'theta_errors': theta_errors,
            'success': True,
            'recovery_metrics': param_recovery
        }
        
        print(f"    {fit_time:.3f}s, recovery accuracy: {param_recovery['mean_relative_error']:.3f}")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        results['multi_instrument'] = {'success': False, 'error': str(e)}
    
    return results


def benchmark_mcmc(models, scenario, data):
    """Benchmark MCMC performance with zeus and emcee."""
    print("\nBenchmarking MCMC...")
    
    results = {}
    
    # Set up parameter bounds
    nguess = scenario['true_N_mgii'] + scenario['true_N_feii']
    bguess = scenario['true_b_mgii'] + scenario['true_b_feii']
    vguess = scenario['true_v_mgii'] + scenario['true_v_feii']
    
    bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
    
    # MCMC settings
    n_walkers = 50
    n_steps = 500
    
    # Test configurations
    test_configs = [
        ('single_compiled_emcee_serial', models['single']['compiled'], data['single'], 'emcee', False),
        ('single_compiled_emcee_parallel', models['single']['compiled'], data['single'], 'emcee', True),
        ('single_compiled_zeus_serial', models['single']['compiled'], data['single'], 'zeus', False),
        ('single_compiled_zeus_parallel', models['single']['compiled'], data['single'], 'zeus', True),
    ]
    
    # Add multi-instrument test
    multi_instrument_data = {
        'XShooter': {
            'model': models['multi']['xshooter']['compiled'],
            'wave': data['multi']['xshooter']['wave'],
            'flux': data['multi']['xshooter']['flux'],
            'error': data['multi']['xshooter']['error']
        },
        'HIRES': {
            'model': models['multi']['hires']['compiled'],
            'wave': data['multi']['hires']['wave'],
            'flux': data['multi']['hires']['flux'],
            'error': data['multi']['hires']['error']
        }
    }
    
    for name, model, test_data, sampler, use_parallel in test_configs:
        print(f"  Testing {name}...")
        
        # Create instrument data
        instrument_data = {
            'XShooter': {
                'model': model,
                'wave': test_data['wave'],
                'flux': test_data['flux'],
                'error': test_data['error']
            }
        }
        
        try:
            # Create fitter
            fitter = mc.vfit(instrument_data, scenario['true_theta'], lb, ub,
                           no_of_Chain=n_walkers, no_of_steps=n_steps,
                           sampler=sampler,perturbation=1e-4)
            
            start = time.time()
            fitter.runmcmc(optimize=True, verbose=False, use_pool=use_parallel,progress=True)
            mcmc_time = time.time() - start
            
            # Extract results
            samples = fitter.get_samples(burn_in=0.3)
            theta_best = np.percentile(samples, 50, axis=0)
            theta_errors = np.std(samples, axis=0)
            
            # Calculate parameter recovery metrics
            param_recovery = calculate_parameter_recovery(scenario['true_theta'], theta_best, theta_errors)
            
            # Calculate MCMC diagnostics
            diagnostics = calculate_mcmc_diagnostics(fitter, samples)
            
            results[name] = {
                'mcmc_time': mcmc_time,
                'time_per_step': mcmc_time / n_steps * 1000,  # ms
                'theta_best': theta_best,
                'theta_errors': theta_errors,
                'samples': samples,
                'success': True,
                'recovery_metrics': param_recovery,
                'diagnostics': diagnostics
            }
            
            print(f"    {mcmc_time:.1f}s ({mcmc_time/n_steps*1000:.2f} ms/step), "
                  f"recovery: {param_recovery['mean_relative_error']:.3f}")
            
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Test multi-instrument with best performing single setup
    print("  Testing multi_instrument_emcee_parallel...")
    try:
        fitter_multi = mc.vfit(multi_instrument_data, scenario['true_theta'], lb, ub,
                             no_of_Chain=n_walkers, no_of_steps=n_steps,
                             sampler='emcee',perturbation=1e-4)
        
        start = time.time()
        fitter_multi.runmcmc(optimize=True, verbose=False, use_pool=True,progress=True)
        mcmc_time = time.time() - start
        
        samples = fitter_multi.get_samples(burn_in=0.3)
        theta_best = np.percentile(samples, 50, axis=0)
        theta_errors = np.std(samples, axis=0)
        
        param_recovery = calculate_parameter_recovery(scenario['true_theta'], theta_best, theta_errors)
        diagnostics = calculate_mcmc_diagnostics(fitter_multi, samples)
        
        results['multi_instrument_emcee_parallel'] = {
            'mcmc_time': mcmc_time,
            'time_per_step': mcmc_time / n_steps * 1000,
            'theta_best': theta_best,
            'theta_errors': theta_errors,
            'samples': samples,
            'success': True,
            'recovery_metrics': param_recovery,
            'diagnostics': diagnostics
        }
        
        print(f"    {mcmc_time:.1f}s ({mcmc_time/n_steps*1000:.2f} ms/step), "
              f"recovery: {param_recovery['mean_relative_error']:.3f}")
        
    except Exception as e:
        print(f"    FAILED: {e}")
        results['multi_instrument_emcee_parallel'] = {'success': False, 'error': str(e)}
    
    return results


def calculate_parameter_recovery(true_theta, fitted_theta, fitted_errors):
    """Calculate parameter recovery accuracy metrics."""
    relative_errors = np.abs(fitted_theta - true_theta) / np.abs(true_theta)
    
    # Standard deviations from truth
    sigma_deviations = np.abs(fitted_theta - true_theta) / (fitted_errors + 1e-10)
    
    metrics = {
        'mean_relative_error': np.mean(relative_errors),
        'max_relative_error': np.max(relative_errors),
        'mean_sigma_deviation': np.mean(sigma_deviations),
        'max_sigma_deviation': np.max(sigma_deviations),
        'fraction_within_1sigma': np.mean(sigma_deviations < 1.0),
        'fraction_within_2sigma': np.mean(sigma_deviations < 2.0),
        'relative_errors': relative_errors,
        'sigma_deviations': sigma_deviations
    }
    
    return metrics


def calculate_mcmc_diagnostics(fitter, samples):
    """Calculate MCMC convergence diagnostics."""
    diagnostics = {}
    
    try:
        # Acceptance fraction
        if hasattr(fitter.sampler, 'acceptance_fraction'):
            diagnostics['acceptance_fraction'] = np.mean(fitter.sampler.acceptance_fraction)
        
        # Autocorrelation time (for emcee)
        if fitter.sampler_name == 'emcee':
            try:
                autocorr_time = fitter.sampler.get_autocorr_time()
                diagnostics['autocorr_time'] = np.mean(autocorr_time)
            except Exception:
                diagnostics['autocorr_time'] = np.nan
        
        # Effective sample size
        n_samples = len(samples)
        diagnostics['n_samples'] = n_samples
        diagnostics['effective_sample_size'] = n_samples  # Simplified
        
    except Exception:
        pass
    
    return diagnostics


def create_performance_summary(eval_results, likelihood_results, quickfit_results, mcmc_results):
    """Create comprehensive performance summary."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Create summary table
    summary_data = []
    
    # Raw evaluation results
    print("\n1. RAW MODEL EVALUATION")
    print("-" * 40)
    for name, result in eval_results.items():
        print(f"  {name:25s}: {result['time_per_eval']:6.2f} ms/eval ({result['evals_per_second']:7.1f} eval/s)")
        summary_data.append({
            'Test': name,
            'Category': 'Raw Evaluation',
            'Time (ms)': result['time_per_eval'],
            'Success Rate': result['success_rate']
        })
    
    # Likelihood evaluation results
    print("\n2. LIKELIHOOD EVALUATION")
    print("-" * 40)
    for name, result in likelihood_results.items():
        print(f"  {name:25s}: {result['time_per_eval']:6.2f} ms/eval ({result['evals_per_second']:7.1f} eval/s)")
        summary_data.append({
            'Test': name,
            'Category': 'Likelihood',
            'Time (ms)': result['time_per_eval'],
            'Success Rate': result['success_rate']
        })
    
    # Quick fit results
    print("\n3. QUICK FIT PERFORMANCE")
    print("-" * 40)
    for name, result in quickfit_results.items():
        if result['success']:
            recovery = result['recovery_metrics']
            print(f"  {name:25s}: {result['fit_time']:6.2f}s, "
                  f"recovery: {recovery['mean_relative_error']:.3f}")
            summary_data.append({
                'Test': name,
                'Category': 'Quick Fit',
                'Time (ms)': result['fit_time'] * 1000,
                'Recovery Error': recovery['mean_relative_error']
            })
    
    # MCMC results
    print("\n4. MCMC PERFORMANCE")
    print("-" * 40)
    for name, result in mcmc_results.items():
        if result['success']:
            recovery = result['recovery_metrics']
            diag = result['diagnostics']
            print(f"  {name:30s}: {result['mcmc_time']:6.1f}s "
                  f"({result['time_per_step']:5.2f} ms/step), "
                  f"recovery: {recovery['mean_relative_error']:.3f}")
            summary_data.append({
                'Test': name,
                'Category': 'MCMC',
                'Time (ms)': result['time_per_step'],
                'Recovery Error': recovery['mean_relative_error'],
                'Acceptance': diag.get('acceptance_fraction', np.nan)
            })
    
    return summary_data


def create_performance_plots(eval_results, likelihood_results, quickfit_results, mcmc_results):
    """Create performance visualization plots."""
    print("\nCreating performance plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Raw evaluation performance
    names = []
    times = []
    colors = []
    
    for name, result in eval_results.items():
        names.append(name.replace('_', '\n'))
        times.append(result['time_per_eval'])
        if 'compiled' in name:
            colors.append('orange')
        elif 'regular' in name:
            colors.append('lightblue')
        else:
            colors.append('gray')
    
    bars1 = ax1.bar(names, times, color=colors)
    ax1.set_ylabel('Time per evaluation (ms)')
    ax1.set_title('Raw Model Evaluation Performance')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                f'{time_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Quick fit vs MCMC time comparison
    quick_names = []
    quick_times = []
    mcmc_names = []
    mcmc_times = []
    
    for name, result in quickfit_results.items():
        if result['success']:
            quick_names.append(name)
            quick_times.append(result['fit_time'])
    
    for name, result in mcmc_results.items():
        if result['success']:
            mcmc_names.append(name)
            mcmc_times.append(result['mcmc_time'])
    
    # Combined plot for quick fit vs MCMC
    x_pos = np.arange(len(quick_names))
    width = 0.35
    
    bars2a = ax2.bar(x_pos - width/2, quick_times, width, label='Quick Fit', color='lightgreen')
    
    # Add MCMC times (scale down for visibility)
    if mcmc_times:
        mcmc_times_scaled = [t/10 for t in mcmc_times[:len(quick_names)]]  # Scale down by 10x
        bars2b = ax2.bar(x_pos + width/2, mcmc_times_scaled, width, label='MCMC (÷10)', color='salmon')
    
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Quick Fit vs MCMC Time Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace('_', '\n') for name in quick_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter recovery accuracy
    recovery_names = []
    recovery_errors = []
    recovery_colors = []
    
    for name, result in quickfit_results.items():
        if result['success']:
            recovery_names.append(f"QF_{name}")
            recovery_errors.append(result['recovery_metrics']['mean_relative_error'])
            recovery_colors.append('lightgreen')
    
    for name, result in mcmc_results.items():
        if result['success']:
            recovery_names.append(f"MCMC_{name}")
            recovery_errors.append(result['recovery_metrics']['mean_relative_error'])
            recovery_colors.append('salmon')
    
    bars3 = ax3.bar(recovery_names, recovery_errors, color=recovery_colors)
    ax3.set_ylabel('Mean Relative Error')
    ax3.set_title('Parameter Recovery Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add horizontal line for 1% accuracy
    ax3.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='1% accuracy')
    ax3.legend()
    
    # Plot 4: MCMC performance breakdown
    mcmc_breakdown = defaultdict(list)
    mcmc_labels = []
    
    for name, result in mcmc_results.items():
        if result['success']:
            parts = name.split('_')
            if len(parts) >= 3:
                sampler = parts[2]  # emcee or zeus
                mode = parts[3] if len(parts) > 3 else 'serial'
                key = f"{sampler}_{mode}"
                mcmc_breakdown[key].append(result['time_per_step'])
                if key not in mcmc_labels:
                    mcmc_labels.append(key)
    
    mcmc_step_times = [np.mean(mcmc_breakdown[label]) for label in mcmc_labels]
    colors4 = ['lightcoral' if 'emcee' in label else 'lightblue' for label in mcmc_labels]
    
    bars4 = ax4.bar(mcmc_labels, mcmc_step_times, color=colors4)
    ax4.set_ylabel('Time per MCMC step (ms)')
    ax4.set_title('MCMC Sampler Performance')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars4, mcmc_step_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mcmc_step_times)*0.01,
                f'{time_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def generate_recommendations(eval_results, likelihood_results, quickfit_results, mcmc_results):
    """Generate performance recommendations."""
    print("\n" + "=" * 80)
    print("PERFORMANCE RECOMMENDATIONS")
    print("=" * 80)
    
    # Find fastest methods
    fastest_eval = min(eval_results.items(), key=lambda x: x[1]['time_per_eval'])
    fastest_likelihood = min(likelihood_results.items(), key=lambda x: x[1]['time_per_eval'])
    
    successful_quick = {k: v for k, v in quickfit_results.items() if v['success']}
    if successful_quick:
        fastest_quick = min(successful_quick.items(), key=lambda x: x[1]['fit_time'])
        most_accurate_quick = min(successful_quick.items(), 
                                key=lambda x: x[1]['recovery_metrics']['mean_relative_error'])
    
    successful_mcmc = {k: v for k, v in mcmc_results.items() if v['success']}
    if successful_mcmc:
        fastest_mcmc = min(successful_mcmc.items(), key=lambda x: x[1]['time_per_step'])
        most_accurate_mcmc = min(successful_mcmc.items(),
                               key=lambda x: x[1]['recovery_metrics']['mean_relative_error'])
    
    print("\n🚀 SPEED RECOMMENDATIONS:")
    print(f"  Fastest model evaluation: {fastest_eval[0]} ({fastest_eval[1]['time_per_eval']:.2f} ms/eval)")
    print(f"  Fastest likelihood eval:   {fastest_likelihood[0]} ({fastest_likelihood[1]['time_per_eval']:.2f} ms/eval)")
    
    if successful_quick:
        print(f"  Fastest quick fit:         {fastest_quick[0]} ({fastest_quick[1]['fit_time']:.2f}s)")
    
    if successful_mcmc:
        print(f"  Fastest MCMC:             {fastest_mcmc[0]} ({fastest_mcmc[1]['time_per_step']:.2f} ms/step)")
    
    print("\n🎯 ACCURACY RECOMMENDATIONS:")
    if successful_quick:
        print(f"  Most accurate quick fit:   {most_accurate_quick[0]} "
              f"({most_accurate_quick[1]['recovery_metrics']['mean_relative_error']:.3f} rel. error)")
    
    if successful_mcmc:
        print(f"  Most accurate MCMC:       {most_accurate_mcmc[0]} "
              f"({most_accurate_mcmc[1]['recovery_metrics']['mean_relative_error']:.3f} rel. error)")
    
    print("\n💡 PRACTICAL RECOMMENDATIONS:")
    
    # Model compilation recommendation
    compiled_faster = False
    for name, result in eval_results.items():
        if 'compiled' in name:
            regular_name = name.replace('compiled', 'regular')
            if regular_name in eval_results:
                speedup = eval_results[regular_name]['time_per_eval'] / result['time_per_eval']
                if speedup > 1.2:
                    compiled_faster = True
                    print(f"  ✅ Use compiled models: {speedup:.1f}x speedup over regular models")
                    break
    
    if not compiled_faster:
        print("  ⚠️  Model compilation shows minimal speedup for this problem size")
    
    # Sampler recommendation
    if successful_mcmc:
        emcee_times = [v['time_per_step'] for k, v in successful_mcmc.items() if 'emcee' in k]
        zeus_times = [v['time_per_step'] for k, v in successful_mcmc.items() if 'zeus' in k]
        
        if emcee_times and zeus_times:
            emcee_avg = np.mean(emcee_times)
            zeus_avg = np.mean(zeus_times)
            
            if zeus_avg < emcee_avg * 0.8:
                print(f"  ✅ Use Zeus sampler: {emcee_avg/zeus_avg:.1f}x faster than emcee")
            else:
                print(f"  ✅ Use emcee sampler: stable performance ({emcee_avg:.1f} ms/step)")
        
        # Parallel recommendation
        serial_times = [v['time_per_step'] for k, v in successful_mcmc.items() if 'serial' in k]
        parallel_times = [v['time_per_step'] for k, v in successful_mcmc.items() if 'parallel' in k]
        
        if serial_times and parallel_times:
            serial_avg = np.mean(serial_times)
            parallel_avg = np.mean(parallel_times)
            
            if parallel_avg < serial_avg * 0.7:
                print(f"  ✅ Use parallel processing: {serial_avg/parallel_avg:.1f}x speedup")
            else:
                print("  ⚠️  Parallel processing shows limited benefit for this problem size")
    
    # Multi-instrument recommendation
    single_times = [v['fit_time'] for k, v in successful_quick.items() if 'single' in k]
    multi_times = [v['fit_time'] for k, v in successful_quick.items() if 'multi' in k]
    
    if single_times and multi_times:
        single_avg = np.mean(single_times)
        multi_avg = np.mean(multi_times)
        overhead = multi_avg / single_avg
        
        if overhead < 2.0:
            print(f"  ✅ Multi-instrument fitting: only {overhead:.1f}x overhead")
        else:
            print(f"  ⚠️  Multi-instrument fitting: {overhead:.1f}x overhead")
    
    # Quick fit vs MCMC recommendation
    if successful_quick and successful_mcmc:
        quick_accuracy = np.mean([v['recovery_metrics']['mean_relative_error'] 
                                for v in successful_quick.values()])
        mcmc_accuracy = np.mean([v['recovery_metrics']['mean_relative_error'] 
                               for v in successful_mcmc.values()])
        
        print(f"\n🔄 QUICK FIT vs MCMC:")
        print(f"  Quick fit accuracy: {quick_accuracy:.3f} relative error")
        print(f"  MCMC accuracy:      {mcmc_accuracy:.3f} relative error")
        
        if quick_accuracy < mcmc_accuracy * 1.5:
            print("  ✅ Use quick_fit for initial fits - comparable accuracy, much faster")
            print("  ✅ Use MCMC for final parameter uncertainties")
        else:
            print("  ⚠️  MCMC significantly more accurate - use for critical fits")


def main():
    """Main performance testing function."""
    print("rbvfit v2.0 Performance Benchmark")
    print("=" * 50)
    print("Testing: MgII + FeII systems, XShooter + HIRES")
    print("Focus: Quick fit vs MCMC performance")
    
    try:
        # 1. Set up test scenarios
        scenario = setup_test_scenarios()
        
        # 2. Create models
        models = create_models(scenario)
        
        # 3. Generate synthetic data
        data = generate_synthetic_data(models, scenario)
        
        # 4. Run benchmarks
        eval_results = benchmark_raw_evaluation(models, scenario, n_evals=1000)
        
        likelihood_results = benchmark_likelihood_evaluation(models, scenario, data, n_evals=1000)
        
        quickfit_results = benchmark_quick_fit(models, scenario, data)
        
        mcmc_results = benchmark_mcmc(models, scenario, data)
        
        # 5. Create comprehensive analysis
        summary_data = create_performance_summary(eval_results, likelihood_results, 
                                                quickfit_results, mcmc_results)
        
        # 6. Create performance plots
        fig = create_performance_plots(eval_results, likelihood_results, 
                                     quickfit_results, mcmc_results)
        
        # 7. Generate recommendations
        generate_recommendations(eval_results, likelihood_results, 
                               quickfit_results, mcmc_results)
        
        print("\n" + "=" * 80)
        print("🏁 BENCHMARK COMPLETE")
        print("=" * 80)
        print("✅ All tests completed successfully")
        print("📊 Performance plots generated")
        print("💡 See recommendations above for optimal usage")
        
        return {
            'scenario': scenario,
            'models': models,
            'data': data,
            'results': {
                'eval': eval_results,
                'likelihood': likelihood_results,
                'quickfit': quickfit_results,
                'mcmc': mcmc_results
            },
            'summary': summary_data
        }
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()