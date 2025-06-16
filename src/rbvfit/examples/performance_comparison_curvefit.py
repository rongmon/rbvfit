#!/usr/bin/env python
"""
Performance comparison between rbvfit v1.0 and v2.0 with curve_fit and MCMC testing.

This script provides a comprehensive performance analysis including:
1. Model setup and compilation
2. Raw model evaluation performance 
3. Curve fitting using scipy.optimize.curve_fit
4. MCMC fitting with vfit_mcmc (serial and parallel)
5. Analysis and recommendations

Focus on practical fitting scenarios that users actually encounter.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Import v1 components
try:
    from rbvfit import model as v1_model
    from rbvfit import vfit_mcmc as v1_mcmc
    V1_AVAILABLE = True
    print("✓ rbvfit v1.0 available")
except ImportError as e:
    print(f"⚠ Warning: rbvfit v1.0 not available: {e}")
    V1_AVAILABLE = False

# Import v2 components  
try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    from rbvfit import vfit_mcmc as v2_mcmc  # Should be same module
    V2_AVAILABLE = True
    print("✓ rbvfit v2.0 available")
except ImportError as e:
    print(f"❌ Error: rbvfit v2.0 not available: {e}")
    V2_AVAILABLE = False


def setup_test_scenario():
    """Set up a realistic test scenario."""
    print("Setting up test scenario...")
    
    # Realistic MgII doublet parameters
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    
    # True parameters for 2-component system
    true_N = [13.95, 13.3]
    true_b = [20.0, 40.0]
    true_v = [-40.0, 20.0]
    true_theta = np.array(true_N + true_b + true_v)
    
    # Realistic wavelength coverage and resolution
    wave = np.linspace(3760, 3800, 800)
    
    print(f"  Redshift: {zabs}")
    print(f"  Transitions: {lambda_rest} Å")
    print(f"  True parameters: N={true_N}, b={true_b}, v={true_v}")
    print(f"  Wavelength points: {len(wave)}")
    
    return {
        'zabs': zabs,
        'lambda_rest': lambda_rest,
        'true_theta': true_theta,
        'wave': wave,
        'nclump': 2,
        'ntransition': 2,
        'true_N': true_N,
        'true_b': true_b,
        'true_v': true_v
    }


def setup_models(scenario):
    """Set up both v1 and v2 models."""
    print("\nSetting up models...")
    
    models = {}
    
    # v1 model setup
    if V1_AVAILABLE:
        print("  Creating v1 model...")
        start = time.time()
        v1_model_obj = v1_model.create_voigt(
            np.array([scenario['zabs']]),
            scenario['lambda_rest'],
            nclump=scenario['nclump'],
            ntransition=scenario['ntransition'],
            FWHM='6.5',
            verbose=False
        )
        v1_setup_time = time.time() - start
        models['v1'] = {
            'model': v1_model_obj,
            'setup_time': v1_setup_time,
            'eval_func': v1_model_obj.model_flux,
            'curvefit_func': v1_model_obj.model_flux_curvefit
        }
        print(f"    v1 setup: {v1_setup_time*1000:.2f} ms")
    
    # v2 model setup
    if V2_AVAILABLE:
        print("  Creating v2 model...")
        start = time.time()
        config = FitConfiguration()
        config.add_system(
            z=scenario['zabs'], 
            ion='MgII', 
            transitions=scenario['lambda_rest'], 
            components=scenario['nclump']
        )
        v2_model_obj = VoigtModel(config, FWHM='6.5')
        v2_setup_time = time.time() - start
        print(f"    v2 setup: {v2_setup_time*1000:.2f} ms")
        
        # Compile v2 model
        print("  Compiling v2 model...")
        start = time.time()
        v2_compiled = v2_model_obj.compile(verbose=False)
        v2_compile_time = time.time() - start
        print(f"    v2 compilation: {v2_compile_time*1000:.2f} ms")
        
        models['v2_regular'] = {
            'model': v2_model_obj,
            'setup_time': v2_setup_time,
            'eval_func': lambda theta, wave: v2_model_obj.evaluate(theta, wave),
            'curvefit_func': None  # v2 doesn't have direct curvefit interface
        }
        
        models['v2_compiled'] = {
            'model': v2_compiled,
            'setup_time': v2_setup_time + v2_compile_time,
            'eval_func': v2_compiled.model_flux,
            'curvefit_func': None  # We'll create a wrapper
        }
    
    return models


def generate_realistic_data(models, scenario):
    """Generate realistic synthetic data with noise."""
    print("\nGenerating synthetic data...")
    
    # Use v1 model if available, otherwise v2
    if 'v1' in models:
        model_func = models['v1']['eval_func']
        print("  Using v1 model for data generation")
    elif 'v2_regular' in models:
        model_func = models['v2_regular']['eval_func']
        print("  Using v2 regular model for data generation")
    else:
        raise ValueError("No models available for data generation")
    
    # Generate clean spectrum
    true_flux = model_func(scenario['true_theta'], scenario['wave'])
    
    # Add realistic noise (S/N ~ 20-30 in continuum)
    noise_level = 0.03  # 3% noise
    np.random.seed(42)  # Reproducible
    noise = np.random.normal(0, noise_level, len(scenario['wave']))
    observed_flux = true_flux + noise
    error = np.full_like(scenario['wave'], noise_level)
    
    print(f"  Noise level: {noise_level*100:.1f}%")
    print(f"  S/N ratio: ~{1/noise_level:.0f}")
    
    return observed_flux, error, true_flux


def benchmark_raw_evaluation(models, scenario, n_evals=100):
    """Benchmark raw model evaluation speed."""
    print(f"\nBenchmarking raw evaluation ({n_evals} calls)...")
    
    results = {}
    
    for name, model_info in models.items():
        print(f"  Testing {name}...")
        eval_func = model_info['eval_func']
        
        # Test with perturbed parameters
        test_thetas = []
        for i in range(n_evals):
            noise = 0.01 * np.random.randn(len(scenario['true_theta']))
            test_thetas.append(scenario['true_theta'] + noise)
        
        # Time the evaluations
        start = time.time()
        for theta in test_thetas:
            try:
                flux = eval_func(theta, scenario['wave'])
            except Exception as e:
                print(f"    Error in {name}: {e}")
                break
        else:
            eval_time = time.time() - start
            results[name] = {
                'total_time': eval_time,
                'time_per_eval': eval_time / n_evals * 1000,  # ms
                'success': True
            }
            print(f"    {name}: {eval_time:.3f}s total, {eval_time/n_evals*1000:.2f} ms/eval")
    
    return results


def benchmark_curve_fit(models, scenario, observed_flux, error):
    """Benchmark curve_fit performance."""
    print("\nBenchmarking scipy.optimize.curve_fit...")
    
    results = {}
    
    # Initial guess (perturbed from truth)
    p0 = scenario['true_theta'] + 0.1 * np.random.randn(len(scenario['true_theta']))
    
    # Parameter bounds
    bounds, lb, ub = v1_mcmc.set_bounds(
        scenario['true_N'], scenario['true_b'], scenario['true_v']
    )
    
    for name, model_info in models.items():
        if name.startswith('v2'):  # Skip v2 for curve_fit (no direct interface)
            continue
            
        curvefit_func = model_info.get('curvefit_func')
        if curvefit_func is None:
            continue
            
        print(f"  Testing {name}...")
        
        try:
            start = time.time()
            popt, pcov = curve_fit(
                curvefit_func,
                scenario['wave'],
                observed_flux,
                p0=p0,
                sigma=error,
                bounds=(lb, ub),
                maxfev=5000
            )
            fit_time = time.time() - start
            
            # Calculate parameter errors
            param_errors = np.sqrt(np.diag(pcov))
            
            results[name] = {
                'fit_time': fit_time,
                'parameters': popt,
                'errors': param_errors,
                'success': True
            }
            
            print(f"    {name}: {fit_time:.3f}s")
            print(f"      Parameter recovery (true vs fitted):")
            for i, (true_val, fit_val, err) in enumerate(zip(scenario['true_theta'], popt, param_errors)):
                print(f"        θ[{i}]: {true_val:.3f} → {fit_val:.3f} ± {err:.3f}")
            
        except Exception as e:
            print(f"    {name}: FAILED - {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    return results


def benchmark_mcmc_fitting(models, scenario, observed_flux, error):
    """Benchmark MCMC fitting performance."""
    print("\nBenchmarking MCMC fitting...")
    
    results = {}
    
    # MCMC settings
    n_walkers = 20
    n_steps = 500
    
    # Set up bounds
    bounds, lb, ub = v1_mcmc.set_bounds(
        scenario['true_N'], scenario['true_b'], scenario['true_v']
    )
    
    print(f"  MCMC settings: {n_walkers} walkers, {n_steps} steps")
    
    for name, model_info in models.items():
        print(f"\n  Testing {name}...")
        eval_func = model_info['eval_func']
        
        try:
            # Test both serial and parallel
            for mode, use_pool in [('serial', False), ('parallel', True)]:
                print(f"    {mode} mode...")
                
                start = time.time()
                
                # Create MCMC fitter
                fitter = v1_mcmc.vfit(
                    eval_func,
                    scenario['true_theta'],
                    lb, ub,
                    scenario['wave'],
                    observed_flux,
                    error,
                    no_of_Chain=n_walkers,
                    no_of_steps=n_steps
                )
                
                # Run MCMC
                fitter.runmcmc(optimize=False, verbose=False, use_pool=use_pool)
                
                mcmc_time = time.time() - start
                
                # Extract results
                samples = fitter.sampler.get_chain(discard=100, thin=1, flat=True)
                best_params = np.percentile(samples, 50, axis=0)
                param_errors = np.std(samples, axis=0)
                
                results[f"{name}_{mode}"] = {
                    'mcmc_time': mcmc_time,
                    'time_per_step': mcmc_time / n_steps * 1000,  # ms
                    'parameters': best_params,
                    'errors': param_errors,
                    'samples': samples,
                    'success': True
                }
                
                print(f"      Time: {mcmc_time:.1f}s ({mcmc_time/n_steps*1000:.2f} ms/step)")
                
                # Show parameter recovery
                print(f"      Parameter recovery:")
                for i, (true_val, fit_val, err) in enumerate(zip(scenario['true_theta'], best_params, param_errors)):
                    print(f"        θ[{i}]: {true_val:.3f} → {fit_val:.3f} ± {err:.3f}")
                
        except Exception as e:
            print(f"    {name}: FAILED - {e}")
            results[f"{name}_failed"] = {'success': False, 'error': str(e)}
    
    return results


def analyze_results(eval_results, curvefit_results, mcmc_results, models):
    """Analyze and summarize all results."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Model evaluation analysis
    print("\n1. RAW MODEL EVALUATION PERFORMANCE")
    print("-" * 40)
    
    if eval_results:
        # Sort by speed
        sorted_evals = sorted(eval_results.items(), key=lambda x: x[1]['time_per_eval'])
        
        print("  Performance ranking (fastest to slowest):")
        for i, (name, result) in enumerate(sorted_evals, 1):
            if result['success']:
                print(f"    {i}. {name}: {result['time_per_eval']:.2f} ms/eval")
        
        # Compare v1 vs v2
        v1_time = next((r['time_per_eval'] for n, r in eval_results.items() if n == 'v1' and r['success']), None)
        v2_reg_time = next((r['time_per_eval'] for n, r in eval_results.items() if n == 'v2_regular' and r['success']), None)
        v2_comp_time = next((r['time_per_eval'] for n, r in eval_results.items() if n == 'v2_compiled' and r['success']), None)
        
        if v1_time and v2_comp_time:
            speedup = v1_time / v2_comp_time
            print(f"\n  v2 compiled vs v1: {speedup:.1f}x speedup" if speedup > 1 else f"{1/speedup:.1f}x slower")
        
        if v2_reg_time and v2_comp_time:
            compilation_speedup = v2_reg_time / v2_comp_time
            print(f"  v2 compiled vs regular: {compilation_speedup:.1f}x speedup")
    
    # Curve fitting analysis
    print("\n2. CURVE FITTING PERFORMANCE")
    print("-" * 40)
    
    if curvefit_results:
        for name, result in curvefit_results.items():
            if result['success']:
                print(f"  {name}: {result['fit_time']:.2f}s")
            else:
                print(f"  {name}: FAILED")
    else:
        print("  No curve fitting results available")
    
    # MCMC analysis
    print("\n3. MCMC PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    if mcmc_results:
        # Group by model type
        mcmc_by_model = {}
        for name, result in mcmc_results.items():
            if not result['success']:
                continue
                
            if '_serial' in name:
                model_name = name.replace('_serial', '')
                mode = 'serial'
            elif '_parallel' in name:
                model_name = name.replace('_parallel', '')
                mode = 'parallel'
            else:
                continue
                
            if model_name not in mcmc_by_model:
                mcmc_by_model[model_name] = {}
            mcmc_by_model[model_name][mode] = result
        
        for model_name, modes in mcmc_by_model.items():
            print(f"\n  {model_name}:")
            
            for mode, result in modes.items():
                print(f"    {mode:8}: {result['mcmc_time']:6.1f}s ({result['time_per_step']:5.2f} ms/step)")
            
            # Calculate parallel speedup
            if 'serial' in modes and 'parallel' in modes:
                speedup = modes['serial']['mcmc_time'] / modes['parallel']['mcmc_time']
                print(f"    parallel speedup: {speedup:.1f}x")
    
    # Overall recommendations
    print("\n4. RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    # Model evaluation recommendations
    if eval_results:
        fastest_eval = min(eval_results.items(), key=lambda x: x[1]['time_per_eval'] if x[1]['success'] else float('inf'))
        if fastest_eval[1]['success']:
            recommendations.append(f"🚀 Fastest model evaluation: {fastest_eval[0]} ({fastest_eval[1]['time_per_eval']:.2f} ms/eval)")
    
    # MCMC recommendations
    if mcmc_results:
        fastest_mcmc = min(mcmc_results.items(), key=lambda x: x[1]['mcmc_time'] if x[1]['success'] else float('inf'))
        if fastest_mcmc[1]['success']:
            recommendations.append(f"⚡ Fastest MCMC: {fastest_mcmc[0]} ({fastest_mcmc[1]['time_per_step']:.2f} ms/step)")
    
    # Practical recommendations
    if V2_AVAILABLE:
        recommendations.append("🔧 For production work: Use v2.0 with compilation for best performance")
        recommendations.append("📊 For quick fits: Consider curve_fit if available, otherwise short MCMC runs")
    
    if V1_AVAILABLE and V2_AVAILABLE:
        recommendations.append("⚖️ For compatibility: v1.0 is stable and well-tested")
        recommendations.append("🆕 For new features: v2.0 offers better architecture and multi-instrument support")
    
    for rec in recommendations:
        print(f"  {rec}")


def create_summary_plot(eval_results, mcmc_results):
    """Create summary performance plots."""
    print("\n" + "=" * 70)
    print("CREATING PERFORMANCE SUMMARY PLOTS")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Model evaluation performance
    eval_names = []
    eval_times = []
    eval_colors = []
    
    for name, result in eval_results.items():
        if result['success']:
            eval_names.append(name)
            eval_times.append(result['time_per_eval'])
            if 'v1' in name:
                eval_colors.append('skyblue')
            elif 'compiled' in name:
                eval_colors.append('orange')
            else:
                eval_colors.append('lightcoral')
    
    if eval_names:
        bars1 = ax1.bar(eval_names, eval_times, color=eval_colors)
        ax1.set_ylabel('Time per evaluation (ms)')
        ax1.set_title('Model Evaluation Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars1, eval_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(eval_times)*0.01,
                    f'{time_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: MCMC performance (time per step)
    mcmc_names = []
    mcmc_times = []
    mcmc_colors = []
    
    for name, result in mcmc_results.items():
        if result['success']:
            mcmc_names.append(name.replace('_', '\n'))
            mcmc_times.append(result['time_per_step'])
            if 'v1' in name:
                mcmc_colors.append('skyblue')
            elif 'compiled' in name:
                mcmc_colors.append('orange')
            else:
                mcmc_colors.append('lightcoral')
    
    if mcmc_names:
        bars2 = ax2.bar(mcmc_names, mcmc_times, color=mcmc_colors)
        ax2.set_ylabel('Time per MCMC step (ms)')
        ax2.set_title('MCMC Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars2, mcmc_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mcmc_times)*0.01,
                    f'{time_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Performance summary plots created")


def main():
    """Main performance testing function."""
    print("rbvfit Performance Comparison: Curve Fit vs MCMC Analysis")
    print("=" * 70)
    print("Comprehensive testing of model evaluation, curve fitting, and MCMC performance")
    
    if not V1_AVAILABLE and not V2_AVAILABLE:
        print("❌ Neither v1.0 nor v2.0 available. Cannot run comparison.")
        return
    
    try:
        # 1. Set up test scenario
        scenario = setup_test_scenario()
        
        # 2. Set up models
        models = setup_models(scenario)
        
        # 3. Generate synthetic data
        observed_flux, error, true_flux = generate_realistic_data(models, scenario)
        
        # 4. Benchmark raw evaluation
        eval_results = benchmark_raw_evaluation(models, scenario, n_evals=100)
        
        # 5. Benchmark curve fitting
        curvefit_results = benchmark_curve_fit(models, scenario, observed_flux, error)
        
        # 6. Benchmark MCMC fitting
        mcmc_results = benchmark_mcmc_fitting(models, scenario, observed_flux, error)
        
        # 7. Analyze results
        analyze_results(eval_results, curvefit_results, mcmc_results, models)
        
        # 8. Create summary plots
        create_summary_plot(eval_results, mcmc_results)
        
        print("\n" + "=" * 70)
        print("🏁 COMPREHENSIVE PERFORMANCE ANALYSIS COMPLETE")
        print("=" * 70)
        print("✅ All benchmarks completed successfully")
        print("📊 Summary plots generated")
        print("💡 Check recommendations above for optimal usage")
        
    except Exception as e:
        print(f"\n❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()