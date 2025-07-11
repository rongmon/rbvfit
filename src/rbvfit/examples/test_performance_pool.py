#!/usr/bin/env python3
"""
Performance test to determine serial vs parallel crossover point for vfit_mcmc.py

This script tests three phases:
1. Single instrument complexity scaling
2. Multi-instrument scaling  
3. Parameter space scaling

Measures wall-clock time to find when parallel becomes beneficial.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import rbvfit components
from rbvfit.vfit_mcmc import vfit
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.fit_configuration import FitConfiguration

# Ion wavelengths (rest frame in Angstrom)
ION_WAVELENGTHS = {
    'HI_Lya': 1215.67,
    'HI_Lyb': 1025.72,
    'OVI': 1031.93,
    'CIV': 1548.20,
    'SiIV': 1393.76,
    'CII': 1334.53
}

def check_wavelength_coverage(ion_waves, redshift, obs_min=1000, obs_max=1800):
    """Check if ion lines fall within observed wavelength range."""
    obs_waves = [wave * (1 + redshift) for wave in ion_waves]
    covered = [obs_min <= wave <= obs_max for wave in obs_waves]
    return all(covered), obs_waves

def create_synthetic_data(n_lines, n_components, instruments=['SINGLE'], 
                         wave_range=(1000, 1800), n_points=1000):
    """Create synthetic spectroscopic data with controlled complexity."""
    
    # Choose redshift to ensure wavelength coverage
    ion_list = list(ION_WAVELENGTHS.keys())[:n_lines]
    rest_waves = [ION_WAVELENGTHS[ion] for ion in ion_list]
    
    # Find redshift that works
    for z in np.arange(0.15, 0.45, 0.05):
        covered, obs_waves = check_wavelength_coverage(rest_waves, z, *wave_range)
        if covered:
            redshift = z
            break
    else:
        raise ValueError(f"No redshift found for {n_lines} lines in range {wave_range}")
    
    # Create wavelength grid
    wave = np.linspace(wave_range[0], wave_range[1], n_points)
    
    # Create FitConfiguration
    config = FitConfiguration()
    
    # Group transitions by ion to avoid duplicates
    ion_transitions = {}
    for ion_key in ion_list:
        ion_name = ion_key.split('_')[0] if '_' in ion_key else ion_key
        wavelength = ION_WAVELENGTHS[ion_key]
        
        if ion_name not in ion_transitions:
            ion_transitions[ion_name] = []
        ion_transitions[ion_name].append(wavelength)
    
    # Add systems for each unique ion
    for ion_name, transitions in ion_transitions.items():
        config.add_system(
            z=redshift,
            ion=ion_name, 
            transitions=transitions,
            components=n_components
        )
    
    # Create instrument data
    instrument_data = {}
    fwhm_values = [2, 4, 8]  # Different LSF for multi-instrument
    
    for i, inst_name in enumerate(instruments):
        # Create VoigtModel
        fwhm = fwhm_values[i % len(fwhm_values)]
        model = VoigtModel(config, FWHM=fwhm)
        
        # Get proper parameter structure from the model
        tt = model.param_manager.generate_theta_bounds()
        lb=tt.lower
        ub=tt.upper
        n_params = len(lb)
        
        # Generate realistic "true" parameters within bounds
        theta_true = []
        for lower, upper in zip(lb, ub):
            if lower > 10:  # Likely column density (log)
                theta_true.append(np.random.uniform(lower + 1, upper - 1))
            else:  # b-parameter or velocity
                theta_true.append(np.random.uniform(lower + 2, upper - 2))
        theta_true = np.array(theta_true)
        
        # Create synthetic flux with noise
        flux_true = model.evaluate(theta_true, wave)
        noise_level = 0.08
        noise = np.random.normal(0, noise_level, len(wave))
        flux_obs = flux_true + noise
        error = np.full_like(wave, noise_level)
        
        instrument_data[inst_name] = {
            'model': model,
            'wave': wave,
            'flux': flux_obs,
            'error': error
        }
    
    # Use the true parameters as starting guess (from first instrument)
    return instrument_data, theta_true, lb, ub

def time_mcmc_run(instrument_data, theta, lb, ub, use_pool, n_steps=200):
    """Time a single MCMC run."""
    n_params = len(theta)
    n_walkers = max(32, 2 * n_params)  # Scale walkers with parameters
    
    try:
        # Create fitter with small perturbations around true values
        fitter = vfit(instrument_data, theta, lb, ub, 
                     no_of_Chain=n_walkers, no_of_steps=n_steps,
                     perturbation=1e-4)  # Small perturbations around truth
        
        # Time the MCMC run
        start_time = time.time()
        fitter.runmcmc(optimize=True, verbose=False, use_pool=use_pool, progress=True)
        end_time = time.time()
        
        return end_time - start_time
        
    except Exception as e:
        print(f"Error in MCMC run: {e}")
        return np.inf

def run_performance_test(test_name, complexity_values, data_generator, n_runs=3):
    """Run performance test for given complexity values."""
    print(f"\n{'='*60}")
    print(f"PHASE: {test_name}")
    print(f"{'='*60}")
    
    results = {
        'complexity': [],
        'serial_time': [],
        'parallel_time': [],
        'speedup': [],
        'n_params': []
    }
    
    for complexity in complexity_values:
        print(f"\nTesting complexity: {complexity}")
        
        # Generate data
        try:
            instrument_data, theta, lb, ub = data_generator(complexity)
            n_params = len(theta)
            print(f"  Parameters: {n_params}")
            
        except Exception as e:
            print(f"  Skipping - data generation failed: {e}")
            continue
        
        # Test serial
        serial_times = []
        for run in range(n_runs):
            print(f"  Serial run {run+1}/{n_runs}...", end=' ')
            t = time_mcmc_run(instrument_data, theta, lb, ub, use_pool=False)
            serial_times.append(t)
            print(f"{t:.1f}s")
        
        # Test parallel
        parallel_times = []
        for run in range(n_runs):
            print(f"  Parallel run {run+1}/{n_runs}...", end=' ')
            t = time_mcmc_run(instrument_data, theta, lb, ub, use_pool=True)
            parallel_times.append(t)
            print(f"{t:.1f}s")
        
        # Calculate averages
        avg_serial = np.mean(serial_times)
        avg_parallel = np.mean(parallel_times)
        speedup = avg_serial / avg_parallel if avg_parallel > 0 else 0
        
        results['complexity'].append(complexity)
        results['serial_time'].append(avg_serial)
        results['parallel_time'].append(avg_parallel)
        results['speedup'].append(speedup)
        results['n_params'].append(n_params)
        
        print(f"  Results: Serial={avg_serial:.1f}s, Parallel={avg_parallel:.1f}s, Speedup={speedup:.2f}x")
    
    return results

def main():
    """Run all performance tests."""
    
    print("MCMC SERIAL vs PARALLEL PERFORMANCE TEST")
    print("="*60)
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Ion coverage: 1000-1800Å observed frame")
    print(f"Test ions: {list(ION_WAVELENGTHS.keys())}")
    
    # Phase 1: Single instrument complexity scaling
    def phase1_data_gen(n_lines):
        return create_synthetic_data(n_lines, n_components=2, instruments=['HIRES'])
    
    phase1_lines = [1, 2, 3, 4, 5, 6]  # Start with very simple cases
    phase1_results = run_performance_test(
        "Phase 1: Single Instrument Line Scaling", 
        phase1_lines, 
        phase1_data_gen
    )
    
    # Phase 2: Multi-instrument scaling  
    # Use optimal complexity from Phase 1 (or default)
    optimal_lines = 3  # Could pick based on Phase 1 results
    
    def phase2_data_gen(n_instruments):
        inst_names = [f'INST_{i}' for i in range(n_instruments)]
        return create_synthetic_data(optimal_lines, n_components=2, instruments=inst_names)
    
    phase2_instruments = [1, 2, 3, 4]
    phase2_results = run_performance_test(
        "Phase 2: Multi-Instrument Scaling",
        phase2_instruments,
        phase2_data_gen  
    )
    
    # Phase 3: Parameter scaling with multiple instruments
    def phase3_data_gen(n_components):
        return create_synthetic_data(3, n_components=n_components, 
                                   instruments=['HIRES', 'FIRE', 'UVES'])
    
    phase3_components = [1, 2, 3, 4, 5]
    phase3_results = run_performance_test(
        "Phase 3: Parameter Scaling (3 instruments)",
        phase3_components,
        phase3_data_gen
    )
    
    # Plot results and save
    figure_filename = plot_results(phase1_results, phase2_results, phase3_results)
    
    # Save results data
    import datetime
    import json
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {
        'metadata': {
            'timestamp': timestamp,
            'cpu_cores': mp.cpu_count(),
            'test_description': 'MCMC Serial vs Parallel Performance Test',
            'wavelength_range': [1000, 1800],
            'ions_tested': list(ION_WAVELENGTHS.keys())
        },
        'phase1': phase1_results,
        'phase2': phase2_results, 
        'phase3': phase3_results
    }
    
    # Save as JSON
    json_filename = f"mcmc_performance_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results data saved as: {json_filename}")
    print(f"Figure saved as: {figure_filename}")
    
    return all_results

def plot_results(phase1, phase2, phase3):
    """Plot performance results and identify crossover points."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MCMC Serial vs Parallel Performance Analysis', fontsize=14)
    
    # Phase 1: Lines scaling
    ax1 = axes[0, 0]
    ax1.plot(phase1['complexity'], phase1['serial_time'], 'b-o', label='Serial')
    ax1.plot(phase1['complexity'], phase1['parallel_time'], 'r-o', label='Parallel') 
    ax1.set_xlabel('Number of Lines')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Phase 1: Single Instrument')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phase 2: Instruments scaling
    ax2 = axes[0, 1]
    ax2.plot(phase2['complexity'], phase2['serial_time'], 'b-o', label='Serial')
    ax2.plot(phase2['complexity'], phase2['parallel_time'], 'r-o', label='Parallel')
    ax2.set_xlabel('Number of Instruments') 
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Phase 2: Multi-Instrument')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase 3: Components scaling  
    ax3 = axes[1, 0]
    ax3.plot(phase3['complexity'], phase3['serial_time'], 'b-o', label='Serial')
    ax3.plot(phase3['complexity'], phase3['parallel_time'], 'r-o', label='Parallel')
    ax3.set_xlabel('Components per Ion')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Phase 3: Parameter Scaling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Speedup analysis
    ax4 = axes[1, 1]
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax4.plot(phase1['n_params'], phase1['speedup'], 'g-o', label='Phase 1')
    ax4.plot(phase2['n_params'], phase2['speedup'], 'm-s', label='Phase 2') 
    ax4.plot(phase3['n_params'], phase3['speedup'], 'c-^', label='Phase 3')
    ax4.set_xlabel('Total Parameters')
    ax4.set_ylabel('Speedup (Serial/Parallel)')
    ax4.set_title('Parallel Speedup vs Model Complexity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mcmc_performance_test_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {filename}")
    
    # Print crossover analysis
    print("\n" + "="*60)
    print("CROSSOVER ANALYSIS")
    print("="*60)
    
    def find_crossover(results, name):
        speedups = np.array(results['speedup'])
        params = np.array(results['n_params'])
        complexity = np.array(results['complexity'])
        
        # Find where speedup > 1.0
        beneficial = speedups > 1.0
        if any(beneficial):
            first_beneficial = np.where(beneficial)[0][0]
            print(f"{name}:")
            print(f"  Parallel becomes beneficial at:")
            print(f"    Complexity: {complexity[first_beneficial]}")
            print(f"    Parameters: {params[first_beneficial]}")
            print(f"    Speedup: {speedups[first_beneficial]:.2f}x")
        else:
            print(f"{name}: Parallel never beneficial in tested range")
        print()
    
    find_crossover(phase1, "Phase 1 (Single Instrument)")
    find_crossover(phase2, "Phase 2 (Multi-Instrument)")  
    find_crossover(phase3, "Phase 3 (Parameter Scaling)")
    
    # Overall recommendation
    all_speedups = phase1['speedup'] + phase2['speedup'] + phase3['speedup']
    all_params = phase1['n_params'] + phase2['n_params'] + phase3['n_params']
    
    beneficial_mask = np.array(all_speedups) > 1.0
    if any(beneficial_mask):
        min_beneficial_params = min(np.array(all_params)[beneficial_mask])
        print(f"RECOMMENDATION:")
        print(f"Use parallel when parameters > {min_beneficial_params}")
        print(f"Or when using multiple instruments")
    
    plt.show()
    
    return filename

if __name__ == "__main__":
    main()