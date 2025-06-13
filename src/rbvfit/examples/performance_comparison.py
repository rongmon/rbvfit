#!/usr/bin/env python
"""
Performance comparison between rbvfit v1.0 and v2.0.

This script identifies bottlenecks and compares timing for:
1. Model setup and initialization
2. Single model evaluation
3. Parameter mapping and transformation
4. MCMC likelihood calculation
5. Full MCMC run comparison
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import psutil
import gc
from dataclasses import dataclass

# Import both v1 and v2 components
# v1 imports
try:
    from rbvfit import model as v1_model
    from rbvfit import vfit_mcmc as v1_mcmc
    from rbvfit.rb_vfit import create_model_simple
    V1_AVAILABLE = True
except ImportError:
    print("Warning: rbvfit v1.0 not available for comparison")
    V1_AVAILABLE = False

# v2 imports  
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.voigt_fitter import VoigtFitter, Dataset, MCMCSettings
from rbvfit.core.parameter_manager import ParameterManager


@dataclass
class TimingResult:
    """Container for timing measurements."""
    operation: str
    version: str
    time_ms: float
    memory_mb: float
    details: Dict = None


class PerformanceProfiler:
    """Utility class for performance profiling."""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
    
    def time_operation(self, operation_name: str, version: str, func, *args, **kwargs):
        """Time a function call and record memory usage."""
        # Force garbage collection before timing
        gc.collect()
        
        # Record initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the operation
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Record final memory
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Store timing result
        timing = TimingResult(
            operation=operation_name,
            version=version,
            time_ms=(end_time - start_time) * 1000,
            memory_mb=final_memory - initial_memory
        )
        
        self.results.append(timing)
        return result, timing
    
    def get_summary(self) -> str:
        """Get formatted summary of all timing results."""
        lines = ["Performance Profiling Results", "=" * 60]
        
        # Group by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)
        
        for op_name, timings in operations.items():
            lines.append(f"\n{op_name}:")
            lines.append("-" * 40)
            
            for timing in timings:
                lines.append(f"  {timing.version}: {timing.time_ms:.2f} ms "
                           f"(+{timing.memory_mb:.1f} MB)")
            
            # Compare if we have both versions
            if len(timings) == 2:
                v1_time = next(t.time_ms for t in timings if 'v1' in t.version)
                v2_time = next(t.time_ms for t in timings if 'v2' in t.version)
                speedup = v1_time / v2_time
                if speedup > 1:
                    lines.append(f"  → v2 is {speedup:.1f}x faster")
                else:
                    lines.append(f"  → v1 is {1/speedup:.1f}x faster")
        
        return "\n".join(lines)


def setup_common_test_case():
    """Set up common test case for both versions."""
    # MgII doublet at z=0.348
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    
    # 2-component system
    nguess = [13.8, 13.3]
    bguess = [20.0, 30.0] 
    vguess = [-40.0, 20.0]
    
    theta = np.concatenate([nguess, bguess, vguess])
    
    # Wavelength grid
    wave = np.linspace(3760, 3800, 2000)
    
    return {
        'zabs': zabs,
        'lambda_rest': lambda_rest,
        'theta': theta,
        'wave': wave,
        'nclump': len(nguess),
        'ntransition': len(lambda_rest)
    }


def benchmark_model_creation(profiler: PerformanceProfiler):
    """Benchmark model creation for both versions."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Model Creation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    
    # v1 model creation
    if V1_AVAILABLE:
        def create_v1_model():
            return v1_model.create_voigt(
                np.array([test_case['zabs']]),
                test_case['lambda_rest'],
                test_case['nclump'],
                ntransition=test_case['ntransition'],
                FWHM='6.5',
                verbose=False
            )
        
        v1_model_obj, v1_timing = profiler.time_operation(
            "Model Creation", "v1.0", create_v1_model
        )
        print(f"v1.0 model creation: {v1_timing.time_ms:.2f} ms")
    
    # v2 model creation
    def create_v2_model():
        config = FitConfiguration()
        config.add_system(
            z=test_case['zabs'],
            ion='MgII', 
            transitions=test_case['lambda_rest'],
            components=test_case['nclump']
        )
        return VoigtModel(config, FWHM='6.5')
    
    v2_model_obj, v2_timing = profiler.time_operation(
        "Model Creation", "v2.0", create_v2_model
    )
    print(f"v2.0 model creation: {v2_timing.time_ms:.2f} ms")
    
    return v1_model_obj if V1_AVAILABLE else None, v2_model_obj


def benchmark_single_evaluation(profiler: PerformanceProfiler, v1_model, v2_model):
    """Benchmark single model evaluation."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Single Model Evaluation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    
    # v1 single evaluation
    if V1_AVAILABLE and v1_model is not None:
        def v1_evaluate():
            return v1_model.model_flux(test_case['theta'], test_case['wave'])
        
        v1_flux, v1_timing = profiler.time_operation(
            "Single Evaluation", "v1.0", v1_evaluate
        )
        print(f"v1.0 single evaluation: {v1_timing.time_ms:.2f} ms")
    
    # v2 single evaluation
    def v2_evaluate():
        return v2_model.evaluate(test_case['theta'], test_case['wave'])
    
    v2_flux, v2_timing = profiler.time_operation(
        "Single Evaluation", "v2.0", v2_evaluate
    )
    print(f"v2.0 single evaluation: {v2_timing.time_ms:.2f} ms")
    
    # Verify results are equivalent
    if V1_AVAILABLE and v1_model is not None:
        diff = np.max(np.abs(v1_flux - v2_flux))
        print(f"Maximum difference between v1/v2: {diff:.2e}")
        if diff > 1e-10:
            print("⚠ Warning: Significant difference in model outputs!")
        else:
            print("✓ Model outputs are equivalent")
    
    return v1_flux if V1_AVAILABLE else None, v2_flux


def benchmark_batch_evaluation(profiler: PerformanceProfiler, v1_model, v2_model):
    """Benchmark batch model evaluations (like in MCMC)."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Batch Model Evaluation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    
    # Generate random parameter sets
    n_evaluations = 100
    theta_base = test_case['theta']
    
    # Add noise to create different parameter sets
    np.random.seed(42)  # For reproducibility
    theta_batch = []
    for i in range(n_evaluations):
        noise = 0.1 * np.random.randn(len(theta_base))
        theta_batch.append(theta_base + noise)
    
    print(f"Performing {n_evaluations} model evaluations...")
    
    # v1 batch evaluation
    if V1_AVAILABLE and v1_model is not None:
        def v1_batch_evaluate():
            results = []
            for theta in theta_batch:
                flux = v1_model.model_flux(theta, test_case['wave'])
                results.append(flux)
            return results
        
        v1_results, v1_timing = profiler.time_operation(
            "Batch Evaluation", "v1.0", v1_batch_evaluate
        )
        print(f"v1.0 batch evaluation: {v1_timing.time_ms:.2f} ms "
              f"({v1_timing.time_ms/n_evaluations:.2f} ms/eval)")
    
    # v2 batch evaluation
    def v2_batch_evaluate():
        results = []
        for theta in theta_batch:
            flux = v2_model.evaluate(theta, test_case['wave'])
            results.append(flux)
        return results
    
    v2_results, v2_timing = profiler.time_operation(
        "Batch Evaluation", "v2.0", v2_batch_evaluate
    )
    print(f"v2.0 batch evaluation: {v2_timing.time_ms:.2f} ms "
          f"({v2_timing.time_ms/n_evaluations:.2f} ms/eval)")
    
    return v1_results if V1_AVAILABLE else None, v2_results


def benchmark_parameter_management(profiler: PerformanceProfiler):
    """Benchmark parameter mapping and management overhead."""
    print("\n" + "=" * 60)
    print("BENCHMARK 4: Parameter Management")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    
    # v2 parameter management (v1 doesn't have equivalent)
    config = FitConfiguration()
    config.add_system(
        z=test_case['zabs'],
        ion='MgII',
        transitions=test_case['lambda_rest'], 
        components=test_case['nclump']
    )
    
    param_manager = ParameterManager(config)
    theta = test_case['theta']
    
    # Benchmark theta to parameters conversion
    def theta_to_params():
        return param_manager.theta_to_parameters(theta)
    
    params, timing1 = profiler.time_operation(
        "Theta to Parameters", "v2.0", theta_to_params
    )
    print(f"v2.0 theta→parameters: {timing1.time_ms:.2f} ms")
    
    # Benchmark parameters to theta conversion
    def params_to_theta():
        return param_manager.parameters_to_theta(params)
    
    theta_reconstructed, timing2 = profiler.time_operation(
        "Parameters to Theta", "v2.0", params_to_theta
    )
    print(f"v2.0 parameters→theta: {timing2.time_ms:.2f} ms")
    
    # Verify round-trip accuracy
    diff = np.max(np.abs(theta - theta_reconstructed))
    print(f"Round-trip accuracy: {diff:.2e}")
    
    # Benchmark line parameters expansion
    def theta_to_line_params():
        return param_manager.theta_to_line_parameters(theta)
    
    line_params, timing3 = profiler.time_operation(
        "Theta to Line Parameters", "v2.0", theta_to_line_params
    )
    print(f"v2.0 theta→line_params: {timing3.time_ms:.2f} ms")
    
    return params, line_params


def benchmark_likelihood_calculation(profiler: PerformanceProfiler, v1_model, v2_model):
    """Benchmark likelihood calculation for MCMC."""
    print("\n" + "=" * 60)
    print("BENCHMARK 5: Likelihood Calculation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    
    # Generate synthetic data
    wave = test_case['wave']
    true_flux = v2_model.evaluate(test_case['theta'], wave)
    noise = np.random.normal(0, 0.02, len(wave))
    observed_flux = true_flux + noise
    error = np.full_like(wave, 0.02)
    
    # v1 likelihood calculation
    if V1_AVAILABLE and v1_model is not None:
        def v1_likelihood():
            model_flux = v1_model.model_flux(test_case['theta'], wave)
            chi2 = np.sum((observed_flux - model_flux)**2 / error**2)
            return -0.5 * chi2
        
        v1_loglike, v1_timing = profiler.time_operation(
            "Likelihood Calculation", "v1.0", v1_likelihood
        )
        print(f"v1.0 likelihood: {v1_timing.time_ms:.2f} ms")
    
    # v2 likelihood calculation
    def v2_likelihood():
        model_flux = v2_model.evaluate(test_case['theta'], wave)
        chi2 = np.sum((observed_flux - model_flux)**2 / error**2)
        return -0.5 * chi2
    
    v2_loglike, v2_timing = profiler.time_operation(
        "Likelihood Calculation", "v2.0", v2_likelihood
    )
    print(f"v2.0 likelihood: {v2_timing.time_ms:.2f} ms")
    
    # Compare likelihood values
    if V1_AVAILABLE and v1_model is not None:
        print(f"Likelihood difference: {abs(v1_loglike - v2_loglike):.2e}")
    
    return observed_flux, error


def benchmark_mcmc_setup(profiler: PerformanceProfiler, v1_model, v2_model):
    """Benchmark MCMC setup and initialization."""
    print("\n" + "=" * 60)
    print("BENCHMARK 6: MCMC Setup")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    
    # Generate synthetic data
    wave = test_case['wave']
    true_flux = v2_model.evaluate(test_case['theta'], wave)
    noise = np.random.normal(0, 0.02, len(wave))
    observed_flux = true_flux + noise
    error = np.full_like(wave, 0.02)
    
    # v1 MCMC setup
    if V1_AVAILABLE and v1_model is not None:
        def v1_mcmc_setup():
            # Set up bounds
            nguess = test_case['theta'][:test_case['nclump']]
            bguess = test_case['theta'][test_case['nclump']:2*test_case['nclump']]
            vguess = test_case['theta'][2*test_case['nclump']:]
            
            bounds, lb, ub = v1_mcmc.set_bounds(nguess, bguess, vguess)
            
            # Create fitter object
            fitter = v1_mcmc.vfit(
                v1_model.model_flux,
                test_case['theta'],
                lb, ub,
                wave, observed_flux, error,
                no_of_Chain=20,
                no_of_steps=10  # Minimal for setup timing
            )
            return fitter
        
        v1_fitter, v1_timing = profiler.time_operation(
            "MCMC Setup", "v1.0", v1_mcmc_setup
        )
        print(f"v1.0 MCMC setup: {v1_timing.time_ms:.2f} ms")
    
    # v2 MCMC setup
    def v2_mcmc_setup():
        dataset = Dataset(wave, observed_flux, error, name="test")
        mcmc_settings = MCMCSettings(n_walkers=20, n_steps=10)
        fitter = VoigtFitter(v2_model, dataset, mcmc_settings)
        return fitter
    
    v2_fitter, v2_timing = profiler.time_operation(
        "MCMC Setup", "v2.0", v2_mcmc_setup
    )
    print(f"v2.0 MCMC setup: {v2_timing.time_ms:.2f} ms")
    
    return v1_fitter if V1_AVAILABLE else None, v2_fitter, observed_flux, error


def benchmark_short_mcmc_run(profiler: PerformanceProfiler, v1_fitter, v2_fitter, 
                            observed_flux, error, test_case):
    """Benchmark short MCMC runs."""
    print("\n" + "=" * 60)
    print("BENCHMARK 7: Short MCMC Run")
    print("=" * 60)
    
    n_steps = 500
    n_walkers = 20
    
    print(f"Running {n_steps} MCMC steps with {n_walkers} walkers...")
    
    # v1 short MCMC
    if V1_AVAILABLE and v1_fitter is not None:
        def v1_short_mcmc():
            # Reset the fitter with short run
            v1_fitter.no_of_steps = n_steps
            v1_fitter.no_of_Chain = n_walkers
            v1_fitter.runmcmc(optimize=False, verbose=False)
            return v1_fitter
        
        v1_result, v1_timing = profiler.time_operation(
            "Short MCMC Run", "v1.0", v1_short_mcmc
        )
        print(f"v1.0 MCMC ({n_steps} steps): {v1_timing.time_ms:.2f} ms "
              f"({v1_timing.time_ms/n_steps:.2f} ms/step)")
    
    # v2 short MCMC
    def v2_short_mcmc():
        # Update settings for short run
        v2_fitter.mcmc_settings.n_steps = n_steps
        v2_fitter.mcmc_settings.n_walkers = n_walkers
        v2_fitter.mcmc_settings.n_burn = 10
        v2_fitter.mcmc_settings.progress = True
        v2_fitter.mcmc_settings.parallel = True        
        
        result = v2_fitter.fit(test_case['theta'], optimize_first=False)
        return result
    
    v2_result, v2_timing = profiler.time_operation(
        "Short MCMC Run", "v2.0", v2_short_mcmc
    )
    print(f"v2.0 MCMC ({n_steps} steps): {v2_timing.time_ms:.2f} ms "
          f"({v2_timing.time_ms/n_steps:.2f} ms/step)")
    
    return v1_result if V1_AVAILABLE else None, v2_result


def analyze_bottlenecks(profiler: PerformanceProfiler):
    """Analyze where the main bottlenecks are."""
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Group results by operation
    operations = {}
    for result in profiler.results:
        if result.operation not in operations:
            operations[result.operation] = {}
        operations[result.operation][result.version] = result
    
    print("\nTiming Breakdown (v2.0):")
    print("-" * 30)
    
    v2_operations = [(op, data.get('v2.0')) for op, data in operations.items() 
                     if 'v2.0' in data]
    v2_operations = [(op, timing) for op, timing in v2_operations if timing is not None]
    v2_operations.sort(key=lambda x: x[1].time_ms, reverse=True)
    
    total_v2_time = sum(timing.time_ms for _, timing in v2_operations)
    
    for operation, timing in v2_operations:
        percentage = (timing.time_ms / total_v2_time) * 100
        print(f"{operation:<25} {timing.time_ms:>8.2f} ms ({percentage:>5.1f}%)")
    
    if V1_AVAILABLE:
        print(f"\nSpeedup Analysis:")
        print("-" * 20)
        
        for operation, data in operations.items():
            if 'v1.0' in data and 'v2.0' in data:
                v1_time = data['v1.0'].time_ms
                v2_time = data['v2.0'].time_ms
                speedup = v1_time / v2_time
                
                if speedup > 1:
                    print(f"{operation:<25} v2 is {speedup:>5.1f}x faster")
                else:
                    print(f"{operation:<25} v1 is {1/speedup:>5.1f}x faster")
    
    # Identify main bottlenecks
    print(f"\nBottleneck Identification:")
    print("-" * 25)
    
    if v2_operations:
        slowest_op, slowest_timing = v2_operations[0]
        print(f"Slowest operation: {slowest_op} ({slowest_timing.time_ms:.2f} ms)")
        
        if slowest_timing.time_ms > 100:
            print("⚠ Major bottleneck identified (>100ms)")
        elif slowest_timing.time_ms > 50:
            print("⚠ Minor bottleneck identified (>50ms)")
        else:
            print("✓ No significant bottlenecks")
    
    # Memory usage analysis
    print(f"\nMemory Usage Analysis:")
    print("-" * 22)
    
    total_memory = sum(abs(timing.memory_mb) for _, timing in v2_operations)
    print(f"Total memory overhead: {total_memory:.1f} MB")
    
    if total_memory > 100:
        print("⚠ High memory usage")
    elif total_memory > 50:
        print("⚠ Moderate memory usage")
    else:
        print("✓ Low memory usage")


def optimization_recommendations(profiler: PerformanceProfiler):
    """Provide optimization recommendations based on benchmarks."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    # Analyze results
    operations = {}
    for result in profiler.results:
        if result.operation not in operations:
            operations[result.operation] = {}
        operations[result.operation][result.version] = result
    
    recommendations = []
    
    # Check model evaluation speed
    if 'Single Evaluation' in operations and 'v2.0' in operations['Single Evaluation']:
        eval_time = operations['Single Evaluation']['v2.0'].time_ms
        if eval_time > 10:
            recommendations.append(
                f"🔧 Model evaluation is slow ({eval_time:.1f}ms). Consider:\n"
                f"   - Vectorizing Voigt profile calculations\n"
                f"   - Caching line atomic parameters\n"
                f"   - Optimizing convolution operations"
            )
    
    # Check parameter management overhead
    param_ops = ['Theta to Parameters', 'Parameters to Theta', 'Theta to Line Parameters']
    param_total = sum(operations.get(op, {}).get('v2.0', type('', (), {'time_ms': 0})).time_ms 
                     for op in param_ops)
    
    if param_total > 5:
        recommendations.append(
            f"🔧 Parameter management overhead is high ({param_total:.1f}ms). Consider:\n"
            f"   - Caching parameter mappings\n"
            f"   - Pre-computing index arrays\n"
            f"   - Using numpy fancy indexing"
        )
    
    # Check MCMC setup time
    if 'MCMC Setup' in operations and 'v2.0' in operations['MCMC Setup']:
        setup_time = operations['MCMC Setup']['v2.0'].time_ms
        if setup_time > 50:
            recommendations.append(
                f"🔧 MCMC setup is slow ({setup_time:.1f}ms). Consider:\n"
                f"   - Lazy initialization of sampler\n"
                f"   - Pre-compiled probability functions\n"
                f"   - Reduced validation overhead"
            )
    
    # Check if v1 is faster in any area
    if V1_AVAILABLE:
        slower_ops = []
        for operation, data in operations.items():
            if 'v1.0' in data and 'v2.0' in data:
                v1_time = data['v1.0'].time_ms
                v2_time = data['v2.0'].time_ms
                if v2_time > v1_time * 1.5:  # v2 significantly slower
                    slower_ops.append((operation, v2_time/v1_time))
        
        if slower_ops:
            recommendations.append(
                f"🔧 v2.0 is slower than v1.0 in some areas:\n" +
                "\n".join(f"   - {op}: {ratio:.1f}x slower" for op, ratio in slower_ops) +
                f"\n   Consider adopting v1.0 algorithms for these operations"
            )
    
    # Memory recommendations
    high_memory_ops = []
    for operation, data in operations.items():
        if 'v2.0' in data and data['v2.0'].memory_mb > 10:
            high_memory_ops.append((operation, data['v2.0'].memory_mb))
    
    if high_memory_ops:
        recommendations.append(
            f"🔧 High memory usage detected:\n" +
            "\n".join(f"   - {op}: +{mem:.1f}MB" for op, mem in high_memory_ops) +
            f"\n   Consider memory pooling or object reuse"
        )
    
    # Print recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n✅ Performance looks good! No major optimizations needed.")
    
    # General v2.0 specific recommendations
    print(f"\n" + "💡 General v2.0 Optimization Tips:")
    print(f"   - Use joint fitting to amortize setup costs")
    print(f"   - Cache VoigtModel objects for repeated use")
    print(f"   - Consider zeus sampler for large parameter spaces")
    print(f"   - Use optimize_first=True to improve convergence")
    print(f"   - Batch model evaluations when possible")


def create_performance_plots(profiler: PerformanceProfiler):
    """Create visualization of performance results."""
    print("\n" + "=" * 60)
    print("CREATING PERFORMANCE PLOTS")
    print("=" * 60)
    
    # Group results by operation
    operations = {}
    for result in profiler.results:
        if result.operation not in operations:
            operations[result.operation] = {}
        operations[result.operation][result.version] = result
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Timing comparison
    op_names = []
    v1_times = []
    v2_times = []
    
    for operation, data in operations.items():
        if 'v2.0' in data:
            op_names.append(operation.replace(' ', '\n'))
            v2_times.append(data['v2.0'].time_ms)
            
            if V1_AVAILABLE and 'v1.0' in data:
                v1_times.append(data['v1.0'].time_ms)
            else:
                v1_times.append(0)
    
    x = np.arange(len(op_names))
    width = 0.35
    
    if V1_AVAILABLE:
        ax1.bar(x - width/2, v1_times, width, label='rbvfit v1.0', color='skyblue')
    ax1.bar(x + width/2, v2_times, width, label='rbvfit v2.0', color='lightcoral')
    
    ax1.set_xlabel('Operation')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(op_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Memory usage
    memory_usage = [operations[op.replace('\n', ' ')]['v2.0'].memory_mb 
                   for op in op_names]
    
    bars = ax2.bar(op_names, memory_usage, color='lightgreen')
    ax2.set_xlabel('Operation')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage (v2.0)')
    ax2.set_xticklabels(op_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, memory_usage):
        if abs(value) > 0.1:  # Only show significant memory changes
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Create detailed timing breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    
    v2_ops = [(op.replace('\n', ' '), operations[op.replace('\n', ' ')]['v2.0'].time_ms) 
              for op in op_names]
    v2_ops.sort(key=lambda x: x[1], reverse=True)
    
    operations_clean, times = zip(*v2_ops)
    
    # Create pie chart of time distribution
    colors = plt.cm.Set3(np.linspace(0, 1, len(operations_clean)))
    wedges, texts, autotexts = ax.pie(times, labels=operations_clean, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
    
    ax.set_title('v2.0 Time Distribution by Operation')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Performance plots generated")


def main():
    """Main performance comparison function."""
    print("rbvfit Performance Comparison: v1.0 vs v2.0")
    print("=" * 60)
    print("This benchmark identifies bottlenecks and optimization opportunities")
    
    profiler = PerformanceProfiler()
    
    try:
        # Run benchmarks
        v1_model, v2_model = benchmark_model_creation(profiler)
        v1_flux, v2_flux = benchmark_single_evaluation(profiler, v1_model, v2_model)
        benchmark_batch_evaluation(profiler, v1_model, v2_model)
        benchmark_parameter_management(profiler)
        observed_flux, error = benchmark_likelihood_calculation(profiler, v1_model, v2_model)
        
        test_case = setup_common_test_case()
        v1_fitter, v2_fitter, observed_flux, error = benchmark_mcmc_setup(
            profiler, v1_model, v2_model
        )
        
        benchmark_short_mcmc_run(profiler, v1_fitter, v2_fitter, 
                               observed_flux, error, test_case)
        
        # Analysis
        print(profiler.get_summary())
        analyze_bottlenecks(profiler)
        optimization_recommendations(profiler)
        create_performance_plots(profiler)
        
        print("\n" + "=" * 60)
        print("🏁 PERFORMANCE COMPARISON COMPLETE")
        print("=" * 60)
        print("✅ Benchmarks completed successfully")
        print("📊 Performance plots generated")
        print("💡 Check optimization recommendations above")
        
        if not V1_AVAILABLE:
            print("⚠️  v1.0 not available - only v2.0 benchmarked")
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("⚠️  Partial results available:")
        if profiler.results:
            print(profiler.get_summary())
        print("=" * 60)


if __name__ == "__main__":
    main()