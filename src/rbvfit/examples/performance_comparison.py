#!/usr/bin/env python
"""
Enhanced Performance comparison between rbvfit v1.0 and v2.0 with MCMC focus.

This script identifies bottlenecks and compares timing for:
1. Model setup and initialization
2. Single model evaluation
3. Model compilation (v2.0 feature)
4. Parameter mapping and transformation
5. MCMC likelihood calculation
6. Full MCMC run comparison (serial vs parallel)

Key improvements:
- Uses vfit_mcmc for both v1 and v2
- Tests both serial and parallel MCMC modes
- Better error handling
- More comprehensive benchmarks
- Fixed v2.0 interface issues
- Added compilation benchmarking
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import psutil
import gc
from dataclasses import dataclass
import traceback
import warnings
warnings.filterwarnings('ignore')

# Import both v1 and v2 components
# v1 imports
try:
    from rbvfit import model as v1_model
    from rbvfit import vfit_mcmc as v1_mcmc
    V1_AVAILABLE = True
    print("✓ rbvfit v1.0 available")
except ImportError as e:
    print(f"⚠ Warning: rbvfit v1.0 not available: {e}")
    V1_AVAILABLE = False

# v2 imports  
try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    from rbvfit import vfit_mcmc as v2_mcmc  # Should be same module
    V2_AVAILABLE = True
    print("✓ rbvfit v2.0 core modules available")
except ImportError as e:
    print(f"❌ Error: rbvfit v2.0 not available: {e}")
    V2_AVAILABLE = False


@dataclass
class TimingResult:
    """Container for timing measurements."""
    operation: str
    version: str
    time_ms: float
    memory_mb: float
    details: Optional[Dict] = None
    success: bool = True
    error_msg: Optional[str] = None


class PerformanceProfiler:
    """Enhanced utility class for performance profiling."""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
        self.verbose = True
    
    def time_operation(self, operation_name: str, version: str, func, *args, **kwargs):
        """Time a function call and record memory usage with error handling."""
        # Force garbage collection before timing
        gc.collect()
        
        # Record initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Time the operation
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Record final memory
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Store successful timing result
            timing = TimingResult(
                operation=operation_name,
                version=version,
                time_ms=(end_time - start_time) * 1000,
                memory_mb=final_memory - initial_memory,
                success=True
            )
            
        except Exception as e:
            # Record failed timing result
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            timing = TimingResult(
                operation=operation_name,
                version=version,
                time_ms=(end_time - start_time) * 1000,
                memory_mb=final_memory - initial_memory,
                success=False,
                error_msg=str(e)
            )
            result = None
            
            if self.verbose:
                print(f"⚠ Error in {operation_name} ({version}): {str(e)}")
        
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
                if timing.success:
                    lines.append(f"  {timing.version}: {timing.time_ms:.2f} ms "
                               f"(+{timing.memory_mb:.1f} MB)")
                else:
                    lines.append(f"  {timing.version}: FAILED - {timing.error_msg}")
            
            # Compare if we have both successful versions
            successful_timings = [t for t in timings if t.success]
            if len(successful_timings) >= 2:
                # Find v1 and v2 timings for comparison
                v1_times = [t.time_ms for t in successful_timings if 'v1' in t.version]
                v2_times = [t.time_ms for t in successful_timings if 'v2' in t.version]
                
                if v1_times and v2_times:
                    v1_avg = np.mean(v1_times)
                    v2_avg = np.mean(v2_times)
                    speedup = v1_avg / v2_avg
                    
                    if speedup > 1:
                        lines.append(f"  → v2 is {speedup:.1f}x faster on average")
                    else:
                        lines.append(f"  → v1 is {1/speedup:.1f}x faster on average")
        
        return "\n".join(lines)


def setup_common_test_case():
    """Set up common test case for both versions."""
    # MgII doublet at z=0.348 - smaller for faster testing
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    
    # 4-component system
    nguess = [13.8, 13.3,13.8, 13.3]
    bguess = [20.0, 30.0,15,8] 
    vguess = [-40.0,-20.,0., 20.0]
    
    theta = np.concatenate([nguess, bguess, vguess])
    
    # Smaller wavelength grid for faster testing
    wave = np.linspace(3760, 3800, 500)
    
    return {
        'zabs': zabs,
        'lambda_rest': lambda_rest,
        'theta': theta,
        'wave': wave,
        'nclump': len(nguess),
        'ntransition': len(lambda_rest),
        'nguess': nguess,
        'bguess': bguess,
        'vguess': vguess
    }


def benchmark_model_creation(profiler: PerformanceProfiler):
    """Benchmark model creation for both versions."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Model Creation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    v1_model_obj = None
    v2_model_obj = None
    
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
        if v1_timing.success:
            print(f"v1.0 model creation: {v1_timing.time_ms:.2f} ms")
        else:
            print(f"v1.0 model creation: FAILED")
    
    # v2 model creation
    if V2_AVAILABLE:
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
        if v2_timing.success:
            print(f"v2.0 model creation: {v2_timing.time_ms:.2f} ms")
        else:
            print(f"v2.0 model creation: FAILED")
    
    return v1_model_obj, v2_model_obj


def benchmark_model_compilation(profiler: PerformanceProfiler, v2_model):
    """Benchmark v2.0 model compilation feature."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Model Compilation (v2.0 feature)")
    print("=" * 60)
    
    compiled_model = None
    
    if V2_AVAILABLE and v2_model is not None:
        def compile_v2_model():
            return v2_model.compile(verbose=False)
        
        compiled_model, timing = profiler.time_operation(
            "Model Compilation", "v2.0", compile_v2_model
        )
        
        if timing.success:
            print(f"v2.0 model compilation: {timing.time_ms:.2f} ms")
            print("✓ Compilation creates optimized model for fast evaluation")
        else:
            print(f"v2.0 model compilation: FAILED")
    else:
        print("v2.0 model not available for compilation")
    
    return compiled_model


def benchmark_single_evaluation(profiler: PerformanceProfiler, v1_model, v2_model, v2_compiled):
    """Benchmark single model evaluation for all variants."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Single Model Evaluation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    results = {}
    
    # v1 single evaluation
    if V1_AVAILABLE and v1_model is not None:
        def v1_evaluate():
            return v1_model.model_flux(test_case['theta'], test_case['wave'])
        
        v1_flux, v1_timing = profiler.time_operation(
            "Single Evaluation", "v1.0", v1_evaluate
        )
        if v1_timing.success:
            print(f"v1.0 single evaluation: {v1_timing.time_ms:.2f} ms")
            results['v1'] = v1_flux
    
    # v2 regular evaluation
    if V2_AVAILABLE and v2_model is not None:
        def v2_evaluate():
            return v2_model.evaluate(test_case['theta'], test_case['wave'])
        
        v2_flux, v2_timing = profiler.time_operation(
            "Single Evaluation", "v2.0 regular", v2_evaluate
        )
        if v2_timing.success:
            print(f"v2.0 regular evaluation: {v2_timing.time_ms:.2f} ms")
            results['v2_regular'] = v2_flux
    
    # v2 compiled evaluation
    if v2_compiled is not None:
        def v2_compiled_evaluate():
            return v2_compiled.model_flux(test_case['theta'], test_case['wave'])
        
        v2_compiled_flux, v2_compiled_timing = profiler.time_operation(
            "Single Evaluation", "v2.0 compiled", v2_compiled_evaluate
        )
        if v2_compiled_timing.success:
            print(f"v2.0 compiled evaluation: {v2_compiled_timing.time_ms:.2f} ms")
            results['v2_compiled'] = v2_compiled_flux
    
    return results


def generate_synthetic_data(model_func, theta, wave, noise_level=0.02):
    """Generate synthetic data for MCMC testing."""
    try:
        true_flux = model_func(theta, wave)
        noise = np.random.normal(0, noise_level, len(wave))
        observed_flux = true_flux + noise
        error = np.full_like(wave, noise_level)
        return observed_flux, error, true_flux
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return None, None, None


def benchmark_mcmc_setup(profiler: PerformanceProfiler, v1_model, v2_model, v2_compiled):
    """Benchmark MCMC setup for all model variants."""
    print("\n" + "=" * 60)
    print("BENCHMARK 4: MCMC Setup")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    wave = test_case['wave']
    theta = test_case['theta']
    
    # Set up bounds (same for all)
    bounds, lb, ub = v1_mcmc.set_bounds(
        test_case['nguess'], 
        test_case['bguess'], 
        test_case['vguess']
    )
    
    mcmc_configs = {}
    
    # v1 MCMC setup
    if V1_AVAILABLE and v1_model is not None:
        # Generate synthetic data using v1 model
        observed_flux, error, _ = generate_synthetic_data(v1_model.model_flux, theta, wave)
        
        if observed_flux is not None:
            def v1_mcmc_setup():
                return v1_mcmc.vfit(
                    v1_model.model_flux,
                    theta, lb, ub,
                    wave, observed_flux, error,
                    no_of_Chain=50,
                    no_of_steps=500  # Small for setup testing
                )
            
            v1_fitter, v1_timing = profiler.time_operation(
                "MCMC Setup", "v1.0", v1_mcmc_setup
            )
            
            if v1_timing.success:
                print(f"v1.0 MCMC setup: {v1_timing.time_ms:.2f} ms")
                mcmc_configs['v1'] = {
                    'fitter': v1_fitter,
                    'data': (wave, observed_flux, error),
                    'model_func': v1_model.model_flux
                }
    
    # v2 regular MCMC setup
    if V2_AVAILABLE and v2_model is not None:
        # Generate synthetic data using v2 model
        observed_flux, error, _ = generate_synthetic_data(v2_model.evaluate, theta, wave)
        
        if observed_flux is not None:
            def v2_mcmc_setup():
                return v1_mcmc.vfit(v2_model.evaluate,
                    theta, lb, ub,
                    wave, observed_flux, error,
                    no_of_Chain=50,
                    no_of_steps=500
                )
            
            v2_fitter, v2_timing = profiler.time_operation(
                "MCMC Setup", "v2.0 regular", v2_mcmc_setup
            )
            
            if v2_timing.success:
                print(f"v2.0 regular MCMC setup: {v2_timing.time_ms:.2f} ms")
                mcmc_configs['v2_regular'] = {
                    'fitter': v2_fitter,
                    'data': (wave, observed_flux, error),
                    'model_func': lambda theta, wave: v2_model.evaluate(theta, wave)
                }
    
    # v2 compiled MCMC setup
    if v2_compiled is not None:
        # Generate synthetic data using compiled model
        observed_flux, error, _ = generate_synthetic_data(v2_compiled.model_flux, theta, wave)
        
        if observed_flux is not None:
            def v2_compiled_mcmc_setup():
                return v1_mcmc.vfit(  # Using same vfit_mcmc interface
                    v2_compiled.model_flux,
                    theta, lb, ub,
                    wave, observed_flux, error,
                    no_of_Chain=50,
                    no_of_steps=500
                )
            
            v2_compiled_fitter, v2_compiled_timing = profiler.time_operation(
                "MCMC Setup", "v2.0 compiled", v2_compiled_mcmc_setup
            )
            
            if v2_compiled_timing.success:
                print(f"v2.0 compiled MCMC setup: {v2_compiled_timing.time_ms:.2f} ms")
                mcmc_configs['v2_compiled'] = {
                    'fitter': v2_compiled_fitter,
                    'data': (wave, observed_flux, error),
                    'model_func': v2_compiled.model_flux
                }
    
    return mcmc_configs


def benchmark_mcmc_serial_vs_parallel(profiler: PerformanceProfiler, mcmc_configs):
    """Benchmark MCMC performance in serial vs parallel modes."""
    print("\n" + "=" * 60)
    print("BENCHMARK 5: MCMC Serial vs Parallel Performance")
    print("=" * 60)
    
    # MCMC settings
    n_steps = 500  # Reasonable for timing
    n_walkers = 50
    
    print(f"Running {n_steps} MCMC steps with {n_walkers} walkers...")
    print("Testing both serial (use_pool=False) and parallel (use_pool=True) modes")
    
    for config_name, config in mcmc_configs.items():
        print(f"\n--- {config_name.upper()} ---")
        fitter = config['fitter']
        
        # Update fitter settings
        fitter.no_of_steps = n_steps
        fitter.no_of_Chain = n_walkers
        
        # Test serial mode
        def run_serial_mcmc():
            fitter.runmcmc(optimize=False, verbose=False, use_pool=False)
            return fitter
        
        _, serial_timing = profiler.time_operation(
            "MCMC Serial", f"{config_name}", run_serial_mcmc
        )
        
        if serial_timing.success:
            print(f"  Serial mode: {serial_timing.time_ms:.0f} ms "
                  f"({serial_timing.time_ms/n_steps:.2f} ms/step)")
            
            # Reset fitter for parallel test
            fitter = config['fitter']  # Get fresh fitter
            fitter.no_of_steps = n_steps
            fitter.no_of_Chain = n_walkers
            
            # Test parallel mode
            def run_parallel_mcmc():
                fitter.runmcmc(optimize=False, verbose=False, use_pool=True)
                return fitter
            
            _, parallel_timing = profiler.time_operation(
                "MCMC Parallel", f"{config_name}", run_parallel_mcmc
            )
            
            if parallel_timing.success:
                print(f"  Parallel mode: {parallel_timing.time_ms:.0f} ms "
                      f"({parallel_timing.time_ms/n_steps:.2f} ms/step)")
                
                # Calculate speedup
                speedup = serial_timing.time_ms / parallel_timing.time_ms
                print(f"  Parallel speedup: {speedup:.2f}x")
            else:
                print(f"  Parallel mode: FAILED")
        else:
            print(f"  Serial mode: FAILED")


def benchmark_likelihood_calculation(profiler: PerformanceProfiler, mcmc_configs):
    """Benchmark likelihood calculation speed for different models."""
    print("\n" + "=" * 60)
    print("BENCHMARK 6: Likelihood Calculation Speed")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    theta = test_case['theta']
    
    # Test likelihood calculation speed
    n_evaluations = 100
    print(f"Testing {n_evaluations} likelihood evaluations per model...")
    
    for config_name, config in mcmc_configs.items():
        fitter = config['fitter']
        
        def benchmark_likelihood():
            # Perform multiple likelihood evaluations
            for _ in range(n_evaluations):
                # Add small random perturbations
                theta_perturbed = theta + 0.01 * np.random.randn(len(theta))
                _ = fitter.lnprob(theta_perturbed)
        
        _, timing = profiler.time_operation(
            "Likelihood Calculation", config_name, benchmark_likelihood
        )
        
        if timing.success:
            print(f"{config_name}: {timing.time_ms:.2f} ms total "
                  f"({timing.time_ms/n_evaluations:.3f} ms/eval)")


def analyze_performance_results(profiler: PerformanceProfiler):
    """Analyze and summarize all performance results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Group results by operation
    operations = {}
    for result in profiler.results:
        if result.operation not in operations:
            operations[result.operation] = []
        operations[result.operation].append(result)
    
    # Analyze each operation
    for op_name, timings in operations.items():
        successful_timings = [t for t in timings if t.success]
        if not successful_timings:
            continue
            
        print(f"\n{op_name}:")
        print("-" * 40)
        
        # Sort by performance
        successful_timings.sort(key=lambda x: x.time_ms)
        
        for timing in successful_timings:
            print(f"  {timing.version:<15}: {timing.time_ms:>8.2f} ms")
        
        # Find best performer
        if len(successful_timings) > 1:
            fastest = successful_timings[0]
            slowest = successful_timings[-1]
            speedup = slowest.time_ms / fastest.time_ms
            print(f"  → {fastest.version} is {speedup:.1f}x faster than {slowest.version}")
    
    # Overall recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Find v2 compiled performance
    v2_compiled_evals = [r for r in profiler.results 
                        if 'compiled' in r.version and 'Evaluation' in r.operation and r.success]
    v1_evals = [r for r in profiler.results 
               if r.version == 'v1.0' and 'Evaluation' in r.operation and r.success]
    
    if v2_compiled_evals and v1_evals:
        v2_time = min(r.time_ms for r in v2_compiled_evals)
        v1_time = min(r.time_ms for r in v1_evals)
        
        if v2_time < v1_time:
            speedup = v1_time / v2_time
            print(f"🎉 v2.0 compiled models are {speedup:.1f}x faster than v1.0!")
            print("   Recommendation: Use v2.0 with compilation for production work")
        else:
            slowdown = v2_time / v1_time
            print(f"⚠ v2.0 compiled models are {slowdown:.1f}x slower than v1.0")
            print("   Recommendation: Optimize v2.0 compilation or use v1.0 for now")
    
    # MCMC performance analysis
    mcmc_results = [r for r in profiler.results if 'MCMC' in r.operation and r.success]
    if mcmc_results:
        print(f"\n🔧 MCMC Performance:")
        serial_results = [r for r in mcmc_results if 'Serial' in r.operation]
        parallel_results = [r for r in mcmc_results if 'Parallel' in r.operation]
        
        if serial_results and parallel_results:
            avg_serial = np.mean([r.time_ms for r in serial_results])
            avg_parallel = np.mean([r.time_ms for r in parallel_results])
            parallel_speedup = avg_serial / avg_parallel
            
            print(f"   Average parallel speedup: {parallel_speedup:.1f}x")
            if parallel_speedup > 1.5:
                print("   Recommendation: Use parallel MCMC for production runs")
            else:
                print("   Recommendation: Parallel overhead may not be worth it for small problems")


def create_performance_plots(profiler: PerformanceProfiler):
    """Create visualization of performance results."""
    print("\n" + "=" * 60)
    print("CREATING PERFORMANCE PLOTS")
    print("=" * 60)
    
    # Group results by operation type
    eval_results = [r for r in profiler.results if 'Evaluation' in r.operation and r.success]
    mcmc_results = [r for r in profiler.results if 'MCMC' in r.operation and r.success]
    
    if not eval_results and not mcmc_results:
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Model evaluation performance
    if eval_results:
        versions = [r.version for r in eval_results]
        times = [r.time_ms for r in eval_results]
        colors = ['skyblue' if 'v1' in v else 'lightcoral' if 'regular' in v else 'orange' 
                 for v in versions]
        
        bars1 = axes[0].bar(versions, times, color=colors)
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Model Evaluation Performance')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars1, times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                        f'{time_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: MCMC performance comparison
    if mcmc_results:
        # Group MCMC results by version and mode
        mcmc_data = {}
        for r in mcmc_results:
            if 'Serial' in r.operation:
                mode = 'Serial'
            elif 'Parallel' in r.operation:
                mode = 'Parallel'
            else:
                continue
                
            version = r.version
            if version not in mcmc_data:
                mcmc_data[version] = {}
            mcmc_data[version][mode] = r.time_ms
        
        # Create grouped bar chart
        versions = list(mcmc_data.keys())
        serial_times = [mcmc_data[v].get('Serial', 0) for v in versions]
        parallel_times = [mcmc_data[v].get('Parallel', 0) for v in versions]
        
        x = np.arange(len(versions))
        width = 0.35
        
        axes[1].bar(x - width/2, serial_times, width, label='Serial', color='lightblue')
        axes[1].bar(x + width/2, parallel_times, width, label='Parallel', color='lightgreen')
        
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_title('MCMC Performance: Serial vs Parallel')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(versions, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Performance plots generated")


def main():
    """Main performance comparison function."""
    print("rbvfit Performance Comparison: v1.0 vs v2.0 with MCMC Focus")
    print("=" * 70)
    print("Testing model evaluation and MCMC performance in serial/parallel modes")
    
    if not V1_AVAILABLE and not V2_AVAILABLE:
        print("❌ Neither v1.0 nor v2.0 available. Cannot run comparison.")
        return
    
    profiler = PerformanceProfiler()
    
    try:
        # Run benchmarks
        print("\n🚀 Starting performance benchmarks...")
        
        # 1. Model creation
        v1_model, v2_model = benchmark_model_creation(profiler)
        
        # 2. Model compilation (v2.0 only)
        v2_compiled = benchmark_model_compilation(profiler, v2_model)
        
        # 3. Single evaluations
        results = benchmark_single_evaluation(profiler, v1_model, v2_model, v2_compiled)
        
        # 4. MCMC setup
        mcmc_configs = benchmark_mcmc_setup(profiler, v1_model, v2_model, v2_compiled)
        
        if mcmc_configs:
            # 5. Serial vs Parallel MCMC
            benchmark_mcmc_serial_vs_parallel(profiler, mcmc_configs)
            
            # 6. Likelihood calculation speed
            benchmark_likelihood_calculation(profiler, mcmc_configs)
        
        # Analysis and visualization
        print(profiler.get_summary())
        analyze_performance_results(profiler)
        create_performance_plots(profiler)
        
        print("\n" + "=" * 70)
        print("🏁 PERFORMANCE COMPARISON COMPLETE")
        print("=" * 70)
        print("✅ Benchmarks completed successfully")
        print("📊 Performance plots generated")
        print("💡 Check analysis and recommendations above")
        
        if not V1_AVAILABLE:
            print("⚠️  v1.0 not available - only v2.0 benchmarked")
        elif not V2_AVAILABLE:
            print("⚠️  v2.0 not available - only v1.0 benchmarked")
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("⚠️  Partial results available:")
        if profiler.results:
            print(profiler.get_summary())
        print("=" * 70)


if __name__ == "__main__":
    main()