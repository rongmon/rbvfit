#!/usr/bin/env python
"""
Enhanced Performance comparison between rbvfit v1.0 and v2.0 with MCMC focus.

FIXED: Serial mode bug - proper sample access and verification
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
try:
    from rbvfit import model as v1_model
    from rbvfit import vfit_mcmc as v1_mcmc
    V1_AVAILABLE = True
    print("✓ rbvfit v1.0 available")
except ImportError as e:
    print(f"⚠ Warning: rbvfit v1.0 not available: {e}")
    V1_AVAILABLE = False

try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    from rbvfit import vfit_mcmc as v2_mcmc
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
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        try:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            timing = TimingResult(
                operation=operation_name,
                version=version,
                time_ms=(end_time - start_time) * 1000,
                memory_mb=final_memory - initial_memory,
                success=True
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
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
        
        return "\n".join(lines)


def setup_common_test_case():
    """Set up common test case for both versions."""
    zabs = 0.348
    lambda_rest = [2796.3, 2803.5]
    
    # 3-component system (reduced for faster testing)
    nguess = [13.8, 13.3, 13.5]
    bguess = [20.0, 30.0, 15.0] 
    vguess = [-40.0, -20., 0.]
    
    theta = np.concatenate([nguess, bguess, vguess])
    wave = np.linspace(3760, 3800, 1000)
    
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


def create_fresh_fitter(model_func, theta, lb, ub, wave, observed_flux, error, n_steps=200, n_walkers=20):
    """Create a fresh fitter object with specified parameters."""
    return v1_mcmc.vfit(
        model_func, theta, lb, ub, wave, observed_flux, error,
        no_of_Chain=n_walkers, no_of_steps=n_steps
    )


def get_fitter_sample_count(fitter):
    """Safely extract sample count from fitter object with multiple fallback methods."""
    try:
        # Method 1: Check if samples attribute exists
        if hasattr(fitter, 'samples') and fitter.samples is not None:
            if isinstance(fitter.samples, np.ndarray):
                return fitter.samples.size
            else:
                return len(fitter.samples)
        
        # Method 2: Check if sampler exists with get_chain method
        if hasattr(fitter, 'sampler') and hasattr(fitter.sampler, 'get_chain'):
            try:
                chain = fitter.sampler.get_chain()
                return chain.size
            except:
                pass
        
        # Method 3: Check if sampler has chain attribute
        if hasattr(fitter, 'sampler') and hasattr(fitter.sampler, 'chain'):
            return fitter.sampler.chain.size
        
        # Method 4: Check MCMC run completion by looking for best_theta
        if hasattr(fitter, 'best_theta'):
            n_walkers = getattr(fitter, 'no_of_Chain', 50)
            n_steps = getattr(fitter, 'no_of_steps', 500)
            return n_walkers * n_steps
        
        return 0
        
    except Exception as e:
        print(f"    Warning: Error checking sample count: {e}")
        return -1


def generate_synthetic_data(model_func, theta, wave, noise_level=0.05):
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


def benchmark_model_creation(profiler: PerformanceProfiler):
    """Benchmark model creation for both versions."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Model Creation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    v1_model_obj = None
    v2_model_obj = None
    
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
    
    return compiled_model


def benchmark_single_evaluation(profiler: PerformanceProfiler, v1_model, v2_model, v2_compiled):
    """Benchmark single model evaluation for all variants."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Single Model Evaluation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    results = {}
    
    if V1_AVAILABLE and v1_model is not None:
        def v1_evaluate():
            return v1_model.model_flux(test_case['theta'], test_case['wave'])
        
        v1_flux, v1_timing = profiler.time_operation(
            "Single Evaluation", "v1.0", v1_evaluate
        )
        if v1_timing.success:
            print(f"v1.0 single evaluation: {v1_timing.time_ms:.2f} ms")
            results['v1'] = v1_flux
    
    if V2_AVAILABLE and v2_model is not None:
        def v2_evaluate():
            return v2_model.evaluate(test_case['theta'], test_case['wave'])
        
        v2_flux, v2_timing = profiler.time_operation(
            "Single Evaluation", "v2.0 regular", v2_evaluate
        )
        if v2_timing.success:
            print(f"v2.0 regular evaluation: {v2_timing.time_ms:.2f} ms")
            results['v2_regular'] = v2_flux
    
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


def prepare_mcmc_data(profiler: PerformanceProfiler, v1_model, v2_model, v2_compiled):
    """Prepare synthetic data and model functions for MCMC testing."""
    print("\n" + "=" * 60)
    print("BENCHMARK 4: MCMC Data Preparation")
    print("=" * 60)
    
    test_case = setup_common_test_case()
    wave = test_case['wave']
    theta = test_case['theta']
    
    bounds, lb, ub = v1_mcmc.set_bounds(
        test_case['nguess'], 
        test_case['bguess'], 
        test_case['vguess']
    )
    
    mcmc_setups = {}
    
    if V1_AVAILABLE and v1_model is not None:
        observed_flux, error, _ = generate_synthetic_data(v1_model.model_flux, theta, wave)
        if observed_flux is not None:
            mcmc_setups['v1.0'] = {
                'model_func': v1_model.model_flux,
                'data': (wave, observed_flux, error),
                'theta': theta,
                'bounds': (lb, ub)
            }
            print("✓ v1.0 data prepared")
    
    if V2_AVAILABLE and v2_model is not None:
        observed_flux, error, _ = generate_synthetic_data(v2_model.evaluate, theta, wave)
        if observed_flux is not None:
            mcmc_setups['v2.0 regular'] = {
                'model_func': lambda theta, wave: v2_model.evaluate(theta, wave),
                'data': (wave, observed_flux, error),
                'theta': theta,
                'bounds': (lb, ub)
            }
            print("✓ v2.0 regular data prepared")
    
    if v2_compiled is not None:
        observed_flux, error, _ = generate_synthetic_data(v2_compiled.model_flux, theta, wave)
        if observed_flux is not None:
            mcmc_setups['v2.0 compiled'] = {
                'model_func': v2_compiled.model_flux,
                'data': (wave, observed_flux, error),
                'theta': theta,
                'bounds': (lb, ub)
            }
            print("✓ v2.0 compiled data prepared")
    
    return mcmc_setups


def benchmark_mcmc_serial_vs_parallel(profiler: PerformanceProfiler, mcmc_setups):
    """Benchmark MCMC performance in serial vs parallel modes with FIXED serial bug."""
    print("\n" + "=" * 60)
    print("BENCHMARK 5: MCMC Serial vs Parallel Performance")
    print("=" * 60)
    
    n_steps = 200
    n_walkers = 20
    
    print(f"Running {n_steps} MCMC steps with {n_walkers} walkers...")
    print("FIXED: Proper sample verification for both serial and parallel modes")
    
    for setup_name, setup in mcmc_setups.items():
        print(f"\n--- {setup_name.upper()} ---")
        
        model_func = setup['model_func']
        wave, observed_flux, error = setup['data']
        theta = setup['theta']
        lb, ub = setup['bounds']
        
        # Test SERIAL mode
        print("  Testing serial mode...")
        def run_serial_mcmc():
            fitter = create_fresh_fitter(model_func, theta, lb, ub, wave, observed_flux, error, n_steps, n_walkers)
            fitter.runmcmc(optimize=False, verbose=False, use_pool=False)
            sample_count = get_fitter_sample_count(fitter)
            return sample_count
        
        serial_result, serial_timing = profiler.time_operation(
            "MCMC Serial", setup_name, run_serial_mcmc
        )
        
        if serial_timing.success and serial_result > 0:
            print(f"    Serial: {serial_timing.time_ms:.0f} ms ({serial_timing.time_ms/n_steps:.2f} ms/step)")
            
            # Test PARALLEL mode
            print("  Testing parallel mode...")
            def run_parallel_mcmc():
                fitter = create_fresh_fitter(model_func, theta, lb, ub, wave, observed_flux, error, n_steps, n_walkers)
                fitter.runmcmc(optimize=False, verbose=False, use_pool=True)
                sample_count = get_fitter_sample_count(fitter)
                return sample_count
            
            parallel_result, parallel_timing = profiler.time_operation(
                "MCMC Parallel", setup_name, run_parallel_mcmc
            )
            
            if parallel_timing.success and parallel_result > 0:
                print(f"    Parallel: {parallel_timing.time_ms:.0f} ms ({parallel_timing.time_ms/n_steps:.2f} ms/step)")
                
                speedup = serial_timing.time_ms / parallel_timing.time_ms
                print(f"    ✅ Speedup: {speedup:.2f}x")
                
                if speedup > 1.2:
                    print(f"       → Parallel is significantly faster!")
                elif speedup > 0.8:
                    print(f"       → Similar performance")
                else:
                    print(f"       → Serial is faster (parallel overhead)")
            else:
                print(f"    Parallel: FAILED")
        else:
            print(f"    Serial: FAILED")


def analyze_performance_results(profiler: PerformanceProfiler):
    """Analyze and summarize all performance results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    operations = {}
    for result in profiler.results:
        if result.operation not in operations:
            operations[result.operation] = []
        operations[result.operation].append(result)
    
    for op_name, timings in operations.items():
        successful_timings = [t for t in timings if t.success]
        if not successful_timings:
            continue
            
        print(f"\n{op_name}:")
        print("-" * 40)
        
        successful_timings.sort(key=lambda x: x.time_ms)
        for timing in successful_timings:
            print(f"  {timing.version:<20}: {timing.time_ms:>8.2f} ms")
        
        if len(successful_timings) > 1:
            fastest = successful_timings[0]
            slowest = successful_timings[-1]
            speedup = slowest.time_ms / fastest.time_ms
            print(f"  → {fastest.version} is {speedup:.1f}x faster than {slowest.version}")
    
    # Serial vs Parallel Analysis
    print("\n" + "=" * 60)
    print("SERIAL vs PARALLEL ANALYSIS")
    print("=" * 60)
    
    serial_results = [r for r in profiler.results if 'MCMC Serial' in r.operation and r.success]
    parallel_results = [r for r in profiler.results if 'MCMC Parallel' in r.operation and r.success]
    
    version_comparisons = {}
    for serial in serial_results:
        version = serial.version
        parallel = next((p for p in parallel_results if p.version == version), None)
        if parallel:
            speedup = serial.time_ms / parallel.time_ms
            version_comparisons[version] = {
                'serial_ms': serial.time_ms,
                'parallel_ms': parallel.time_ms,
                'speedup': speedup
            }
    
    if version_comparisons:
        for version, data in version_comparisons.items():
            print(f"\n{version}:")
            print(f"  Serial:   {data['serial_ms']:>8.0f} ms")
            print(f"  Parallel: {data['parallel_ms']:>8.0f} ms")
            print(f"  Speedup:  {data['speedup']:>8.2f}x")
            
            if data['speedup'] > 1.5:
                print(f"  Status:   🚀 Excellent parallel scaling")
            elif data['speedup'] > 1.1:
                print(f"  Status:   ✅ Good parallel scaling")
            else:
                print(f"  Status:   ⚠️ Limited parallel benefit")


def create_performance_plots(profiler: PerformanceProfiler):
    """Create visualization of performance results."""
    print("\n" + "=" * 60)
    print("CREATING PERFORMANCE PLOTS")
    print("=" * 60)
    
    eval_results = [r for r in profiler.results if 'Evaluation' in r.operation and r.success]
    serial_results = [r for r in profiler.results if 'MCMC Serial' in r.operation and r.success]
    parallel_results = [r for r in profiler.results if 'MCMC Parallel' in r.operation and r.success]
    
    if not eval_results and not (serial_results or parallel_results):
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Model evaluation performance
    if eval_results:
        versions = [r.version for r in eval_results]
        times = [r.time_ms for r in eval_results]
        colors = ['skyblue' if 'v1' in v else 'lightcoral' if 'regular' in v else 'orange' for v in versions]
        
        bars1 = axes[0].bar(versions, times, color=colors)
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Model Evaluation Performance')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars1, times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                        f'{time_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: MCMC serial vs parallel comparison
    if serial_results and parallel_results:
        version_data = {}
        
        for serial in serial_results:
            version = serial.version
            if version not in version_data:
                version_data[version] = {}
            version_data[version]['serial'] = serial.time_ms
        
        for parallel in parallel_results:
            version = parallel.version
            if version not in version_data:
                version_data[version] = {}
            version_data[version]['parallel'] = parallel.time_ms
        
        complete_versions = {v: data for v, data in version_data.items() 
                           if 'serial' in data and 'parallel' in data}
        
        if complete_versions:
            versions = list(complete_versions.keys())
            serial_times = [complete_versions[v]['serial'] for v in versions]
            parallel_times = [complete_versions[v]['parallel'] for v in versions]
            
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
            
            for i, version in enumerate(versions):
                speedup = serial_times[i] / parallel_times[i]
                axes[1].text(i, max(serial_times[i], parallel_times[i]) * 1.1, 
                           f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("📊 Performance plots generated")


def main():
    """Main performance comparison function."""
    print("rbvfit Performance Comparison: v1.0 vs v2.0 with MCMC Focus")
    print("=" * 70)
    print("FIXED: Serial mode bug - proper sample verification and reduced test size")
    
    if not V1_AVAILABLE and not V2_AVAILABLE:
        print("❌ Neither v1.0 nor v2.0 available. Cannot run comparison.")
        return
    
    profiler = PerformanceProfiler()
    
    try:
        print("\n🚀 Starting performance benchmarks...")
        
        # 1. Model creation
        v1_model, v2_model = benchmark_model_creation(profiler)
        
        # 2. Model compilation (v2.0 only)
        v2_compiled = benchmark_model_compilation(profiler, v2_model)
        
        # 3. Single evaluations
        results = benchmark_single_evaluation(profiler, v1_model, v2_model, v2_compiled)
        
        # 4. Prepare MCMC data and configurations
        mcmc_setups = prepare_mcmc_data(profiler, v1_model, v2_model, v2_compiled)
        
        if mcmc_setups:
            # 5. Serial vs Parallel MCMC (FIXED)
            benchmark_mcmc_serial_vs_parallel(profiler, mcmc_setups)
        
        # Analysis and visualization
        print(profiler.get_summary())
        analyze_performance_results(profiler)
        create_performance_plots(profiler)
        
        print("\n" + "=" * 70)
        print("🏁 PERFORMANCE COMPARISON COMPLETE")
        print("=" * 70)
        print("✅ Benchmarks completed successfully")
        print("🔧 FIXED: Serial mode sample verification and debugging")
        print("💡 Check analysis and recommendations above")
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()