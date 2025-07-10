#!/usr/bin/env python3
"""
VoigtModel Performance Profiler

This script profiles the performance of VoigtModel creation, compilation,
and evaluation to identify bottlenecks.

Usage:
    python voigt_profiler.py

Requirements:
    - Your rbvfit package with VoigtModel
    - line_profiler (optional, for detailed profiling): pip install line_profiler
"""

import time
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add your rbvfit path if needed
# sys.path.append('/path/to/your/rbvfit/src')

try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels
except ImportError as e:
    print(f"Error importing rbvfit modules: {e}")
    print("Make sure rbvfit is in your Python path")
    sys.exit(1)

# Try to import line_profiler for detailed profiling
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False
    print("line_profiler not available. Install with: pip install line_profiler")


class VoigtProfiler:
    """
    Comprehensive profiler for VoigtModel performance analysis.
    """
    
    def __init__(self):
        self.results = {}
        self.timings = {}
        
    def create_test_config(self, complexity: str = "simple") -> FitConfiguration:
        """
        Create test configurations of varying complexity.
        
        Parameters
        ----------
        complexity : str
            'simple', 'medium', or 'complex'
        """
        config = FitConfiguration()
        
        if complexity == "simple":
            # Single system, single ion, single component
            config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=1)
            
        elif complexity == "medium":
            # Single system, multiple ions, multiple components
            config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=2)
            config.add_system(z=0.1, ion='FeII', transitions=[2600.17, 2586.65], components=1)
            
        elif complexity == "complex":
            # Multiple systems, multiple ions, multiple components
            config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=3)
            config.add_system(z=0.1, ion='FeII', transitions=[2600.17, 2586.65, 2374.46], components=2)
            config.add_system(z=0.05, ion='SiII', transitions=[1526.71, 1808.01], components=1)
            config.add_system(z=0.2, ion='OVI', transitions=[1031.93, 1037.62], components=2)
            
        return config
    
    def create_test_wavelength_grid(self, n_points: int = 2000) -> np.ndarray:
        """Create a test wavelength grid."""
        return np.linspace(1200, 1700, n_points)
    
    def create_test_parameters(self, config: FitConfiguration) -> np.ndarray:
        """Create realistic test parameters for the configuration."""
        structure = config.get_parameter_structure()
        n_params = structure['total_parameters']
        n_components = n_params // 3
        
        # Generate realistic parameters
        N_values = np.random.uniform(13.0, 15.0, n_components)  # log N
        b_values = np.random.uniform(10.0, 50.0, n_components)  # b in km/s
        v_values = np.random.uniform(-100.0, 100.0, n_components)  # v in km/s
        
        theta = np.concatenate([N_values, b_values, v_values])
        return theta
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    
    def profile_model_creation(self, complexity: str = "medium"):
        """Profile VoigtModel creation and setup."""
        print(f"\n{'='*60}")
        print(f"PROFILING MODEL CREATION ({complexity.upper()})")
        print(f"{'='*60}")
        
        # Create configuration
        config, config_time = self.time_function(self.create_test_config, complexity)
        print(f"✓ Configuration creation: {config_time*1000:.2f} ms")
        
        # Create wavelength grid
        wave, wave_time = self.time_function(self.create_test_wavelength_grid, 2000)
        print(f"✓ Wavelength grid creation: {wave_time*1000:.2f} ms")
        
        # Convert FWHM
        FWHM_vel = 18.0  # km/s
        fwhm_pixels, fwhm_time = self.time_function(mean_fwhm_pixels, FWHM_vel, wave)
        print(f"✓ FWHM conversion: {fwhm_time*1000:.2f} ms")
        
        # Create VoigtModel
        model, model_time = self.time_function(VoigtModel, config, str(fwhm_pixels))
        print(f"✓ VoigtModel creation: {model_time*1000:.2f} ms")
        
        # Compilation
        compiled_model, compile_time = self.time_function(model.compile, verbose=False)
        print(f"✓ Model compilation: {compile_time*1000:.2f} ms")
        
        # Store results
        self.timings[f'{complexity}_creation'] = {
            'config': config_time,
            'wavelength': wave_time,
            'fwhm_conversion': fwhm_time,
            'model_init': model_time,
            'compilation': compile_time,
            'total': config_time + wave_time + fwhm_time + model_time + compile_time
        }
        
        print(f"\n📊 Total creation time: {self.timings[f'{complexity}_creation']['total']*1000:.2f} ms")
        
        return model, compiled_model, config, wave
    
    def profile_model_evaluation(self, model, compiled_model, config, wave, n_evaluations: int = 100):
        """Profile both uncompiled and compiled model evaluation performance."""
        print(f"\n{'='*60}")
        print(f"PROFILING MODEL EVALUATION ({n_evaluations} evaluations)")
        print(f"{'='*60}")
        
        # Create test parameters
        theta = self.create_test_parameters(config)
        print(f"✓ Test parameters created: {len(theta)} parameters")
        
        # Compare uncompiled vs compiled model performance
        print(f"\n--- UNCOMPILED MODEL ---")
        flux_uncompiled, uncompiled_time = self.time_function(model.evaluate, theta, wave)
        print(f"✓ Uncompiled single evaluation: {uncompiled_time*1000:.2f} ms")
        
        print(f"\n--- COMPILED MODEL ---")
        flux_compiled, compiled_time = self.time_function(compiled_model.model_flux, theta, wave)
        print(f"✓ Compiled single evaluation: {compiled_time*1000:.2f} ms")
        
        speedup = uncompiled_time / compiled_time if compiled_time > 0 else float('inf')
        print(f"✓ Compilation speedup: {speedup:.2f}x")
        
        # Test caching behavior (same wavelength grid)
        print(f"\n--- CACHING TEST (same wavelength grid) ---")
        cache_times = []
        for i in range(5):
            theta_perturbed = theta + np.random.normal(0, 0.01, len(theta))
            _, eval_time = self.time_function(compiled_model.model_flux, theta_perturbed, wave)
            cache_times.append(eval_time)
            print(f"  Evaluation {i+1}: {eval_time*1000:.2f} ms")
        
        first_eval = cache_times[0]
        subsequent_avg = np.mean(cache_times[1:])
        cache_speedup = first_eval / subsequent_avg if subsequent_avg > 0 else 1.0
        print(f"✓ First evaluation: {first_eval*1000:.2f} ms")
        print(f"✓ Subsequent average: {subsequent_avg*1000:.2f} ms")
        print(f"✓ Caching speedup: {cache_speedup:.2f}x")
        
        # Test with different wavelength grid (cache miss)
        print(f"\n--- CACHE MISS TEST (different wavelength grid) ---")
        wave_different = wave * 1.1  # Slightly different grid
        _, cache_miss_time = self.time_function(compiled_model.model_flux, theta, wave_different)
        print(f"✓ Different wavelength grid: {cache_miss_time*1000:.2f} ms")
        
        # Time multiple evaluations (MCMC simulation)
        print(f"\n--- MCMC SIMULATION ({n_evaluations} evaluations) ---")
        start_time = time.perf_counter()
        for i in range(n_evaluations):
            # Slightly perturb parameters to simulate MCMC
            theta_perturbed = theta + np.random.normal(0, 0.01, len(theta))
            flux = compiled_model.model_flux(theta_perturbed, wave)
        multi_time = time.perf_counter() - start_time
        
        avg_time = multi_time / n_evaluations
        print(f"✓ {n_evaluations} evaluations: {multi_time*1000:.2f} ms total")
        print(f"✓ Average per evaluation: {avg_time*1000:.2f} ms")
        print(f"✓ Evaluations per second: {1/avg_time:.1f}")
        
        # Test function call overhead
        print(f"\n--- FUNCTION CALL OVERHEAD ---")
        # Time just the function call without computation
        dummy_theta = theta.copy()
        dummy_wave = wave[:10]  # Very small array
        _, overhead_time = self.time_function(compiled_model.model_flux, dummy_theta, dummy_wave)
        print(f"✓ Function call overhead (10 wavelengths): {overhead_time*1000:.2f} ms")
        
        # Store results
        self.timings['evaluation'] = {
            'uncompiled': uncompiled_time,
            'compiled': compiled_time,
            'compilation_speedup': speedup,
            'first_eval': first_eval,
            'subsequent_avg': subsequent_avg,
            'cache_speedup': cache_speedup,
            'cache_miss': cache_miss_time,
            'single': compiled_time,
            'average': avg_time,
            'total_multi': multi_time,
            'eval_per_sec': 1/avg_time,
            'overhead': overhead_time
        }
        
        return flux_compiled
    
    def profile_with_cprofile(self, compiled_model, config, wave, n_evaluations: int = 50):
        """Profile using cProfile for detailed function-level analysis."""
        print(f"\n{'='*60}")
        print(f"DETAILED PROFILING WITH cProfile")
        print(f"{'='*60}")
        
        theta = self.create_test_parameters(config)
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the evaluation loop
        profiler.enable()
        for i in range(n_evaluations):
            theta_perturbed = theta + np.random.normal(0, 0.01, len(theta))
            flux = compiled_model.model_flux(theta_perturbed, wave)
        profiler.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        print(profile_output)
        
        # Save to file
        with open('voigt_profile_results.txt', 'w') as f:
            f.write(profile_output)
        print(f"✓ Detailed profile saved to 'voigt_profile_results.txt'")
        
        return profile_output
    
    def profile_memory_usage(self, compiled_model, config, wave):
        """Profile memory usage during evaluation."""
        print(f"\n{'='*60}")
        print(f"MEMORY USAGE ANALYSIS")
        print(f"{'='*60}")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            theta = self.create_test_parameters(config)
            
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run evaluation
            flux = compiled_model.model_flux(theta, wave)
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"✓ Memory before evaluation: {mem_before:.1f} MB")
            print(f"✓ Memory after evaluation: {mem_after:.1f} MB")
            print(f"✓ Memory increase: {mem_after - mem_before:.1f} MB")
            
            # Estimate array sizes
            wave_size = wave.nbytes / 1024 / 1024
            flux_size = flux.nbytes / 1024 / 1024
            
            print(f"✓ Wavelength array size: {wave_size:.2f} MB")
            print(f"✓ Flux array size: {flux_size:.2f} MB")
            
        except ImportError:
            print("❌ psutil not available. Install with: pip install psutil")
    
    def profile_scaling(self):
        """Profile how performance scales with problem size."""
        print(f"\n{'='*60}")
        print(f"SCALING ANALYSIS")
        print(f"{'='*60}")
        
        complexities = ['simple', 'medium', 'complex']
        wavelength_sizes = [500, 1000, 2000, 4000]
        
        scaling_results = {}
        
        for complexity in complexities:
            print(f"\n--- {complexity.upper()} MODEL ---")
            
            # Create model once
            config = self.create_test_config(complexity)
            model = VoigtModel(config, '6.5')
            compiled_model = model.compile(verbose=False)
            theta = self.create_test_parameters(config)
            
            structure = config.get_parameter_structure()
            print(f"Parameters: {structure['total_parameters']}, Lines: {model.n_lines}")
            
            scaling_results[complexity] = {}
            
            for n_wave in wavelength_sizes:
                wave = self.create_test_wavelength_grid(n_wave)
                
                # Time multiple evaluations
                start_time = time.perf_counter()
                for i in range(10):
                    flux = compiled_model.model_flux(theta, wave)
                elapsed = time.perf_counter() - start_time
                
                avg_time = elapsed / 10
                scaling_results[complexity][n_wave] = avg_time
                
                print(f"  {n_wave:4d} wavelengths: {avg_time*1000:6.2f} ms/eval")
        
        # Plot scaling results
        self.plot_scaling_results(scaling_results)
        
        return scaling_results
    
    def plot_scaling_results(self, scaling_results):
        """Plot scaling analysis results."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Performance vs wavelength size
            plt.subplot(2, 2, 1)
            for complexity, results in scaling_results.items():
                wavelengths = list(results.keys())
                times = [results[w] * 1000 for w in wavelengths]  # Convert to ms
                plt.plot(wavelengths, times, 'o-', label=complexity, linewidth=2, markersize=6)
            
            plt.xlabel('Number of Wavelength Points')
            plt.ylabel('Evaluation Time (ms)')
            plt.title('Performance vs Wavelength Grid Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Evaluations per second
            plt.subplot(2, 2, 2)
            for complexity, results in scaling_results.items():
                wavelengths = list(results.keys())
                eval_per_sec = [1/results[w] for w in wavelengths]
                plt.plot(wavelengths, eval_per_sec, 's-', label=complexity, linewidth=2, markersize=6)
            
            plt.xlabel('Number of Wavelength Points')
            plt.ylabel('Evaluations per Second')
            plt.title('Throughput vs Wavelength Grid Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Timing breakdown
            plt.subplot(2, 2, 3)
            if hasattr(self, 'timings'):
                categories = []
                times = []
                
                for complexity in ['simple', 'medium', 'complex']:
                    if f'{complexity}_creation' in self.timings:
                        timing_data = self.timings[f'{complexity}_creation']
                        categories.extend([f'{complexity}_config', f'{complexity}_model', f'{complexity}_compile'])
                        times.extend([timing_data['config']*1000, timing_data['model_init']*1000, 
                                    timing_data['compilation']*1000])
                
                if categories:
                    plt.bar(categories, times)
                    plt.xticks(rotation=45)
                    plt.ylabel('Time (ms)')
                    plt.title('Model Creation Breakdown')
            
            # Plot 4: Summary
            plt.subplot(2, 2, 4)
            if 'evaluation' in self.timings:
                eval_data = self.timings['evaluation']
                metrics = ['Single Eval (ms)', 'Avg Eval (ms)', 'Evals/sec']
                values = [eval_data['single']*1000, eval_data['average']*1000, eval_data['eval_per_sec']]
                
                bars = plt.bar(metrics, values)
                plt.ylabel('Value')
                plt.title('Evaluation Performance Summary')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('voigt_profiling_results.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Scaling plots saved to 'voigt_profiling_results.png'")
            
        except ImportError:
            print("❌ matplotlib not available for plotting")
    
    def run_comprehensive_profile(self):
        """Run comprehensive profiling analysis."""
        print(f"🚀 COMPREHENSIVE VOIGT MODEL PROFILING")
        print(f"{'='*80}")
        
        # Profile model creation for different complexities
        models = {}
        for complexity in ['simple', 'medium', 'complex']:
            model, compiled_model, config, wave = self.profile_model_creation(complexity)
            models[complexity] = (model, compiled_model, config, wave)
        
        # Use medium complexity for detailed analysis
        model, compiled_model, config, wave = models['medium']
        
        # Profile evaluation performance (now tests both uncompiled and compiled)
        self.profile_model_evaluation(model, compiled_model, config, wave, n_evaluations=100)
        
        # Detailed profiling
        self.profile_with_cprofile(compiled_model, config, wave, n_evaluations=50)
        
        # Memory usage
        self.profile_memory_usage(compiled_model, config, wave)
        
        # Scaling analysis
        self.profile_scaling()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of profiling results."""
        print(f"\n{'='*60}")
        print(f"PROFILING SUMMARY")
        print(f"{'='*60}")
        
        if 'evaluation' in self.timings:
            eval_data = self.timings['evaluation']
            print(f"📊 Model Evaluation Performance:")
            print(f"  • Uncompiled evaluation: {eval_data['uncompiled']*1000:.2f} ms")
            print(f"  • Compiled evaluation: {eval_data['compiled']*1000:.2f} ms")
            print(f"  • Compilation speedup: {eval_data['compilation_speedup']:.2f}x")
            print(f"  • First eval (cache miss): {eval_data['first_eval']*1000:.2f} ms")
            print(f"  • Subsequent avg (cache hit): {eval_data['subsequent_avg']*1000:.2f} ms")
            print(f"  • Caching speedup: {eval_data['cache_speedup']:.2f}x")
            print(f"  • MCMC average: {eval_data['average']*1000:.2f} ms")
            print(f"  • Evaluations per second: {eval_data['eval_per_sec']:.1f}")
            print(f"  • Function call overhead: {eval_data['overhead']*1000:.2f} ms")
        
        print(f"\n🎯 Bottleneck Analysis:")
        print(f"  • Check 'voigt_profile_results.txt' for detailed function timings")
        print(f"  • Check 'voigt_profiling_results.png' for scaling analysis")
        
        print(f"\n💡 Optimization Recommendations:")
        if 'evaluation' in self.timings and self.timings['evaluation']['eval_per_sec'] < 100:
            print(f"  • Model evaluation is slow (<100 evals/sec)")
            print(f"  • Consider optimizing array operations and caching")
        else:
            print(f"  • Model evaluation performance looks good!")
        
        print(f"\n📁 Files created:")
        print(f"  • voigt_profile_results.txt - Detailed function profiling")
        print(f"  • voigt_profiling_results.png - Performance scaling plots")


def main():
    """Main profiling routine."""
    profiler = VoigtProfiler()
    
    try:
        profiler.run_comprehensive_profile()
    except KeyboardInterrupt:
        print("\n⏹️  Profiling interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()