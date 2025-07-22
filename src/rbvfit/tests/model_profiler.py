#!/usr/bin/env python3
"""
VoigtModel Evaluation Profiler

Focused profiler for model.evaluate() and compiled_model.model_flux() performance.
Tests wavelength array of 2000 points across different model complexities.
"""

import time
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import gc

try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
except ImportError as e:
    print(f"Error importing rbvfit modules: {e}")
    sys.exit(1)


class VoigtEvaluationProfiler:
    """Profile VoigtModel evaluation performance."""
    
    def __init__(self):
        self.results = {}
        self.wave = np.linspace(1200, 1700, 2000)  # Fixed 2000 points
        
    def create_configs(self) -> Dict[str, FitConfiguration]:
        configs = {}
        
        # Simple: 1 system, 1 ion, 1 component
        config = FitConfiguration()
        config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=1)
        configs['simple'] = config
        
        # Medium: 1 system, 2 ions, mixed components  
        config = FitConfiguration()
        config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=2)
        config.add_system(z=0.1, ion='FeII', transitions=[2600.17, 2586.65], components=1)
        configs['medium'] = config
        
        # Complex: 2 systems, 3 ions, multiple components
        config = FitConfiguration()
        config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=3)
        config.add_system(z=0.1, ion='FeII', transitions=[2600.17, 2586.65, 2374.46], components=2)
        config.add_system(z=0.05, ion='SiII', transitions=[1526.71, 1808.01], components=1)
        configs['complex'] = config
        
        # Very Complex: 3 systems, 4 ions, many components
        config = FitConfiguration()
        config.add_system(z=0.05, ion='HI', transitions=[1215.67], components=1)
        config.add_system(z=0.05, ion='CIV', transitions=[1548.20, 1550.77], components=2)
        config.add_system(z=0.1, ion='MgII', transitions=[2796.35, 2803.53], components=3)
        config.add_system(z=0.1, ion='FeII', transitions=[2600.17, 2586.65, 2374.46, 2382.77], components=2)
        config.add_system(z=0.2, ion='OVI', transitions=[1031.93, 1037.62], components=2)
        configs['very_complex'] = config
        
        return configs
    
    def create_theta(self, config: FitConfiguration) -> np.ndarray:
        """Create realistic test parameters."""
        structure = config.get_parameter_structure()
        n_params = structure['total_parameters']
        n_comp = n_params // 3
        
        N_vals = np.random.uniform(13.0, 15.0, n_comp)
        b_vals = np.random.uniform(10.0, 50.0, n_comp) 
        v_vals = np.random.uniform(-100.0, 100.0, n_comp)
        
        return np.concatenate([N_vals, b_vals, v_vals])
    
    def time_eval(self, func, *args):
        """Time single function call."""
        gc.collect()
        start = time.perf_counter()
        result = func(*args)
        return time.perf_counter() - start, result
    
    def profile_single_evaluations(self):
        """Profile single evaluation calls for all complexities."""
        print("EVALUATION PERFORMANCE PROFILING")
        print("="*50)
        
        configs = self.create_configs()
        results = {}
        
        for name, config in configs.items():
            print(f"\n{name.upper()}:")
            
            # Setup
            model = VoigtModel(config, FWHM='6.5')
            compiled_model = model.compile(verbose=False)
            theta = self.create_theta(config)
            
            structure = config.get_parameter_structure()
            n_params = structure['total_parameters']
            n_lines = model.n_lines
            
            print(f"  Lines: {n_lines}, Params: {n_params}")
            
            # Test uncompiled
            uncompiled_time, flux1 = self.time_eval(model.evaluate, theta, self.wave)
            print(f"  Uncompiled: {uncompiled_time*1000:.2f} ms")
            
            # Test compiled
            compiled_time, flux2 = self.time_eval(compiled_model.model_flux, theta, self.wave)
            print(f"  Compiled:   {compiled_time*1000:.2f} ms")
            
            speedup = uncompiled_time / compiled_time if compiled_time > 0 else 0
            print(f"  Speedup:    {speedup:.1f}x")
            
            results[name] = {
                'n_lines': n_lines,
                'n_params': n_params,
                'uncompiled_time': uncompiled_time,
                'compiled_time': compiled_time,
                'speedup': speedup
            }
        
        self.results['single_eval'] = results
        return results
    
    def profile_batch_evaluations(self, n_evals=100):
        """Profile batch evaluations (MCMC-like)."""
        print(f"\nBATCH EVALUATION ({n_evals} calls)")
        print("="*40)
        
        configs = self.create_configs()
        results = {}
        
        for name, config in configs.items():
            model = VoigtModel(config, FWHM='6.5')
            compiled_model = model.compile(verbose=False)
            base_theta = self.create_theta(config)
            
            # Generate parameter variations
            thetas = []
            for i in range(n_evals):
                noise = np.random.normal(0, 0.01, len(base_theta))
                thetas.append(base_theta + noise)
            
            # Time batch uncompiled
            start = time.perf_counter()
            for theta in thetas:
                model.evaluate(theta, self.wave)
            uncompiled_batch = time.perf_counter() - start
            
            # Time batch compiled
            start = time.perf_counter()
            for theta in thetas:
                compiled_model.model_flux(theta, self.wave)
            compiled_batch = time.perf_counter() - start
            
            uncompiled_avg = uncompiled_batch / n_evals
            compiled_avg = compiled_batch / n_evals
            
            print(f"{name:12s}: {uncompiled_avg*1000:6.2f} ms → {compiled_avg*1000:6.2f} ms "
                  f"({uncompiled_avg/compiled_avg:.1f}x)")
            
            results[name] = {
                'uncompiled_avg': uncompiled_avg,
                'compiled_avg': compiled_avg,
                'uncompiled_total': uncompiled_batch,
                'compiled_total': compiled_batch
            }
        
        self.results['batch_eval'] = results
        return results
    
    def profile_wavelength_scaling(self):
        """Test how evaluation scales with wavelength grid size."""
        print("\nWAVELENGTH SCALING")
        print("="*30)
        
        config = self.create_configs()['medium']  # Use medium complexity
        model = VoigtModel(config, FWHM='6.5')
        compiled_model = model.compile(verbose=False)
        theta = self.create_theta(config)
        
        wave_sizes = [500, 1000, 2000, 4000]
        results = {}
        
        print("Wave Points | Compiled Time")
        print("-"*30)
        
        for n_wave in wave_sizes:
            wave = np.linspace(1200, 1700, n_wave)
            eval_time, _ = self.time_eval(compiled_model.model_flux, theta, wave)
            
            print(f"{n_wave:10d} | {eval_time*1000:8.2f} ms")
            results[n_wave] = eval_time
        
        self.results['wavelength_scaling'] = results
        return results
    
    def profile_caching_behavior(self):
        """Test wavelength grid caching behavior."""
        print("\nCACHING BEHAVIOR")
        print("="*25)
        
        config = self.create_configs()['medium']
        model = VoigtModel(config, FWHM='6.5')
        compiled_model = model.compile(verbose=False)
        theta = self.create_theta(config)
        
        # Same wavelength grid (cache hit)
        times_same = []
        for i in range(5):
            eval_time, _ = self.time_eval(compiled_model.model_flux, theta, self.wave)
            times_same.append(eval_time)
            print(f"Same grid {i+1}: {eval_time*1000:.2f} ms")
        
        print()
        
        # Different wavelength grids (cache miss)
        times_diff = []
        for i in range(5):
            wave_shifted = self.wave * (1 + i * 0.001)  # Slight shifts
            eval_time, _ = self.time_eval(compiled_model.model_flux, theta, wave_shifted)
            times_diff.append(eval_time)
            print(f"Diff grid {i+1}: {eval_time*1000:.2f} ms")
        
        avg_same = np.mean(times_same[1:])  # Skip first (warmup)
        avg_diff = np.mean(times_diff)
        
        print(f"\nCache hit avg:  {avg_same*1000:.2f} ms")
        print(f"Cache miss avg: {avg_diff*1000:.2f} ms")
        print(f"Cache benefit:  {avg_diff/avg_same:.1f}x")
        
        self.results['caching'] = {
            'cache_hit_avg': avg_same,
            'cache_miss_avg': avg_diff,
            'cache_benefit': avg_diff/avg_same
        }
    
    def profile_detailed_breakdown(self):
        """Detailed profiling of evaluation internals."""
        print("\nDETAILED BREAKDOWN (cProfile)")
        print("="*40)
        
        config = self.create_configs()['complex']
        model = VoigtModel(config, FWHM='6.5') 
        compiled_model = model.compile(verbose=False)
        theta = self.create_theta(config)
        
        profiler = cProfile.Profile()
        
        # Profile compiled evaluation
        profiler.enable()
        for i in range(50):
            compiled_model.model_flux(theta, self.wave)
        profiler.disable()
        
        # Get results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(15)  # Top 15 functions
        
        profile_output = s.getvalue()
        print(profile_output)
        
        # Save to file
        with open('voigt_eval_profile.txt', 'w') as f:
            f.write("VoigtModel Evaluation Profiling\n")
            f.write("="*40 + "\n\n")
            f.write(profile_output)
        
        print("Detailed profile saved to 'voigt_eval_profile.txt'")
    
    def plot_results(self):
        """Plot profiling results."""
        if 'single_eval' not in self.results:
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('VoigtModel Evaluation Performance', fontsize=14)
            
            # Plot 1: Single evaluation times
            ax = axes[0, 0]
            data = self.results['single_eval']
            complexities = list(data.keys())
            uncompiled = [data[c]['uncompiled_time']*1000 for c in complexities]
            compiled = [data[c]['compiled_time']*1000 for c in complexities]
            
            x = range(len(complexities))
            width = 0.35
            ax.bar([i-width/2 for i in x], uncompiled, width, label='Uncompiled', alpha=0.7)
            ax.bar([i+width/2 for i in x], compiled, width, label='Compiled', alpha=0.7)
            
            ax.set_xlabel('Model Complexity')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Single Evaluation Times')
            ax.set_xticks(x)
            ax.set_xticklabels(complexities)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Speedup factors
            ax = axes[0, 1]
            speedups = [data[c]['speedup'] for c in complexities]
            ax.bar(complexities, speedups, alpha=0.7)
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Compilation Speedup')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Wavelength scaling
            if 'wavelength_scaling' in self.results:
                ax = axes[1, 0]
                data = self.results['wavelength_scaling']
                wave_points = list(data.keys())
                times = [data[n]*1000 for n in wave_points]
                
                ax.plot(wave_points, times, 'o-', linewidth=2)
                ax.set_xlabel('Wavelength Points')
                ax.set_ylabel('Time (ms)')
                ax.set_title('Wavelength Grid Scaling')
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Performance vs complexity
            ax = axes[1, 1]
            if 'batch_eval' in self.results:
                data = self.results['batch_eval']
                complexities = list(data.keys())
                times = [data[c]['compiled_avg']*1000 for c in complexities]
                n_lines = [self.results['single_eval'][c]['n_lines'] for c in complexities]
                
                ax.scatter(n_lines, times, s=80, alpha=0.7)
                for i, comp in enumerate(complexities):
                    ax.annotate(comp, (n_lines[i], times[i]), xytext=(5,5), 
                              textcoords='offset points', fontsize=8)
                
                ax.set_xlabel('Number of Lines')
                ax.set_ylabel('Avg Time (ms)')
                ax.set_title('Performance vs Model Size')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('voigt_eval_performance.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("EVALUATION PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'single_eval' in self.results:
            print("\nSINGLE EVALUATION TIMES:")
            for name, data in self.results['single_eval'].items():
                print(f"  {name:12s}: {data['compiled_time']*1000:6.2f} ms "
                      f"({data['n_lines']:2d} lines, {data['speedup']:.1f}x speedup)")
        
        if 'batch_eval' in self.results:
            print("\nBATCH AVERAGE TIMES:")
            for name, data in self.results['batch_eval'].items():
                print(f"  {name:12s}: {data['compiled_avg']*1000:6.2f} ms/eval")
        
        if 'wavelength_scaling' in self.results:
            print("\nWAVELENGTH SCALING:")
            data = self.results['wavelength_scaling']
            base_time = data[500]
            for n_wave, time_val in data.items():
                factor = time_val / base_time
                print(f"  {n_wave:4d} points: {time_val*1000:6.2f} ms ({factor:.1f}x)")
        
        if 'caching' in self.results:
            data = self.results['caching']
            print(f"\nCACHING PERFORMANCE:")
            print(f"  Cache hit:  {data['cache_hit_avg']*1000:.2f} ms")
            print(f"  Cache miss: {data['cache_miss_avg']*1000:.2f} ms")
            print(f"  Benefit:    {data['cache_benefit']:.1f}x")
        
        print("\nFILES CREATED:")
        print("  • voigt_eval_profile.txt - Detailed function profiling")
        print("  • voigt_eval_performance.png - Performance plots")
    
    def run_full_profile(self):
        """Run complete evaluation profiling."""
        print("VOIGT MODEL EVALUATION PROFILER")
        print("="*50)
        print("Testing 2000-point wavelength grids\n")
        
        self.profile_single_evaluations()
        self.profile_batch_evaluations()
        self.profile_wavelength_scaling()
        self.profile_caching_behavior()
        self.profile_detailed_breakdown()
        self.plot_results()
        self.print_summary()


if __name__ == "__main__":
    profiler = VoigtEvaluationProfiler()
    profiler.run_full_profile()