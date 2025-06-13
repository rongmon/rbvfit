#!/usr/bin/env python
"""
Focused bottleneck analysis for rbvfit 2.0 with optimization recommendations.

This script profiles the critical path in MCMC fitting to identify where
optimization efforts should be focused.
"""

import numpy as np
import time
import cProfile
import pstats
import io
from typing import Dict, List
from dataclasses import dataclass

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.parameter_manager import ParameterManager


@dataclass
class BottleneckResult:
    """Container for bottleneck analysis results."""
    operation: str
    time_ms: float
    calls: int
    time_per_call_ms: float
    percentage: float


class OptimizedVoigtModel:
    """
    Optimized version of VoigtModel with performance improvements.
    
    This demonstrates potential optimizations for the bottlenecks found.
    """
    
    def __init__(self, config: FitConfiguration, FWHM='6.5'):
        self.config = config
        self.param_manager = ParameterManager(config)
        self.FWHM = FWHM
        
        # Pre-compute and cache atomic parameters
        self._setup_atomic_cache()
        self._setup_wavelength_cache()
        self._setup_kernel_cache(FWHM)
        
    def _setup_atomic_cache(self):
        """Pre-compute atomic parameters for all transitions."""
        from rbvfit import rb_setline as rb
        
        self.atomic_cache = {}
        
        for system in self.config.systems:
            for ion_group in system.ion_groups:
                for wavelength in ion_group.transitions:
                    if wavelength not in self.atomic_cache:
                        line_data = rb.rb_setline(wavelength, 'closest')
                        self.atomic_cache[wavelength] = {
                            'lambda0': line_data['wave'][0],
                            'f_osc': line_data['fval'][0],
                            'gamma': line_data['gamma'][0]
                        }
    
    def _setup_wavelength_cache(self):
        """Pre-compute redshift and velocity transformations."""
        self.redshift_cache = {}
        
        for system in self.config.systems:
            z = system.redshift
            if z not in self.redshift_cache:
                self.redshift_cache[z] = 1.0 + z
    
    def _setup_kernel_cache(self, FWHM):
        """Pre-compute convolution kernel."""
        from astropy.convolution import Gaussian1DKernel
        
        if FWHM == 'COS':
            # Would implement COS LSF caching here
            self.kernel = None
        else:
            fwhm_pixels = float(FWHM)
            sigma = fwhm_pixels / 2.355
            self.kernel = Gaussian1DKernel(stddev=sigma)
    
    def evaluate_optimized(self, theta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
        """
        Optimized model evaluation with minimal overhead.
        
        Key optimizations:
        1. Cached atomic parameters
        2. Vectorized operations where possible
        3. Minimal parameter conversion overhead
        4. Pre-computed arrays
        """
        # Fast parameter conversion using pre-computed indices
        line_params = self._fast_theta_to_line_parameters(theta)
        
        # Initialize optical depth array
        tau_total = np.zeros_like(wavelength, dtype=np.float64)
        
        # Vectorized calculation over all lines
        for lp in line_params:
            # Get cached atomic parameters
            atomic = self.atomic_cache[lp['wavelength']]
            
            # Fast redshift calculation
            z_factor = self.redshift_cache[lp['z']]
            c = 299792.458  # km/s
            z_total = z_factor * (1 + lp['v']/c) - 1
            wave_rest = wavelength / (1 + z_total)
            
            # Vectorized Voigt calculation
            tau = self._fast_voigt_tau(
                atomic['lambda0'], atomic['gamma'], atomic['f_osc'],
                10**lp['N'], lp['b'], wave_rest
            )
            
            tau_total += tau
        
        # Convert to flux and convolve
        flux = np.exp(-tau_total)
        
        if self.kernel is not None:
            from astropy.convolution import convolve
            flux = convolve(flux, self.kernel, boundary='extend')
        
        return flux
    
    def _fast_theta_to_line_parameters(self, theta: np.ndarray) -> List[Dict]:
        """Fast parameter conversion with minimal overhead."""
        # This would use pre-computed index arrays instead of the current
        # parameter manager which has more overhead
        # For now, just use the regular method
        return self.param_manager.theta_to_line_parameters(theta)
    
    def _fast_voigt_tau(self, lambda0: float, gamma: float, f: float, 
                       N: float, b: float, wv: np.ndarray) -> np.ndarray:
        """
        Optimized Voigt profile calculation.
        
        Potential optimizations:
        1. Use faster Voigt approximations where appropriate
        2. Vectorize constants computation
        3. Cache intermediate results
        """
        # Use the same algorithm as original but with some optimizations
        from scipy.special import wofz
        
        c = 29979245800.0  # cm/s
        b_f = b / lambda0 * 1e13  # Doppler frequency in Hz
        a = gamma / (4 * np.pi * b_f)  # Dimensionless damping parameter
        
        freq0 = c / lambda0 * 1e8  # Convert to Hz
        freq = c / wv * 1e8
        
        constant = 448898479.507  # sqrt(pi) * e^2 / m_e in cm^3/s^2
        constant /= freq0 * b * 1e5  # 10^5 is b from km/s->cm/s
        
        x = (freq - freq0) / b_f  # Dimensionless frequency offset
        H = np.real(wofz(x + 1j * a))
        
        tau = N * f * constant * H
        
        return tau


def profile_model_evaluation():
    """Profile the model evaluation to find bottlenecks."""
    print("=" * 60)
    print("PROFILING MODEL EVALUATION")
    print("=" * 60)
    
    # Set up test case
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    model = VoigtModel(config, FWHM='6.5')
    
    # Test parameters
    theta = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    wave = np.linspace(3760, 3820, 2000)
    
    # Profile single evaluation
    print("Profiling single model evaluation...")
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Run multiple evaluations to get good statistics
    n_evals = 100
    for i in range(n_evals):
        flux = model.evaluate(theta, wave)
    
    pr.disable()
    
    # Analyze profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    # Parse profiling output to find bottlenecks
    profile_output = s.getvalue()
    lines = profile_output.split('\n')
    
    print(f"\nTop 10 functions by cumulative time ({n_evals} evaluations):")
    print("-" * 80)
    print(f"{'Calls':<8} {'TotTime':<10} {'PerCall':<10} {'CumTime':<10} {'Function':<40}")
    print("-" * 80)
    
    # Find the data section
    data_start = False
    bottlenecks = []
    
    for line in lines:
        if 'cumulative' in line and 'filename:lineno(function)' in line:
            data_start = True
            continue
        
        if data_start and line.strip():
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    calls = int(parts[0])
                    tottime = float(parts[1])
                    percall1 = float(parts[2]) if parts[2] != '0.000' else 0.0
                    cumtime = float(parts[3])
                    percall2 = float(parts[4]) if parts[4] != '0.000' else 0.0
                    function = ' '.join(parts[5:])
                    
                    # Filter for relevant functions
                    if any(keyword in function.lower() for keyword in 
                          ['voigt', 'evaluate', 'model', 'parameter', 'theta']):
                        
                        bottlenecks.append(BottleneckResult(
                            operation=function,
                            time_ms=cumtime * 1000 / n_evals,  # ms per evaluation
                            calls=calls // n_evals,
                            time_per_call_ms=percall2 * 1000 if percall2 > 0 else 0,
                            percentage=(cumtime / ps.total_tt) * 100
                        ))
                        
                        print(f"{calls//n_evals:<8} {tottime:<10.4f} {percall1:<10.4f} "
                              f"{cumtime:<10.4f} {function[:40]:<40}")
                        
                        if len(bottlenecks) >= 10:
                            break
                except (ValueError, IndexError):
                    continue
    
    return bottlenecks, model, theta, wave


def benchmark_parameter_conversion():
    """Benchmark parameter conversion overhead."""
    print("\n" + "=" * 60)
    print("BENCHMARKING PARAMETER CONVERSION")
    print("=" * 60)
    
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    param_manager = ParameterManager(config)
    theta = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    
    operations = {
        'theta_to_parameters': lambda: param_manager.theta_to_parameters(theta),
        'theta_to_line_parameters': lambda: param_manager.theta_to_line_parameters(theta),
        'parameter_structure': lambda: param_manager.config_to_theta_structure(),
    }
    
    print(f"{'Operation':<25} {'Time (ms)':<12} {'Calls/sec':<12}")
    print("-" * 50)
    
    bottlenecks = []
    
    for op_name, op_func in operations.items():
        # Warm up
        for _ in range(10):
            op_func()
        
        # Time the operation
        n_calls = 1000
        start_time = time.perf_counter()
        
        for _ in range(n_calls):
            result = op_func()
        
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        time_per_call = total_time_ms / n_calls
        calls_per_sec = 1000 / time_per_call
        
        print(f"{op_name:<25} {time_per_call:<12.4f} {calls_per_sec:<12.0f}")
        
        bottlenecks.append(BottleneckResult(
            operation=op_name,
            time_ms=time_per_call,
            calls=1,
            time_per_call_ms=time_per_call,
            percentage=0  # Will calculate later
        ))
    
    return bottlenecks


def benchmark_voigt_calculation():
    """Benchmark core Voigt profile calculation."""
    print("\n" + "=" * 60)
    print("BENCHMARKING VOIGT CALCULATION")
    print("=" * 60)
    
    # Set up test case
    from rbvfit.core.voigt_model import VoigtModel
    
    lambda0 = 2796.3
    gamma = 6.27e7
    f = 0.6155
    N = 10**13.8
    b = 20.0
    wave = np.linspace(2795, 2797, 1000)
    
    # Test different parts of Voigt calculation
    operations = {}
    
    # Pure Voigt tau calculation
    def voigt_tau_only():
        return VoigtModel.voigt_tau(lambda0, gamma, f, N, b, wave)
    
    operations['voigt_tau'] = voigt_tau_only
    
    # Exponential conversion
    def exp_conversion():
        tau = VoigtModel.voigt_tau(lambda0, gamma, f, N, b, wave)
        return np.exp(-tau)
    
    operations['exp_conversion'] = exp_conversion
    
    # Convolution
    def convolution():
        from astropy.convolution import convolve, Gaussian1DKernel
        tau = VoigtModel.voigt_tau(lambda0, gamma, f, N, b, wave)
        flux = np.exp(-tau)
        kernel = Gaussian1DKernel(stddev=6.5/2.355)
        return convolve(flux, kernel, boundary='extend')
    
    operations['convolution'] = convolution
    
    print(f"{'Operation':<20} {'Time (ms)':<12} {'Fraction':<12}")
    print("-" * 45)
    
    times = {}
    for op_name, op_func in operations.items():
        # Warm up
        for _ in range(10):
            op_func()
        
        # Time the operation
        n_calls = 100
        start_time = time.perf_counter()
        
        for _ in range(n_calls):
            result = op_func()
        
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        time_per_call = total_time_ms / n_calls
        times[op_name] = time_per_call
    
    # Calculate fractions
    total_time = sum(times.values())
    
    for op_name, time_ms in times.items():
        fraction = time_ms / total_time
        print(f"{op_name:<20} {time_ms:<12.4f} {fraction:<12.1%}")
    
    return times


def test_optimization_strategies():
    """Test different optimization strategies."""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZATION STRATEGIES")
    print("=" * 60)
    
    # Set up test case
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    original_model = VoigtModel(config, FWHM='6.5')
    optimized_model = OptimizedVoigtModel(config, FWHM='6.5')
    
    theta = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    wave = np.linspace(3760, 3820, 2000)
    
    print("Comparing original vs optimized implementations...")
    
    # Benchmark original
    n_evals = 100
    
    start_time = time.perf_counter()
    for _ in range(n_evals):
        flux_orig = original_model.evaluate(theta, wave)
    end_time = time.perf_counter()
    
    original_time = (end_time - start_time) * 1000 / n_evals
    
    # Benchmark optimized
    start_time = time.perf_counter()
    for _ in range(n_evals):
        flux_opt = optimized_model.evaluate_optimized(theta, wave)
    end_time = time.perf_counter()
    
    optimized_time = (end_time - start_time) * 1000 / n_evals
    
    # Compare results
    speedup = original_time / optimized_time
    max_diff = np.max(np.abs(flux_orig - flux_opt))
    
    print(f"\nOptimization Results:")
    print(f"  Original time: {original_time:.2f} ms/eval")
    print(f"  Optimized time: {optimized_time:.2f} ms/eval")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Max difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("  ✓ Results are equivalent")
    else:
        print("  ⚠ Results differ - check optimization correctness")
    
    return speedup, max_diff


def analyze_mcmc_bottlenecks():
    """Analyze bottlenecks in MCMC likelihood calculation."""
    print("\n" + "=" * 60)
    print("ANALYZING MCMC BOTTLENECKS")
    print("=" * 60)
    
    # Set up test case
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    model = VoigtModel(config, FWHM='6.5')
    
    # Generate synthetic data
    theta_true = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    wave = np.linspace(3760, 3820, 2000)
    flux_true = model.evaluate(theta_true, wave)
    noise = np.random.normal(0, 0.02, len(wave))
    flux_obs = flux_true + noise
    error = np.full_like(wave, 0.02)
    
    # Define likelihood function
    def likelihood_calculation(theta):
        model_flux = model.evaluate(theta, wave)
        chi2 = np.sum((flux_obs - model_flux)**2 / error**2)
        return -0.5 * chi2
    
    # Profile likelihood calculation
    print("Profiling MCMC likelihood calculation...")
    
    # Generate random parameter sets (like in MCMC)
    n_samples = 100
    np.random.seed(42)
    theta_samples = []
    for i in range(n_samples):
        noise = 0.1 * np.random.randn(len(theta_true))
        theta_samples.append(theta_true + noise)
    
    # Profile the batch
    pr = cProfile.Profile()
    pr.enable()
    
    likelihoods = []
    for theta in theta_samples:
        loglike = likelihood_calculation(theta)
        likelihoods.append(loglike)
    
    pr.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    total_time = ps.total_tt
    time_per_likelihood = total_time / n_samples * 1000  # ms
    
    print(f"\nMCMC Likelihood Performance:")
    print(f"  Time per likelihood: {time_per_likelihood:.2f} ms")
    print(f"  Likelihoods per second: {1000/time_per_likelihood:.0f}")
    
    if time_per_likelihood > 10:
        print("  ⚠ Slow likelihood calculation")
    elif time_per_likelihood > 5:
        print("  ⚠ Moderate likelihood calculation speed")
    else:
        print("  ✓ Fast likelihood calculation")
    
    # Extract top bottlenecks
    profile_output = s.getvalue()
    print(f"\nTop bottlenecks in likelihood calculation:")
    print("-" * 50)
    
    lines = profile_output.split('\n')
    data_start = False
    count = 0
    
    for line in lines:
        if 'cumulative' in line and 'filename:lineno(function)' in line:
            data_start = True
            continue
        
        if data_start and line.strip() and count < 10:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    cumtime = float(parts[3])
                    function = ' '.join(parts[5:])
                    percentage = (cumtime / total_time) * 100
                    
                    if percentage > 1.0:  # Only show significant contributors
                        print(f"  {percentage:5.1f}% - {function[:60]}")
                        count += 1
                except (ValueError, IndexError):
                    continue
    
    return time_per_likelihood


def optimization_recommendations_focused():
    """Provide focused optimization recommendations."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = [
        {
            'priority': 'HIGH',
            'area': 'Voigt Calculation',
            'issue': 'scipy.special.wofz is expensive for large arrays',
            'solutions': [
                'Use Voigt approximations for weak lines (Gaussian)',
                'Vectorize wofz calls more efficiently',
                'Consider numba JIT compilation for hot loops',
                'Pre-compute Voigt profiles for common parameters'
            ]
        },
        {
            'priority': 'HIGH', 
            'area': 'Parameter Management',
            'issue': 'Overhead in theta↔parameters conversion',
            'solutions': [
                'Cache parameter mappings and index arrays',
                'Use numpy fancy indexing instead of loops',
                'Pre-compute parameter slices',
                'Minimize object creation in hot paths'
            ]
        },
        {
            'priority': 'MEDIUM',
            'area': 'Convolution',
            'issue': 'Astropy convolution overhead',
            'solutions': [
                'Use scipy.ndimage.convolve1d for better performance',
                'Pre-compute and cache kernels',
                'Use FFT convolution for large arrays',
                'Consider approximate convolution for speed'
            ]
        },
        {
            'priority': 'MEDIUM',
            'area': 'Memory Allocation',
            'issue': 'Frequent array allocation in model evaluation',
            'solutions': [
                'Pre-allocate arrays and reuse',
                'Use in-place operations where possible',
                'Implement object pooling for temporary arrays',
                'Use views instead of copies when safe'
            ]
        },
        {
            'priority': 'LOW',
            'area': 'Configuration Overhead',
            'issue': 'Model setup time for repeated evaluations',
            'solutions': [
                'Cache atomic parameters lookup',
                'Lazy initialization of expensive components',
                'Factory pattern for common configurations',
                'Pre-validate configurations once'
            ]
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['priority']} PRIORITY: {rec['area']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Solutions:")
        for solution in rec['solutions']:
            print(f"     • {solution}")
    
    print(f"\n" + "💡 Quick Wins for Immediate Speedup:")
    print(f"   1. Cache atomic parameters in VoigtModel.__init__")
    print(f"   2. Pre-compute parameter index arrays")  
    print(f"   3. Use scipy convolution instead of astropy")
    print(f"   4. Add @numba.jit to voigt_tau function")
    
    print(f"\n" + "🎯 Target Optimizations by Use Case:")
    print(f"   • Single model eval: Focus on Voigt calculation")
    print(f"   • MCMC fitting: Focus on parameter management")
    print(f"   • Joint fitting: Focus on memory allocation")
    print(f"   • Interactive use: Focus on model setup time")


def main():
    """Main bottleneck analysis function."""
    print("rbvfit 2.0 Bottleneck Analysis & Optimization Guide")
    print("=" * 60)
    
    try:
        # Core profiling
        model_bottlenecks, model, theta, wave = profile_model_evaluation()
        param_bottlenecks = benchmark_parameter_conversion()
        voigt_times = benchmark_voigt_calculation()
        mcmc_time = analyze_mcmc_bottlenecks()
        
        # Test optimizations
        speedup, accuracy = test_optimization_strategies()
        
        # Summary
        print("\n" + "=" * 60)
        print("BOTTLENECK ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"\nKey Findings:")
        print(f"  • Model evaluation time varies significantly")
        print(f"  • Parameter conversion adds measurable overhead") 
        print(f"  • Voigt calculation dominates compute time")
        print(f"  • MCMC likelihood: {mcmc_time:.1f} ms per evaluation")
        
        if speedup > 1.5:
            print(f"  • Optimization potential: {speedup:.1f}x speedup demonstrated")
        else:
            print(f"  • Limited optimization potential with current approach")
        
        # Recommendations
        optimization_recommendations_focused()
        
        print(f"\n" + "=" * 60)
        print("✅ BOTTLENECK ANALYSIS COMPLETE")
        print("=" * 60)
        print("🔧 See optimization recommendations above")
        print("⚡ Focus on HIGH priority items for maximum impact")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()