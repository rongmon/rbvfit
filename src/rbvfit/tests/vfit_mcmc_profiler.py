#!/usr/bin/env python3
"""
Simple VFIT MCMC Performance Profiler - Real Performance Testing

This script profiles the actual performance using your real example data
with realistic MCMC settings: 20 walkers, 500 steps, multiprocessing enabled.

No toy examples, no artificial constraints - just measure where time actually goes.

Usage:
    python simple_vfit_profiler.py
"""

import time
import cProfile
import pstats
import io
import numpy as np
import sys

# Import exactly like your example
try:
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels
    from rbvfit import vfit_mcmc as mc
    from rbcodes.utils.rb_spectrum import rb_spectrum
    from pkg_resources import resource_filename
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure rbvfit and rbcodes are installed and in your path")
    sys.exit(1)

try:
    import zeus
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False


class SimpleVFitProfiler:
    """Simple profiler using real data and real MCMC settings."""
    
    def __init__(self):
        self.timings = {}
    
    def setup_exactly_like_example(self):
        """Set up everything exactly like your working example."""
        print("Setting up exactly like your working example...")
        
        # Load real data exactly like your example
        filename = resource_filename('rbvfit', 'example-data/test.fits')
        sp = rb_spectrum.from_file(filename)
        
        # Process exactly like your example
        zabs = 0.0
        wave_full = sp.wavelength.value
        flux = sp.flux.value / sp.co.value
        error = sp.sig.value / sp.co.value
        
        # Clean data exactly like your example
        qt = np.isnan(flux)
        flux[qt] = 0.
        error[qt] = 0.
        
        # Select wavelength range exactly like your example
        q = ((wave_full/(1.+zabs) > 1189.5) * (wave_full/(1.+zabs) < 1195.))
        wave = wave_full[q]
        flux = flux[q]
        error = error[q]
        
        print(f"✓ Loaded {len(wave)} data points")
        print(f"✓ Wavelength range: {wave.min():.2f} - {wave.max():.2f} Å")
        
        # Parameters exactly like your example
        lambda_rest = [1190.5, 1193.5]
        lambda_rest1 = [1025.7]
        nguess = [14.2, 14.5]
        bguess = [40., 30.]
        vguess = [0., 0.]
        
        # Bounds exactly like your example
        bounds, lb, ub = mc.set_bounds(
            nguess, bguess, vguess,
            Nlow=[12.0, 12.0], blow=[5.0, 5.0], vlow=[-300.0, -300.0],
            Nhi=[17.0, 17.0], bhi=[100.0, 100.0], vhi=[300.0, 300.0]
        )
        
        # Theta exactly like your example
        theta = np.concatenate((nguess, bguess, vguess))
        
        # FWHM exactly like your example
        FWHM_vel = 18.0
        FWHM = str(mean_fwhm_pixels(FWHM_vel, wave))
        print(f"✓ Model FWHM: {FWHM} pixels")
        
        # Model exactly like your example
        config = FitConfiguration()
        config.add_system(z=zabs, ion='SiII', transitions=lambda_rest, components=1)
        config.add_system(z=0.162005, ion='HI', transitions=lambda_rest1, components=1)
        
        v2_model = VoigtModel(config, FWHM=FWHM)
        print(f"✓ Model created with {len(theta)} parameters")
        
        # Instrument data exactly like your example
        instrument_data = {
            'COS': {
                'model': v2_model,
                'wave': wave,
                'flux': flux,
                'error': error
            }
        }
        
        # MCMC settings exactly like your example
        n_steps = 500
        n_walkers = 20
        sampler = 'zeus' if HAS_ZEUS else 'emcee'
        
        print(f"✓ MCMC settings: {n_walkers} walkers, {n_steps} steps, {sampler} sampler")
        
        return instrument_data, theta, lb, ub, n_walkers, n_steps, sampler
    
    def profile_setup_time(self, instrument_data, theta, lb, ub, n_walkers, n_steps, sampler):
        """Profile the fitter creation time."""
        print(f"\n{'='*60}")
        print("PROFILING FITTER SETUP")
        print(f"{'='*60}")
        
        start_time = time.perf_counter()
        fitter = mc.vfit(
            instrument_data, theta, lb, ub,
            no_of_Chain=n_walkers,
            no_of_steps=n_steps,
            sampler=sampler,
            perturbation=1e-4
        )
        setup_time = time.perf_counter() - start_time
        
        print(f"✓ Fitter creation: {setup_time*1000:.2f} ms")
        
        self.timings['setup'] = setup_time
        return fitter
    
    def profile_likelihood_evaluation(self, fitter, n_evals=100):
        """Profile likelihood evaluation performance."""
        print(f"\n{'='*60}")
        print(f"PROFILING LIKELIHOOD EVALUATION ({n_evals} calls)")
        print(f"{'='*60}")
        
        # Test initial likelihood
        start_time = time.perf_counter()
        lnprob = fitter.lnprob(fitter.theta)
        single_time = time.perf_counter() - start_time
        
        print(f"✓ Single likelihood: {single_time*1000:.2f} ms")
        print(f"✓ Log probability: {lnprob:.3f}")
        
        if not np.isfinite(lnprob):
            print("⚠️ Initial likelihood not finite - optimizing...")
            fitter.theta = fitter.optimize_guess(fitter.theta)
            lnprob = fitter.lnprob(fitter.theta)
            print(f"✓ Optimized log probability: {lnprob:.3f}")
        
        # Profile multiple evaluations
        start_time = time.perf_counter()
        valid_evals = 0
        for i in range(n_evals):
            theta_pert = fitter.theta + np.random.normal(0, 0.001, len(fitter.theta))
            theta_pert = np.clip(theta_pert, fitter.lb + 1e-10, fitter.ub - 1e-10)
            lnprob = fitter.lnprob(theta_pert)
            if np.isfinite(lnprob):
                valid_evals += 1
        multi_time = time.perf_counter() - start_time
        
        avg_time = multi_time / n_evals
        evals_per_sec = 1 / avg_time
        
        print(f"✓ {n_evals} evaluations: {multi_time*1000:.2f} ms total")
        print(f"✓ Average per evaluation: {avg_time*1000:.2f} ms")
        print(f"✓ Evaluations per second: {evals_per_sec:.1f}")
        print(f"✓ Valid evaluations: {valid_evals}/{n_evals}")
        
        self.timings['likelihood'] = {
            'single': single_time,
            'average': avg_time,
            'evals_per_sec': evals_per_sec,
            'success_rate': valid_evals / n_evals
        }
        
        return avg_time
    
    def profile_real_mcmc_run(self, fitter, use_multiprocessing=True):
        """Profile the real MCMC run with actual settings."""
        print(f"\n{'='*60}")
        print(f"PROFILING REAL MCMC RUN")
        print(f"{'='*60}")
        print(f"Settings: {fitter.no_of_Chain} walkers × {fitter.no_of_steps} steps")
        print(f"Sampler: {fitter.sampler_name}")
        print(f"Multiprocessing: {use_multiprocessing}")
        print(f"Total evaluations: {fitter.no_of_Chain * fitter.no_of_steps}")
        
        try:
            start_time = time.perf_counter()
            fitter.runmcmc(
                optimize=True, 
                verbose=True, 
                use_pool=use_multiprocessing, 
                progress=True
            )
            mcmc_time = time.perf_counter() - start_time
            
            total_evals = fitter.no_of_Chain * fitter.no_of_steps
            time_per_eval = mcmc_time / total_evals
            time_per_step = mcmc_time / fitter.no_of_steps
            
            print(f"\n✓ MCMC completed successfully!")
            print(f"✓ Total time: {mcmc_time:.2f} seconds")
            print(f"✓ Time per step: {time_per_step*1000:.2f} ms")
            print(f"✓ Time per evaluation: {time_per_eval*1000:.2f} ms")
            print(f"✓ Evaluations per second: {1/time_per_eval:.1f}")
            
            # Try to get acceptance fraction
            try:
                if hasattr(fitter, 'sampler') and fitter.sampler:
                    if hasattr(fitter.sampler, 'acceptance_fraction'):
                        accept_frac = np.mean(fitter.sampler.acceptance_fraction)
                        print(f"✓ Acceptance fraction: {accept_frac:.3f}")
                    elif fitter.sampler_name == 'zeus':
                        print("✓ Zeus sampler - check R-hat for convergence")
            except:
                pass
            
            self.timings['mcmc'] = {
                'total_time': mcmc_time,
                'time_per_step': time_per_step,
                'time_per_eval': time_per_eval,
                'total_evals': total_evals,
                'multiprocessing': use_multiprocessing
            }
            
            return True
            
        except Exception as e:
            print(f"❌ MCMC failed: {e}")
            print("This indicates a real problem with your setup!")
            return False
    
    def profile_with_cprofile(self, fitter, use_multiprocessing=True, n_steps=100):
        """Profile with cProfile using realistic but shorter settings."""
        print(f"\n{'='*60}")
        print(f"DETAILED FUNCTION PROFILING ({n_steps} steps)")
        print(f"{'='*60}")
        
        # Create a copy of fitter with shorter steps for profiling
        instrument_data = fitter.instrument_data
        theta = fitter.theta.copy()
        lb = fitter.lb.copy()
        ub = fitter.ub.copy()
        
        try:
            # Create profiling fitter with shorter run
            profile_fitter = mc.vfit(
                instrument_data, theta, lb, ub,
                no_of_Chain=fitter.no_of_Chain,  # Same walkers
                no_of_steps=n_steps,             # Shorter steps
                sampler=fitter.sampler_name,
                perturbation=1e-4
            )
            
            profiler = cProfile.Profile()
            
            print(f"Running {profile_fitter.no_of_Chain} walkers × {n_steps} steps with cProfile...")
            profiler.enable()
            profile_fitter.runmcmc(
                optimize=True, 
                verbose=False, 
                use_pool=use_multiprocessing, 
                progress=False
            )
            profiler.disable()
            
            # Analyze results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(25)  # Top 25 functions
            
            profile_output = s.getvalue()
            print("\nTOP FUNCTION TIMINGS:")
            print("=" * 50)
            print(profile_output)
            
            # Save detailed results
            with open('vfit_mcmc_real_profile.txt', 'w') as f:
                f.write("VFIT MCMC Real Performance Profile\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Settings: {profile_fitter.no_of_Chain} walkers × {n_steps} steps\n")
                f.write(f"Sampler: {profile_fitter.sampler_name}\n")
                f.write(f"Multiprocessing: {use_multiprocessing}\n\n")
                f.write(profile_output)
            
            print(f"\n✓ Detailed profile saved to 'vfit_mcmc_real_profile.txt'")
            
        except Exception as e:
            print(f"❌ cProfile failed: {e}")
    
    def print_performance_summary(self):
        """Print comprehensive performance summary."""
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        if 'setup' in self.timings:
            print(f"📊 Setup Time: {self.timings['setup']*1000:.2f} ms")
        
        if 'likelihood' in self.timings:
            like = self.timings['likelihood']
            print(f"\n📊 Likelihood Performance:")
            print(f"  • Average time: {like['average']*1000:.2f} ms per evaluation")
            print(f"  • Throughput: {like['evals_per_sec']:.1f} evaluations/second")
            print(f"  • Success rate: {like['success_rate']*100:.1f}%")
        
        if 'mcmc' in self.timings:
            mcmc = self.timings['mcmc']
            print(f"\n📊 MCMC Performance:")
            print(f"  • Total time: {mcmc['total_time']:.2f} seconds")
            print(f"  • Time per step: {mcmc['time_per_step']*1000:.2f} ms")
            print(f"  • Time per evaluation: {mcmc['time_per_eval']*1000:.2f} ms")
            print(f"  • Throughput: {1/mcmc['time_per_eval']:.1f} evaluations/second")
            print(f"  • Total evaluations: {mcmc['total_evals']:,}")
            print(f"  • Multiprocessing: {mcmc['multiprocessing']}")
        
        # Performance assessment
        print(f"\n🎯 Performance Assessment:")
        if 'likelihood' in self.timings:
            evals_per_sec = self.timings['likelihood']['evals_per_sec']
            if evals_per_sec > 200:
                print(f"  ✅ Excellent likelihood performance: {evals_per_sec:.1f} evals/sec")
            elif evals_per_sec > 100:
                print(f"  ✅ Good likelihood performance: {evals_per_sec:.1f} evals/sec")
            elif evals_per_sec > 50:
                print(f"  ⚠️ Moderate likelihood performance: {evals_per_sec:.1f} evals/sec")
            else:
                print(f"  ❌ Slow likelihood performance: {evals_per_sec:.1f} evals/sec")
        
        if 'mcmc' in self.timings:
            mcmc_throughput = 1 / self.timings['mcmc']['time_per_eval']
            if mcmc_throughput > 100:
                print(f"  ✅ Good MCMC throughput: {mcmc_throughput:.1f} evals/sec")
            elif mcmc_throughput > 50:
                print(f"  ⚠️ Moderate MCMC throughput: {mcmc_throughput:.1f} evals/sec")
            else:
                print(f"  ❌ Slow MCMC throughput: {mcmc_throughput:.1f} evals/sec")
        
        print(f"\n💡 Next Steps:")
        print(f"  • Check 'vfit_mcmc_real_profile.txt' for detailed bottleneck analysis")
        print(f"  • Look for functions with highest 'cumtime' values")
        print(f"  • Focus optimization on the slowest functions")
        
        print(f"\n📁 Files created:")
        print(f"  • vfit_mcmc_real_profile.txt - Detailed cProfile results")
    
    def run_real_performance_test(self):
        """Run the complete real performance test."""
        print("🚀 VFIT MCMC REAL PERFORMANCE PROFILER")
        print("=" * 60)
        print("Testing with realistic settings: 20 walkers, 500 steps, multiprocessing")
        
        # Setup exactly like working example
        instrument_data, theta, lb, ub, n_walkers, n_steps, sampler = self.setup_exactly_like_example()
        
        # Profile setup time
        fitter = self.profile_setup_time(instrument_data, theta, lb, ub, n_walkers, n_steps, sampler)
        
        # Profile likelihood evaluation
        self.profile_likelihood_evaluation(fitter, n_evals=100)
        
        # Ask user about full MCMC run
        print(f"\n" + "="*60)
        print("FULL MCMC PROFILING OPTIONS")
        print("="*60)
        print("1. Run full MCMC (500 steps, 20 walkers) - takes ~5-10 minutes")
        print("2. Run shorter MCMC for cProfile (100 steps) - takes ~1-2 minutes")
        print("3. Skip MCMC, show results so far")
        
        try:
            choice = input("\nEnter choice (1-3, default=2): ").strip()
            if not choice:
                choice = "2"
            
            if choice == "1":
                # Run full MCMC
                mcmc_success = self.profile_real_mcmc_run(fitter, use_multiprocessing=True)
                if mcmc_success:
                    # Run cProfile on shorter version
                    self.profile_with_cprofile(fitter, use_multiprocessing=True, n_steps=50)
                    
            elif choice == "2":
                # Run shorter MCMC with cProfile
                self.profile_with_cprofile(fitter, use_multiprocessing=True, n_steps=100)
                
            elif choice == "3":
                print("Skipping MCMC profiling...")
                
        except KeyboardInterrupt:
            print("\nSkipping MCMC profiling...")
        
        # Print summary
        self.print_performance_summary()


def main():
    """Run the real performance test."""
    try:
        profiler = SimpleVFitProfiler()
        profiler.run_real_performance_test()
    except KeyboardInterrupt:
        print("\n⏹️ Profiling interrupted")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()