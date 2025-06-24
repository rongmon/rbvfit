# Fitting Methods

[← Back to Main Documentation](../README.md)

Deep dive into rbvfit's fitting algorithms: quick scipy optimization vs full Bayesian MCMC.

## Overview

rbvfit provides two complementary fitting approaches:

| Method | Algorithm | Speed | Use Case |
|--------|-----------|-------|----------|
| **Quick Fit** | scipy.optimize.curve_fit | Seconds | Initial estimates, simple systems |
| **MCMC Fit** | emcee/zeus samplers | Minutes-Hours | Robust uncertainties, complex systems |

## Quick Fitting

### Algorithm: scipy.optimize.curve_fit

Fast parameter estimation using Levenberg-Marquardt algorithm:

```python
# Quick fit usage
fitter.fit_quick(verbose=True)

# Results immediately available
best_params = fitter.theta_best
param_errors = fitter.theta_best_error
```

### Under the Hood

```python
from scipy.optimize import curve_fit

# rbvfit wrapper for curve_fit
def curve_fit_wrapper(xdata, *params):
    theta = np.array(params)
    return model_func(theta, xdata)

# Actual fitting call
popt, pcov = curve_fit(
    curve_fit_wrapper,
    wave,                    # x-data (wavelength)
    flux,                    # y-data (observed flux)
    p0=theta_guess,          # initial parameters
    sigma=error,             # error weights
    bounds=(lb, ub),         # parameter bounds
    maxfev=5000              # max function evaluations
)
```

### Advantages
- **Fast**: Seconds even for complex systems
- **Robust**: Well-tested scipy implementation
- **Bounded**: Respects parameter bounds
- **Weighted**: Uses error arrays properly

### Limitations
- **Local minima**: Can get stuck in local optima
- **Gaussian assumptions**: Errors assumed Gaussian
- **No correlations**: Doesn't sample full posterior
- **Linear approximation**: Uncertainties from linear approximation

### When to Use Quick Fit

✅ **Good for**:
- Initial parameter estimation
- Simple 1-2 component systems
- Parameter exploration
- Checking model setup
- Starting point for MCMC

❌ **Avoid for**:
- Final publication results
- Complex multi-component systems
- Non-Gaussian parameter spaces
- Parameter correlation analysis

### Quick Fit Example

```python
import numpy as np
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
import rbvfit.vfit_mcmc as mc

# Setup (data, config, model)
config = FitConfiguration()
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=1)
model = VoigtModel(config, FWHM='2.0')
compiled = model.compile()

# Quick fit
fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
fitter.fit_quick(verbose=True)

# Immediate results
print(f"Best-fit: {fitter.theta_best}")
print(f"Errors: {fitter.theta_best_error}")

# Quick visualization
mc.plot_quick_fit(model, fitter, show_residuals=True)
```

## MCMC Fitting

### Algorithms: emcee & zeus

Full Bayesian parameter estimation using ensemble samplers:

```python
# MCMC setup
fitter.no_of_Chain = 20      # Number of walkers
fitter.no_of_steps = 1000    # Number of steps
fitter.sampler = 'emcee'     # or 'zeus'

# Run MCMC
fitter.runmcmc(optimize=True, verbose=True, use_pool=True)
```

### Sampler Comparison

**emcee (default)**:
- Affine-invariant ensemble sampler
- Robust and well-tested
- Good for moderate dimensions (< 50 parameters)
- Slower scaling with parameter number

**zeus**:
- Slice sampling in ensemble
- Better for high-dimensional spaces (> 20 parameters)
- Often faster convergence
- Newer algorithm, less battle-tested

```python
# Sampler selection
fitter.sampler = 'emcee'  # Conservative choice
fitter.sampler = 'zeus'   # High-dimensional systems
```

### MCMC Configuration

**Walker Setup**:
```python
# Rule of thumb: 2-3x number of parameters
n_params = len(theta_guess)
fitter.no_of_Chain = max(20, 2 * n_params)

# Walker initialization
fitter.perturbation = 1e-4  # Small random perturbations around guess
```

**Step Configuration**:
```python
# Convergence typically needs:
fitter.no_of_steps = max(500, 50 * n_params)  # Minimum steps

# For production runs:
fitter.no_of_steps = 2000   # More robust
```

**Optimization**:
```python
fitter.runmcmc(
    optimize=True,    # Pre-optimize starting positions (recommended)
    verbose=True,     # Print progress
    use_pool=True     # Multiprocessing (faster)
)
```

### Likelihood Function

rbvfit uses chi-squared likelihood:

```python
def log_likelihood(theta, wave, flux, error, model_func):
    """Log likelihood for absorption line fitting."""
    model_flux = model_func(theta, wave)
    
    # Chi-squared
    chi2 = np.sum(((flux - model_flux) / error)**2)
    
    # Log likelihood (assuming Gaussian errors)
    log_like = -0.5 * chi2
    
    return log_like
```

### Prior Function

Uniform priors within bounds:

```python
def log_prior(theta, bounds_lower, bounds_upper):
    """Uniform priors within parameter bounds."""
    if np.all((theta >= bounds_lower) & (theta <= bounds_upper)):
        return 0.0  # log(1) = 0 for uniform prior
    else:
        return -np.inf  # Zero probability outside bounds
```

### MCMC Example

```python
# Full MCMC workflow
config = FitConfiguration()
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
model = VoigtModel(config, FWHM='2.0')
compiled = model.compile()

# MCMC setup
theta_guess = [13.5, 13.2, 25.0, 15.0, -50.0, 20.0]  # 2 components
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)

fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
fitter.no_of_Chain = 24  # 2x parameters
fitter.no_of_steps = 1000
fitter.sampler = 'emcee'

# Run with optimization
fitter.runmcmc(optimize=True, verbose=True, use_pool=True)

# Enhanced analysis
from rbvfit.core import fit_results as fr
results = fr.FitResults(fitter, model)
results.print_fit_summary()
results.corner_plot()
results.convergence_diagnostics()
```

## Method Comparison

### Performance Comparison

**Speed Test Example**:
```python
import time

# Quick fit timing
start = time.time()
fitter.fit_quick()
quick_time = time.time() - start
print(f"Quick fit: {quick_time:.1f} seconds")

# MCMC timing  
start = time.time()
fitter.runmcmc(optimize=True)
mcmc_time = time.time() - start
print(f"MCMC fit: {mcmc_time:.1f} seconds")

print(f"Speedup: {mcmc_time/quick_time:.0f}x faster for quick fit")
```

### Uncertainty Comparison

**Parameter Uncertainties**:
```python
# Quick fit uncertainties (from covariance matrix)
quick_errors = fitter.theta_best_error

# MCMC uncertainties (from posterior samples)
mcmc_samples = results.get_samples()
mcmc_errors = np.std(mcmc_samples, axis=0)

# Comparison
for i, (quick_err, mcmc_err) in enumerate(zip(quick_errors, mcmc_errors)):
    ratio = mcmc_err / quick_err
    print(f"Parameter {i}: Quick={quick_err:.3f}, MCMC={mcmc_err:.3f}, Ratio={ratio:.2f}")
```

### Result Quality

| Aspect | Quick Fit | MCMC |
|---------|-----------|------|
| Parameter values | Good if converged | Robust |
| Uncertainties | Approximate | True posterior |
| Correlations | Covariance only | Full posterior |
| Outlier handling | Sensitive | Robust |
| Multi-modal | Single mode | All modes |
| Systematic errors | Not captured | Better handling |

## Best Practices

### Workflow Strategy

**Recommended Workflow**:
```python
# 1. Start with quick fit
fitter.fit_quick()
print(f"Quick fit chi2: {np.sum(((flux - model(fitter.theta_best, wave))/error)**2)}")

# 2. Check if reasonable
if np.all(np.isfinite(fitter.theta_best_error)):
    print("Quick fit successful, using as MCMC starting point")
    theta_guess = fitter.theta_best
else:
    print("Quick fit failed, using manual guess")
    # Keep original theta_guess

# 3. Run MCMC with optimized starting point
fitter.theta = theta_guess  # Update starting point
fitter.runmcmc(optimize=True, verbose=True)
```

### Parameter Bounds

**Physical Bounds**:
```python
# Column density bounds (log cm^-2)
N_bounds = (11.0, 17.0)  # Reasonable for most ions

# Doppler parameter bounds (km/s)  
b_bounds = (1.0, 200.0)  # Thermal to highly turbulent

# Velocity bounds (km/s)
v_bounds = (-1000.0, 1000.0)  # Wide but reasonable
```

**Adaptive Bounds**:
```python
# Use quick fit to set MCMC bounds
fitter.fit_quick()
best = fitter.theta_best
errors = fitter.theta_best_error

# Expand bounds around quick fit result
n_sigma = 5  # 5-sigma bounds
new_lb = np.maximum(lb, best - n_sigma * errors)
new_ub = np.minimum(ub, best + n_sigma * errors)
```

### Convergence Assessment

**Diagnostic Tools**:
```python
# Convergence diagnostics
conv = results.convergence_diagnostics()
print(f"Overall status: {conv['overall_status']}")

# Manual checks
samples = results.get_samples()
autocorr_time = results.autocorr_time()
effective_samples = results.effective_sample_size()

print(f"Autocorrelation time: {autocorr_time}")
print(f"Effective samples: {effective_samples}")
```

**Convergence Criteria**:
- Gelman-Rubin R̂ < 1.1 for all parameters
- Effective sample size > 100 per parameter  
- Autocorrelation time < n_steps/50
- Visual inspection of chain traces

### Multi-Instrument Considerations

**Computational Scaling**:
```python
# Multi-instrument increases computation
# - More data points to evaluate
# - Joint likelihood calculation
# - Careful sampler tuning needed

# Recommendations for multi-instrument:
fitter.sampler = 'zeus'        # Often better scaling
fitter.no_of_Chain = 30        # More walkers
fitter.use_pool = True         # Essential for speed
```

## Troubleshooting

### Common Issues

**Quick Fit Fails**:
```python
# Check for NaN/infinite values
if np.any(~np.isfinite(fitter.theta_best_error)):
    print("Quick fit uncertainties contain NaN/inf")
    # Possible causes:
    # - Poor initial guess
    # - Model evaluation fails
    # - Singular covariance matrix
```

**MCMC Not Converging**:
```python
# Increase steps
fitter.no_of_steps = 2000

# Tighter initialization
fitter.perturbation = 1e-5

# More walkers
fitter.no_of_Chain = 40

# Better starting point
fitter.fit_quick()  # Get good starting point first
```

**Poor Parameter Constraints**:
```python
# Check data quality
chi2_reduced = np.sum(((flux - model_flux)/error)**2) / (len(flux) - len(theta))
print(f"Reduced chi2: {chi2_reduced:.2f}")

# Check parameter degeneracies
results.corner_plot()  # Look for correlations
```

---

## Advanced Topics

### Custom Samplers

```python
# Direct emcee usage (advanced)
import emcee

sampler = emcee.EnsembleSampler(
    nwalkers=fitter.no_of_Chain,
    ndim=len(theta_guess), 
    log_prob_fn=fitter.lnprob
)

# Custom sampling
state = sampler.run_mcmc(initial_state, nsteps=1000)
```

### Parallel Computing

```python
# Multiprocessing setup
from multiprocessing import Pool

# Use all available cores
fitter.use_pool = True

# Or specify number of processes
with Pool(processes=4) as pool:
    fitter.pool = pool
    fitter.runmcmc()
```

### Model Selection

```python
# Compare different models using information criteria
results_1comp = fit_1_component_model()
results_2comp = fit_2_component_model()

# Calculate AIC/BIC
aic_1 = results_1comp.aic()
aic_2 = results_2comp.aic()

print(f"1-component AIC: {aic_1:.1f}")
print(f"2-component AIC: {aic_2:.1f}")
print(f"Preferred: {'2-component' if aic_2 < aic_1 else '1-component'}")
```

---

[← Back to Main Documentation](../README.md) | [Next: Examples Gallery →](examples-gallery.md)