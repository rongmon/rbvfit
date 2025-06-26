# rbvfit User Guide

[← Back to Main Documentation](../README.md)

Comprehensive guide to absorption line fitting with rbvfit.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Workflow Overview](#workflow-overview)
- [Data Preparation](#data-preparation)
- [System Configuration](#system-configuration)
- [Model Creation](#model-creation)
- [Parameter Setup](#parameter-setup)
- [Fitting Methods](#fitting-methods)
- [Results Analysis](#results-analysis)
- [Advanced Topics](#advanced-topics)

## Core Concepts

### Physical vs Instrumental Parameters

rbvfit separates **physical** absorption properties from **instrumental** effects:

**Physical Parameters (per velocity component):**
- **N**: Column density [log cm⁻²]
- **b**: Doppler parameter [km/s] - combines thermal + turbulent broadening
- **v**: Velocity offset [km/s] - relative to systemic redshift

**Instrumental Parameters:**
- **FWHM**: Spectral resolution [pixels or km/s] - defined at configuration stage
- **LSF**: Line spread function shape (Gaussian default)

### Automatic Ion Parameter Tying

Key innovation in rbvfit: **Same ion at same redshift = shared parameters**

```python
# Important model imports
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel

#Analysis module import
from rbvfit.core import fit_results as fr
```

```python
# This configuration automatically ties MgII parameters and sets FWHM
config = FitConfiguration(FWHM='2.5')
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)

# Result: Fits [N1, N2, b1, b2, v1, v2] shared between both transitions
# NOT separate parameters for each transition
```

### Multi-System Architecture

Handle complex contamination naturally:

```python
# Foreground MgII + background CIV with instrument-specific FWHM
config = FitConfiguration(FWHM='2.2')
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=1.5, ion='CIV', transitions=[1548.2, 1550.3], components=1)

# Total parameters: [N_Mg1, N_Mg2, N_CIV, b_Mg1, b_Mg2, b_CIV, v_Mg1, v_Mg2, v_CIV]
```

## Workflow Overview

```
1. Data Preparation → 2. System Configuration → 3. Model Creation
         ↓                        ↓                       ↓
4. Parameter Setup → 5. Fitting (Quick/MCMC) → 6. Results Analysis
```

### Typical Session

```python
# 1. Load and prepare data
wave, flux, error = load_your_spectrum()

# 2. Configure physical systems with instrumental setup
config = FitConfiguration(FWHM='2.0')
config.add_system(z=redshift, ion='Ion', transitions=[...], components=N)

# 3. Create model (FWHM automatically extracted from config)
model = VoigtModel(config)
compiled = model.compile()

# 4. Set up parameters and bounds
theta_guess = [N_values, b_values, v_values]
bounds = mc.set_bounds(N_guess, b_guess, v_guess)

# 5. Fit
fitter = mc.vfit(compiled.model_flux, theta_guess, bounds, wave, flux, error)
fitter.runmcmc()  # or fitter.fit_quick()

# 6. Analyze results
results = fr.FitResults(fitter, model)
results.print_fit_summary()
```

## Data Preparation

### Loading Spectra

```python
import numpy as np
from astropy.io import fits

# Example: FITS file loading
with fits.open('spectrum.fits') as hdul:
    wave = hdul[1].data['WAVELENGTH']
    flux = hdul[1].data['FLUX']
    error = hdul[1].data['ERROR']
```

### Data Quality Control

```python
# Handle bad pixels
mask = np.isnan(flux) | np.isnan(error) | (error <= 0)
flux[mask] = 1.0      # Set to continuum
error[mask] = 1e10    # Large error = ignore

# Wavelength selection
z_abs = 0.5
rest_range = (1540, 1560)  # CIV region
obs_range = np.array(rest_range) * (1 + z_abs)
mask = (wave >= obs_range[0]) & (wave <= obs_range[1])
wave, flux, error = wave[mask], flux[mask], error[mask]
```

### Normalization

```python
# Continuum normalization (if needed)
# rbvfit expects normalized flux (continuum = 1.0)

# Simple polynomial continuum
from numpy.polynomial import Polynomial
p = Polynomial.fit(wave, flux, deg=2)
flux_norm = flux / p(wave)
error_norm = error / p(wave)
```

## System Configuration

### Basic Ion System

```python
# Single ion system with FWHM configuration
config = FitConfiguration(FWHM='3.0')  # Define resolution at configuration
config.add_system(
    z=0.348,                         # Absorption redshift
    ion='MgII',                      # Ion species
    transitions=[2796.3, 2803.5],   # Rest wavelengths [Å]
    components=2                     # Number of velocity components
)
```

### Multi-Ion Systems

```python
# Multiple ions at same redshift
config = FitConfiguration(FWHM='4.5')
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.5, ion='FeII', transitions=[2344.2, 2374.5], components=2)
# Note: Same z means shared kinematics (v1, v2) between MgII and FeII
```

### Multi-Redshift Systems

```python
# Foreground and background systems
config = FitConfiguration(FWHM='2.8')
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=1)
config.add_system(z=1.2, ion='CIV', transitions=[1548.2, 1550.3], components=2)
# Independent velocity structure for each redshift
```

### Advanced Configuration Options

```python
# Custom line selection and validation
config = FitConfiguration(FWHM='6.0')
config.add_system(
    z=0.8, 
    ion='SiII', 
    transitions=[1190.4, 1193.3, 1260.4],  # Multiple transitions
    components=3,
    validate_ion=True  # Verify transitions belong to declared ion
)
```

## Model Creation

### Basic Model

```python
from rbvfit.core.voigt_model import VoigtModel

# Create model (FWHM automatically extracted from configuration)
config = FitConfiguration(FWHM='2.5')
config.add_system(z=0.5, ion='CIV', transitions=[1548.2, 1550.3], components=2)

model = VoigtModel(config)  # No FWHM parameter needed
compiled = model.compile()
```

### Parameter Structure

Understanding the parameter vector `theta`:

```python
# For 2-component CIV system: theta = [N1, N2, b1, b2, v1, v2]
# N: log10(column density in cm^-2)
# b: Doppler parameter in km/s  
# v: Velocity offset in km/s

theta_example = [13.5, 13.2, 15.0, 25.0, -150.0, 20.0]
```

### Multi-System Parameter Structure

```python
# Example: 2-comp MgII + 1-comp CIV
config = FitConfiguration(FWHM='3.0')
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=1.5, ion='CIV', transitions=[1548.2, 1550.3], components=1)

# theta = [N_Mg1, N_Mg2, N_CIV, b_Mg1, b_Mg2, b_CIV, v_Mg1, v_Mg2, v_CIV]
#         |------ N ------|  |------ b ------|  |------ v ------|
```

### Model Evaluation

```python
# Generate synthetic spectrum
wave = np.linspace(3700, 3820, 10000)
theta = [13.5, 13.2, 15.0, 25.0, -150.0, 20.0]
flux_model = compiled.model_flux(theta, wave)

# Plot comparison
import matplotlib.pyplot as plt
plt.plot(wave, flux_model, 'r-', label='Model')
plt.plot(wave, flux_obs, 'k-', alpha=0.7, label='Data')
plt.legend()
```

## Parameter Setup

### Initial Guesses

```python
# Based on visual inspection or previous fits
N_guess = [13.5, 13.2]    # log column densities
b_guess = [15.0, 25.0]    # Doppler parameters [km/s]
v_guess = [-150.0, 20.0]  # Velocity offsets [km/s]

# Combine into theta vector
theta_guess = np.concatenate([N_guess, b_guess, v_guess])
```

### Parameter Bounds

```python
import rbvfit.vfit_mcmc as mc

# Automatic bounds with defaults
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)

# Custom bounds
bounds, lb, ub = mc.set_bounds(
    N_guess, b_guess, v_guess,
    Nlow=[12.0, 12.0], Nhi=[16.0, 16.0],      # Column density range
    blow=[5.0, 5.0], bhi=[80.0, 80.0],        # Doppler parameter range  
    vlow=[-200.0, -200.0], vhi=[200.0, 200.0] # Velocity range
)
```

### Interactive Parameter Estimation

```python
# Visual parameter guessing (recommended for complex systems)
from rbvfit import guess_profile_parameters_interactive as g

# Interactive component identification
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
tab.input_b_guess()  # GUI parameter input

# Extract parameters
N_guess = tab.nguess
b_guess = tab.bguess
v_guess = tab.vguess
```

## Fitting Methods

### Quick Fitting (scipy.optimize)

Fast approximate fitting using least-squares optimization:

```python
# Quick fit for initial exploration
fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
result = fitter.fit_quick()

print(f"Best-fit parameters: {result.x}")
print(f"Reduced chi-squared: {result.fun / (len(wave) - len(theta_guess))}")
```

**When to use**: Initial parameter estimation, simple systems, quick checks

### MCMC Fitting (emcee/zeus)

Robust Bayesian parameter estimation with full uncertainty quantification:

```python
# Set up MCMC
fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
fitter.no_of_Chain = 50     # Number of walkers
fitter.no_of_steps = 1000   # Number of MCMC steps
fitter.sampler = 'emcee'    # or 'zeus'

# Run MCMC
fitter.runmcmc(optimize=True, verbose=True)

# Create results object
results = fr.FitResults(fitter, model)
```

**When to use**: Final analysis, complex systems, publication-quality uncertainties

### Method Comparison

| Feature | Quick Fit | MCMC |
|---------|-----------|------|
| Speed | Seconds | Minutes-Hours |
| Uncertainties | Approximate | Robust |
| Correlations | No | Full covariance |
| Complex systems | Limited | Excellent |
| Starting point | Any reasonable | Needs good guess |

## Results Analysis

### Basic Results

```python
# Print summary
results.print_fit_summary()

# Parameter summary
param_summary = results.parameter_summary()
print(param_summary.names)
print(param_summary.best_fit)
print(param_summary.errors)
```

### Visualization

```python
# Corner plot (parameter correlations)
results.corner_plot(save_path='corner.pdf')

# Model comparison
mc.plot_model(model, fitter, show_residuals=True)

# Convergence diagnostics
results.convergence_diagnostics()

# Chain traces
results.chain_trace_plot()
```

### Saving Results

```python
# Save to HDF5
results.save('my_fit_results.h5')

# Load later
results = fr.FitResults.load('my_fit_results.h5')

# Export parameter table
results.export_parameter_table('parameters.txt')
```

## Advanced Topics

### Multi-Instrument Fitting

Joint fitting of data from multiple telescopes:

```python
# See detailed examples in:
# - rbvfit2-multi-instrument-tutorial.py
# - rbvfit2-multi-instrument-tutorial2.py

# Basic concept: Each instrument gets its own configuration with FWHM
config_A = FitConfiguration(FWHM='2.2')  # XShooter configuration
config_A.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

config_B = FitConfiguration(FWHM='4.0')  # FIRE configuration
config_B.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

# Create models (FWHM extracted from configurations)
model_A = VoigtModel(config_A)  # Uses FWHM='2.2'
model_B = VoigtModel(config_B)  # Uses FWHM='4.0'

# Multi-instrument compilation with automatic FWHM handling
instrument_configs = {
    'XShooter': config_A,
    'FIRE': config_B
}
compiled = model_A.compile(instrument_configs=instrument_configs)
```

### Custom Line Lists

```python
# Using rb_setline for line identification
from rbvfit import rb_setline as rb

# Find line information
line_info = rb.rb_setline(1548.2, 'closest')
print(f"Line: {line_info['name'][0]}")
print(f"Exact wavelength: {line_info['wave'][0]:.3f} Å")
print(f"f-value: {line_info['fval'][0]:.3e}")
```

### Performance Optimization

```python
# Vectorized evaluation for large datasets
wave_grid = np.linspace(1200, 1600, 50000)  # High-resolution grid
flux_model = compiled.model_flux(theta, wave_grid)

# Parallel MCMC (automatic on multi-core systems)
fitter.no_of_Chain = 100  # More walkers for better sampling
```

### Systematic Effects

```python
# Velocity zero-point corrections
v_systemic = -50.0  # km/s correction
v_guess_corrected = [v - v_systemic for v in v_guess]

# Wavelength calibration uncertainties
wave_shift = 0.1  # Å
wave_corrected = wave + wave_shift
```

### Complex Systems

```python
# Multiple redshift systems with contamination
config = FitConfiguration(FWHM='2.5')

# Primary absorption system
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=3)
config.add_system(z=0.5, ion='FeII', transitions=[2344.2, 2374.5], components=3)

# Intervening system
config.add_system(z=0.8, ion='CIV', transitions=[1548.2, 1550.3], components=1)

# Background quasar system  
config.add_system(z=2.1, ion='LyA', transitions=[1215.7], components=2)
```

---

## Troubleshooting

### Common Issues

**Poor Convergence**:
- Increase MCMC steps: `fitter.no_of_steps = 2000`
- Better initial guess: Use interactive parameter estimation
- Check parameter bounds: Ensure physically reasonable ranges

**Model-Data Mismatch**:
- Verify wavelength calibration
- Check continuum normalization
- Consider additional velocity components
- Review instrumental resolution (FWHM)

**Parameter Degeneracies**:
- Check corner plots for correlations
- Simplify model (fewer components)
- Add constraints from other transitions
- Use multi-instrument data

### Performance Tips

- Start with quick fits before MCMC
- Use interactive tools for complex systems
- Save intermediate results frequently
- Monitor convergence diagnostics

---

## Best Practices

1. **Always start simple**: Single component, then add complexity
2. **Use interactive tools**: Visual parameter estimation saves time
3. **Validate with synthetics**: Test fitting procedure on known parameters
4. **Check convergence**: Monitor MCMC diagnostics
5. **Save everything**: Use HDF5 persistence for reproducibility
6. **Document assumptions**: Record modeling choices and limitations

For more examples and advanced techniques, see the [Tutorials](tutorials.md) and [Examples Gallery](examples-gallery.md).