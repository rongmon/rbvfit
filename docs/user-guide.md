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
- **FWHM**: Spectral resolution [pixels or km/s]
- **LSF**: Line spread function shape (Gaussian default)

### Automatic Ion Parameter Tying

Key innovation in rbvfit: **Same ion at same redshift = shared parameters**

```python
# Important model imports
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel

#Analysis module import
from rbvfit.core import fit_results as f
```

```python

# This configuration automatically ties MgII parameters
config = FitConfiguration()
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)

# Result: Fits [N1, N2, b1, b2, v1, v2] shared between both transitions
# NOT separate parameters for each transition
```

### Multi-System Architecture

Handle complex contamination naturally:

```python
# Foreground MgII + background CIV
config = FitConfiguration()
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

# 2. Configure physical systems
config = FitConfiguration()
config.add_system(z=redshift, ion='Ion', transitions=[...], components=N)

# 3. Create model with instrumental setup
model = VoigtModel(config, FWHM='2.0')
compiled = model.compile()

# 4. Set up parameters and bounds
theta_guess = [N_values, b_values, v_values]
bounds = mc.set_bounds(N_guess, b_guess, v_guess)

# 5. Fit
fitter = mc.vfit(compiled.model_flux, theta_guess, bounds, wave, flux, error)
fitter.runmcmc()  # or fitter.fit_quick()

# 6. Analyze

results = FitResults(fitter, model)
results.print_fit_summary()
```

## Data Preparation

### Required Data Format

```python
wave = np.array([...])   # Observed wavelength [Angstroms]
flux = np.array([...])   # Normalized flux (continuum = 1.0)
error = np.array([...])  # 1-sigma error on flux
```

### Data Quality Checks

```python
# Handle NaN values
mask = np.isnan(flux) | np.isnan(error)
flux[mask] = 0.0
error[mask] = 0.0

# Select wavelength range around transitions
z_abs = 0.348
wrest = 2796.3  # MgII
wobs = wrest * (1 + z_abs)
margin = 50  # Angstroms

mask = (wave > wobs - margin) & (wave < wobs + margin)
wave = wave[mask]
flux = flux[mask]
error = error[mask]
```

### Continuum Normalization

```python
# Ensure continuum is normalized to 1.0
# Methods:
# 1. Manual: flux = flux / continuum_estimate
# 2. Using rbcodes (if available):
from rbcodes.utils.rb_spectrum import rb_spectrum
sp = rb_spectrum.from_file('spectrum.fits')
flux_norm = sp.flux.value / sp.co.value  # Normalized flux
error_norm = sp.sig.value / sp.co.value  # Normalized error
```

## System Configuration

### FitConfiguration Class

The heart of rbvfit - defines what you're fitting:

```python
config = FitConfiguration()

# Add each absorption system
config.add_system(
    z=0.348,                          # Redshift
    ion='MgII',                       # Ion name
    transitions=[2796.3, 2803.5],     # Rest wavelengths
    components=2                       # Velocity components
)
```

### Common Ion Systems

```python
# MgII doublet
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)

# CIV doublet  
config.add_system(z=1.2, ion='CIV', transitions=[1548.2, 1550.3], components=1)

# FeII multiplet
config.add_system(z=0.8, ion='FeII', transitions=[2374.5, 2382.8, 2586.7], components=3)

# Lyman alpha
config.add_system(z=2.1, ion='HI', transitions=[1215.67], components=1)
```

### Multi-System Examples

**Foreground + Background**:
```python
config = FitConfiguration()
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=1.5, ion='CIV', transitions=[1548.2, 1550.3], components=1)
```

**Multiple Ions at Same Redshift**:
```python
config = FitConfiguration()
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.5, ion='FeII', transitions=[2374.5, 2382.8], components=2)
# Ion parameters automatically tied for same z!
```

## Model Creation

### VoigtModel Class

Converts configuration into fittable model:

```python
# Basic model
model = VoigtModel(config)

# With instrumental resolution
model = VoigtModel(config, FWHM='2.0')  # 2.0 pixels FWHM

# Compilation for fitting
compiled = model.compile(verbose=True)
model_function = compiled.model_flux  # Ready for fitting
```

### Instrumental Resolution

```python
# Different ways to specify FWHM:
VoigtModel(config, FWHM='2.0')        # Pixels
VoigtModel(config, FWHM='15.0')       # km/s (if > 10)
VoigtModel(config, FWHM=2.0)          # Direct float
```

### Model Information

```python
# Check model structure
print(model.get_info())

# Parameter structure
structure = model.param_manager.structure
print(f"Total parameters: {structure['total_parameters']}")
print(f"Parameter names: {structure['parameter_names']}")
```

## Parameter Setup

### Parameter Organization

For each velocity component, rbvfit fits three parameters:
- **N**: log₁₀(column density / cm⁻²)
- **b**: Doppler parameter (km/s)  
- **v**: Velocity offset (km/s)

### Parameter Arrays

```python
# For 2-component MgII system:
N_guess = [13.5, 13.2]     # log column densities
b_guess = [25.0, 15.0]     # Doppler parameters  
v_guess = [-50.0, 20.0]    # Velocities

# Combine into theta array
theta = np.concatenate([N_guess, b_guess, v_guess])
# Result: [13.5, 13.2, 25.0, 15.0, -50.0, 20.0]
```

### Setting Bounds

```python
# Automatic bounds
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)

# Custom bounds
bounds, lb, ub = mc.set_bounds(
    N_guess, b_guess, v_guess,
    Nlow=[12.0, 12.0],     # Custom N lower bounds
    Nhi=[16.0, 16.0],      # Custom N upper bounds
    blow=[5.0, 5.0],       # Custom b lower bounds
    bhi=[100.0, 100.0],    # Custom b upper bounds
    vlow=[-300.0, -300.0], # Custom v lower bounds
    vhi=[300.0, 300.0]     # Custom v upper bounds
)
```

### Physical Interpretation

```python
# Column density (N)
N = 13.5  # log cm^-2
print(f"Column density: 10^{N:.1f} = {10**N:.1e} cm^-2")

# Doppler parameter (b)
b = 25.0  # km/s
T_thermal = (b**2 * 24.3) / 2  # For MgII (approximate)
print(f"Thermal temperature: ~{T_thermal:.0f} K")

# Velocity (v)
v = -50.0  # km/s (negative = blueshift)
print(f"Velocity offset: {v} km/s")
```

## Fitting Methods

### Quick Fitting (scipy.optimize)

Fast parameter estimation using curve_fit:

```python
# Quick fit
fitter.fit_quick(verbose=True)

# Results immediately available
best_params = fitter.theta_best
param_errors = fitter.theta_best_error

# Quick visualization
mc.plot_quick_fit(model, fitter, show_residuals=True)
```

**When to use**: Initial parameter estimation, simple systems, quick checks

### MCMC Fitting (emcee/zeus)

Full Bayesian analysis with robust uncertainties:

```python
# Configure MCMC
fitter.no_of_Chain = 20     # Number of walkers
fitter.no_of_steps = 1000   # Number of steps
fitter.sampler = 'emcee'    # or 'zeus'

# Run MCMC
fitter.runmcmc(
    optimize=True,    # Pre-optimize starting positions
    verbose=True,     # Print progress
    use_pool=True     # Multiprocessing
)

# Enhanced analysis
from rbvfit.core import fit_results as fr
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

# Basic concept:
config_A = FitConfiguration()  # Same physical system
config_B = FitConfiguration()  # Different instrumental setup

model_A = VoigtModel(config_A, FWHM='2.2')  # XShooter
model_B = VoigtModel(config_B, FWHM='4.0')  # FIRE

# Joint fitting combines datasets with shared physics
```

### Performance Optimization

```python
# For large parameter spaces:
fitter.sampler = 'zeus'        # Often faster than emcee
fitter.use_pool = True         # Multiprocessing
fitter.perturbation = 1e-4     # Tighter walker initialization

# For quick exploration:
fitter.fit_quick()             # Get good starting point
theta_guess = fitter.theta_best
# Then use for MCMC
```

### Custom Line Lists

```python
# Add custom transitions
config.add_system(
    z=0.5, 
    ion='CustomIon',
    transitions=[1234.5, 1567.8],  # Your wavelengths
    components=2
)
```

## Troubleshooting

### Common Issues

**Model evaluation fails**:
```python
# Check wavelength coverage
z = 0.348
wrest = 2796.3
wobs = wrest * (1 + z)
print(f"Transition at {wobs:.1f} Å")
print(f"Data range: {wave.min():.1f} - {wave.max():.1f} Å")
```

**Poor convergence**:
```python
# Check initial likelihood
print(f"Initial likelihood: {fitter.lnprob(theta_guess)}")

# Try wider bounds
# Use quick fit as starting point
```

**Parameter values unreasonable**:
```python
# Check parameter interpretation
N, b, v = theta_guess[0], theta_guess[n_comp], theta_guess[2*n_comp]
print(f"N = {N:.1f} (log cm^-2), b = {b:.1f} km/s, v = {v:.1f} km/s")
```

---

## Next Steps

- [Tutorials](tutorials.md) - Detailed worked examples
- [Fitting Methods](fitting-methods.md) - Deep dive into algorithms  
- [Examples Gallery](examples-gallery.md) - Visual showcase

---

[← Back to Main Documentation](../README.md) | [Next: Tutorials →](tutorials.md)