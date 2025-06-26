# Quick Start Guide

[← Back to Main Documentation](../README.md)

Get your first rbvfit2 absorption line fit running in under 5 minutes!

## Installation Check

First, verify rbvfit2 is installed:

```python
import rbvfit
print(f"rbvfit version: {rbvfit.__version__}")
```

## Your First Fit (2 minutes)

### 1. Import Essential Modules

```python
import numpy as np
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
import rbvfit.vfit_mcmc as mc
```

### 2. Create Synthetic Absorption Spectrum

```python
# Create synthetic MgII absorption data using rbvfit
print("Creating synthetic absorption spectrum...")

# Set up the physical system with instrumental resolution
config = FitConfiguration(FWHM='2.5')  # Define FWHM at configuration stage
config.add_system(
    z=0.348,                          # Redshift
    ion='MgII',                       # Ion species  
    transitions=[2796.3, 2803.5],     # Rest wavelengths (Å)
    components=2                       # Number of velocity components
)

# Create model (FWHM automatically extracted from configuration)
model = VoigtModel(config)
compiled_model = model.compile()

# Define wavelength grid (observed frame)
wave = np.linspace(3760, 3800, 1500)  # Around MgII at z=0.348

# True parameters for synthetic data: [N1, N2, b1, b2, v1, v2]
true_params = np.array([
    14.2, 13.8,    # log column densities
    20.0, 35.0,    # Doppler parameters (km/s)
    -40.0, +25.0   # Velocity offsets (km/s)
])

# Generate clean synthetic spectrum
flux_clean = compiled_model.model_flux(true_params, wave)

# Add realistic noise
np.random.seed(42)  # For reproducible results
error = np.full_like(wave, 0.015)  # 1.5% error
noise = np.random.normal(0, error)
flux_obs = flux_clean + noise

print(f"✓ Created synthetic spectrum with {len(wave)} wavelength points")
print(f"✓ Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å")
print(f"✓ Added {error[0]*100:.1f}% noise level")
```

### 3. Set Up Fitting Parameters

```python
# Initial parameter guess (slightly different from truth)
N_guess = [14.0, 13.5]     # log column densities
b_guess = [25.0, 30.0]     # Doppler parameters
v_guess = [-30.0, +20.0]   # Velocity offsets

# Set parameter bounds
bounds, lb, ub = mc.set_bounds(
    N_guess, b_guess, v_guess,
    Nlow=[12.0, 12.0], Nhi=[16.0, 16.0], #optional custom bounds
    blow=[5.0, 5.0], bhi=[80.0, 80.0], #optional custom bounds
    vlow=[-150.0, -150.0], vhi=[150.0, 150.0]#optional custom bounds
)

# Combine into theta array
theta_guess = np.concatenate([N_guess, b_guess, v_guess])
print(f"Initial guess: {theta_guess}")
print(f"True values:   {true_params}")
```

### 4. Run the Fit

```python
# Create fitter
fitter = mc.vfit(
    compiled_model.model_flux, theta_guess, 
    lb, ub, wave, flux_obs, error,sampler='emcee'
)


# Set MCMC parameters
fitter.no_of_Chain = 20    # Number of walkers
fitter.no_of_steps = 500   # Number of MCMC steps

# Run MCMC
fitter.runmcmc(optimize=True, verbose=True)

```

### 5. Analyze Results

```python
# Create results object
from rbvfit.core import fit_results as fr
results = fr.FitResults(fitter, model)

# Print parameter summary
results.print_fit_summary()

# Quick visualization
import matplotlib.pyplot as plt

# Plot the fit
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                               gridspec_kw={'height_ratios': [3, 1]})

# Best-fit model
best_params = results.parameter_summary().best_fit
flux_best = compiled_model.model_flux(best_params, wave)

# Main plot
ax1.plot(wave, flux_obs, 'k-', alpha=0.7, label='Synthetic Data')
ax1.plot(wave, flux_best, 'r-', linewidth=2, label='Best Fit')
ax1.fill_between(wave, flux_obs-error, flux_obs+error, alpha=0.3, color='gray')
ax1.set_ylabel('Normalized Flux')
ax1.legend()
ax1.set_title('rbvfit2 Quick Start: MgII Doublet Fit')

# Residuals
residuals = flux_obs - flux_best
ax2.plot(wave, residuals, 'k-', alpha=0.7)
ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Wavelength (Å)')
ax2.set_ylabel('Residuals')

plt.tight_layout()
plt.show()

print("✓ Quick Start completed successfully!")
print("✓ Check the plot to see your first rbvfit2 absorption line fit")
```

## Understanding the Results

The quick start fit should recover parameters close to the true values:

```python
# Compare fitted vs true parameters
param_names = ['N1', 'N2', 'b1', 'b2', 'v1', 'v2']
print("\nParameter Recovery:")
print("Parameter | True Value | Fitted Value | Difference")
print("-" * 50)
for i, name in enumerate(param_names):
    true_val = true_params[i]
    fit_val = best_params[i]
    diff = fit_val - true_val
    print(f"{name:8s} | {true_val:9.2f} | {fit_val:11.2f} | {diff:9.2f}")
```

## Next Steps

Now that you have rbvfit2 working, explore these topics:

### 🔬 **Real Data Analysis**
```python
# Load your own spectrum
wave, flux, error = load_your_spectrum()  # Replace with your data loading

# Configure for your absorption system
config = FitConfiguration(FWHM='3.0')  # Adjust FWHM for your instrument
config.add_system(z=your_redshift, ion='YourIon', 
                  transitions=[rest_wavelengths], components=N_components)
```

### 🎮 **Interactive Parameter Estimation**
```python
# Visual component identification (highly recommended!)
from rbvfit import guess_profile_parameters_interactive as g

tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
tab.input_b_guess()  # Interactive parameter input

# Extract automatically estimated parameters
N_guess = tab.nguess
b_guess = tab.bguess  
v_guess = tab.vguess
```

### 🔭 **Multi-Instrument Fitting**
```python
# Joint fitting of data from multiple telescopes
config_A = FitConfiguration(FWHM='2.2')  # XShooter
config_A.add_system(z=z_abs, ion='OI', transitions=[1302.17], components=1)

config_B = FitConfiguration(FWHM='4.0')  # FIRE  
config_B.add_system(z=z_abs, ion='OI', transitions=[1302.17], components=1)

# Models automatically use correct FWHM from configurations
model_A = VoigtModel(config_A)
model_B = VoigtModel(config_B)
```

### 📊 **Advanced Analysis**
```python
# Comprehensive results analysis
results.corner_plot(save_path='corner_plot.pdf')
results.plot_velocity_fits(save_path='velocity_plot.pdf')
results.convergence_diagnostics()

# Save results for later
results.save('my_first_fit.h5')
```

## Common Patterns

### Single Ion, Multiple Components
```python
# Multiple velocity components of same ion
config = FitConfiguration(FWHM='2.5')
config.add_system(z=0.5, ion='CIV', transitions=[1548.2, 1550.3], components=3)
# Fits: [N1, N2, N3, b1, b2, b3, v1, v2, v3]
```

### Multiple Ions, Same Redshift  
```python
# Different ions at same redshift (shared kinematics)
config = FitConfiguration(FWHM='3.0')
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.5, ion='FeII', transitions=[2344.2, 2374.5], components=2)
# Shared velocity structure between ions
```

### Multiple Redshift Systems
```python
# Independent systems at different redshifts
config = FitConfiguration(FWHM='2.8')
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=1)
config.add_system(z=1.2, ion='CIV', transitions=[1548.2, 1550.3], components=2) 
# Independent velocity structures
```

## Troubleshooting Quick Start

### Issue: Poor Fit Quality
```python
# Try these solutions:
fitter.no_of_steps = 1000     # More MCMC steps
fitter.no_of_Chain = 50       # More walkers

# Better initial guess with interactive tools
from rbvfit import guess_profile_parameters_interactive as g
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=2796.3)
```

### Issue: Slow Performance
```python
# Quick fit first for initial parameters
quick_result = fitter.fit_quick()
theta_better = quick_result.x

# Then run MCMC with better starting point
fitter.theta = theta_better
fitter.runmcmc()
```

### Issue: Parameter Bounds
```python
# Check if parameters hit bounds
if np.any(best_params <= lb) or np.any(best_params >= ub):
    print("Warning: Parameters at bounds - consider expanding ranges")
    
# Expand bounds if needed
bounds, lb, ub = mc.set_bounds(
    N_guess, b_guess, v_guess,
    Nlow=[10.0, 10.0], Nhi=[18.0, 18.0],  # Wider N range
    blow=[3.0, 3.0], bhi=[100.0, 100.0],  # Wider b range
    vlow=[-300.0, -300.0], vhi=[300.0, 300.0]  # Wider v range
)
```

## Performance Expectations

**Quick Start Performance (typical laptop):**
- Data generation: < 1 second
- MCMC fitting (500 steps, 20 walkers): 10-30 seconds  
- Results analysis: < 5 seconds
- **Total time**: Under 1 minute

**Scaling Guidelines:**
- More components: ~linear scaling
- More MCMC steps: linear scaling  
- Multi-instrument: ~2-3x single instrument
- Complex systems (5+ components): 5-15 minutes

---

**🎉 Congratulations!** You've successfully run your first rbvfit2 absorption line fit. 

**Next recommended steps:**
1. Try the [Interactive Tutorial](tutorials.md#3-single-instrument-analysis) 
2. Explore [Multi-Instrument Fitting](tutorials.md#4-multi-instrument-fitting)
3. Read the [User Guide](user-guide.md) for comprehensive documentation
4. Check out [Examples Gallery](examples-gallery.md) for more use cases