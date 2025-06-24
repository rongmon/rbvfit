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

# Set up the physical system
config = FitConfiguration()
config.add_system(
    z=0.348,                          # Redshift
    ion='MgII',                       # Ion species  
    transitions=[2796.3, 2803.5],     # Rest wavelengths (Å)
    components=2                       # Number of velocity components
)

# Create model with instrumental resolution
model = VoigtModel(config, FWHM='2.5')
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
    Nlow=[12.0, 12.0], Nhi=[16.0, 16.0],
    blow=[5.0, 5.0], bhi=[80.0, 80.0], 
    vlow=[-150.0, -150.0], vhi=[150.0, 150.0]
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
    lb, ub, wave, flux_obs, error
)

# Quick fit (fast, good for initial estimates-- NOT MCMC)
print("Running quick fit...")
fitter.fit_quick()
print("✓ Quick fit completed!")

# Get results
best_params = fitter.theta_best
print(f"Best-fit parameters: {best_params}")
print(f"Parameter recovery: {np.abs(best_params - true_params)}")
```

### 5. View Results

```python
# Plot the fit
mc.plot_quick_fit(model, fitter, show_residuals=True)

# Print parameter comparison
print("\nParameter Recovery Analysis:")
print("=" * 50)
param_names = ['logN1', 'logN2', 'b1', 'b2', 'v1', 'v2']
for i, name in enumerate(param_names):
    true_val = true_params[i]
    fit_val = best_params[i]
    diff = fit_val - true_val
    print(f"{name:>6}: True={true_val:6.1f}, Fit={fit_val:6.1f}, Diff={diff:+6.1f}")
```

## Quick vs Full MCMC (3 minutes)

### Option A: Quick Fitting (seconds)
Perfect for initial parameter estimation and simple systems:

```python
# Already shown above - uses scipy.optimize.curve_fit
fitter.fit_quick()
print(f"Quick fit uncertainties: {fitter.theta_best_error}")
```

### Option B: Full Bayesian MCMC (minutes)
For robust uncertainties and complex systems:

```python
# Set MCMC parameters
fitter.no_of_Chain = 20    # Number of walkers
fitter.no_of_steps = 500   # Number of MCMC steps

# Run MCMC
fitter.runmcmc(optimize=True, verbose=True)

# Enhanced results analysis
from rbvfit.core import fit_results as fr
results = fr.FitResults(fitter, model)
results.print_fit_summary()
results.corner_plot()
```

## Complete Working Example

Here's a complete script you can copy and run:

```python
import numpy as np
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
import rbvfit.vfit_mcmc as mc

# Create synthetic data
config = FitConfiguration()
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
model = VoigtModel(config, FWHM='2.5')
compiled_model = model.compile()

wave = np.linspace(3760, 3800, 1500)
true_params = np.array([14.2, 13.8, 20.0, 35.0, -40.0, +25.0])
flux_clean = compiled_model.model_flux(true_params, wave)

np.random.seed(42)
error = np.full_like(wave, 0.015)
flux_obs = flux_clean + np.random.normal(0, error)

# Fit the data
N_guess, b_guess, v_guess = [14.0, 13.5], [25.0, 30.0], [-30.0, +20.0]
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)
theta_guess = np.concatenate([N_guess, b_guess, v_guess])

fitter = mc.vfit(compiled_model.model_flux, theta_guess, lb, ub, wave, flux_obs, error)
fitter.fit_quick()

# View results
mc.plot_model(model, fitter, show_residuals=True)
print(f"Recovery: {np.abs(fitter.theta_best - true_params)}")
```

### Single Ion System
```python
config = FitConfiguration()
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
```

### Multiple Ions at Same Redshift
```python
config = FitConfiguration()
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.5, ion='FeII', transitions=[2374.5, 2382.8], components=2)
# Parameters automatically tied for same redshift!
```

### Multi-Component System
```python
config = FitConfiguration()
config.add_system(z=0.3, ion='CIV', transitions=[1548.2, 1550.3], components=3)
# Fits 3 velocity components: [N1,N2,N3, b1,b2,b3, v1,v2,v3]
```

## Quick Troubleshooting

### Fit Not Converging?
```python
# 1. Check initial parameters are reasonable
print(f"Initial likelihood: {fitter.lnprob(theta_guess)}")

# 2. Adjust bounds
bounds, lb, ub = mc.set_bounds(
    N_guess, b_guess, v_guess,
    Nlow=[11.0, 11.0], Nhi=[17.0, 17.0],  # Wider bounds
    blow=[1.0, 1.0], bhi=[200.0, 200.0],
    vlow=[-500.0, -500.0], vhi=[500.0, 500.0]
)

# 3. Try quick fit first
fitter.fit_quick()
theta_guess = fitter.theta_best  # Use quick fit as starting point
```

### Model Evaluation Errors?
```python
# Check your wavelength range covers the transitions
z = 0.348
for wrest in [2796.3, 2803.5]:
    wobs = wrest * (1 + z)
    print(f"Transition at {wobs:.1f} Å")
    
# Ensure this falls within your wavelength array
print(f"Data range: {wave.min():.1f} - {wave.max():.1f} Å")

# Test model evaluation
try:
    test_flux = compiled_model.model_flux(theta_guess, wave)
    print(f"✓ Model evaluation successful: {np.isfinite(test_flux).all()}")
except Exception as e:
    print(f"✗ Model evaluation failed: {e}")
```

## Next Steps

🎯 **Ready for more?** Check out:

- [Complete User Guide](user-guide.md) - Detailed concepts and workflow
- [Tutorials](tutorials.md) - Step-by-step examples with real data
- [Examples Gallery](examples-gallery.md) - Visual showcase of capabilities

## Working Examples

All code snippets above are based on working examples in:
- [`../src/rbvfit/examples/example_voigt_model.py`](../src/rbvfit/examples/example_voigt_model.py)
- [`../src/rbvfit/examples/example_voigt_fitter.py`](../src/rbvfit/examples/example_voigt_fitter.py)

---

[← Back to Main Documentation](../README.md) | [Next: User Guide →](user-guide.md)