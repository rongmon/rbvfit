# rbvfit Tutorials

[← Back to Main Documentation](../README.md)

Step-by-step tutorials with working examples from simple to advanced use cases.

## Learning Path

```
📚 Basic Concepts → 🔬 Single System → 🌐 Multi-System → 🔭 Multi-Instrument
```

## Tutorial Overview

| Tutorial | Level | Time | Topics Covered |
|----------|-------|------|----------------|
| [1. Model Creation](#1-model-creation) | Beginner | 10 min | VoigtModel basics |
| [2. Single System Fitting](#2-single-system-fitting) | Beginner | 15 min | Complete workflow |
| [3. Single Instrument Analysis](#3-single-instrument-analysis) | Intermediate | 20 min | Real data analysis |
| [4. Multi-Instrument Fitting](#4-multi-instrument-fitting) | Advanced | 30 min | Joint fitting |
| [5. Complex Multi-System](#5-complex-multi-system) | Advanced | 45 min | Multiple redshifts |

---

## 1. Model Creation
**File**: [`../src/rbvfit/examples/example_voigt_model.py`](../src/rbvfit/examples/example_voigt_model.py)

### Learning Objectives
- Create VoigtModel from configuration
- Understand parameter structure  
- Generate synthetic spectra
- Compare different ion systems

### Key Concepts

**Basic Model Creation**:
```python
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel

# Simple MgII doublet
config = FitConfiguration()
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
model = VoigtModel(config)
```

**Parameter Understanding**:
```python
# For 2-component system: [N1, N2, b1, b2, v1, v2]
theta = [13.5, 13.2, 15.0, 25.0, -150.0, 20.0]

# Evaluate model
wave = np.linspace(3700, 3820, 10000)
flux = model.evaluate(theta, wave)
```

### Expected Output
- Synthetic MgII doublet spectrum
- Parameter structure explanation
- Multiple ion comparison plots

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python example_voigt_model.py
```

---

## 2. Single System Fitting  
**File**: [`../src/rbvfit/examples/example_voigt_fitter.py`](../src/rbvfit/examples/example_voigt_fitter.py)

### Learning Objectives
- Complete fitting workflow
- Data preparation steps
- Parameter setup and bounds
- MCMC vs quick fitting
- Results analysis

### Key Concepts

**Data Preparation**:
```python
# Load your spectrum data
wave, flux, error = load_spectrum_data()

# Handle bad pixels
mask = np.isnan(flux)
flux[mask] = 0.0
error[mask] = 0.0

# Select wavelength range
q = (wave/(1+z_abs) > 1189.5) & (wave/(1+z_abs) < 1195.0)
wave, flux, error = wave[q], flux[q], error[q]
```

**Complete Workflow**:
```python
# 1. Configure system
config = FitConfiguration()
config.add_system(z=z_abs, ion='SiII', transitions=[1190.5, 1193.5], components=2)

# 2. Create and compile model
model = VoigtModel(config, FWHM='6.5')
compiled = model.compile()

# 3. Set parameters and bounds  
theta_guess = [14.2, 14.5, 40.0, 30.0, 0.0, 0.0]
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)

# 4. Fit with MCMC
fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
fitter.runmcmc(optimize=True, verbose=True)

# 5. Analyze results
#Analysis module import
from rbvfit.core import fit_results as f
results = f.FitResults(fitter, model)
results.print_fit_summary()

#6. Extract best fit and plot it
param_summary = results.parameter_summary()

#Best fit model parameters
best_fit=param_summary.best_fit

# Access all percentiles
percentiles_16th = param_summary.percentiles['16th']  # 16th percentile values
percentiles_50th = param_summary.percentiles['50th']  # 50th percentile (median/best_fit)
percentiles_84th = param_summary.percentiles['84th']  # 84th percentile values

# Calculate asymmetric errors
lower_errors = param_summary.best_fit - percentiles_16th  # Lower error bars
upper_errors = percentiles_84th - param_summary.best_fit  # Upper error bars

print(f"\nBest-fit parameters with asymmetric errors:")
for name, value, lower_err, upper_err in zip(param_summary.names, 
                                           param_summary.best_fit,
                                           lower_errors, 
                                           upper_errors):
    print(f"  {name}: {value:.3f} +{upper_err:.3f} -{lower_err:.3f}")

#Create best fit model flux

best_fit_flux=compiled.model_flux(best_fit,wave)

import matplotlib.pyplot as plt

plt.plot(wave,flux,label='Data')
plt.plot(wave,best_fit_flux,label='best_fit')

plt.show()


```

### Expected Output
- Fitted absorption line profile
- Parameter summary with uncertainties
- Corner plot showing correlations
- Convergence diagnostics

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python example_voigt_fitter.py
```

---

## 3. Single Instrument Analysis
**File**: [`../src/rbvfit/examples/rbvfit2-single-instrument-tutorial.py`](../src/rbvfit/examples/rbvfit2-single-instrument-tutorial.py)

### Learning Objectives
- Real spectroscopic data analysis
- Multi-component CIV system
- Advanced parameter estimation
- Publication-quality results

### Key Concepts

**Real Data Handling**:
```python
# Load real spectrum (example uses rb_spec format)
from rbcodes.GUIs.rb_spec import load_rb_spec_object
spectrum = load_rb_spec_object('your_spectrum.pkl')

# Extract data
wave = spectrum.wave.value
flux = spectrum.flux.value / spectrum.co.value  # Normalize
error = spectrum.sig.value / spectrum.co.value

# Redshift to observed frame
z_qso = 6.074762
wave_obs = wave * (1 + z_qso)
```

**Multi-Component System**:
```python
# Complex CIV system with 3 components
config = FitConfiguration()
config.add_system(
    z=4.9484, 
    ion='CIV', 
    transitions=[1548.2, 1550.3], 
    components=3
)

# Parameter arrays for 3 components
N_guess = [13.15, 13.58, 13.5]    # log column densities
b_guess = [23.0, 25.0, 30.0]      # Doppler parameters
v_guess = [-67.0, 0.0, 10.0]      # Velocity offsets
```

**Advanced MCMC Setup**:
```python
# Optimized MCMC parameters
fitter = mc.vfit(
    compiled.model_flux, theta_guess, lb, ub, wave_obs, flux, error,
    no_of_Chain=20,           # 20 walkers
    no_of_steps=1000,         # 1000 steps
    sampler='emcee',          # MCMC sampler
    perturbation=1e-4         # Tight initialization
)

# Run with optimization
fitter.runmcmc(optimize=True, verbose=True, use_pool=True)
```

### Expected Output
- CIV doublet fit with 3 velocity components
- Individual component decomposition
- Robust parameter uncertainties
- Model residuals analysis

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python rbvfit2-single-instrument-tutorial.py
```

---

## 4. Multi-Instrument Fitting
**File**: [`../src/rbvfit/examples/rbvfit2-multi-instrument-tutorial.py`](../src/rbvfit/examples/rbvfit2-multi-instrument-tutorial.py)

### Learning Objectives
- Joint fitting of multiple datasets
- Instrumental resolution differences
- Shared physical parameters
- Enhanced constraints from combined data

### Key Concepts

**Same Physics, Different Instruments**:
```python
# Identical physical system observed by different instruments
config_A = FitConfiguration()  # XShooter configuration
config_A.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

config_B = FitConfiguration()  # FIRE configuration  
config_B.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

# Different instrumental resolutions
model_A = VoigtModel(config_A, FWHM='2.2')  # Higher resolution
model_B = VoigtModel(config_B, FWHM='4.0')  # Lower resolution
```

**Multi-Instrument Compilation**:
```python
# Create joint model
joint_datasets = {
    'XShooter': {'wave': wave_A, 'flux': flux_A, 'error': error_A},
    'FIRE': {'wave': wave_B, 'flux': flux_B, 'error': error_B}
}

# Compile for joint fitting
multi_compiled = mc.compile_multi_instrument([model_A, model_B], joint_datasets)
```

**Joint Parameter Estimation**:
```python
# Same physical parameters fit both datasets
theta_guess = [13.0, 25.0, 0.0]  # [N, b, v] shared between instruments

# Joint likelihood combines both datasets
fitter = mc.vfit_multi_instrument(
    multi_compiled.joint_model, theta_guess, bounds,
    datasets=joint_datasets
)
```

### Expected Output
- Joint fit showing both datasets
- Improved parameter constraints
- Instrument-specific residuals
- Combined chi-squared statistics

### Advanced Features
- Velocity space plots by ion
- Cross-validation between instruments
- Systematic uncertainty analysis

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python rbvfit2-multi-instrument-tutorial.py
```

---

## 5. Complex Multi-System
**File**: [`../src/rbvfit/examples/rbvfit2-multi-instrument-tutorial2.py`](../src/rbvfit/examples/rbvfit2-multi-instrument-tutorial2.py)

### Learning Objectives
- Multiple absorption systems
- Different redshifts and ions
- Complex contamination scenarios
- Advanced results visualization

### Key Concepts

**Multi-System Configuration**:
```python
# Complex system: CIV + OI + SiII at different redshifts
config = FitConfiguration()

# Background CIV (high redshift)
config.add_system(z=4.9484, ion='CIV', transitions=[1548.2, 1550.3], components=2)

# Foreground systems (low redshift)  
config.add_system(z=6.074762, ion='OI', transitions=[1302.17], components=1)
config.add_system(z=6.074762, ion='SiII', transitions=[1304.5], components=1)

# Automatic parameter organization:
# [N_CIV1, N_CIV2, N_OI, N_SiII, b_CIV1, b_CIV2, b_OI, b_SiII, v_CIV1, v_CIV2, v_OI, v_SiII]
```

**Parameter Management**:
```python
# Different systems have independent parameters
N_guess = [13.5, 13.2, 13.0, 12.8]  # Four components
b_guess = [25.0, 20.0, 30.0, 25.0]
v_guess = [-50.0, 10.0, 0.0, 5.0]

# But same ions at same z are automatically tied
# (OI and SiII both at z=6.074762 share z but have independent N,b,v)
```

**Advanced Visualization**:
```python
# Velocity plots organized by ion
results.plot_velocity_fits(
    show_components=True,     # Individual components
    show_rail_system=True,    # Component markers
    velocity_range=(-600, 600)  # Custom range
)

# System-by-system analysis
for i, system in enumerate(results.model.config.systems):
    print(f"System {i+1}: z={system.redshift:.6f}")
    results.plot_system_fit(i, save_path=f'system_{i+1}.pdf')
```

### Expected Output
- Multi-system absorption profile
- Component-by-component decomposition
- Ion-specific velocity plots
- System contamination analysis

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python rbvfit2-multi-instrument-tutorial2.py
```

---

## Tutorial Tips

### Before Running Examples

1. **Check Dependencies**:
```bash
python -c "import rbvfit, emcee, corner; print('All dependencies OK')"
```

2. **Data Requirements**:
- Some tutorials use `rbcodes` for data loading
- Synthetic data examples work without external data
- Real data examples may need modification for your data format

3. **Computational Resources**:
- Quick fits: seconds
- MCMC fits: minutes to hours depending on complexity
- Multi-instrument: longer due to joint likelihood evaluation

### Modifying Examples

**Use Your Data**:
```python
# Replace data loading section with:
wave, flux, error = load_your_spectrum_function()

# Adjust redshifts and transitions:
config.add_system(z=your_redshift, ion='YourIon', transitions=[...], components=N)
```

**Customize Parameters**:
```python
# Adjust initial guesses based on your system
N_guess = [your_column_densities]
b_guess = [your_doppler_parameters]  
v_guess = [your_velocity_offsets]

# Modify bounds for your parameter ranges
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess,
                               Nlow=[...], Nhi=[...], ...)
```

### Common Modifications

**Different Ions**:
```python
# Replace MgII with CIV
config.add_system(z=1.5, ion='CIV', transitions=[1548.2, 1550.3], components=2)

# Replace with FeII
config.add_system(z=0.8, ion='FeII', transitions=[2374.5, 2382.8], components=1)
```

**Different Instruments**:
```python
# High resolution spectrograph
model = VoigtModel(config, FWHM='1.5')  # UVES-like

# Low resolution spectrograph  
model = VoigtModel(config, FWHM='5.0')  # SDSS-like
```

---

## Next Steps

After completing these tutorials:

- [Fitting Methods](fitting-methods.md) - Algorithm details
- [Examples Gallery](examples-gallery.md) - Visual results showcase
- [Advanced Topics](user-guide.md#advanced-topics) - Specialized techniques

### Getting Help

- **Issues with examples**: Check [GitHub Issues](https://github.com/rongmon/rbvfit/issues)
- **Questions**: Use [GitHub Discussions](https://github.com/rongmon/rbvfit/discussions)
- **Email**: rongmon@andrew.cmu.edu

---

[← Back to Main Documentation](../README.md) | [Next: Fitting Methods →](fitting-methods.md)