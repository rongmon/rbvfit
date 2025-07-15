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
model = VoigtModel(config, FWHM='6.5')  # FWHM in pixels
```

**Parameter Understanding**:
```python
# For 2-component system: [N1, N2, b1, b2, v1, v2]
theta = [13.5, 13.2, 15.0, 25.0, -150.0, 20.0]

# Evaluate model
wave = np.linspace(3700, 3820, 1000)
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
flux[mask] = 1.0
error[mask] = 1e10

# Select wavelength range
q = (wave/(1+z_abs) > 1189.5) & (wave/(1+z_abs) < 1195.0)
wave, flux, error = wave[q], flux[q], error[q]
```

**Complete Workflow with Unified Interface**:
```python
# 1. Configure system
config = FitConfiguration()
config.add_system(z=z_abs, ion='SiII', transitions=[1190.5, 1193.5], components=2)

# 2. Create model with FWHM
model = VoigtModel(config, FWHM='6.5')  # FWHM in pixels

# 3. Set parameters and bounds  
theta_guess = [14.2, 14.5, 40.0, 30.0, 0.0, 0.0]
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)

# 4. Create Instrument Data dictionary for unified interface
instrument_data = {
    'COS': {
        'model': model,     # VoigtModel object
        'wave': wave,       # Wavelength grid in Angstroms
        'flux': flux,       # Normalized flux array
        'error': error      # Normalized error array
    }
}

# 5. Fit and analyze with unified interface
fitter = mc.vfit(
    instrument_data, theta_guess, lb, ub, 
    sampler='zeus', 
    no_of_Chain=10, 
    no_of_steps=1000,
    perturbation=1e-4
)
fitter.runmcmc()

# 6. Create results object
from rbvfit.core import unified_results as u 
results = u.UnifiedResults(fitter)
```

### Expected Output
- Best-fit parameters with uncertainties
- Model vs data comparison plot
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
- Real observational data analysis
- Interactive parameter estimation
- Advanced MCMC techniques
- Publication-quality results

### Key Concepts

**Interactive Parameter Estimation**:
```python
from rbvfit import guess_profile_parameters_interactive as g

# Visual component identification
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
tab.input_b_guess()  # Interactive parameter input

# Extract parameters
N_guess = tab.nguess
b_guess = tab.bguess
v_guess = tab.vguess
```

**Advanced Configuration**:
```python
# Multi-component CIV system
config = FitConfiguration()
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], components=3)

model = VoigtModel(config, FWHM='2.2')
```

**Production MCMC Setup with Unified Interface**:
```python
# Optimized for convergence
instrument_data = {
    'COS': {
        'model': model,
        'wave': wave,       # Wavelength grid in Angstroms
        'flux': flux,       # Normalized flux array
        'error': error      # Normalized error array
    }
}

fitter = mc.vfit(
    instrument_data, theta_guess, lb, ub, 
    sampler='zeus', 
    no_of_Chain=10, 
    no_of_steps=1000,
    perturbation=1e-4
)
fitter.runmcmc(optimize=True, verbose=True)
```

### Expected Output
- High-quality parameter constraints
- Velocity space analysis
- Systematic uncertainty assessment
- HDF5 results file

### Advanced Features
- Multi-component kinematic analysis
- Parameter correlation assessment
- Model selection criteria
- Error budget analysis

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
config = FitConfiguration()
config.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

# Define models for each instrument with different FWHM
model_xshooter = VoigtModel(config, FWHM='2.2')  # XShooter resolution
model_fire = VoigtModel(config, FWHM='4.0')      # FIRE resolution
```

**Multi-Instrument Unified Interface**:
```python
# Create unified instrument data dictionary
instrument_data = {
    'XShooter': {
        'model': model_xshooter,    # XShooter model with FWHM=2.2
        'wave': wave_xshooter,      # XShooter wavelength array
        'flux': flux_xshooter,      # XShooter flux array
        'error': error_xshooter     # XShooter error array
    },
    'FIRE': {
        'model': model_fire,        # FIRE model with FWHM=4.0
        'wave': wave_fire,          # FIRE wavelength array
        'flux': flux_fire,          # FIRE flux array
        'error': error_fire         # FIRE error array
    }
}
```

**Joint Parameter Estimation**:
```python
# Same physical parameters fit both datasets
theta_guess = [13.0, 25.0, 0.0]  # [N, b, v] shared between instruments

# Create vfit object with unified interface
fitter = mc.vfit(
    instrument_data,              # All instruments in one dictionary
    theta_guess, lb, ub,         # Parameters and bounds
    no_of_Chain=20,
    no_of_steps=500,
    perturbation=1e-4,
    sampler='zeus'
)

# Same fitting call regardless of number of instruments!
fitter.runmcmc()

# Results automatically handle multi-instrument
results = u.UnifiedResults(fitter)
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
- Instrument verification diagnostics

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python rbvfit2-multi-instrument-tutorial.py
```

---

## 5. Complex Multi-System
**File**: [`../src/rbvfit/examples/rbvfit2-multi-instrument-tutorial2.py`](../src/rbvfit/examples/rbvfit2-multi-instrument-tutorial2.py)

### Learning Objectives
- Multiple redshift systems
- Multi-ion parameter tying
- Complex contamination modeling
- Three-instrument joint fitting

### Key Concepts

**Multi-System Configuration**:
```python
# Complex absorption with multiple redshifts
config = FitConfiguration()
config.add_system(z=4.97, ion='CIV', transitions=[1548.2,1550.3], components=2)
config.add_system(z=6.1, ion='OI', transitions=[1302.17], components=1)
config.add_system(z=6.1, ion='SiII', transitions=[1304.5], components=1)
```

**Three-Instrument Unified Interface**:
```python
# Create models with different FWHM for each instrument
model_xshooter = VoigtModel(config, FWHM='2.2')   # XShooter
model_fire = VoigtModel(config, FWHM='4.0')       # FIRE
model_hires = VoigtModel(config, FWHM='4.285')    # HIRES

# Multi-instrument unified data dictionary
instrument_data = {
    'XShooter': {
        'model': model_xshooter,    # XShooter model with correct FWHM
        'wave': wave_xshooter,      # XShooter wavelength array
        'flux': flux_xshooter,      # XShooter flux array
        'error': error_xshooter     # XShooter error array
    },
    'FIRE': {
        'model': model_fire,        # FIRE model with correct FWHM
        'wave': wave_fire,          # FIRE wavelength array
        'flux': flux_fire,          # FIRE flux array
        'error': error_fire         # FIRE error array
    },
    'HIRES': {
        'model': model_hires,       # HIRES model with correct FWHM
        'wave': wave_hires,         # HIRES wavelength array
        'flux': flux_hires,         # HIRES flux array
        'error': error_hires        # HIRES error array
    }
}

# Same unified fitting interface!
fitter = mc.vfit(instrument_data, theta_guess, lb, ub)
fitter.runmcmc()
```

**Complex Parameter Structure**:
```python
# Parameter organization for multi-system fitting
# CIV: 2 components at z=4.97
# OI:  1 component at z=6.1  
# SiII: 1 component at z=6.1
# Total: 4 velocity components = 12 parameters

theta_guess = np.concatenate([
    [13.25, 13.63, 13.12, 13.2],    # N values [log cm^-2]
    [23.0, 25., 50., 13.2],         # b values [km/s]
    [-67., 0., -20., -20.]          # v values [km/s]
])
```

### Expected Output
- Multi-instrument joint fit
- Complex velocity structure analysis
- Ion-specific parameter correlations
- Instrument verification plots

### Advanced Features
- Redshift-dependent parameter tying
- Contamination assessment
- Multi-scale kinematic analysis
- Publication-ready multi-panel plots

### Run This Tutorial
```bash
cd src/rbvfit/examples/
python rbvfit2-multi-instrument-tutorial2.py
```

---

## Interactive Mode Deep Dive

### Getting Started with Interactive Tools

**Basic Interactive Setup**:
```python
from rbvfit import guess_profile_parameters_interactive as g

# Launch interactive parameter estimation
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
```

**Interactive Workflow**:
1. **Visual Component Identification**: Click on absorption features
2. **Parameter Input**: Use GUI to set N, b, v estimates  
3. **Real-time Preview**: See model updates as you adjust parameters
4. **Parameter Export**: Extract estimates for MCMC fitting

**Integration with Unified Interface**:
```python
# Interactive parameter estimation
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
tab.input_b_guess()

# Extract parameters
N_guess = tab.nguess
b_guess = tab.bguess
v_guess = tab.vguess

# Use in unified interface
config = FitConfiguration()
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], 
                  components=len(N_guess))

model = VoigtModel(config, FWHM='2.5')

instrument_data = {
    'COS': {
        'model': model,
        'wave': wave,
        'flux': flux,
        'error': error
    }
}

theta_guess = np.concatenate([N_guess, b_guess, v_guess])
fitter = mc.vfit(instrument_data, theta_guess, lb, ub)
```

**Best Practices**:
- Start with obvious, strong components
- Use velocity space for component identification
- Iteratively refine parameter estimates
- Save parameter sets for different models

### Advanced Interactive Features

**Custom Velocity Ranges**:
```python
# Focus on specific velocity range
tab.set_velocity_range(-200, +100)  # km/s
tab.update_display()
```

**Multi-Ion Interactive**:
```python
# Interactive estimation for multiple ions
tab_mgii = g.gui_set_clump(wave, flux, error, z_abs, wrest=2796.3)
tab_feii = g.gui_set_clump(wave, flux, error, z_abs, wrest=2344.2)
```

---

## Tutorial Comparison Matrix

| Feature | Tutorial 1 | Tutorial 2 | Tutorial 3 | Tutorial 4 | Tutorial 5 |
|---------|------------|------------|------------|------------|------------|
| **Complexity** | Basic | Simple | Intermediate | Advanced | Expert |
| **Data Type** | Synthetic | Synthetic | Real | Real | Real |
| **Components** | 2 | 2-3 | 3+ | 1 | 4+ |
| **Instruments** | 1 | 1 | 1 | 2 | 3 |
| **Redshifts** | 1 | 1 | 1 | 1 | 2 |
| **Ions** | 1 | 1-2 | 1 | 1 | 3 |
| **Interactive** | No | No | Yes | Optional | Optional |
| **Time** | 10 min | 15 min | 20 min | 30 min | 45 min |
| **Interface** | Basic | Unified | Unified | Unified | Unified |

---

## Common Analysis Patterns

### Pattern 1: Single Ion, Multiple Components
```python
# Use for: Complex velocity structure analysis
config = FitConfiguration()
config.add_system(z=0.5, ion='CIV', transitions=[1548.2, 1550.3], components=4)
model = VoigtModel(config, FWHM='2.5')
```

### Pattern 2: Multiple Ions, Same Redshift
```python
# Use for: Abundance ratio analysis, shared kinematics
config = FitConfiguration()
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.5, ion='FeII', transitions=[2344.2, 2374.5], components=2)
model = VoigtModel(config, FWHM='3.0')
```

### Pattern 3: Multi-Redshift Systems
```python
# Use for: Contamination, intervening systems
config = FitConfiguration()
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=1)
config.add_system(z=1.2, ion='CIV', transitions=[1548.2, 1550.3], components=2)
model = VoigtModel(config, FWHM='2.8')
```

### Pattern 4: Multi-Instrument Joint Fitting
```python
# Use for: Enhanced constraints, systematic checks
model_A = VoigtModel(config, FWHM='2.2')  # High resolution
model_B = VoigtModel(config, FWHM='4.0')  # Lower resolution

instrument_data = {
    'HighRes': {'model': model_A, 'wave': wave1, 'flux': flux1, 'error': error1},
    'LowRes': {'model': model_B, 'wave': wave2, 'flux': flux2, 'error': error2}
}
# Identical physical systems, different instrumental setups
```

---

## Troubleshooting Guide

### Issue: Poor Convergence
**Symptoms**: 
- Low acceptance rates
- Parameter chains not mixing
- Large Gelman-Rubin statistics

**Solutions**:
```python
# Increase MCMC steps
fitter.no_of_steps = 2000

# More walkers
fitter.no_of_Chain = 100

# Better initial guess
from rbvfit import guess_profile_parameters_interactive as g
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=rest_wave)

# Optimize before MCMC
fitter.runmcmc(optimize=True)
```

### Issue: Parameter Degeneracies
**Symptoms**:
- Highly correlated parameters in corner plots
- Large uncertainties
- Bimodal posteriors

**Solutions**:
```python
# Simplify model
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], components=2)  # Reduce from 3

# Add more transitions
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], components=2)
config.add_system(z=z_abs, ion='SiIV', transitions=[1393.8, 1402.8], components=2)

# Multi-instrument constraints
# Use different resolution data to break degeneracies
```

### Issue: Model-Data Mismatch
**Symptoms**:
- Systematic residuals
- High χ² values
- Poor visual fit quality

**Solutions**:
```python
# Check wavelength calibration
wave_corrected = wave * (1 + velocity_correction/c)

# Verify FWHM
model = VoigtModel(config, FWHM='3.5')  # Try different values

# Add velocity components
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], components=3)  # Increase from 2

# Check continuum normalization
flux_renorm = flux / polynomial_fit(wave, flux)
```

### Issue: Unified Interface Problems
**Symptoms**:
- KeyError for instrument data
- Array length mismatches
- Model compilation errors

**Solutions**:
```python
# Check instrument data format
instrument_data = {
    'INSTRUMENT_NAME': {
        'model': voigt_model_object,  # Must be VoigtModel
        'wave': wave_array,           # 1D numpy array
        'flux': flux_array,           # 1D numpy array
        'error': error_array          # 1D numpy array (same length as wave/flux)
    }
}

# Verify all arrays have same length
print(f"Wave: {len(wave)}, Flux: {len(flux)}, Error: {len(error)}")

# Ensure model is VoigtModel object
print(f"Model type: {type(model)}")
```

---

## Performance Guidelines

### Speed Optimization
```python
# Quick fitting for initial parameters
quick_result = fitter.fit_quick()
theta_init = quick_result.x

# Parallel MCMC (automatic on multi-core)
fitter = mc.vfit(instrument_data, theta_init, lb, ub, no_of_Chain=2*n_cores)

# Vectorized evaluation for large grids
wave_hires = np.linspace(wave.min(), wave.max(), 10000)
```

### Memory Management
```python
# For large datasets or many instruments
# Process instruments sequentially if memory limited
results_list = []
for name, data in instrument_data.items():
    single_instrument = {name: data}
    fitter_single = mc.vfit(single_instrument, theta, lb, ub)
    # ... fit individual instruments
    results_list.append(fitter_single)
```

### Convergence Monitoring
```python
# Real-time convergence checking
fitter.runmcmc(optimize=True, verbose=True)
results = u.UnifiedResults(fitter)
results.convergence_diagnostics()

# Auto-extend if not converged
if not results.is_converged():
    fitter.no_of_steps += 1000
    fitter.runmcmc(continue_chain=True)
```

---

## Next Steps After Tutorials

### 🔬 **Research Applications**
- Intergalactic medium studies
- Galaxy-CGM absorption analysis  
- Quasar proximity effect measurements
- Metal abundance determinations

### 🛠 **Advanced Techniques**
- Custom line lists with rb_setline
- Systematic uncertainty propagation
- Model selection and comparison
- Bayesian evidence calculation

### 📊 **Publication Workflow**
- Parameter table generation
- Publication-quality plotting
- Results persistence and sharing
- Reproducible analysis pipelines

### 💻 **Code Development**
- Custom line spread functions
- New ion database integration
- Alternative MCMC samplers
- Performance profiling and optimization

---

For questions, issues, or contributions, visit the [GitHub repository](https://github.com/rongmon/rbvfit) or check the [Examples Gallery](examples-gallery.md) for more specialized use cases.