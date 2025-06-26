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

# Simple MgII doublet with FWHM configuration
config = FitConfiguration(FWHM='2.5')
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
model = VoigtModel(config)  # FWHM automatically extracted from config
```

**Parameter Understanding**:
```python
# For 2-component system: [N1, N2, b1, b2, v1, v2]
theta = [13.5, 13.2, 15.0, 25.0, -150.0, 20.0]

# Evaluate model
wave = np.linspace(3700, 3820, 10000)
compiled_model = model.compile()
flux = compiled_model.model_flux(theta, wave)
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
# 1. Configure system with FWHM
config = FitConfiguration(FWHM='6.5')
config.add_system(z=z_abs, ion='SiII', transitions=[1190.5, 1193.5], components=2)

# 2. Create and compile model
model = VoigtModel(config)  # FWHM from config
compiled = model.compile()

# 3. Set parameters and bounds  
theta_guess = [14.2, 14.5, 40.0, 30.0, 0.0, 0.0]
bounds, lb, ub = mc.set_bounds(N_guess, b_guess, v_guess)

# 4. Fit and analyze
fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
fitter.runmcmc()
results = fr.FitResults(fitter, model)
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
# Multi-component CIV system with optimized FWHM
config = FitConfiguration(FWHM='2.2')
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], components=3)

model = VoigtModel(config)
compiled = model.compile()
```

**Production MCMC Setup**:
```python
# Optimized for convergence
fitter = mc.vfit(compiled.model_flux, theta_guess, lb, ub, wave, flux, error)
fitter.no_of_Chain = 100
fitter.no_of_steps = 2000
fitter.sampler = 'emcee'
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
# FWHM defined at configuration stage for each instrument
config_A = FitConfiguration(FWHM='2.2')  # XShooter configuration
config_A.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

config_B = FitConfiguration(FWHM='4.0')  # FIRE configuration  
config_B.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)

# Models automatically use correct FWHM from configurations
model_A = VoigtModel(config_A)  # Uses FWHM='2.2'
model_B = VoigtModel(config_B)  # Uses FWHM='4.0'
```

**Multi-Instrument Compilation**:
```python
# Create joint model with instrument-specific configurations
instrument_configs = {
    'XShooter': config_A,  # Contains FWHM='2.2'
    'FIRE': config_B       # Contains FWHM='4.0'
}

# Compile for joint fitting (FWHM extracted automatically)
compiled = model_A.compile(instrument_configs=instrument_configs)
```

**Joint Parameter Estimation**:
```python
# Same physical parameters fit both datasets
theta_guess = [13.0, 25.0, 0.0]  # [N, b, v] shared between instruments

# Joint likelihood combines both datasets
joint_datasets = {
    'XShooter': {'wave': wave_A, 'flux': flux_A, 'error': error_A},
    'FIRE': {'wave': wave_B, 'flux': flux_B, 'error': error_B}
}

fitter = mc.vfit_multi_instrument(
    compiled.joint_model, theta_guess, bounds,
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
# Each instrument gets its own configuration with correct FWHM
FWHM_XShooter = '2.2'   
FWHM_FIRE = '4.0'      
FWHM_HIRES = '4.285'   

config_A = FitConfiguration(FWHM=FWHM_XShooter)  # XShooter
config_A.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=2)
config_A.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
config_A.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)

config_B = FitConfiguration(FWHM=FWHM_FIRE)      # FIRE
config_B.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=2)
config_B.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
config_B.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)

config_C = FitConfiguration(FWHM=FWHM_HIRES)     # HIRES
config_C.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=2)
config_C.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
config_C.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)
```

**Three-Instrument Compilation**:
```python
# Create models (FWHM automatically extracted from configurations)
model_A = VoigtModel(config_A)  # XShooter FWHM
model_B = VoigtModel(config_B)  # FIRE FWHM
model_C = VoigtModel(config_C)  # HIRES FWHM

# Multi-instrument compilation with automatic FWHM handling
instrument_configs = {
    'XShooter': config_A,  # Contains FWHM='2.2'
    'FIRE': config_B,      # Contains FWHM='4.0'  
    'HIRES': config_C      # Contains FWHM='4.285'
}

compiled = model_A.compile(instrument_configs=instrument_configs)
```

**Complex Parameter Structure**:
```python
# Parameter organization for multi-system fitting
# CIV: 2 components at z=4.9484
# OI:  1 component at z=6.075  
# SiII: 1 component at z=6.075
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

---

## Common Analysis Patterns

### Pattern 1: Single Ion, Multiple Components
```python
# Use for: Complex velocity structure analysis
config = FitConfiguration(FWHM='2.5')
config.add_system(z=0.5, ion='CIV', transitions=[1548.2, 1550.3], components=4)
```

### Pattern 2: Multiple Ions, Same Redshift
```python
# Use for: Abundance ratio analysis, shared kinematics
config = FitConfiguration(FWHM='3.0')
config.add_system(z=0.5, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.5, ion='FeII', transitions=[2344.2, 2374.5], components=2)
```

### Pattern 3: Multi-Redshift Systems
```python
# Use for: Contamination, intervening systems
config = FitConfiguration(FWHM='2.8')
config.add_system(z=0.3, ion='MgII', transitions=[2796.3, 2803.5], components=1)
config.add_system(z=1.2, ion='CIV', transitions=[1548.2, 1550.3], components=2)
```

### Pattern 4: Multi-Instrument Joint Fitting
```python
# Use for: Enhanced constraints, systematic checks
config_A = FitConfiguration(FWHM='2.2')  # High resolution
config_B = FitConfiguration(FWHM='4.0')  # Lower resolution
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
config = FitConfiguration(FWHM='3.5')  # Try different values

# Add velocity components
config.add_system(z=z_abs, ion='CIV', transitions=[1548.2, 1550.3], components=3)  # Increase from 2

# Check continuum normalization
flux_renorm = flux / polynomial_fit(wave, flux)
```

---

## Performance Guidelines

### Speed Optimization
```python
# Quick fitting for initial parameters
quick_result = fitter.fit_quick()
theta_init = quick_result.x

# Parallel MCMC (automatic on multi-core)
fitter.no_of_Chain = 2 * n_cores

# Vectorized evaluation for large grids
wave_hires = np.linspace(wave.min(), wave.max(), 10000)
```

### Memory Management
```python
# For large datasets or many instruments
# Process instruments sequentially if memory limited
results_list = []
for config in instrument_configs:
    model = VoigtModel(config)
    # ... fit individual instruments
    results_list.append(results)
```

### Convergence Monitoring
```python
# Real-time convergence checking
fitter.runmcmc(optimize=True, verbose=True)
results = fr.FitResults(fitter, model)
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