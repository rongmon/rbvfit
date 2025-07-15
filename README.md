# rbvfit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10403231.svg)](https://doi.org/10.5281/zenodo.10403231)

> **Version 2.0**  
> Major release with multi-component support, interactive tools, and optimized performance.  

**Bayesian Voigt Profile Fitting for Absorption Line Spectroscopy**

`rbvfit` performs forward modeling of absorption line spectra using Bayesian Voigt profile fitting.  
Version 2.0 introduces:
- Multi-system and multi-ion fitting
- Automatic parameter tying
- Joint fitting across instruments
- Interactive parameter exploration
- Support for [*emcee*](https://emcee.readthedocs.io/) and [*zeus*](https://zeus-mcmc.readthedocs.io/) samplers



![rbvfit Example](docs/images/rbvfit_example.png)  
*Example: Multi-component MgII absorption line fit with `rbvfit`*

---

## 📋 Quick Navigation

| Section                | Link                                      |
|------------------------|-------------------------------------------|
| 🚀 Quick Start         | [Jump to Quick Start](#-quick-start)     |
| 💾 Installation        | [Jump to Installation](#installation)    |
| 🎮 Interactive Mode    | [Jump to Interactive Mode](#-interactive-mode) |
| 📚 Documentation       | [Jump to Documentation](#-documentation) |
| 📁 Examples            | [Jump to Examples](#-examples)           |

---
## Installation

### Recommended Installation (conda)
```bash
# Create new conda environment
conda create -n rbvfit python=3.9
conda activate rbvfit
# Clone the repository
git clone https://github.com/rongmon/rbvfit.git
cd rbvfit
# Install dependencies and rbvfit
pip install -r requirements.txt
pip install -e .
```
### Alternative Installation Methods

**Legacy method**:
```bash
git clone https://github.com/rongmon/rbvfit.git
cd rbvfit
python setup.py develop
```
### Dependencies

- **Core**: numpy, scipy, matplotlib, emcee, corner, astropy (for convolution)
- **Interactive**: ipywidgets (Jupyter), tkinter/Qt (command-line)
- **Optional**: linetools (for COS-LSF), zeus (alternative MCMC sampler), h5py (results persistence)


## 🚀 Quick Start


# Basic usage
```python
import numpy as np
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
import rbvfit.vfit_mcmc as mc

# Set up configuration with FWHM at the start
config = FitConfiguration() 
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)

FWHM_pixels='6.5'#FWHM in pixels
# Note: If you have FWHM in km/s, convert to pixels:
# from rbvfit.core.voigt_model import mean_fwhm_pixels
# FWHM_pixels = mean_fwhm_pixels(FWHM_vel_kms, wave_obs_grid)

# Create model (FWHM automatically extracted from configuration)
model = VoigtModel(config,FWHM=FWHM_pixels)

# Set initial parameter guesses
nguess = [14.2, 14.5]  # log10(column density) in cm^-2
bguess = [40., 30.]    # Doppler parameter in km/s
vguess = [0., 0.]      # Velocity offset in km/s

# Set parameter bounds
bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
theta_guess = np.concatenate([nguess, bguess, vguess])

# Create Instrument Data dictionary for fitter
instrument_data = {
    'COS': {
        'model': model,
        'wave': wave, # Wavelength grid in Angstroms
        'flux': flux, # Normalized flux array
        'error': error # Normalized error array
    }
}

# Fit with MCMC
fitter = mc.vfit(
    instrument_data, theta, lb, ub,
    no_of_Chain=n_walkers, 
    no_of_steps=n_steps,
    sampler='zeus',
    perturbation=1e-4
)



        
# Run MCMC
fitter.runmcmc(optimize=True, verbose=True, use_pool=False)



# Analyze results
from rbvfit.core import unified_results as u 
results = u.UnifiedResults(fitter)
results.print_summary()   
results.help()
```

## What's New in Version 2.0

| Feature                  | v1.0               | v2.0                          |
| ------------------------ | ------------------ | ----------------------------- |
| Interactive Tools        | ✗ Basic            | ✅ GUI with widget support     |
| Parameter Estimation     | ✗ Manual           | ✅ Visual + GUI guessing       |
| Multi-System Setup       | ✗ Manual config    | ✅ Ion-based auto setup        |
| Parameter Tying          | Single redshift    | ✅ Multi-redshift              |
| Multi-Instrument Fitting | 2 instruments max  | ✅ Full joint N-instrument fit |
| Fitting Methods          | emcee only         | ✅ emcee + zeus + curve\_fit   |
| Results Analysis         | Basic output       | ✅ Diagnostics + plots         |
| Data Persistence         | Manual             | ✅ HDF5 w/ metadata            |
| Code Architecture        | Monolithic         | ✅ Modular                     |
| Fitting Engine           | Standard           | ✅ Vectorized +optimized       |
|FWHM Configuration        | ✗ Manual handling  |✅ Automatic at setup stage    |

## 📁 Examples

Explore working examples in [`src/rbvfit/examples/`](src/rbvfit/examples/):

- `example_voigt_model.py` - Basic model creation
- `example_voigt_fitter.py` - Single system fitting  
- `rbvfit2-single-instrument-tutorial.py` - Complete single dataset workflow
- `rbvfit2-single-instrument-interactive-tutorial.py` - **Interactive mode demonstration**
- `rbvfit2-multi-instrument-tutorial.py` - Joint fitting multiple datasets

## Description

**Main Modules (Version 2.0)**:

**Core Architecture (`rbvfit/core/`)**:
- **voigt_model.py**: Main model class for creating and evaluating Voigt profiles with automatic ion parameter tying
- **fit_configuration.py**: Configuration system for defining multi-system absorption setups with FWHM integration
- **parameter_manager.py**: Handles parameter mapping between configurations and fitting arrays
- **unified_results.py**: Enhanced results management with HDF5 persistence and analysis capabilities
- **quick_fit_interface.py**: Fast scipy.optimize-based fitting interface
- **results_plot.py**: Advanced plotting utilities for fit diagnostics and visualizations

**Interactive Tools**:
- **guess_profile_parameters_interactive.py**: **Enhanced interactive parameter estimation with GUI support**

**Fitting Engine**:
- **vfit_mcmc.py**: MCMC fitter supporting emcee and zeus samplers for Bayesian parameter estimation

**Utility Modules**:
- **rb_setline.py**: Line properties reader using approximate rest wavelength guess (active in v2.0)


**Key Version 2.0 Improvements**:
- **🎯 Enhanced Interactive Tools**: Visual component identification with cross-platform GUI support
- **Ion-specific model setup**: Clean configuration system with automatic ion detection and parameter organization
- **Enhanced multi-instrument support**: Full N-instrument joint fitting (v1.0 limited to 2 instruments)  
- **Multi-redshift parameter tying**: Automatic parameter sharing for same ions across different redshift systems
- **Multiple fitting algorithms**: Support for emcee, zeus (MCMC) and scipy curve_fit (quick fitting)
- **Advanced results analysis**: Comprehensive fit diagnostics, convergence analysis, and publication-quality visualization
- **HDF5 persistence**: Complete save/load functionality for complex fitting results
- **Modular architecture**: Clean separation between models, fitting, and analysis components
- **Streamlined FWHM configuration**: FWHM defined at configuration stage for better multi-instrument handling

## 🎯 Typical Workflow

### 1. Interactive Parameter Estimation
```python
from rbvfit import guess_profile_parameters_interactive as g

# Visual component identification
tab = g.gui_set_clump(wave, flux, error, zabs, wrest=1548.5)
tab.input_b_guess()  # Interactive parameter input
```

### 2. Model Configuration
```python
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel

# Configure with FWHM at setup stage
config = FitConfiguration()
config.add_system(z=zabs, ion='CIV', transitions=[1548.2, 1550.3], 
                  components=len(tab.nguess))

# Create model (FWHM automatically extracted from configuration)
model = VoigtModel(config,FWHM='2.5')
```

### 3. MCMC Fitting
```python
import rbvfit.vfit_mcmc as mc

theta = np.concatenate([tab.nguess, tab.bguess, tab.vguess])
bounds, lb, ub = mc.set_bounds(tab.nguess, tab.bguess, tab.vguess)

instrument_data = {
    'COS': {
        'model': model,
        'wave': wave,  # Wavelength grid in Angstroms
        'flux': flux,  # Normalized flux array
        'error': error  # Normalized error array
    }
}
fitter = mc.vfit(instrument_data, theta, lb, ub)
fitter.runmcmc()
```

### 4. Results Analysis
```python
from rbvfit.core import unified_results as u 
results = u.UnifiedResults(fitter)

results.help() # Help and documentation
results.print_summary()   # Print fit summary

out = results.convergence_diagnostics() # convergence diagnostics
results.corner_plot() # Corner plot of parameters
results.velocity_plot() # Velocity plot of fitted components
```

## 🎮 Interactive Mode

rbvfit provides powerful interactive tools for visual parameter estimation:

### Jupyter Notebook Interface
```python
# Perfect for exploratory analysis
from rbvfit import guess_profile_parameters_interactive as g

tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
# Interactive widgets appear automatically in notebook
```

### Command Line Interface  
```python
# Cross-platform GUI support
tab = g.gui_set_clump(wave, flux, error, z_abs, wrest=1548.5)
tab.input_b_guess()  # Launches native GUI
```

### Key Interactive Features
- **Visual Component Identification**: Click to identify absorption components
- **Real-time Parameter Adjustment**: See model updates as you modify parameters
- **Multiple Ion Support**: Handle complex multi-ion systems interactively
- **Automatic Parameter Export**: Seamlessly transition to MCMC fitting

### Interactive Workflow Example
```python
# Step 1: Load your data
wave, flux, error = load_your_spectrum()

# Step 2: Interactive parameter estimation
from rbvfit import guess_profile_parameters_interactive as g
tab = g.gui_set_clump(wave, flux, error, z_abs=0.5, wrest=1548.5)

# Step 3: Refine parameters visually
tab.input_b_guess()  # Use GUI to adjust N, b, v values

# Step 4: Extract parameters for fitting
config = FitConfiguration()
config.add_system(z=0.5, ion='CIV', transitions=[1548.2, 1550.3], 
                  components=len(tab.nguess))

# Step 5: Run MCMC with interactive estimates as starting point
theta_guess = np.concatenate([tab.nguess, tab.bguess, tab.vguess])
# ... continue with MCMC fitting
```

**Interactive Mode is especially powerful for**:
- Complex multi-component systems
- Blended absorption features
- Parameter degeneracy exploration
- Teaching and learning absorption line analysis

## 📚 Documentation

Comprehensive documentation available in [`docs/`](docs/):

- **[Quick Start Guide](docs/quick-start-guide.md)** - Get running in 5 minutes
- **[User Guide](docs/user-guide.md)** - Complete feature overview
- **[Tutorials](docs/tutorials.md)** - Step-by-step examples
- **[Examples Gallery](docs/examples-gallery.md)** - Curated use cases
- **[API Reference](docs/api-reference.md)** - Detailed function documentation
- **[Interactive Mode Guide](docs/interactive-mode-guide.md)** - GUI tools walkthrough

### Quick Documentation Links

| Topic | Link | Description |
|-------|------|-------------|
| First Steps | [Quick Start](docs/quick-start-guide.md) | 5-minute introduction |
| Complete Guide | [User Guide](docs/user-guide.md) | Comprehensive documentation |
| Learning Path | [Tutorials](docs/tutorials.md) | Progressive examples |
| Use Cases | [Examples](docs/examples-gallery.md) | Real-world applications |
| Interactive Tools | [Interactive Guide](docs/interactive-mode-guide.md) | GUI documentation |

## Citation

If you use rbvfit in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10403231.svg)](https://doi.org/10.5281/zenodo.10403231)

## Support

- **Issues**: [GitHub Issues](https://github.com/rongmon/rbvfit/issues)
- **Interactive Mode Help**: See [Interactive Mode Guide](docs/interactive-mode-guide.md)

**Note**: 
- **Version 1.0**: Written By: Rongmon Bordoloi. July 2019. Tested on: Python 3.7+
- **Version 2.0**: Enhanced by: Rongmon Bordoloi. 2025. Tested on: Python 3.8+

## ![License](https://img.shields.io/badge/license-MIT-green)

This project is licensed under the [MIT License](LICENSE).