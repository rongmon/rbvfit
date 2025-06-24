# rbvfit
======

**Bayesian Voigt Profile Fitting for Absorption Line Spectroscopy**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10403232.svg)](https://doi.org/10.5281/zenodo.10403232)

This suite of code will do a forward modeling analysis of absorption line spectrum using Bayesian Voigt profile fitting. This version 2.0 features multi-system support, automatic ion parameter tying, and multi-instrument joint fitting capabilities.

![rbvfit Example](docs/images/rbvfit_example.png)
*Example: Multi-component MgII absorption line fit with rbvfit*

## 🚀 Quick Start

```bash
# Installation  
git clone https://github.com/rongmon/rbvfit.git
cd rbvfit
python setup.py develop

# Basic usage
from rbvfit.core import FitConfiguration, VoigtModel
import rbvfit.vfit_mcmc as mc

# Create configuration
config = FitConfiguration()
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)

# Fit your data
model = VoigtModel(config)
fitter = mc.vfit(model, theta_guess, bounds, wave, flux, error)
fitter.runmcmc()
```

## 📚 Documentation

| Guide | Description | Level |
|-------|-------------|-------|
| [Quick Start Guide](docs/quick-start-guide.md) | Get your first fit running in 5 minutes | Beginner |
| [User Guide](docs/user-guide.md) | Comprehensive workflow and concepts | Intermediate |
| [Tutorials](docs/tutorials.md) | Step-by-step examples with code | All levels |
| [Fitting Methods](docs/fitting-methods.md) | Quick fit vs MCMC comparison | Advanced |
| [Examples Gallery](docs/examples-gallery.md) | Visual showcase of capabilities | All levels |

## ✨ Key Features

- **Multi-System Support**: Fit multiple absorption systems simultaneously
- **Automatic Ion Tying**: Parameters shared correctly for same ions at same redshift
- **Multi-Instrument Fitting**: Joint analysis of data from different telescopes
- **Fast & Robust**: Both quick scipy fitting and full Bayesian MCMC
- **Rich Visualization**: Corner plots, velocity plots, convergence diagnostics

## Installation

### From source
```bash
git clone https://github.com/rongmon/rbvfit.git
cd rbvfit
python setup.py develop
```

**Alternative (modern pip)**:
```bash
git clone https://github.com/rongmon/rbvfit.git
cd rbvfit
pip install -e .
```

## Dependencies

- **Core**: numpy, scipy, matplotlib, emcee, corner
- **Optional**: linetools (for COS-LSF), zeus (alternative MCMC sampler), h5py (results persistence)

## What's New in Version 2.0

| Feature | Version 1.0 | Version 2.0 |
|---------|-------------|-------------|
| **Multi-system setup** | Manual configuration | Ion-specific automatic configuration |
| **Parameter tying** | Single redshift only | Multi-redshift system support |
| **Multi-instrument** | Limited (2 instruments) | Full N-instrument joint fitting |
| **Fitting algorithms** | emcee only | emcee + zeus + scipy curve_fit |
| **Results analysis** | Basic parameter output | Advanced diagnostics + visualization |
| **Data persistence** | Manual save/load | HDF5 with complete metadata |
| **Architecture** | Monolithic scripts | Modular core components |

## 📁 Examples

Explore working examples in [`src/rbvfit/examples/`](src/rbvfit/examples/):

- `example_voigt_model.py` - Basic model creation
- `example_voigt_fitter.py` - Single system fitting  
- `rbvfit2-single-instrument-tutorial.py` - Complete single dataset workflow
- `rbvfit2-multi-instrument-tutorial.py` - Joint fitting multiple datasets

## Description

**Main Modules (Version 2.0)**:

**Core Architecture (`rbvfit/core/`)**:
- **voigt_model.py**: Main model class for creating and evaluating Voigt profiles with automatic ion parameter tying
- **fit_configuration.py**: Configuration system for defining multi-system absorption setups
- **parameter_manager.py**: Handles parameter mapping between configurations and fitting arrays
- **fit_results.py**: Enhanced results management with HDF5 persistence and analysis capabilities
- **quick_fit_interface.py**: Fast scipy.optimize-based fitting interface

**Fitting Engine**:
- **vfit_mcmc.py**: MCMC fitter supporting emcee and zeus samplers for Bayesian parameter estimation

**Legacy Modules (Version 1.0 - still available)**:
- **model.py**: Original top-level code for complex multi-component/multi-species Voigt profiles
- **rb_vfit.py**: General code to create individual Voigt profiles
- **rb_setline.py**: Line properties reader using approximate rest wavelength guess
- **rb_interactive_vpfit.py**: Interactive Voigt profile fitter with least squares and MCMC options

**Key Version 2.0 Improvements**:
- **Ion-specific model setup**: Clean configuration system with automatic ion detection and parameter organization
- **Enhanced multi-instrument support**: Full N-instrument joint fitting (v1.0 limited to 2 instruments)  
- **Multi-redshift parameter tying**: Automatic parameter sharing for same ions across different redshift systems
- **Multiple fitting algorithms**: Support for emcee, zeus (MCMC) and scipy curve_fit (quick fitting)
- **Advanced results analysis**: Comprehensive fit diagnostics, convergence analysis, and publication-quality visualization
- **HDF5 persistence**: Complete save/load functionality for complex fitting results
- **Modular architecture**: Clean separation between models, fitting, and analysis components

## Citation

If you use rbvfit in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10403232.svg)](https://doi.org/10.5281/zenodo.10403232)

## Support

- **Issues**: [GitHub Issues](https://github.com/rongmon/rbvfit/issues)

**Note**: 
- **Version 1.0**: Written By: Rongmon Bordoloi. July 2019. Tested on: Python 3.7+
- **Version 2.0**: Enhanced by: Rongmon Bordoloi. 2025. Tested on: Python 3.8+

**License**: MIT