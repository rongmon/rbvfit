=========
Changelog
=========
Version 2.0.1
==============
- GUI bug fix: Resolved limited x-axis range in model comparison plots
- GUI bug fix: Fixed issue with correctly auto assigning b values during interactive component selection
-GUI bug fix: Fixed crash issue with while deleting interactive components [patch applied]

Version 2.0.0
==============

Major Features
--------------
- **Multi-system support**: Fit absorption systems at different redshifts simultaneously
- **Automatic ion parameter tying**: Same ion at same redshift shares physical parameters
- **Multi-instrument joint fitting**: Combine data from different instruments with automatic LSF handling
- **Interactive parameter estimation**: GUI tools for initial parameter guessing
- **Enhanced MCMC samplers**: Support for both *emcee* and *zeus* samplers
- **Vectorized fitting engine**: Optimized performance for complex multi-system fits
- **HDF5 data persistence**: Save and load complete fitting results with metadata
- **Modular architecture**: Clean, maintainable codebase with clear separation of concerns

GUI Enhancements
----------------
- **Modern PyQt5 interface**: Complete 4-tab workflow design
- **Command-line launcher**: ``rbvfit_gui`` command for easy access
- **Project management**: Save/load complete analysis sessions (.rbv files)
- **Interactive plotting**: Real-time model comparison and parameter visualization
- **Configuration management**: Multi-instrument setup with FWHM handling
- **Data processing tools**: Wavelength trimming, union building, and validation
- **Results analysis**: Corner plots, trace plots, and model comparison views
- **Export capabilities**: CSV, JSON, and LaTeX table output formats

Command-Line Interface
----------------------
- **Entry point integration**: Automatic ``rbvfit_gui`` command installation
- **Project file support**: Launch GUI with pre-loaded configurations
- **Version detection**: Dynamic version reporting from package metadata
- **Error handling**: Graceful handling of missing dependencies and files

Unified vfit Interface
----------------------
- **Single API**: Consistent interface for single and multi-instrument fitting
- **Automatic compilation**: VoigtModel objects with per-instrument FWHM
- **Ion-aware bounds**: Built-in parameter bounds for common astronomical ions
- **Sampler flexibility**: Easy switching between emcee and zeus backends
- **Performance optimization**: Multiprocessing context detection and optimization

Breaking Changes
----------------
- Complete rewrite with new API (v1.0 interface available in a separate branch called v1)
- New configuration-based setup replaces manual parameter arrays
- Enhanced plotting and results analysis tools
- GUI requires PyQt5 dependency 
- Project files use new .rbv format with comprehensive metadata

Installation Updates
--------------------
- **Entry points**: Automatic command-line script creation via setuptools
- **Optional dependencies**: GUI components available as optional install
- **Requirements updates**: Added PyQt5, updated matplotlib and corner versions
- **Cross-platform support**: Improved Windows compatibility with multiprocessing

Documentation
-------------
- **Comprehensive GUI tutorial**: Step-by-step workflow guide with screenshots
- **API documentation**: Complete reference for programmatic usage
- **Example workflows**: Common use cases and best practices
- **Troubleshooting guide**: Solutions for common installation and usage issues

Version 1.0
============

Initial Features
----------------
- Basic Voigt profile fitting with MCMC
- Single absorption system support
- Manual parameter configuration
- Basic matplotlib plotting
- *emcee* MCMC sampler support