=========
Changelog
=========
Version 2.2.1
==============

Bug Fixes & Cleanup
--------------------
- **Graceful rbcodes import**: ``gui/io.py`` now uses a try/except for the ``rbcodes`` import; missing ``rbcodes`` raises a clear ``ImportError`` with install instructions instead of crashing at startup
- **Export functions implemented**: ``export_results_csv()`` and ``export_results_latex()`` in ``gui/io.py`` were stubs; now fully implemented using ``UnifiedResults.parameter_summary()``
- **Removed dead import**: Unused ``export_results_csv``/``export_results_latex`` import removed from ``gui/results_tab.py``
- **Debug prints removed**: Two ``# DEBUG`` ``print()`` calls removed from ``gui/interactive_param_dialog.py``
- **Dependency alignment**: ``numpy`` pinned to ``>=1.22.3,<1.24`` in both ``setup.cfg`` and ``requirements.txt`` (was ``>=1.18.0`` in requirements); version specifiers added to all core deps in ``setup.cfg``
- **Missing astropy dep**: ``astropy>=5.3.3`` added to ``setup.cfg`` ``install_requires`` (was already required by core code but not declared)
- **Dynamic version in saved projects**: ``gui/io.py`` now writes the actual ``rbvfit.__version__`` into saved project files instead of the hardcoded string ``'2.0'``

Version 2.2.0
==============

Packaging & Compatibility
--------------------------
- **Removed ``pkg_resources`` dependency**: Replaced ``pkg_resources.resource_filename`` with ``importlib.resources.files`` (stdlib) in ``rb_setline``, examples, and tests
- **Explicit package data**: Added ``package_data`` declaration in ``setup.cfg`` to reliably include line lists and example data in built distributions
- **Python version aligned with rbcodes**: Updated ``python_requires`` to ``>=3.9.6,<3.11``

Version 2.1.0
==============

GUI Enhancements
----------------
- **FWHM unit toggle**: Configurations now support FWHM entry in km/s or pixels; km/s is converted to pixels at compile time via ``mean_fwhm_pixels``; each configuration can independently use different units
- **Component tick marks**: Velocity space plot and model vs data plot now draw vertical tick marks at each component's position
- **Tick mark bug fix**: Fixed incorrect tick positions in model vs data plot (velocity was applied twice via ``z_total``); ticks are now correctly placed at ``λ0 × (1 + z_total)``
- **Consistent tick aesthetics**: Tick marks in velocity space and model vs data plots use the same style (black, linewidth 1.5) and are positioned relative to the y-axis maximum
- **Velocity plot range controls**: "Set Plot Range" button in velocity space tab now correctly persists x and y range changes across redraws; dialog initializes from stored user ranges rather than tick-extended axes limits
- **export_script() fix**: Fixed broken Python output for multi-instrument cases (closing brace was inside loop; FWHM value was unquoted)

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