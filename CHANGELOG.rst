=========
Changelog
=========

Version 2.0
===========

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

Breaking Changes
----------------
- Complete rewrite with new API (v1.0 interface deprecated but still available)
- New configuration-based setup replaces manual parameter arrays
- Enhanced plotting and results analysis tools

Version 1.0
===========

Initial Features
----------------
- Basic Voigt profile fitting with MCMC
- Single absorption system support
- Manual parameter configuration
- Basic matplotlib plotting
- *emcee* MCMC sampler support
