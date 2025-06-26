#!/usr/bin/env python
"""
===============================================================================
rbvfit 2.0 Tutorial: Multi-Instrument Absorption Line Fitting
===============================================================================

This tutorial demonstrates how to simultaneously fit the same absorption line
observed with two different instruments (XShooter and FIRE) using rbvfit 2.0.

WHAT THIS SCRIPT DOES:
1. Loads real spectroscopic data from two instruments
2. Sets up shared physical parameters with different instrumental resolutions
3. Runs joint MCMC fitting to constrain absorption line properties
4. Visualizes results showing how both datasets contribute to the fit

SCIENTIFIC MOTIVATION:
- Joint fitting provides better parameter constraints than individual fits
- Accounts for different instrumental resolutions automatically
- Validates results across multiple instruments
- Combines data from different wavelength coverages/sensitivities

LEARNING OBJECTIVES:
- Understand multi-instrument configuration setup
- Learn parameter sharing concepts in rbvfit 2.0
- Practice MCMC fitting with joint datasets
- Interpret multi-instrument fitting results
"""

import numpy as np
import sys
import os

# rbvfit 2.0 imports - new architecture for multi-group fitting
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.parameter_manager import ParameterManager
from rbvfit import vfit_mcmc as mc  # MCMC fitting engine

import matplotlib.pyplot as plt

from rbcodes.GUIs.rb_spec import load_rb_spec_object

# Global settings
verbose = True  # Print detailed information during execution

# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================
# This section loads real spectroscopic data from both instruments
# In your own work, replace this with your data loading procedure

def load_spectrum(slice_name):
    """
    Load spectrum from saved rb_spec object.
    
    This is a custom data loading function. For your own data, replace this
    with whatever method you use to load wavelength, flux, and error arrays.
    
    Parameters
    ----------
    slice_name : str
        Filename of the saved spectrum object
        
    Returns
    -------
    wave, flux, error : np.ndarray
        Wavelength (Angstroms), normalized flux, and error arrays
    """
    print(f"Loading spectrum: {slice_name}")
    s_HIRES = load_rb_spec_object(filename=slice_name, verbose=verbose)
    
    # Clean and extract the spectrum data
    flux, error, wave = s_HIRES.fnorm, s_HIRES.enorm, s_HIRES.wave_slice
    
    print(f"  Loaded {len(wave)} wavelength points")
    print(f"  Wavelength range: {wave.min():.2f} - {wave.max():.2f} Å")
    
    return wave, flux, error

# Load both datasets
print("=" * 60)
print("LOADING OBSERVATIONAL DATA")
print("=" * 60)

# XShooter spectrum (higher resolution)
wave, flux, error = load_spectrum('J159_7921_XShooter_OI1302.json')
print(f"XShooter: {len(wave)} points, {wave.min():.1f}-{wave.max():.1f} Å")

# FIRE spectrum (lower resolution)  
wave1, flux1, error1 = load_spectrum('J159_7921_FIRE_OI1302.json')
print(f"FIRE: {len(wave1)} points, {wave1.min():.1f}-{wave1.max():.1f} Å")

print("\n✓ Data loading complete")

# ============================================================================
# PART 2: PHYSICAL SYSTEM CONFIGURATION
# ============================================================================
# This section sets up the physical absorption system that both instruments observe
# KEY CONCEPT: Same physics, different instrumental responses

print("\n" + "=" * 60)
print("CONFIGURING PHYSICAL ABSORPTION SYSTEM")
print("=" * 60)

# Both instruments observe the same physical system (OI 1302 at z=0.0)
# but with different instrumental resolutions

# Define instrumental resolutions (Full Width at Half Maximum in pixels)
FWHM_XShooter = '2.2'  # Higher spectral resolution (sharper lines)
FWHM_FIRE = '4.0'      # Lower spectral resolution (broader lines)

config_A = FitConfiguration(FWHM=FWHM_XShooter)
config_A.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)
print("XShooter configuration: OI 1302 at z=0.0, 1 component")

config_B = FitConfiguration(FWHM=FWHM_FIRE)
config_B.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)
print("FIRE configuration: identical physical system")

# Display the configuration details
print(f"\nPhysical system details:")
print(f"  Ion: OI (neutral oxygen)")
print(f"  Rest wavelength: 1302.17 Å")
print(f"  Redshift: z = 0.0 (systemic)")
print(f"  Components: 1 velocity component")
print(f"  Parameters to fit: N (column density), b (Doppler), v (velocity)")

print("\n✓ Physical system configured")

# ============================================================================
# PART 3: INSTRUMENTAL SETUP
# ============================================================================
# This section defines the different instrumental characteristics
# KEY CONCEPT: Different FWHM values account for different spectral resolutions

print("\n" + "=" * 60)
print("SETTING UP INSTRUMENTAL PARAMETERS")
print("=" * 60)



# Create instrument-specific models
# Each model applies different instrumental broadening to the same physics
model_A = VoigtModel(config_A)  
print(f"XShooter model: convolves with {FWHM_XShooter}-pixel Gaussian")

model_B = VoigtModel(config_B)      
print(f"FIRE model: convolves with {FWHM_FIRE}-pixel Gaussian")

print("\n✓ Instrumental models created")

# ============================================================================
# PART 4: MULTI-INSTRUMENT MODEL COMPILATION
# ============================================================================
# This is the key step that enables joint fitting
# KEY CONCEPT: Unified parameter space with per-instrument evaluation

print("\n" + "=" * 60)
print("COMPILING MULTI-INSTRUMENT MODEL")
print("=" * 60)

# Dictionary maps instrument names to their configurations
instrument_configs = {
    'XShooter': config_A,  # High-resolution configuration
    'FIRE': config_B       # Lower-resolution configuration
}

print("Instrument mapping:")
for name, config in instrument_configs.items():
    print(f"  {name}: {config.get_parameter_structure()['total_parameters']} parameters")

# Compile unified multi-instrument model
# This creates a master configuration that handles parameter sharing
compiled = model_A.compile(instrument_configs=instrument_configs, verbose=True)
print("\n✓ Multi-instrument model compiled")

# What the compilation does:
print("\nCompilation effects:")
print("  - Merges identical physics parameters (N, b, v) across instruments")
print("  - Creates unified parameter space (single theta array)")
print("  - Enables per-instrument evaluation with correct FWHM")
print("  - Maintains parameter sharing while allowing different resolutions")

# ============================================================================
# PART 5: MODEL EVALUATION FUNCTIONS
# ============================================================================
# Create wrapper functions for MCMC compatibility
# KEY CONCEPT: Single theta array controls both instrument models

print("\n" + "=" * 60)
print("CREATING MODEL EVALUATION FUNCTIONS")
print("=" * 60)

def model_xshooter(theta, wave):
    """
    Evaluate XShooter model with high resolution (2.2-pixel FWHM).
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter array [N, b, v] for all components
    wave : np.ndarray  
        Wavelength array for evaluation
        
    Returns
    -------
    np.ndarray
        Model flux convolved with XShooter instrumental response
    """
    return compiled.model_flux(theta, wave, instrument='XShooter')

def model_fire(theta, wave):
    """
    Evaluate FIRE model with lower resolution (4.0-pixel FWHM).
    
    Parameters
    ----------
    theta : np.ndarray
        Same parameter array as XShooter (shared physics!)
    wave : np.ndarray
        Wavelength array for evaluation
        
    Returns
    -------
    np.ndarray
        Model flux convolved with FIRE instrumental response
    """
    return compiled.model_flux(theta, wave, instrument='FIRE')

print("Model evaluation functions created:")
print("  model_xshooter(): applies 2.2-pixel FWHM convolution")
print("  model_fire():     applies 4.0-pixel FWHM convolution")
print("  Both use same theta parameters (shared physics)")

print("\n✓ Evaluation functions ready")

# ============================================================================
# PART 6: PARAMETER ESTIMATION AND BOUNDS
# ============================================================================
# Set up initial guesses and parameter bounds for MCMC
# KEY CONCEPT: Physical reasoning guides parameter ranges

print("\n" + "=" * 60)
print("SETTING UP MCMC PARAMETERS")
print("=" * 60)

# Initial parameter guesses based on visual inspection or previous fits
nguess = [14.4]  # log10(column density in cm^-2) - typical for OI
bguess = [18.0]  # Doppler parameter in km/s - thermal + turbulent broadening
vguess = [200.0] # Velocity offset in km/s - relative to systemic redshift

print("Initial parameter guesses:")
print(f"  N (log column density): {nguess[0]:.1f} [log cm^-2]")
print(f"  b (Doppler parameter):  {bguess[0]:.1f} km/s")
print(f"  v (velocity offset):    {vguess[0]:.1f} km/s")

# Create theta array for MCMC (concatenated parameter vector)
theta = np.concatenate([nguess, bguess, vguess])
print(f"\nTheta array structure: {theta}")
print("  theta[0] = N, theta[1] = b, theta[2] = v")

# Set parameter bounds using rbvfit's bound-setting utility
bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
print(f"\nParameter bounds:")
print(f"  N: [{lb[0]:.1f}, {ub[0]:.1f}] [log cm^-2]")
print(f"  b: [{lb[1]:.1f}, {ub[1]:.1f}] km/s") 
print(f"  v: [{lb[2]:.1f}, {ub[2]:.1f}] km/s")

print("\n✓ MCMC parameters configured")

# ============================================================================
# PART 7: JOINT MCMC FITTING
# ============================================================================
# Run the actual fitting using both datasets simultaneously
# KEY CONCEPT: Combined likelihood from both instruments

print("\n" + "=" * 60)
print("RUNNING JOINT MCMC FITTING")
print("=" * 60)

print("Setting up multi-instrument fitter...")

# Create vfit_mcmc object with multi-instrument support
fitter = mc.vfit(
    model_xshooter,           # Primary model function (XShooter)
    theta, lb, ub,            # Parameters and bounds
    wave, flux, error,        # Primary dataset (XShooter data)
    no_of_Chain=20,
    no_of_steps=500,
    perturbation=1e-4,
    sampler='zeus',
    multi_instrument=True,    # Enable multi-instrument mode
    instrument_data={         # Additional instruments
        'FIRE': {
            'model': model_fire,  # FIRE model function
            'wave': wave1,        # FIRE wavelength array
            'flux': flux1,        # FIRE flux array
            'error': error1       # FIRE error array
        }
    }
)

print("Fitter configuration:")
print("  Primary instrument: XShooter")
print("  Additional instruments: FIRE")
print("  Shared parameters: N, b, v")
print("  Different instrumental responses: Yes")

print("\nStarting MCMC sampling...")
print("This may take several minutes depending on data size and convergence")

# Run MCMC with optimization
fitter.runmcmc(optimize=True)  # optimize=True finds better starting point

print("\n✓ MCMC fitting completed")

# ============================================================================
# PART 8: RESULTS ANALYSIS
# ============================================================================
# Display and analyze the fitting results
# KEY CONCEPT: Joint constraints from both datasets

print("\n" + "=" * 60)
print("ANALYZING FITTING RESULTS")
print("=" * 60)

# Display corner plot (parameter correlations and posteriors)
print("Generating corner plot (parameter posterior distributions)...")
fitter.plot_corner()

# Extract key results
print("\nExtracting results...")
samples = fitter.samples      # MCMC samples (posterior chains)
best_theta = fitter.best_theta # Best-fit parameters (median of posterior)

# Print best-fit results
print(f"\nBest-fit parameters:")
print(f"  N = {best_theta[0]:.2f} ± {np.std(samples[:,0]):.2f} [log cm^-2]")
print(f"  b = {best_theta[1]:.1f} ± {np.std(samples[:,1]):.1f} km/s")
print(f"  v = {best_theta[2]:.1f} ± {np.std(samples[:,2]):.1f} km/s")

# Calculate derived quantities
linear_N = 10**best_theta[0]
print(f"\nDerived quantities:")
print(f"  Linear column density: {linear_N:.2e} cm^-2")
print(f"  Thermal velocity (for T=10^4 K): ~12.9 km/s")
print(f"  Turbulent component: ~{np.sqrt(best_theta[1]**2 - 12.9**2):.1f} km/s")

print("\n✓ Results analysis complete")

# ============================================================================
# PART 9: DATA AND MODEL VISUALIZATION
# ============================================================================
# Create publication-quality plots showing the joint fit
# KEY CONCEPT: Visualize how both datasets contribute to the constraints

print("\n" + "=" * 60)
print("CREATING VISUALIZATION")
print("=" * 60)

print("Preparing data/model comparison plots...")

fig = mc.plot_model(model_A, fitter, 
                outfile=False,           # or 'output.png' to save
                show_residuals=False,     # Include residual plots
                velocity_marks=True,     # Mark component velocities
                verbose=True)            # Print parameter summary

print("\n✓ Visualization complete")

# ============================================================================
# TUTORIAL SUMMARY AND NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("TUTORIAL COMPLETE - SUMMARY")
print("=" * 80)

print("\nWhat you learned:")
print("  ✓ How to load and prepare multi-instrument spectroscopic data")
print("  ✓ How to configure shared physical systems in rbvfit 2.0")
print("  ✓ How to handle different instrumental resolutions")
print("  ✓ How to compile and use multi-instrument models")
print("  ✓ How to run joint MCMC fitting")
print("  ✓ How to analyze and visualize results")

print("\nKey advantages of multi-instrument fitting:")
print("  • Better parameter constraints from combined data")
print("  • Automatic handling of different instrumental responses")
print("  • Consistency checks across multiple datasets")
print("  • Reduced systematic uncertainties")

print(f"\nYour results:")
print(f"  OI 1302 column density: N = {best_theta[0]:.2f} ± {np.std(samples[:,0]):.2f} [log cm^-2]")
print(f"  Doppler parameter:      b = {best_theta[1]:.1f} ± {np.std(samples[:,1]):.1f} km/s")
print(f"  Velocity offset:        v = {best_theta[2]:.1f} ± {np.std(samples[:,2]):.1f} km/s")

print("\nNext steps for your research:")
print("  1. Try adding more velocity components if needed")
print("  2. Fit additional transitions (e.g., OI 1355) jointly")
print("  3. Add more instruments if available")
print("  4. Compare results with single-instrument fits")
print("  5. Explore different ion species in the same system")

print("\nFor questions or issues, consult the rbvfit 2.0 documentation")
print("Happy fitting! 🎉")