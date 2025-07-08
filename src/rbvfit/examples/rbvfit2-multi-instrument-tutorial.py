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

NEW IN THIS VERSION:
- Uses the new unified interface for cleaner multi-instrument setup
- Demonstrates automatic instrument detection
- Shows improved plotting capabilities
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

config = FitConfiguration()
config.add_system(z=0.0, ion='OI', transitions=[1302.17], components=1)
print("Absorber configuration: OI 1302 at z=0.0, 1 component")


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
model_xshooter = VoigtModel(config,FWHM=FWHM_XShooter)  
print(f"XShooter model: convolves with {FWHM_XShooter}-pixel Gaussian")

model_FIRE = VoigtModel(config,FWHM=FWHM_FIRE)      
print(f"FIRE model: convolves with {FWHM_FIRE}-pixel Gaussian")

print("\n✓ Instrumental models created")


print("\n" + "=" * 60)


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
# PART 7: JOINT MCMC FITTING - NEW UNIFIED INTERFACE
# ============================================================================
# Run the actual fitting using both datasets simultaneously
# KEY CONCEPT: Combined likelihood from both instruments using new clean interface

print("\n" + "=" * 60)
print("RUNNING JOINT MCMC FITTING - NEW UNIFIED INTERFACE")
print("=" * 60)

print("Setting up multi-instrument fitter with new unified interface...")

# NEW UNIFIED INTERFACE: Define instrument data outside the call
instrument_data = {
    'XShooter': {
        'model': model_xshooter,    # XShooter model function
        'wave': wave,               # XShooter wavelength array
        'flux': flux,               # XShooter flux array
        'error': error              # XShooter error array
    },
    'FIRE': {
        'model': model_FIRE,        # FIRE model function
        'wave': wave1,              # FIRE wavelength array
        'flux': flux1,              # FIRE flux array
        'error': error1             # FIRE error array
    }
}

# Create vfit_mcmc object with new unified interface
fitter = mc.vfit(
    instrument_data,              # All instruments in one dictionary
    theta, lb, ub,               # Parameters and bounds
    no_of_Chain=20,
    no_of_steps=500,
    perturbation=1e-4,
    sampler='zeus'
)

print("New interface benefits:")
print("  ✓ Symmetric treatment of all instruments")
print("  ✓ No primary/secondary distinction")
print("  ✓ Automatic multi-instrument detection")
print("  ✓ Cleaner, more intuitive setup")

print("Fitter configuration:")
print(f"  Instruments: {', '.join(instrument_data.keys())}")
print("  Shared parameters: N, b, v")
print("  Different instrumental responses: Yes")
print("  Multi-instrument mode: Automatically detected")



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



from rbvfit.core import unified_results as u 

results= u.UnifiedResults(fitter)


results.help()


results.print_summary()           # Overview
results.convergence_diagnostics() # Check zeus     convergence
#results.velocity_plot(velocity_range=(-5200, 5200))

results.residuals_plot()
