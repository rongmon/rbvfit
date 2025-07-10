#!/usr/bin/env python
"""
===============================================================================
rbvfit 2.0 Tutorial: 3-Instrument Multi-System Absorption Line Fitting
===============================================================================

This tutorial demonstrates advanced multi-instrument fitting with rbvfit 2.0,
fitting multiple ion systems across three different instruments simultaneously.

WHAT THIS SCRIPT DOES:
1. Loads spectroscopic data from three instruments (XShooter, FIRE, HIRES)
2. Sets up complex multi-system absorption models (CIV, OI, SiII)
3. Handles different instrumental resolutions correctly
4. Runs joint MCMC fitting with shared physical parameters
5. Demonstrates advanced analysis and visualization capabilities

SCIENTIFIC MOTIVATION:
- Joint fitting of multiple ion systems provides physical insights
- Different instruments probe different wavelength ranges and sensitivities
- Proper instrumental response handling is critical for accurate results
- Advanced diagnostics ensure robust parameter constraints

LEARNING OBJECTIVES:
- Master complex multi-instrument, multi-system configurations
- Understand advanced parameter management across instruments
- Practice sophisticated MCMC diagnostics and analysis
- Learn publication-quality visualization techniques

NEW IN THIS VERSION:
- Uses the new unified interface for all instruments
- Demonstrates automatic instrument detection for 3+ instruments
- Shows enhanced plotting and analysis capabilities
- Includes complete legacy interface preservation for reference
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

# User-specific imports for data loading
from rbcodes.GUIs.rb_spec import load_rb_spec_object

# Global settings
verbose = True  # Print detailed information during execution

# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================

def load_spectrum(slice_name):
    print(f"Loading spectrum: {slice_name}")
    s_HIRES = load_rb_spec_object(filename=slice_name, verbose=verbose)
    
    # Clean and extract the spectrum data
    flux, error, wave = s_HIRES.fnorm, s_HIRES.enorm, s_HIRES.wave_slice
    
    print(f"  Loaded {len(wave)} wavelength points")
    print(f"  Wavelength range: {wave.min():.2f} - {wave.max():.2f} Å")
    
    return wave, flux, error

# Load all three datasets
print("=" * 60)
print("LOADING OBSERVATIONAL DATA")
print("=" * 60)

# XShooter spectrum (medium resolution)
wave, flux, error = load_spectrum('J1030_9089_XShooter_OI1302.json')
print(f"XShooter: {len(wave)} points, {wave.min():.1f}-{wave.max():.1f} Å")

# FIRE spectrum (lower resolution)  
wave1, flux1, error1 = load_spectrum('J1030_9089_FIRE_OI1302.json')
print(f"FIRE: {len(wave1)} points, {wave1.min():.1f}-{wave1.max():.1f} Å")

# HIRES spectrum (highest resolution)  
wave2, flux2, error2 = load_spectrum('J1030_9089_HIRES_OI_air2vac_updated.json')
print(f"HIRES: {len(wave2)} points, {wave2.min():.1f}-{wave2.max():.1f} Å")

# Some tweaking to get everything to CIV redshift. This is rb_spec object specific
z = 6.074762
zabs_CIV = 4.9484

wave_obs = wave * (z + 1.)
wave1_obs = wave1 * (z + 1.)
wave2_obs = wave2 * (z + 1.) 

print("\n✓ Data loading complete")

# ============================================================================
# PART 2: PHYSICAL SYSTEM CONFIGURATION WITH INSTRUMENTAL PARAMETERS
# ============================================================================

print("\n" + "=" * 60)
print("CONFIGURING PHYSICAL SYSTEMS WITH INSTRUMENTAL PARAMETERS")
print("=" * 60)

# NEW APPROACH: Create configurations with FWHM built-in
# This ensures each instrument gets the correct convolution kernel

# Define instrumental resolutions (Full Width at Half Maximum in pixels)
FWHM_XShooter = '2.2'   # Medium resolution
FWHM_FIRE = '4.0'       # Lower resolution  
FWHM_HIRES = '4.285'    # Highest resolution

print(f"Instrumental resolutions:")
print(f"  XShooter FWHM: {FWHM_XShooter} pixels (medium resolution)")
print(f"  FIRE FWHM:     {FWHM_FIRE} pixels (lower resolution)")
print(f"  HIRES FWHM:    {FWHM_HIRES} pixels (highest resolution)")

# Create instrument-specific configurations with FWHM
# Each configuration contains both physics and instrumental setup
config = FitConfiguration()  # XShooter configuration
config.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=4)
#config.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
#config.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)


print("\nPhysical system details:")
print(f"  CIV doublet at z={zabs_CIV:.6f}: 2 velocity components")
print(f"  OI 1302 at z={z:.6f}: 1 velocity component") 
print(f"  SiII 1304 at z={z:.6f}: 1 velocity component")
print(f"  Total: 4 velocity components = 12 parameters (N, b, v each)")

print("\n✓ Physical systems and instrumental setup configured")

# ============================================================================
# PART 3: MODEL CREATION AND MULTI-INSTRUMENT COMPILATION
# ============================================================================

print("\n" + "=" * 60)
print("CREATING MODELS AND COMPILING MULTI-INSTRUMENT SUPPORT")
print("=" * 60)

# Create individual models (FWHM automatically extracted from configurations)
model_xshooter = VoigtModel(config,FWHM=FWHM_XShooter)  # No need to pass FWHM - it's in the config!
model_fire = VoigtModel(config,FWHM=FWHM_FIRE)  
model_hires = VoigtModel(config,FWHM=FWHM_HIRES)


# ============================================================================
# PART 5: PARAMETER ESTIMATION AND BOUNDS
# ============================================================================

print("\n" + "=" * 60)
print("SETTING UP MCMC PARAMETERS")
print("=" * 60)

# Initial parameter guesses based on visual inspection or previous fits
nguess = [13.25, 13.63, 13.12, 13.2]  # log10(column density in cm^-2)
bguess = [23.0, 25., 50., 13.2]        # Doppler parameter in km/s
vguess = [-67., 0., -20., -20.]        # Velocity offset in km/s

print("Initial parameter guesses for 4 components:")
print(f"  N (log column density): {nguess}")
print(f"  b (Doppler parameter):  {bguess} km/s")
print(f"  v (velocity offset):    {vguess} km/s")

# Create theta array for MCMC (concatenated parameter vector)
theta = np.concatenate([nguess, bguess, vguess])
print(f"\nTheta array structure: {len(theta)} parameters")
print("  theta[0:4]  = N values for all components")
print("  theta[4:8]  = b values for all components") 
print("  theta[8:12] = v values for all components")

# Set parameter bounds using rbvfit's bound-setting utility
bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
print(f"\nParameter bounds set for MCMC exploration")

print("\n✓ MCMC parameters configured")

# ============================================================================
# PART 6: JOINT MCMC FITTING - NEW UNIFIED INTERFACE
# ============================================================================

print("\n" + "=" * 60)
print("RUNNING JOINT MCMC FITTING - NEW UNIFIED INTERFACE")
print("=" * 60)

print("Setting up 3-instrument joint fitter with new unified interface...")
print("KEY: Each instrument uses its correct FWHM during evaluation")

# NEW UNIFIED INTERFACE: Define all instrument data outside the call
instrument_data = {
    'XShooter': {
        'model': model_xshooter,    # XShooter model with correct FWHM
        'wave': wave_obs,           # XShooter wavelength array
        'flux': flux,               # XShooter flux array
        'error': error              # XShooter error array
    },
    'FIRE': {
        'model': model_fire,        # FIRE model with correct FWHM
        'wave': wave1_obs,          # FIRE wavelength array
        'flux': flux1,              # FIRE flux array
        'error': error1             # FIRE error array
    },
    'HIRES': {
        'model': model_hires,       # HIRES model with correct FWHM
        'wave': wave2_obs,          # HIRES wavelength array
        'flux': flux2,              # HIRES flux array
        'error': error2             # HIRES error array
    }
}

# Create vfit_mcmc object with new unified interface
fitter = mc.vfit(
    instrument_data,              # All 3 instruments in one dictionary
    theta, lb, ub,               # Parameters and bounds
    no_of_Chain=40,
    no_of_steps=500,
    perturbation=1e-4,
    sampler='emcee'
    # Note: No multi_instrument flag needed - automatically detected!
)

print("\nStarting MCMC sampling...")
print("This may take several minutes depending on data size and convergence")

# Run MCMC with optimization
fitter.runmcmc(optimize=True,use_pool=True)

print("\n✓ MCMC fitting completed")
print("✓ All 3 instruments used their correct FWHM values during fitting")

# ============================================================================
# PART 7: RESULTS ANALYSIS
# ============================================================================


from rbvfit.core import unified_results as u 

results= u.UnifiedResults(fitter)


results.help()


results.print_summary()           # Overview
results.convergence_diagnostics() # Check zeus     convergence
results.velocity_plot(velocity_range=(-5200, 5200))

results.residuals_plot()
