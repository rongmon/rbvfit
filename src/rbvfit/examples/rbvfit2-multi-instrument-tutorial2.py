

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

# Load both datasets
print("=" * 60)
print("LOADING OBSERVATIONAL DATA")
print("=" * 60)

# XShooter spectrum (higher resolution)
wave, flux, error = load_spectrum('J1030_9089_XShooter_OI1302.json')
print(f"XShooter: {len(wave)} points, {wave.min():.1f}-{wave.max():.1f} Å")

# FIRE spectrum (lower resolution)  
wave1, flux1, error1 = load_spectrum('J1030_9089_FIRE_OI1302.json')
print(f"FIRE: {len(wave1)} points, {wave1.min():.1f}-{wave1.max():.1f} Å")


# FIRE spectrum (lower resolution)  
wave2, flux2, error2 = load_spectrum('J1030_9089_HIRES_OI_air2vac_updated.json')
print(f"HIRES: {len(wave2)} points, {wave2.min():.1f}-{wave2.max():.1f} Å")

#some tweaking to get everything to CIV redshift. This is rb_spec object specfic

z=6.074762

zabs_CIV = 4.9484

wave_obs = wave * (z+1.)
wave1_obs = wave1 * (z+1.)
wave2_obs = wave2 * (z+1.) 


print("\n✓ Data loading complete")

# ============================================================================
# PART 2: PHYSICAL SYSTEM CONFIGURATION
# ============================================================================

print("\n" + "=" * 60)
print("CONFIGURING PHYSICAL ABSORPTION SYSTEM")
print("=" * 60)

# Both instruments observe the same physical system (OI 1302 at z=0.0)
# but with different instrumental resolutions




config_A = FitConfiguration()
config_A.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=2)
config_A.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
config_A.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)


config_B = FitConfiguration()
config_B.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=2)
config_B.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
config_B.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)


config_C = FitConfiguration()
config_C.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=2)
config_C.add_system(z=z, ion='OI', transitions=[1302.17], components=1)
config_C.add_system(z=z, ion='SiII', transitions=[1304.5], components=1)


print("\n✓ Physical system configured")

# ============================================================================
# PART 3: INSTRUMENTAL SETUP
# ============================================================================
# This section defines the different instrumental characteristics
# KEY CONCEPT: Different FWHM values account for different spectral resolutions

print("\n" + "=" * 60)
print("SETTING UP INSTRUMENTAL PARAMETERS")
print("=" * 60)

# Define instrumental resolutions (Full Width at Half Maximum in pixels)
FWHM_XShooter = '2.2' # XShooter
FWHM_HIRES = '4.285' # HIRES
FWHM_FIRE = '4.0'

print(f"Instrumental resolutions:")
print(f"  HIRES FWHM:     {FWHM_HIRES} pixels (highest resolution)")
print(f"  XShooter FWHM: {FWHM_XShooter} pixels (medium resolution)")
print(f"  FIRE FWHM:     {FWHM_FIRE} pixels (lower resolution)")

# Create instrument-specific models
# Each model applies different instrumental broadening to the same physics
model_A = VoigtModel(config_A, FWHM=FWHM_XShooter)  
print(f"XShooter model: convolves with {FWHM_XShooter}-pixel Gaussian")

model_B = VoigtModel(config_B, FWHM=FWHM_FIRE)      
print(f"FIRE model: convolves with {FWHM_FIRE}-pixel Gaussian")


model_C = VoigtModel(config_C, FWHM=FWHM_HIRES)      
print(f"FIRE model: convolves with {FWHM_HIRES}-pixel Gaussian")


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
    'XShooter': config_A,  
    'FIRE': config_B,
    'HIRES': config_C       
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
    return compiled.model_flux(theta, wave, instrument='XShooter')

def model_fire(theta, wave):
    return compiled.model_flux(theta, wave, instrument='FIRE')

def model_hires(theta, wave):
    return compiled.model_flux(theta, wave, instrument='HIRES')


print("Model evaluation functions created:")

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
nguess = [13.25, 13.63, 13.12,13.2]  # log10(column density in cm^-2) - typical for OI
bguess = [23.0,25.,50.,13.2]  # Doppler parameter in km/s - thermal + turbulent broadening
vguess = [-67.,0.,-20.,-20.] # Velocity offset in km/s - relative to systemic redshift

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
    wave_obs, flux, error,        # Primary dataset (XShooter data)
    no_of_Chain=50,
    no_of_steps=500,
    perturbation=1e-6,
    multi_instrument=True,    # Enable multi-instrument mode
    instrument_data={         # Additional instruments
        'FIRE': {
            'model': model_fire,  # FIRE model function
            'wave': wave1_obs,        # FIRE wavelength array
            'flux': flux1,        # FIRE flux array
            'error': error1       # FIRE error array
        },
        'HIRES': {
            'model': model_hires,  # FIRE model function
            'wave': wave2_obs,        # FIRE wavelength array
            'flux': flux2,        # FIRE flux array
            'error': error2       # FIRE error array
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
fig = mc.plot_model(model_A, fitter, 
                outfile=False,           # or 'output.png' to save
                show_residuals=False,     # Include residual plots
                velocity_marks=True,     # Mark component velocities
                verbose=True)            # Print parameter summary
