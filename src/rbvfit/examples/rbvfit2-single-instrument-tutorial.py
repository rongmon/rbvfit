

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


#some tweaking to get everything to CIV redshift. This is rb_spec object specfic

z=6.074762
zabs_CIV = 4.9484
wave_obs = wave * (1.+z)

q=(wave_obs>9198)*(wave_obs<9235)

wave_obs=wave_obs[q]
flux=flux[q]
error=error[q]


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
config_A.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=3)


print("\n✓ Physical system configured")

# ============================================================================
# PART 3: INSTRUMENTAL SETUP
# ============================================================================

print("\n" + "=" * 60)
print("SETTING UP INSTRUMENTAL PARAMETERS")
print("=" * 60)

# Define instrumental resolutions (Full Width at Half Maximum in pixels)
FWHM_XShooter = '2.2' # XShooter


# Create instrument-specific models
model_A = VoigtModel(config_A, FWHM=FWHM_XShooter)  



print("\n✓ Instrumental models created")

# ============================================================================
# PART 4: MODEL COMPILATION
# ============================================================================

print("\n" + "=" * 60)
print("COMPILING SINGLE-INSTRUMENT MODEL")
print("=" * 60)


compiled = model_A.compile(verbose=True)
print("\n✓ Single-instrument model compiled")



# ============================================================================
# PART 5: PARAMETER ESTIMATION AND BOUNDS
# ============================================================================
# Set up initial guesses and parameter bounds for MCMC
# KEY CONCEPT: Physical reasoning guides parameter ranges

print("\n" + "=" * 60)
print("SETTING UP MCMC PARAMETERS")
print("=" * 60)

# Initial parameter guesses based on visual inspection or previous fits
nguess = [13.15, 13.58, 13.5]  # log10(column density in cm^-2) 
bguess = [23.0,25.,30.]  # Doppler parameter in km/s - thermal + turbulent broadening
vguess = [-67.,0.,10.] # Velocity offset in km/s - relative to systemic redshift

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
# PART 6: MCMC FITTING
# ============================================================================

print("\n" + "=" * 60)
print("RUNNING JOINT MCMC FITTING")
print("=" * 60)

print("Setting up mcmc fitter...")

# Create vfit_mcmc object with multi-instrument support
fitter = mc.vfit(
    compiled.model_flux,           # Primary model function (XShooter)
    theta, lb, ub,            # Parameters and bounds
    wave_obs, flux, error,        # Primary dataset (XShooter data)
    no_of_Chain=50,
    no_of_steps=500,
    perturbation=1e-4,
    sampler='zeus'
    )



print("\nStarting MCMC sampling...")
print("This may take several minutes depending on data size and convergence")

# Run MCMC with optimization
fitter.runmcmc(optimize=True)  # optimize=True finds better starting point

print("\n✓ MCMC fitting completed")

# ============================================================================
# PART 7: RESULTS ANALYSIS
# ============================================================================
# Display and analyze the fitting results
# KEY CONCEPT: Joint constraints from both datasets

print("\n" + "=" * 60)
print("ANALYZING FITTING RESULTS")
print("=" * 60)

# Display corner plot (parameter correlations and posteriors)
#print("Generating corner plot (parameter posterior distributions)...")
#fitter.plot_corner()


# Extract key results
#print("\nExtracting results...")
#fig = mc.plot_model(model_A, fitter, 
#                outfile=False,           # or 'output.png' to save
#                show_residuals=True,     # Include residual plots
#                velocity_marks=True,     # Mark component velocities
#                verbose=True)            # Print parameter summary


from rbvfit.core import fit_results as f
# Save results
results = f.FitResults(fitter, model_A)
#results.save('my_fit.h5')

# Load and analyze
#results = f.FitResults.load('my_fit.h5')
results.print_fit_summary()
print("Generating corner plot (parameter posterior distributions)...")
results.corner_plot()#save_path='corner.pdf')
results.convergence_diagnostics()

# Visual chain inspection
results.chain_trace_plot()#save_path='trace_plots.pdf')

# This is the main new feature - velocity space plots by ion!
velocity_plots = results.plot_velocity_fits(
    show_components=True,      # Show individual components
    show_rail_system=True     # Show component position markers
)

# For single ion systems, also try velocity range control:
results.plot_velocity_fits(velocity_range=(-600, 600))
