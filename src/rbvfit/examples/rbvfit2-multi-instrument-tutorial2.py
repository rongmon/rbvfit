

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
from rbvfit.core import fit_results as f

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
#                show_residuals=False,     # Include residual plots
#                velocity_marks=True,     # Mark component velocities
#                verbose=True)            # Print parameter summary

print("Creating FitResults object from 3-instrument joint fitter...")
print("Physical model:")
print(f"  CIV doublet at z={zabs_CIV:.6f}: 2 components")
print(f"  OI 1302 at z={z:.6f}: 1 component") 
print(f"  SiII 1304 at z={z:.6f}: 1 component")
print(f"Instruments: XShooter (FWHM={FWHM_XShooter}), FIRE (FWHM={FWHM_FIRE}), HIRES (FWHM={FWHM_HIRES})")

# Create FitResults object from your multi-instrument fitter
results = f.FitResults(fitter, model_A)  # Use model_A as the base (contains config)

print(f"✓ Results created: {results}")
print(f"  Multi-instrument: {results.is_multi_instrument}")
print(f"  Instruments: {list(results.instrument_data.keys()) if results.instrument_data else 'None'}")

# =============================================================================
# STEP 2: SAVE AND LOAD RESULTS (RECOMMENDED FOR LONG FITS)
# =============================================================================

# Save comprehensive results including all MCMC chains and configurations
#print("\nSaving complete 3-instrument fit results...")
#results.save('civ_oi_siii_3instrument_joint_fit.h5')

# Later, you can reload everything for analysis
#print("Reloading results for analysis...")
#results = f.FitResults.load('civ_oi_siii_3instrument_joint_fit.h5')

# =============================================================================
# STEP 3: COMPREHENSIVE FIT SUMMARY
# =============================================================================

print("\n" + "="*80)
print("3-INSTRUMENT JOINT FIT ANALYSIS")
print("="*80)

# Overall fit summary with multi-instrument info
results.print_fit_summary()

# This will show:
# - Model: rbvfit 2.0 VoigtModel  
# - Sampler: emcee (or zeus)
# - Multi-instrument fit: 3 datasets
# - Physical model: 2 systems, 3 ion groups, 4 components total
# - Combined χ² and χ²/ν from all instruments
# - Convergence status

# =============================================================================
# STEP 4: CONVERGENCE DIAGNOSTICS (CRITICAL FOR MCMC VALIDATION)
# =============================================================================

print("\nChecking MCMC convergence for joint fit...")
convergence = results.convergence_diagnostics()

# The diagnostics will show:
# ✅ Good acceptance fraction (0.2-0.7)
# ✅ Adequate chain length (>50x autocorr time)  
# ✅ Good effective sample size (>100 per parameter)
# ✅ Overall status: GOOD/MARGINAL/POOR
# 📈 Recommendations for improvement if needed

print(f"Convergence status: {convergence['overall_status']}")

# =============================================================================
# STEP 5: PARAMETER SUMMARY WITH ION ORGANIZATION
# =============================================================================

print("\nDetailed parameter summary...")
param_summary = results.parameter_summary()

# This organizes parameters by:
# System 1 (z = zabs_CIV):
#   CIV Component 1: logN, b, v ± errors
#   CIV Component 2: logN, b, v ± errors  
# System 2 (z = z):
#   OI Component 1: logN, b, v ± errors
#   SiII Component 1: logN, b, v ± errors

# =============================================================================
# STEP 6: VISUAL CONVERGENCE ASSESSMENT  
# =============================================================================

print("\nGenerating chain trace plots for visual inspection...")
trace_fig = results.chain_trace_plot()#save_path='3instrument_trace_plots.pdf')

# Critical for validating MCMC:
# ✅ Good traces: stable, well-mixed, no trends
# ❌ Bad traces: trending, stuck walkers, poor mixing
# Shows convergence status on each parameter panel

# =============================================================================
# STEP 7: PARAMETER CORRELATIONS AND CORNER PLOT
# =============================================================================

print("\nAnalyzing parameter correlations...")
correlation_matrix = results.correlation_matrix()#plot=True, save_path='3instrument_correlations.pdf')

print("Generating corner plot (parameter posterior distributions)...")
corner_fig = results.corner_plot()#save_path='3instrument_corner.pdf')

# Corner plot shows:
# - Posterior distributions for all parameters
# - Parameter correlations (especially for tied CIV doublet)
# - Convergence quality assessment
# - Best-fit values marked

# =============================================================================
# STEP 8: ION-SPECIFIC VELOCITY SPACE ANALYSIS (NEW FEATURE!)
# =============================================================================

print("\n" + "="*60)
print("ION-SPECIFIC VELOCITY SPACE VISUALIZATION")
print("="*60)

# This is the main new feature - separate plots for each ion!
print("Creating velocity space plots organized by ion...")

velocity_plots = results.plot_velocity_fits(
    show_components=True,       # Show individual velocity components
    show_rail_system=True,      # Show component position markers  
    velocity_range=(-400, 400), # Velocity range for all plots
    #save_path='velocity_plots'  # Will create separate files per ion
)

print("Velocity plots created:")
for ion_key, fig in velocity_plots.items():
    print(f"  {ion_key}: Shows all 3 instruments × transitions for this ion")

# The velocity plots show:
# - Layout: transitions (rows) × instruments (columns)
# - CIV plot: 2 transitions × 3 instruments = 6 panels
#   - Shows doublet parameter tying across all instruments
#   - Rail system shows 2 velocity components
# - OI plot: 1 transition × 3 instruments = 3 panels  
#   - Shows resolution differences between instruments
#   - Single component marked on rail
# - SiII plot: 1 transition × 3 instruments = 3 panels
#   - Single component, resolution comparison

# =============================================================================
# STEP 9: GOODNESS OF FIT ASSESSMENT
# =============================================================================

print("\nAssessing fit quality across all instruments...")
chi2_stats = results.chi_squared()

print("Chi-squared breakdown:")
print(f"  XShooter χ² = {chi2_stats.get('chi2', 'N/A'):.2f}")
print(f"  FIRE χ² = {chi2_stats.get('chi2_FIRE', 'N/A'):.2f}")  
print(f"  HIRES χ² = {chi2_stats.get('chi2_HIRES', 'N/A'):.2f}")
print(f"  Combined χ² = {chi2_stats.get('chi2_total', 'N/A'):.2f}")
print(f"  Combined χ²/ν = {chi2_stats.get('reduced_chi2_total', 'N/A'):.3f}")

# Good fit indicators:
# ✅ χ²/ν ≈ 1.0 (not much larger or smaller)
# ✅ Similar χ²/ν across instruments (consistency)
# ✅ No systematic residuals in any instrument

# =============================================================================
# STEP 10: EXPORT RESULTS FOR PUBLICATION
# =============================================================================

#print("\nExporting results for publication...")

# CSV export with all parameters and uncertainties
#results.export_csv('3instrument_joint_fit_parameters.csv', include_errors=True)

# LaTeX table for publication
#results.export_latex('3instrument_fit_table.tex', 
#                    table_format='publication',
#                    caption="Joint 3-instrument absorption line fit results for CIV, OI, and SiII systems",
#                    label="tab:3instrument_absorption")

# Comprehensive summary report
#results.export_summary_report('3instrument_fit_report.txt', include_plots=True)

#print("✅ All exports completed!")

# =============================================================================
# STEP 11: SCIENTIFIC INTERPRETATION SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SCIENTIFIC RESULTS SUMMARY") 
print("="*80)

print("Joint 3-instrument fitting advantages:")
print("✅ Better parameter constraints from combined datasets")
print("✅ Validation across different spectral resolutions")
print("✅ Consistent physical parameters despite different LSFs")
print("✅ Robust uncertainty estimates including systematic errors")

print(f"\nKey results:")
if param_summary:
    # Extract some key results for interpretation
    print("CIV system (high-ionization):")
    print("  Component 1: Column density, Doppler parameter, velocity")
    print("  Component 2: Column density, Doppler parameter, velocity")
    print("OI system (neutral gas):")
    print("  Single component: Column density, Doppler parameter, velocity") 
    print("SiII system (low-ionization):")
    print("  Single component: Column density, Doppler parameter, velocity")

print(f"\nQuality assessment:")
print(f"  Convergence: {convergence['overall_status']}")
print(f"  Fit quality: χ²/ν = {chi2_stats.get('reduced_chi2_total', 'N/A')}")
print(f"  Parameter correlations: Available in corner plot")

print("\n🎉 Multi-instrument analysis complete!")
print("📊 Check velocity plots for ion-specific behavior")
print("📈 Review trace plots if convergence was not GOOD")
print("📄 Use exported tables/figures for your publication")

# =============================================================================
# ADVANCED ANALYSIS OPTIONS
# =============================================================================

print("\n" + "="*60)
print("ADDITIONAL ANALYSIS OPTIONS")
print("="*60)

# Custom velocity range for specific ions
print("Example: Focused analysis of CIV system...")
civ_plots = results.plot_velocity_fits(
    velocity_range=(-600, 600),  # Wider range for high-velocity CIV
    show_components=True,
    show_rail_system=True
)

# Correlation analysis for tied parameters
print("Analyzing parameter correlations...")
corr_matrix = results.correlation_matrix(plot=False)

# Look for strong correlations (>0.5) between tied parameters
# This validates that ion tying is working correctly

print("Analysis complete! 🚀")
print("Happy multi-instrument fitting! 🎉")