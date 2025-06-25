import numpy as np
import sys
import os

# rbvfit 2.0 imports - new architecture for multi-group fitting
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.parameter_manager import ParameterManager
from rbvfit import vfit_mcmc as mc  # MCMC fitting engine

from rbvfit import guess_profile_parameters_interactive as g # interactive parameter guesser

import matplotlib.pyplot as plt

# User-specific imports for data loading
from rbcodes.GUIs.rb_spec import load_rb_spec_object

# Global settings
verbose=True

# Set up matplotlib for interactive use - CRITICAL FIX
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for proper interactivity
plt.ion()  # Turn on interactive mode

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
# PART 2: PARAMETER ESTIMATION (INTERACTIVE GUESSING)
# ============================================================================
# Interactive parameter guessing - this creates initial estimates for MCMC
# Note: This MUST be done before setting up physical system configurations

print("\n" + "=" * 60)
print("INTERACTIVE PARAMETER ESTIMATION")
print("=" * 60)

print("Setting up interactive GUI...")
print("INSTRUCTIONS:")
print("1. A spectrum plot will appear")
print("2. Click on absorption features to mark velocity components")
print("3. Close the plot window when done")
print("4. You'll be prompted to enter b-parameters")

# Set up interactive GUI for component identification and parameter guessing
try:
    tab = g.gui_set_clump(wave_obs, flux, error, zabs_CIV, wrest=1548.5, xlim=[-600,600])
    
    # CRITICAL FIX: Force the plot to show and wait for user interaction
    plt.draw()
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    print("\n>>> CLICK ON ABSORPTION FEATURES IN THE PLOT <<<")
    print(">>> Close the plot window when finished clicking <<<")
    
    # Wait for the user to interact with the plot
    input("Press Enter after you've clicked on features and closed the plot window...")
    
    # Interactive input of parameters - GUI will appear for user input
    print("\nNow you'll be prompted to enter b-parameters...")
    tab.input_b_guess()
    
except Exception as e:
    print(f"Error in interactive session: {e}")
    print("Using default parameters instead...")
    # Fallback parameters if interactive fails
    tab = type('obj', (object,), {
        'nguess': np.array([13.5]),
        'bguess': np.array([20.0]), 
        'vguess': np.array([0.0])
    })()

# Extract parameter guesses from interactive session
nguess = tab.nguess  # log10(column density in cm^-2) 
bguess = tab.bguess  # Doppler parameter in km/s - thermal + turbulent broadening
vguess = tab.vguess  # Velocity offset in km/s - relative to systemic redshift

print("Initial parameter guesses from interactive session:")
for i in range(len(nguess)):
    print(f"  Component {i+1}:")
    print(f"    N (log column density): {nguess[i]:.1f} [log cm^-2]")
    print(f"    b (Doppler parameter):  {bguess[i]:.1f} km/s")
    print(f"    v (velocity offset):    {vguess[i]:.1f} km/s")

# Create theta array for MCMC (concatenated parameter vector)
theta = np.concatenate([nguess, bguess, vguess])
print(f"\nTheta array structure: {theta}")
print(f"  theta[0:{len(nguess)}] = N values")
print(f"  theta[{len(nguess)}:{len(nguess)+len(bguess)}] = b values") 
print(f"  theta[{len(nguess)+len(bguess)}:] = v values")

# Set parameter bounds using rbvfit's bound-setting utility
bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
print(f"\nParameter bounds:")
for i in range(len(nguess)):
    print(f"  Component {i+1}:")
    print(f"    N: [{lb[i]:.1f}, {ub[i]:.1f}] [log cm^-2]")
    print(f"    b: [{lb[i+len(nguess)]:.1f}, {ub[i+len(nguess)]:.1f}] km/s")
    print(f"    v: [{lb[i+len(nguess)+len(bguess)]:.1f}, {ub[i+len(nguess)+len(bguess)]:.1f}] km/s")

print("\n✓ Parameter estimation complete")

# ============================================================================
# PART 3: PHYSICAL SYSTEM CONFIGURATION
# ============================================================================

print("\n" + "=" * 60)
print("CONFIGURING PHYSICAL ABSORPTION SYSTEM")
print("=" * 60)

# Configure the physical absorption system using the number of components
# determined from the interactive parameter guessing session
config = FitConfiguration()
config.add_system(z=zabs_CIV, ion='CIV', transitions=[1548.2,1550.3], components=len(nguess))

print(f"Physical system configured:")
print(f"  Ion: CIV")
print(f"  Redshift: z = {zabs_CIV:.4f}")
print(f"  Transitions: [1548.2, 1550.3] Å")
print(f"  Components: {len(nguess)}")

print("\n✓ Physical system configured")

# ============================================================================
# PART 4: INSTRUMENTAL SETUP
# ============================================================================

print("\n" + "=" * 60)
print("SETTING UP INSTRUMENTAL PARAMETERS")
print("=" * 60)

# Define instrumental resolutions (Full Width at Half Maximum in pixels)
FWHM_XShooter = '2.2' # XShooter

print(f"Instrumental resolution: FWHM = {FWHM_XShooter} pixels")

# Create instrument-specific models
model = VoigtModel(config, FWHM=FWHM_XShooter)  

print("\n✓ Instrumental models created")

# ============================================================================
# PART 5: MODEL COMPILATION
# ============================================================================

print("\n" + "=" * 60)
print("COMPILING SINGLE-INSTRUMENT MODEL")
print("=" * 60)

compiled = model.compile(verbose=True)
print("\n✓ Single-instrument model compiled")

# ============================================================================
# PART 6: MCMC FITTING
# ============================================================================

print("\n" + "=" * 60)
print("RUNNING MCMC FITTING")
print("=" * 60)

print("Setting up MCMC fitter...")

# Create vfit_mcmc object 
fitter = mc.vfit(
    compiled.model_flux,           # Model function
    theta, lb, ub,                 # Parameters and bounds
    wave_obs, flux, error,         # Dataset
    no_of_Chain=50,               # Number of walkers
    no_of_steps=500,              # Number of steps
    perturbation=1e-4,            # Walker initialization perturbation
    sampler='zeus'                # Sampler choice
)

print("\nStarting MCMC sampling...")
print("This may take several minutes depending on data size and convergence")

# Run MCMC with optimization
try:
    fitter.runmcmc(optimize=True)  # optimize=True finds better starting point
    print("\n✓ MCMC fitting completed successfully")
except Exception as e:
    print(f"\n✗ MCMC fitting failed: {e}")
    print("You may need to adjust initial parameters or bounds")
    sys.exit(1)

# ============================================================================
# PART 7: RESULTS ANALYSIS
# ============================================================================
# Display and analyze the fitting results

print("\n" + "=" * 60)
print("ANALYZING FITTING RESULTS")
print("=" * 60)

try:
    from rbvfit.core import fit_results as f
    
    # Create results object
    results = f.FitResults(fitter, model)
    
    # Print summary
    print("Fit summary:")
    results.print_fit_summary()
    
    # Generate plots with proper display handling
    print("\nGenerating diagnostic plots...")
    
    # Corner plot
    print("1. Corner plot (parameter posterior distributions)...")
    fig_corner = results.corner_plot()
    if fig_corner:
        plt.figure(fig_corner.number)
        plt.draw()
        plt.pause(0.1)
    
    # Convergence diagnostics
    print("2. Convergence diagnostics...")
    conv_results = results.convergence_diagnostics()
    
    # Chain trace plot
    print("3. Chain trace plots...")
    fig_trace = results.chain_trace_plot()
    if fig_trace:
        plt.figure(fig_trace.number)
        plt.draw()
        plt.pause(0.1)
    
    # Velocity space plots
    print("4. Velocity space fit plots...")
    velocity_plots = results.plot_velocity_fits(
        show_components=True,      # Show individual components
        show_rail_system=True     # Show component position markers
    )
    
    if velocity_plots:
        plt.figure(velocity_plots.number)
        plt.draw()
        plt.pause(0.1)
    
    print("\n✓ All plots generated successfully")
    
    # Optional: Save results
    save_results = input("\nSave results to file? (y/n): ").lower().strip()
    if save_results in ['y', 'yes']:
        filename = input("Enter filename (or press Enter for 'fit_results.h5'): ").strip()
        if not filename:
            filename = 'fit_results.h5'
        results.save(filename)
        print(f"Results saved to {filename}")

except Exception as e:
    print(f"Error in results analysis: {e}")
    print("Basic fit completed, but results analysis failed")

# ============================================================================
# FINAL DISPLAY
# ============================================================================

print("\n" + "=" * 60)
print("FITTING COMPLETE")
print("=" * 60)
print("All plots should now be displayed.")
print("Close plot windows manually when done examining results.")

# Keep all plots open - THIS IS THE KEY FIX
print("\nKeeping script alive to maintain plot windows...")
print("Press Ctrl+C to exit when done examining plots.")

try:
    # Keep the script running to maintain plot windows
    while True:
        plt.pause(1.0)  # Check every second
except KeyboardInterrupt:
    print("\nExiting...")
    plt.close('all')  # Clean up all figures
    sys.exit(0)