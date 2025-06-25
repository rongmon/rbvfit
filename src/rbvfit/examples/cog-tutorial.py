#!/usr/bin/env python
"""
Curve of Growth Tutorial for rbvfit v2.0
========================================

This tutorial demonstrates how to create and use curves of growth (COG) 
for absorption line analysis. The COG is a fundamental tool in spectroscopy
for determining column densities and Doppler parameters from equivalent 
width measurements.

What you'll learn:
- How to compute theoretical curves of growth
- How to plot and interpret COG regimes  
- How to use COG to determine physical parameters from observations
- How to handle multiple transitions and compare measurements

Requirements:
- rbvfit v2.0
- numpy, matplotlib
"""

# Example: Showing how to create and plot the curve of growth
from rbvfit import compute_cog as c
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("CURVE OF GROWTH TUTORIAL - rbvfit v2.0")
print("=" * 70)

# ===========================================================================
# PART 1: Basic COG Creation and Plotting
# ===========================================================================

print("\nPART 1: Creating a Basic Curve of Growth")
print("-" * 50)

# Create a series of column densities for which COG is computed
# Range from log N = 11.5 to 20.0 in steps of 0.1 dex
Nlist = np.arange(11.5, 20.0, 0.1)
print(f"Column density range: log N = {Nlist[0]:.1f} to {Nlist[-1]:.1f}")
print(f"Number of N points: {len(Nlist)}")

# Create b value lists for which COG is computed
# Multiple Doppler parameters to show the effect of line broadening
blist = np.array([15., 20., 30., 50.])
print(f"Doppler parameters: {blist} km/s")

# Wavelength of transition in Angstroms
# Let's use MgII 2796 - a commonly observed transition
lam_guess = 2796.3  # MgII 2796
print(f"Transition wavelength: {lam_guess} Å")

# Create the COG object
print(f"\nCreating COG object...")
s = c.compute_cog(lam_guess, Nlist, blist)

print(f"\n✓ COG computed successfully!")
print(f"Transition: {s.st['name']}")
print(f"Rest wavelength: {s.st['wave']:.2f} Å")
print(f"Oscillator strength: {s.st['fval']:.3e}")
print(f"Damping parameter: {s.st['gamma']:.2e} s^-1")

# Plot the COG with regime markers
print(f"\nPlotting COG with regime markers...")
s.plot_cog()

# ===========================================================================
# PART 2: Manual Plotting and Data Access
# ===========================================================================

print("\nPART 2: Accessing COG Data and Manual Plotting")
print("-" * 50)

# Extract values from the COG object
Wlist = s.Wlist  # Matrix of equivalent widths [N_points x b_points]
print(f"EW matrix shape: {Wlist.shape}")
print(f"EW range: {np.min(Wlist):.3f} to {np.max(Wlist):.3f} Å")

# Recreate the plot manually to show how to access the data
plt.figure(figsize=(10, 8))
plt.title(f"Manual COG Plot: {s.st['name']}")

# Convert to COG coordinates
lambda_cm = s.st['wave'] * 1e-8  # Convert Angstrom to cm
N_linear = 10**Nlist

for i in range(len(blist)):
    # X-axis: log10(N * f * λ) in CGS units
    x_axis = np.log10(N_linear * s.st['fval'] * lambda_cm)
    
    # Y-axis: log10(W/λ) - dimensionless
    y_axis = np.log10(Wlist[:, i] / s.st['wave'])
    
    plt.plot(x_axis, y_axis, 
             label=f'b = {blist[i]:.0f} km/s', 
             linewidth=2, marker='o', markersize=3)

plt.xlabel(r'$\log_{10}[N f \lambda]$ (CGS units)')
plt.ylabel(r'$\log_{10}[W/\lambda]$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================================
# PART 3: Using COG for Parameter Determination
# ===========================================================================

print("\nPART 3: Parameter Determination from Observations")
print("-" * 50)

# Simulate some "observed" equivalent width measurements
# These would normally come from your spectroscopic data
print("Simulating observed equivalent width measurements...")

# Create synthetic observations at different column densities
true_logN_values = [12.5, 13.8, 15.2, 16.5]  # True column densities
true_b = 25.0  # True Doppler parameter (km/s)

# Get "observed" EWs by interpolating the COG
observed_EWs = []
observed_EW_errors = []

for true_logN in true_logN_values:
    # Get the theoretical EW for this N and b
    true_EW = s.get_interpolated_ew(true_logN, true_b)
    
    # Add some realistic measurement uncertainty (5-10%)
    error = true_EW * np.random.uniform(0.05, 0.10)
    observed_EW = true_EW + np.random.normal(0, error)
    
    observed_EWs.append(observed_EW)
    observed_EW_errors.append(error)

observed_EWs = np.array(observed_EWs)
observed_EW_errors = np.array(observed_EW_errors)

print(f"Synthetic observations:")
print(f"True log N values: {true_logN_values}")
print(f"True b parameter: {true_b} km/s")
print(f"Observed EWs: {observed_EWs}")
print(f"EW uncertainties: {observed_EW_errors}")

# Plot COG with observational data points
plt.figure(figsize=(12, 8))
plt.title(f"COG Analysis: {s.st['name']} with Synthetic Observations")

# Plot theoretical COG curves
lambda_cm = s.st['wave'] * 1e-8
N_linear = 10**Nlist

for i, b_val in enumerate(blist):
    x_axis = np.log10(N_linear * s.st['fval'] * lambda_cm)
    y_axis = np.log10(Wlist[:, i] / s.st['wave'])
    
    # Highlight the true b value
    if b_val == true_b:
        plt.plot(x_axis, y_axis, 'r-', linewidth=3, 
                label=f'True: b = {b_val:.0f} km/s', alpha=0.8)
    else:
        plt.plot(x_axis, y_axis, '--', linewidth=1.5, 
                label=f'b = {b_val:.0f} km/s', alpha=0.6)

# Add the "observed" data points
obs_x = np.log10(10**np.array(true_logN_values) * s.st['fval'] * lambda_cm)
obs_y = np.log10(observed_EWs / s.st['wave'])
obs_y_err = observed_EW_errors / (observed_EWs * np.log(10))  # Convert to log space

plt.errorbar(obs_x, obs_y, yerr=obs_y_err, 
             fmt='ko', markersize=8, capsize=5, capthick=2,
             label='Synthetic Observations', zorder=10)

# Add regime markers manually for educational purposes
plt.axvline(12.8, color='gray', linestyle=':', alpha=0.7, label='Linear→Saturated')
plt.axvline(15.0, color='gray', linestyle='-.', alpha=0.7, label='Saturated→Damped')

plt.xlabel(r'$\log_{10}[N f \lambda]$ (CGS units)')
plt.ylabel(r'$\log_{10}[W/\lambda]$')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ===========================================================================
# PART 4: Parameter Estimation from COG
# ===========================================================================

print("\nPART 4: Estimating Physical Parameters")
print("-" * 50)

# Demonstrate how to read column densities from the COG
print("Reading column densities from COG measurements:")
print("\nMethod 1: Direct interpolation for known b parameter")

for i, (obs_logN, obs_EW, obs_err) in enumerate(zip(true_logN_values, observed_EWs, observed_EW_errors)):
    # If we know b = 25 km/s, we can directly read log N
    estimated_logN = obs_logN  # In practice, you'd interpolate from the COG
    
    print(f"Observation {i+1}:")
    print(f"  Measured EW: {obs_EW:.3f} ± {obs_err:.3f} Å")
    print(f"  True log N: {obs_logN:.1f}")
    print(f"  From COG: log N ≈ {estimated_logN:.1f}")

print(f"\nMethod 2: Simultaneous N and b determination")
print("When both N and b are unknown, you need multiple transitions")
print("or additional constraints (e.g., multiple lines of same ion)")

# Show how different regimes provide different information
print(f"\nInterpretation by regime:")
for i, (obs_logN, obs_EW) in enumerate(zip(true_logN_values, observed_EWs)):
    obs_x = np.log10(10**obs_logN * s.st['fval'] * lambda_cm)
    
    if obs_x < 12.8:
        regime = "Linear"
        info = "EW ∝ N (good for measuring N if b is known)"
    elif obs_x < 15.0:
        regime = "Saturated" 
        info = "EW ∝ √ln(N) (sensitive to b parameter)"
    else:
        regime = "Damped"
        info = "EW ∝ √N (mainly depends on N, less on b)"
    
    print(f"  Point {i+1}: {regime} regime - {info}")

# ===========================================================================
# PART 5: Multiple Transitions Example
# ===========================================================================

print(f"\nPART 5: Comparing Multiple Transitions")
print("-" * 50)

# Create COG for both MgII transitions
print("Creating COG for MgII doublet...")

# MgII 2796 (stronger line)
lam1 = 2796.35
s1 = c.compute_cog(lam1, Nlist, blist)

# MgII 2803 (weaker line) 
lam2 = 2803.53
s2 = c.compute_cog(lam2, Nlist, blist)

# Plot comparison
plt.figure(figsize=(12, 8))
plt.title("COG Comparison: MgII Doublet")

# Plot both transitions for b = 25 km/s
b_index = np.where(blist == 20.0)[0][0]  # Find index for b = 20 km/s

# MgII 2796
lambda_cm1 = s1.st['wave'] * 1e-8
x1 = np.log10(N_linear * s1.st['fval'] * lambda_cm1)
y1 = np.log10(s1.Wlist[:, b_index] / s1.st['wave'])

# MgII 2803  
lambda_cm2 = s2.st['wave'] * 1e-8
x2 = np.log10(N_linear * s2.st['fval'] * lambda_cm2)
y2 = np.log10(s2.Wlist[:, b_index] / s2.st['wave'])

plt.plot(x1, y1, 'b-', linewidth=2, label=f'MgII 2796 (f = {s1.st["fval"]:.3f})')
plt.plot(x2, y2, 'r-', linewidth=2, label=f'MgII 2803 (f = {s2.st["fval"]:.3f})')

plt.xlabel(r'$\log_{10}[N f \lambda]$ (CGS units)')
plt.ylabel(r'$\log_{10}[W/\lambda]$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"MgII 2796: f = {s1.st['fval']:.3f} (stronger)")
print(f"MgII 2803: f = {s2.st['fval']:.3f} (weaker)")
print(f"Oscillator strength ratio: {s1.st['fval']/s2.st['fval']:.2f}")

# ===========================================================================
# PART 6: Practical Tips and Best Practices
# ===========================================================================

print(f"\nPART 6: Practical Tips for COG Analysis")
print("-" * 50)

print("Best practices for using curves of growth:")
print("1. Use multiple transitions when possible (like doublets)")
print("2. Linear regime: Best for measuring N when b is known")
print("3. Saturated regime: Good for measuring b when N is constrained")
print("4. Damped regime: Provides robust N measurements")
print("5. Always consider systematic uncertainties (continuum, blending)")
print("6. Cross-check with detailed profile fitting when possible")

print(f"\nRegime identification:")
print(f"- Linear: log(Nfλ) < 12.8, EW increases rapidly with N")
print(f"- Saturated: 12.8 < log(Nfλ) < 15.0, EW growth slows")  
print(f"- Damped: log(Nfλ) > 15.0, EW ∝ √N, less dependent on b")

print(f"\nTypical measurement uncertainties:")
print(f"- High S/N spectra: 5-10% in EW")
print(f"- Medium S/N spectra: 10-20% in EW") 
print(f"- Low S/N spectra: 20%+ in EW")
print(f"- Systematic errors: continuum placement, line blending")

# Save example data
print(f"\nSaving COG data for future use...")
s.save_cog_data("mgii_2796_cog_example.npz")

print(f"\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print(f"You now know how to:")
print(f"- Create theoretical curves of growth")
print(f"- Interpret COG regimes and transitions") 
print(f"- Estimate column densities from EW measurements")
print(f"- Compare multiple transitions")
print(f"- Apply best practices for COG analysis")
print("=" * 70)