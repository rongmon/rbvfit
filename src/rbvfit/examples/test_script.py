#!/usr/bin/env python
"""
Automatically generated script to reproduce rbvfit 2.0 MCMC analysis.

Generated from UnifiedResults containing:
- 1 instrument(s): COS
- 6 parameters
- zeus sampler, 20 walkers, 500 steps

This script reproduces the complete fitting workflow.
"""

import numpy as np
import rbvfit.vfit_mcmc as mc
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.unified_results import UnifiedResults

def main():
    """Main fitting workflow"""
    print("Reproducing rbvfit 2.0 analysis...")

    # ================================================================
    # DATA SETUP
    # ================================================================

    # Load observational data from files
    cos_data = np.load("cos_data.npz")
    wave_cos = cos_data["wave"]
    flux_cos = cos_data["flux"]
    error_cos = cos_data["error"]

    # ================================================================
    # MODEL CONFIGURATION
    # ================================================================

    # Create fit configuration
    config = FitConfiguration()

    config.add_system(z=0.000000, ion="SiII", transitions=[1190.416, 1193.290], components=1)
    config.add_system(z=0.162005, ion="HI", transitions=[1025.722], components=1)

    # Create VoigtModel objects for each instrument
    models = {}

    models["COS"] = VoigtModel(config, FWHM=2.394991274145626)
    models["COS"].compile()

    # Create instrument data dictionary for vfit
    instrument_data = {
        "COS": {"model": models["COS"], "wave":     wave_cos, "flux": flux_cos, "error":     error_cos}
    }

    # ================================================================
    # PARAMETERS AND BOUNDS
    # ================================================================

    # Initial parameter guess (from fitted results)
    theta = np.array([ 14.4228,  14.5105,  46.4066,  46.7789, -28.2628,  -6.743 ])

    # Parameter bounds
    lb = np.array([  10.,   10.,    1.,    1., -500., -500.])
    ub = np.array([ 20.,  20., 100., 100., 500., 500.])

    # ================================================================
    # MCMC FITTING
    # ================================================================

    print(f"Starting MCMC fitting with {len(instrument_data)} instrument(s)...")

    # Create vfit object
    fitter = mc.vfit(
        instrument_data,
        theta, lb, ub,
        no_of_Chain=20,
        no_of_steps=500,
        sampler="zeus",
        perturbation=1e-4
    )

    # Run MCMC
    fitter.runmcmc(optimize=True,verbose=False)
    print("MCMC fitting completed!")

    # ================================================================
    # RESULTS ANALYSIS
    # ================================================================

    # Create unified results
    results = UnifiedResults(fitter)

    # Save results
    results.save("reproduced_analysis.h5")
    print("Results saved to reproduced_analysis.h5")

    # Quick analysis
    results.print_summary()
    results.convergence_diagnostics()

    # Generate plots
    try:
        #results.corner_plot(save_path="corner_plot.png")
        results.velocity_plot(save_path="velocity_plot.png")
        print("Plots saved: corner_plot.png, velocity_plot.png")
    except Exception as e:
        print(f"Plot generation failed: {e}")

    return results


if __name__ == "__main__":
    results = main()
    print("\nScript completed successfully!")
    print("Use results.help() for analysis options")