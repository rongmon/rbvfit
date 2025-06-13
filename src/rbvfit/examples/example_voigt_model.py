#!/usr/bin/env python
"""
Simple examples for using the VoigtModel in rbvfit 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel


def example_mgii_doublet():
    """Example 1: Simple MgII doublet."""
    print("Example 1: MgII Doublet")
    print("-" * 50)
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    # Create model with default Gaussian LSF
    model = VoigtModel(config)
    print(model.get_info())
    
    # Set up parameters for 2 components
    # theta = [N1, N2, b1, b2, v1, v2]
    theta = np.array([
        13.5, 13.2,     # log N values
        15.0, 25.0,     # b values (km/s)
        -150.0, 20.0     # v values (km/s)
    ])
    
    # Wavelength grid
    z = 0.348
    wave = np.linspace(3700, 3820, 10000)
    
    # Evaluate model
    flux = model.evaluate(theta, wave)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wave, flux, 'b-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title('MgII Doublet at z=0.348')
    plt.grid(True, alpha=0.3)
    
    # Mark transitions
    for wrest in [2796.3, 2803.5]:
        wobs = wrest * (1 + z)
        plt.axvline(wobs, color='red', linestyle='--', alpha=0.5)
        plt.text(wobs, 1.02, f'{wrest:.1f}', ha='center', fontsize=10)
    
    plt.ylim(0, 1.1)
    plt.show()
    
    return model, theta, wave, flux


def example_multi_ion():
    """Example 2: Multiple ions at same redshift."""
    print("\nExample 2: Multi-Ion System")
    print("-" * 50)
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    config.add_system(z=0.348, ion='FeII', transitions=[2600.2], components=1)
    
    # Create model
    model = VoigtModel(config, FWHM='6.5')
    print(model.get_info())
    
    # Parameters: MgII (2 comp) + FeII (1 comp)
    theta = np.array([
        13.5, 13.2, 14.8,    # N values
        15.0, 25.0, 20.0,    # b values
        -50.0, 20.0, 0.0     # v values
    ])
    
    # Wavelength grid
    wave = np.linspace(3400, 3920, 10000)
    
    # Evaluate
    flux = model.evaluate(theta, wave)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(wave, flux, 'b-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title('Multi-Ion System: MgII + FeII at z=0.348')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    # Mark transitions
    z = 0.348
    plt.axvline(2796.3 * (1+z), color='red', linestyle='--', alpha=0.5, label='MgII')
    plt.axvline(2803.5 * (1+z), color='red', linestyle='--', alpha=0.5)
    plt.axvline(2600.2 * (1+z), color='green', linestyle='--', alpha=0.5, label='FeII')
    
    plt.legend()
    plt.show()
    
    return model, theta


def example_different_lsf():
    """Example 3: Different LSF values."""
    print("\nExample 3: Effect of Different LSF")
    print("-" * 50)
    
    # Single narrow line to see LSF effects
    config = FitConfiguration()
    config.add_system(z=0.1, ion='MgII', transitions=[2796.3], components=1)
    
    # Parameters for narrow line
    theta = np.array([13.5, 10.0, 0.0])  # N, b=10 km/s, v=0
    
    # Wavelength grid
    wave = np.linspace(3070, 3080, 500)
    
    # Different FWHM values
    fwhm_values = ['3.0', '6.5', '13.0']
    
    plt.figure(figsize=(10, 6))
    
    for fwhm in fwhm_values:
        model = VoigtModel(config, FWHM=fwhm)
        flux = model.evaluate(theta, wave)
        plt.plot(wave, flux, linewidth=2, label=f'FWHM = {fwhm} pixels')
    
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title('Effect of LSF Convolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.show()


def example_ion_tying():
    """Example 4: Demonstrate ion parameter tying."""
    print("\nExample 4: Ion Parameter Tying")
    print("-" * 50)
    
    # MgII doublet with transitions that must share parameters
    config = FitConfiguration()
    config.add_system(z=0.1, ion='MgII', transitions=[2796.3, 2803.5], components=1)
    
    model = VoigtModel(config)
    
    # Single component
    theta = np.array([13.8, 20.0, 0.0])  # N, b, v
    
    # Evaluate around each transition
    z = 0.1
    
    # Plot both transitions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Around 2796
    wave1 = np.linspace(3070, 3080, 500)
    flux1 = model.evaluate(theta, wave1)
    ax1.plot(wave1, flux1, 'b-', linewidth=2)
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('MgII 2796.3')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(2796.3 * (1+z), color='red', linestyle='--', alpha=0.5)
    
    # Around 2803
    wave2 = np.linspace(3078, 3088, 500)
    flux2 = model.evaluate(theta, wave2)
    ax2.plot(wave2, flux2, 'b-', linewidth=2)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title('MgII 2803.5')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(2803.5 * (1+z), color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Ion Tying: Both lines use same parameters (N={theta[0]:.1f}, b={theta[1]:.0f} km/s, v={theta[2]:.0f} km/s)')
    plt.tight_layout()
    plt.show()
    
    print("Note: Both MgII transitions automatically share the same N, b, v parameters")


def example_multi_redshift():
    """Example 5: Systems at different redshifts."""
    print("\nExample 5: Multiple Redshift Systems")
    print("-" * 50)
    
    # Create configuration with different redshifts
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=1)
    config.add_system(z=0.524, ion='MgII', transitions=[2796.3, 2803.5], components=1)
    
    model = VoigtModel(config)
    print(model.get_info())
    
    # Parameters for both systems
    theta = np.array([
        13.8, 13.5,      # N values
        25.0, 30.0,      # b values
        0.0, 0.0         # v values
    ])
    
    # Wide wavelength range
    wave = np.linspace(3700, 4400, 3000)
    flux = model.evaluate(theta, wave)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(wave, flux, 'b-', linewidth=1)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Normalized Flux')
    plt.title('MgII at Two Different Redshifts')
    plt.grid(True, alpha=0.3)
    
    # Mark systems
    for z, color, label in [(0.348, 'red', 'z=0.348'), (0.524, 'green', 'z=0.524')]:
        for wrest in [2796.3, 2803.5]:
            wobs = wrest * (1 + z)
            plt.axvline(wobs, color=color, linestyle='--', alpha=0.5)
        # Add label once
        plt.axvline(2796.3 * (1+z), color=color, linestyle='--', alpha=0.5, label=label)
    
    plt.legend()
    plt.ylim(-0.05, 1.1)
    plt.show()
    
    return model


if __name__ == "__main__":
    # Run examples
    print("rbvfit 2.0 VoigtModel Examples")
    print("=" * 50)
    
    example_mgii_doublet()
    example_multi_ion()
    example_different_lsf()
    example_ion_tying()
    example_multi_redshift()
    
    print("\nAll examples completed!")