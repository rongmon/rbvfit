from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
from rbvfit import rb_setline as rt
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel

#------------------------
# Compute Curve of Growth (COG) - rbvfit v2.0
#------------------------
# This module computes the curve of growth (COG) for a given transition.
# It uses the Voigt profile to calculate equivalent widths (EWs) for various
# column densities and Doppler parameters.
#
# The COG is a plot of equivalent width (W) versus log(Nfλ), where:
# - N is the column density
# - f is the oscillator strength
# - λ is the rest wavelength of the transition  
#------------------------

def set_one_absorber(N, b, lam_rest, model_compiled):
    """
    Calculate equivalent width for one absorber.
    
    Parameters
    ----------
    N : float
        Log column density
    b : float
        Doppler parameter (km/s)
    lam_rest : float
        Rest wavelength (Angstroms)
    model_compiled : CompiledVoigtModel
        Compiled v2 model object
        
    Returns
    -------
    W : float
        Equivalent width (Angstroms)
    """
    try:
        zabs = 0.
        N = float(N)
        b = float(b)
        v = 0.0
        
        # Create wavelength grid around the transition
        wave = np.linspace(lam_rest - 5., lam_rest + 5., 1000)
        
        # Parameter array for v2: [N, b, v] for single component
        theta = np.array([N, b, v])
        
        # Use model_flux method from compiled model
        flx = model_compiled.model_flux(theta, wave)
        
        # Calculate equivalent width using trapezoidal integration
        W = np.trapz(1. - flx, x=wave)
        return W
        
    except Exception as e:
        print(f"Error in set_one_absorber: N={N}, b={b}, lam_rest={lam_rest}")
        print(f"Error details: {e}")
        raise

def compute_ewlist_from_voigt(Nlist, b, lam_rest, model_compiled):
    """
    Compute equivalent widths for a list of column densities.
    
    Parameters
    ----------
    Nlist : array_like
        Array of log column densities
    b : float
        Doppler parameter (km/s)
    lam_rest : float
        Rest wavelength (Angstroms)
    model_compiled : CompiledVoigtModel
        Compiled v2 model object
        
    Returns
    -------
    Wlist : array
        Array of equivalent widths (Angstroms)
    """
    Wlist = np.zeros(len(Nlist))
    for i in range(len(Nlist)):
        Wlist[i] = set_one_absorber(Nlist[i], b, lam_rest, model_compiled)
    return Wlist

class compute_cog(object):
    def __init__(self, lam_guess, Nlist, blist):
        """
        Create a curve of growth for a given input set of parameters.

        Parameters
        ----------
        lam_guess : float
            Rest frame wavelength of one transition (Angstroms)
        Nlist : array_like  
            Array of log column densities for which COG is to be computed
        blist : array_like
            Array of b values (km/s) for which COG is to be computed

        Attributes
        ----------
        st : dict
            Structure containing transition information
        Wlist : ndarray
            Matrix containing EW for every logN and b value                

        Examples
        --------
        >>> # Create COG for Lyman alpha
        >>> Nlist = np.linspace(12, 16, 50)  # log N from 12 to 16
        >>> blist = [10, 20, 30, 50]        # b values in km/s
        >>> cog = compute_cog(1215.67, Nlist, blist)
        >>> cog.plot_cog()
        """
        # Get transition information
        print(f"Looking up transition for wavelength: {lam_guess} Å")
        self.st = rt.rb_setline(lam_guess, 'closest')
        
        # Debug: print what we got from rb_setline
        print(f"Transition found: {self.st}")
        
        # Extract scalar values from potentially array-like entries
        wave_val = float(self.st['wave']) if hasattr(self.st['wave'], '__len__') else self.st['wave']
        fval = float(self.st['fval']) if hasattr(self.st['fval'], '__len__') else self.st['fval']
        gamma_val = float(self.st['gamma']) if hasattr(self.st['gamma'], '__len__') else self.st['gamma']
        
        # Update st with scalar values for easier handling
        self.st['wave'] = wave_val
        self.st['fval'] = fval
        self.st['gamma'] = gamma_val
        
        # Create v2 configuration
        config = FitConfiguration()
        
        # Try to detect ion automatically, fallback to manual specification
        try:
            config.add_system(z=0., ion='auto', transitions=[wave_val], components=1)
        except:
            # If auto-detection fails, try common ions based on wavelength
            if 1200 < wave_val < 1230:
                ion_name = 'HI'
            elif 2790 < wave_val < 2810:
                ion_name = 'MgII'
            elif 1190 < wave_val < 1200:
                ion_name = 'SiII'
            elif 1240 < wave_val < 1260:
                ion_name = 'NV'
            elif 1520 < wave_val < 1530:
                ion_name = 'CIV'
            elif 1030 < wave_val < 1040:
                ion_name = 'OVI'
            else:
                ion_name = 'HI'  # Default fallback
            
            print(f"Auto-detection failed, using ion: {ion_name}")
            config.add_system(z=0., ion=ion_name, transitions=[wave_val], components=1)
    
        # Create and compile model (no LSF for COG calculations)
        model = VoigtModel(config, FWHM=None)
        self.model_compiled = model.compile()
        
        # Store input parameters
        self.Nlist = np.array(Nlist)
        self.blist = np.array(blist)
        
        # Initialize results matrix
        self.Wlist = np.zeros((len(Nlist), len(blist)))
        
        # Compute COG for all b values
        print(f"Computing COG for transition: {self.st['name']} at {wave_val:.2f} Å")
        print(f"Oscillator strength: {fval:.3e}")
        print(f"Damping parameter: {gamma_val:.3e} s^-1")
        
        for i, b_val in enumerate(blist):
            print(f"Computing for b = {b_val} km/s...")
            self.Wlist[:, i] = compute_ewlist_from_voigt(
                Nlist, b_val, wave_val, self.model_compiled
            )

    def plot_cog(self, show_legend=True, figsize=(10, 8), mark_regimes=True):
        """
        Plot the curve of growth.
        
        Parameters
        ----------
        show_legend : bool, optional
            Whether to show the legend (default: True)
        figsize : tuple, optional
            Figure size (default: (10, 8))
        mark_regimes : bool, optional
            Whether to mark the theoretical regime transitions (default: True)
        """
        plt.figure(figsize=figsize)
        plt.title(f"Curve of Growth: {self.st['name']}")

        # Convert to standard COG units
        # x-axis: log10(N * f * λ) where N is in cm^-2, f dimensionless, λ in cm
        lambda_cm = self.st['wave'] * 1e-8  # Convert Å to cm
        
        # Store x-axis values for regime calculations
        N_linear = 10**self.Nlist  # Column density in cm^-2
        x_axis_values = np.log10(N_linear * self.st['fval'] * lambda_cm)
        
        for i in range(len(self.blist)):
            # y-axis: log10(W/λ) where both W and λ are in same units (Angstroms)
            y_axis = np.log10(self.Wlist[:, i] / self.st['wave'])
            
            plt.plot(x_axis_values, y_axis, 
                    label=f'b = {self.blist[i]} km/s',
                    linewidth=2, marker='o', markersize=3)
        
        plt.xlabel(r'$\log_{10}[N f \lambda]$ (cgs units)')
        plt.ylabel(r'$\log_{10}[W/\lambda]$')
        plt.grid(True, alpha=0.3)
        
        if mark_regimes:
            self._add_regime_markers(x_axis_values, lambda_cm)
        
        if show_legend:
            plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    def _add_regime_markers(self, x_axis_values, lambda_cm):
        """
        Add vertical lines and labels to mark COG regime transitions based on physics.
        """
        # Get current axis limits
        ylim = plt.ylim()
        x_min, x_max = np.min(x_axis_values), np.max(x_axis_values)
        
        # Calculate regime transition points using proper physics
        transitions = self._calculate_regime_transitions_physics()
        
        # Add vertical lines at transitions if they're in our data range
        transition_names = ['Linear→Saturated', 'Saturated→Damped']
        
        for i, (name, x_trans) in enumerate(zip(transition_names, transitions)):
            if x_trans is not None and x_min <= x_trans <= x_max:
                plt.axvline(x_trans, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Add small text label near the line
                plt.text(x_trans + 0.1, ylim[1] - 0.1 - i*0.05*(ylim[1]-ylim[0]), name, 
                        rotation=90, fontsize=9, alpha=0.8,
                        verticalalignment='top')
        
        # Add regime background shading
        regime_colors = ['lightblue', 'lightgreen', 'lightcoral']
        regime_names = ['Linear', 'Saturated', 'Damped']
        
        # Define regime boundaries based on calculated transitions
        if transitions[0] is not None and transitions[1] is not None and \
           x_min <= transitions[0] <= x_max and x_min <= transitions[1] <= x_max:
            # All three regimes visible
            regime_bounds = [x_min, transitions[0], transitions[1], x_max]
        elif transitions[0] is not None and x_min <= transitions[0] <= x_max:
            # Linear and saturated regimes visible
            regime_bounds = [x_min, transitions[0], x_max]
            regime_names = regime_names[:2]
            regime_colors = regime_colors[:2]
        elif transitions[1] is not None and x_min <= transitions[1] <= x_max:
            # Saturated and damped regimes visible
            regime_bounds = [x_min, transitions[1], x_max]
            regime_names = regime_names[1:]
            regime_colors = regime_colors[1:]
        else:
            # No clear transitions in visible range - don't add confusing labels
            return
        
        # Add subtle background shading for each regime
        for i in range(len(regime_names)):
            if i < len(regime_bounds) - 1:
                plt.axvspan(regime_bounds[i], regime_bounds[i+1], 
                           alpha=0.1, color=regime_colors[i])
                
                # Add regime label in the middle of each region
                x_center = (regime_bounds[i] + regime_bounds[i+1]) / 2
                plt.text(x_center, ylim[0] + 0.05 * (ylim[1] - ylim[0]), 
                        regime_names[i], 
                        horizontalalignment='center', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=regime_colors[i], alpha=0.7))
        
    def _calculate_regime_transitions_physics(self):
        """
        Calculate regime transitions using proper physics with rb_setline parameters.
        
        Returns
        -------
        transitions : tuple
            (linear_to_saturated, saturated_to_damped) transition points
        """
        # Get physical parameters
        f = self.st['fval']
        lambda_A = self.st['wave']  # Angstroms
        lambda_cm = lambda_A * 1e-8  # cm
        gamma = self.st['gamma']  # s^-1
        
        # Use median b value for calculations
        b_typical = np.median(self.blist) * 1e5  # Convert km/s to cm/s
        
        # Transition 1: Linear to Saturated regime (happens FIRST at lower N)
        # Occurs when central optical depth τ₀ ≈ 1
        # τ₀ = (1.497e-15 * f * λ_A * N) / b
        # Solving for N when τ₀ = 1: N = b / (1.497e-15 * f * λ_A)
        N_transition_1 = b_typical / (1.497e-15 * f * lambda_A)
        x_transition_1 = np.log10(N_transition_1 * f * lambda_cm)
        
        # Transition 2: Saturated to Damped regime (happens SECOND at higher N)
        # Occurs when natural broadening becomes comparable to Doppler broadening
        # Rough estimate: when √(γ/4π) ≈ b/λ, or equivalently when:
        # N ≈ (4π * b²) / (f * λ² * γ)
        N_transition_2 = (4 * np.pi * b_typical**2) / (f * lambda_cm**2 * gamma)
        x_transition_2 = np.log10(N_transition_2 * f * lambda_cm)
        
        # Ensure proper ordering: transition_1 should be < transition_2
        if x_transition_2 < x_transition_1:
            x_transition_1, x_transition_2 = x_transition_2, x_transition_1
        
        # Check if transitions are reasonable (within typical COG range)
        if not (10 < x_transition_1 < 20):
            x_transition_1 = None
        if not (10 < x_transition_2 < 20):
            x_transition_2 = None
            
        return (x_transition_1, x_transition_2)

    def save_cog_data(self, filename):
        """
        Save COG data to a file.
        
        Parameters
        ----------
        filename : str
            Output filename (recommended: .npz format)
        """
        np.savez(filename,
                Nlist=self.Nlist,
                blist=self.blist, 
                Wlist=self.Wlist,
                transition_info=self.st)
        print(f"COG data saved to {filename}")
        
    def get_interpolated_ew(self, logN, b):
        """
        Get interpolated equivalent width for given logN and b.
        
        Parameters
        ----------
        logN : float
            Log column density
        b : float
            Doppler parameter (km/s)
            
        Returns
        -------
        W : float
            Interpolated equivalent width (Angstroms)
        """
        from scipy.interpolate import interp2d
        
        f_interp = interp2d(self.blist, self.Nlist, self.Wlist, kind='linear')
        return float(f_interp(b, logN))

# Example usage and test function
def test_cog_example():
    """
    Test function demonstrating COG usage for Lyman alpha.
    """
    print("=" * 50)
    print("Testing Curve of Growth for Lyman Alpha")
    print("=" * 50)
    
    # Parameters for Lyman alpha
    lyman_alpha = 1215.67  # Angstroms
    Nlist = np.linspace(12, 20, 40)  # log N from 12 to 20
    blist = [5, 10, 20, 30, 50, 100]  # b values in km/s
    
    # Create and plot COG
    cog = compute_cog(lyman_alpha, Nlist, blist)
    cog.plot_cog()
    
    # Test interpolation
    test_logN = 14.5
    test_b = 25.0
    test_ew = cog.get_interpolated_ew(test_logN, test_b)
    print(f"\nInterpolated EW for logN={test_logN}, b={test_b} km/s: {test_ew:.3f} Å")
    
    return cog

if __name__ == "__main__":
    # Run test
    cog_test = test_cog_example()