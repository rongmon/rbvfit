"""
Minimal quick fitting interface using scipy.optimize.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings


def quick_fit(model_func, theta_guess, wave, flux, error, bounds=None):
    """
    Quick parameter fitting using scipy.optimize.curve_fit.
    
    Parameters
    ----------
    model_func : callable
        Model function: model_func(theta, wave) -> flux
    theta_guess : array_like
        Initial parameter guess
    wave : array_like
        Wavelength array
    flux : array_like
        Observed flux
    error : array_like
        Error array
    bounds : list of tuples, optional
        Parameter bounds [(low, high), ...]
        
    Returns
    -------
    theta_best : np.ndarray
        Best-fit parameters
    theta_best_error : np.ndarray
        Parameter uncertainties (1-sigma)
    """
    # Wrapper for curve_fit (expects xdata as first argument)
    def curve_fit_wrapper(xdata, *params):
        theta = np.array(params)
        return model_func(theta, xdata)
    
    # Convert bounds format for curve_fit
    if bounds is not None:
        bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
    
    try:
        popt, pcov = curve_fit(
            curve_fit_wrapper, 
            wave, 
            flux, 
            p0=theta_guess,
            sigma=error,
            absolute_sigma=True,
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            maxfev=5000
        )
        
        theta_best = popt
        theta_best_error = np.sqrt(np.diag(pcov))
        
    except Exception as e:
        warnings.warn(f"Fitting failed: {e}")
        theta_best = np.array(theta_guess)
        theta_best_error = np.zeros_like(theta_guess)
    
    return theta_best, theta_best_error


def quick_fit_vfit(vfit_obj):
    """
    Quick fit directly from vfit object.
    
    Parameters
    ----------
    vfit_obj : vfit
        The rbvfit vfit object
        
    Returns
    -------
    theta_best : np.ndarray
        Best-fit parameters
    theta_best_error : np.ndarray
        Parameter uncertainties
    """
    bounds = None
    if hasattr(vfit_obj, 'lb') and hasattr(vfit_obj, 'ub'):
        bounds = list(zip(vfit_obj.lb, vfit_obj.ub))
    
    return quick_fit(
        vfit_obj.model, 
        vfit_obj.theta,
        vfit_obj.wave_obs, 
        vfit_obj.fnorm, 
        vfit_obj.enorm,
        bounds=bounds
    )