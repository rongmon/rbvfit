"""
Unified quick fitting interface using scipy.optimize for single and multi-instrument datasets.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
import warnings


def quick_fit(model_func, theta_guess, wave, flux, error, bounds=None):
    """
    Quick parameter fitting using scipy.optimize.curve_fit for single instrument.
    
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


def quick_fit_minimize(vfit_obj):
    """
    Quick fit using scipy.optimize.minimize for single or multi-instrument.
    
    This unified approach works for both single and multi-instrument cases
    by iterating over the instrument_data dictionary.
    
    Parameters
    ----------
    vfit_obj : vfit
        The rbvfit vfit object with instrument_data
        
    Returns
    -------
    theta_best : np.ndarray
        Best-fit parameters
    theta_best_error : np.ndarray
        Parameter uncertainties (estimated from finite differences)
    """
    
    def chi2_objective(params):
        """Chi-squared objective function for all instruments"""
        try:
            total_chi2 = 0.0
            
            # Sum chi-squared across all instruments
            for name, inst_data in vfit_obj.instrument_data.items():
                wave = inst_data['wave']
                flux = inst_data['flux']
                error = inst_data['error']
                model_func = inst_data['model']
                
                # Evaluate model for this instrument
                model = model_func(params, wave)
                
                # Add chi-squared contribution
                chi2_contrib = np.sum(((flux - model) / error) ** 2)
                total_chi2 += chi2_contrib
            
            return total_chi2
            
        except Exception:
            # Return large value if model evaluation fails
            return 1e10
    
    # Set up bounds
    bounds = None
    if hasattr(vfit_obj, 'lb') and hasattr(vfit_obj, 'ub'):
        bounds = list(zip(vfit_obj.lb, vfit_obj.ub))
    
    try:
        # Run scipy optimization
        result = minimize(
            chi2_objective, 
            vfit_obj.theta,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxfun': 5000}
        )
        
        theta_best = result.x
        
        # Estimate parameter uncertainties using finite differences
        # This is a rough approximation - not as good as curve_fit covariance
        theta_best_error = _estimate_parameter_errors(chi2_objective, theta_best, vfit_obj.theta)
        
        if not result.success:
            warnings.warn(f"Optimization may not have converged: {result.message}")
        
    except Exception as e:
        warnings.warn(f"Minimize fitting failed: {e}")
        theta_best = np.array(vfit_obj.theta)
        theta_best_error = np.zeros_like(vfit_obj.theta)
    
    return theta_best, theta_best_error


def _estimate_parameter_errors(chi2_func, theta_best, theta_initial, delta_frac=0.01):
    """
    Estimate parameter uncertainties using finite differences.
    
    This is a rough approximation for cases where we can't get
    a proper covariance matrix from the optimizer.
    """
    n_params = len(theta_best)
    theta_errors = np.zeros(n_params)
    
    try:
        chi2_best = chi2_func(theta_best)
        
        for i in range(n_params):
            # Use adaptive step size
            delta = max(abs(theta_best[i]) * delta_frac, abs(theta_initial[i]) * delta_frac, 1e-6)
            
            # Forward difference
            theta_plus = theta_best.copy()
            theta_plus[i] += delta
            chi2_plus = chi2_func(theta_plus)
            
            # Backward difference  
            theta_minus = theta_best.copy()
            theta_minus[i] -= delta
            chi2_minus = chi2_func(theta_minus)
            
            # Estimate curvature (second derivative)
            d2chi2 = (chi2_plus - 2*chi2_best + chi2_minus) / (delta**2)
            
            # Parameter error from chi2 + 1 criterion
            if d2chi2 > 0:
                theta_errors[i] = np.sqrt(1.0 / d2chi2)
            else:
                # Fallback: use parameter range as rough estimate
                theta_errors[i] = abs(theta_best[i] - theta_initial[i])
                
    except Exception:
        # If error estimation fails, return zeros
        theta_errors = np.zeros(n_params)
    
    return theta_errors


def quick_fit_vfit(vfit_obj):
    """
    Quick fit directly from vfit object - unified interface.
    
    Automatically detects single vs multi-instrument and uses appropriate method.
    
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
    
    # Check if unified interface (has instrument_data)
    if hasattr(vfit_obj, 'instrument_data') and vfit_obj.instrument_data is not None:
        # Use minimize approach for unified interface (works for single or multi-instrument)
        n_instruments = len(vfit_obj.instrument_data)
        if n_instruments == 1:
            # Could use either method for single instrument, but minimize is more consistent
            return quick_fit_minimize(vfit_obj)
        else:
            # Multi-instrument: must use minimize
            return quick_fit_minimize(vfit_obj)
    
    else:
        # Legacy interface: use curve_fit for single instrument
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