"""
Quick fitting interface using scipy.optimize for rbvfit 2.0 V2 unified interface.
"""

import numpy as np
from scipy.optimize import minimize
import warnings


def quick_fit_vfit(vfit_obj):
    """
    Quick fit using scipy.optimize.minimize for V2 unified interface.
    
    Works for both single and multi-instrument cases by iterating over
    the instrument_data dictionary.
    
    Parameters
    ----------
    vfit_obj : vfit
        The rbvfit V2 vfit object with instrument_data dictionary
        
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

