#!/usr/bin/env python
"""
Custom Plotting GUI Module for rbvfit 2.0 Results

This module provides custom plotting routines that work directly with Qt canvases
instead of creating separate matplotlib windows.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


def plot_corner_custom(figure, results, fitter, param_filter="all"):
    """
    Custom corner plot that works with Qt canvas.
    
    Parameters
    ----------
    figure : matplotlib.Figure
        The Qt figure to plot on
    results : FitResults or None
        Results object 
    fitter : vfit
        MCMC fitter object with samples
    param_filter : str
        "all", "N", "b", or "v" to filter parameters
    """
    figure.clear()
    
    try:
        # Get MCMC samples
        if hasattr(fitter, 'samples') and len(fitter.samples) > 0:
            samples = fitter.samples
        else:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No MCMC samples available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return
            
        # Get parameter names
        n_params = samples.shape[1]
        n_comp = n_params // 3
        
        # Generate parameter names
        param_names = []
        for i in range(n_comp):
            param_names.extend([f'N_{i+1}', f'b_{i+1}', f'v_{i+1}'])
            
        # Apply parameter filter
        if param_filter == "N":
            indices = [i for i in range(0, n_comp)]
            filtered_names = [f'N_{i+1}' for i in range(n_comp)]
        elif param_filter == "b":
            indices = [i for i in range(n_comp, 2*n_comp)]
            filtered_names = [f'b_{i+1}' for i in range(n_comp)]
        elif param_filter == "v":
            indices = [i for i in range(2*n_comp, 3*n_comp)]
            filtered_names = [f'v_{i+1}' for i in range(n_comp)]
        else:  # all
            indices = list(range(n_params))
            filtered_names = param_names
            
        # Filter samples
        filtered_samples = samples[:, indices]
        
        if HAS_CORNER and len(indices) > 1:
            # Create corner plot
            corner.corner(filtered_samples, labels=filtered_names, 
                        fig=figure, show_titles=True, title_kwargs={"fontsize": 10})
        else:
            # Fallback: histogram grid
            n_params_filtered = len(indices)
            if n_params_filtered == 1:
                # Single parameter histogram
                ax = figure.add_subplot(111)
                ax.hist(filtered_samples[:, 0], bins=50, alpha=0.7, density=True)
                ax.set_xlabel(filtered_names[0])
                ax.set_ylabel('Density')
                ax.set_title('Parameter Distribution')
            else:
                # Create simple scatter plot matrix
                n_cols = min(3, n_params_filtered)
                n_rows = (n_params_filtered + n_cols - 1) // n_cols
                
                for i in range(n_params_filtered):
                    ax = figure.add_subplot(n_rows, n_cols, i + 1)
                    ax.hist(filtered_samples[:, i], bins=30, alpha=0.7, density=True)
                    ax.set_xlabel(filtered_names[i])
                    ax.set_ylabel('Density')
                    
                figure.suptitle('Parameter Distributions')
                
    except Exception as e:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating corner plot:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_model_comparison_custom(figure, results, fitter, model, show_components=True, show_residuals=True):
    """
    Custom model vs data comparison plot.
    
    Parameters
    ----------
    figure : matplotlib.Figure
        The Qt figure to plot on
    results : FitResults or None
        Results object
    fitter : vfit
        MCMC fitter object with data and best-fit parameters
    model : VoigtModel
        Model object for evaluation
    show_components : bool
        Whether to show individual components
    show_residuals : bool
        Whether to show residuals subplot
    """
    figure.clear()
    
    try:
        # Get data from fitter
        wave = fitter.wave_obs
        flux = fitter.fnorm
        error = fitter.enorm
        best_theta = fitter.best_theta
        
        # Calculate best-fit model
        model_flux = model.evaluate(best_theta, wave)
        
        # Create subplots
        if show_residuals:
            ax1 = figure.add_subplot(211)
            ax2 = figure.add_subplot(212, sharex=ax1)
        else:
            ax1 = figure.add_subplot(111)
            ax2 = None
            
        # Main plot: data vs model
        ax1.step(wave, flux, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax1.fill_between(wave, flux - error, flux + error, 
                       alpha=0.3, color='gray', label='1σ Error')
        ax1.plot(wave, model_flux, 'r-', linewidth=2, label='Best-fit Model')
        
        # Add individual components if requested and available
        if show_components:
            try:
                # Try to get individual components from model
                component_result = model.evaluate(best_theta, wave, return_components=True)
                if isinstance(component_result, dict) and 'components' in component_result:
                    components = component_result['components']
                    for i, comp_flux in enumerate(components):
                        ax1.plot(wave, comp_flux, '--', alpha=0.7, linewidth=1, 
                               label=f'Component {i+1}')
            except Exception:
                # Model doesn't support component decomposition
                pass
                
        ax1.set_ylabel('Normalized Flux')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Model vs Data Comparison')
        
        if not show_residuals:
            ax1.set_xlabel('Wavelength (Å)')
            
        # Residuals subplot
        if show_residuals and ax2:
            residuals = (flux - model_flux) / error
            ax2.step(wave, residuals, 'k-', where='mid', linewidth=1, alpha=0.8)
            ax2.axhline(0, color='r', linestyle='-', alpha=0.7)
            ax2.fill_between(wave, -1, 1, alpha=0.3, color='gray', label='±1σ')
            ax2.set_xlabel('Wavelength (Å)')
            ax2.set_ylabel('Residuals (σ)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
        figure.tight_layout()
        
    except Exception as e:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating model comparison:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_velocity_space_custom(figure, results, fitter, model, velocity_range=None, 
                              show_components=True, show_rail=True):
    """
    Custom velocity space plotting that works with Qt canvas.
    
    Parameters
    ----------
    figure : matplotlib.Figure
        The Qt figure to plot on
    results : FitResults or None
        Results object (can be None)
    fitter : vfit
        MCMC fitter object with data and best-fit parameters
    model : VoigtModel  
        Model object for evaluation
    velocity_range : tuple, optional
        (vmin, vmax) in km/s
    show_components : bool
        Whether to show individual components
    show_rail : bool
        Whether to show component position markers
    """
    figure.clear()
    
    try:
        # Get data from fitter
        wave_obs = fitter.wave_obs
        flux_obs = fitter.fnorm
        error_obs = fitter.enorm
        best_theta = fitter.best_theta
        
        # Get system information from model
        if hasattr(model, 'config') and hasattr(model.config, 'systems'):
            systems = model.config.systems
        else:
            # Fallback: create generic system info
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Model configuration not available for velocity plotting',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
            
        if not systems:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No absorption systems found in model',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
            
        # Count total ion groups for subplot layout
        total_ion_groups = sum(len(system.ion_groups) for system in systems)
        
        if total_ion_groups == 0:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No ion groups found in model',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
            
        # Create subplot grid
        n_cols = min(2, total_ion_groups)
        n_rows = (total_ion_groups + n_cols - 1) // n_cols
        
        subplot_idx = 1
        
        # Extract parameter values for rail plotting
        n_params = len(best_theta)
        n_comp = n_params // 3
        v_components = best_theta[2*n_comp:] if n_comp > 0 else []
        
        for system in systems:
            for ion_group in system.ion_groups:
                ax = figure.add_subplot(n_rows, n_cols, subplot_idx)
                
                # Use the first transition for this ion group for velocity calculation
                if ion_group.transitions:
                    transition = ion_group.transitions[0]
                    
                    # Convert to velocity space
                    c_kms = 299792.458
                    lambda_sys = transition * (1 + system.redshift)
                    velocity = c_kms * (wave_obs / lambda_sys - 1)
                    
                    # Plot data
                    ax.step(velocity, flux_obs, 'k-', where='mid', linewidth=1, 
                           alpha=0.8, label='Data')
                    ax.fill_between(velocity, flux_obs - error_obs, flux_obs + error_obs,
                                   alpha=0.3, color='gray', label='1σ Error')
                    
                    # Plot best-fit model
                    try:
                        model_flux = model.evaluate(best_theta, wave_obs)
                        ax.plot(velocity, model_flux, 'r-', linewidth=2, label='Best Fit')
                    except Exception as e:
                        print(f"Could not evaluate model: {e}")
                    
                    # Add component rail markers if requested
                    if show_rail and len(v_components) > 0:
                        ylim = ax.get_ylim()
                        for i, v_comp in enumerate(v_components):
                            ax.axvline(v_comp, color='red', linestyle='--', alpha=0.7, linewidth=1)
                            ax.text(v_comp, ylim[1] * 0.95, f'C{i+1}', 
                                   ha='center', va='top', color='red', fontsize=8)
                    
                    # Add zero velocity reference
                    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                    
                    # Set velocity range
                    if velocity_range:
                        ax.set_xlim(velocity_range)
                    else:
                        # Auto range around components
                        if len(v_components) > 0:
                            v_center = np.mean(v_components)
                            v_span = max(200, np.ptp(v_components) * 2) if len(v_components) > 1 else 200
                            ax.set_xlim(v_center - v_span, v_center + v_span)
                        else:
                            ax.set_xlim(-300, 300)
                    
                    # Formatting
                    ax.set_ylabel('Normalized Flux')
                    ax.set_xlabel('Velocity (km/s)')
                    ax.set_title(f'{ion_group.ion_name} {transition:.1f}Å (z={system.redshift:.3f})')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.2)
                    
                    if subplot_idx == 1:  # Only show legend on first subplot
                        ax.legend(fontsize=8, loc='upper right')
                        
                subplot_idx += 1
                
        figure.suptitle('Velocity Space Analysis', fontsize=14)
        figure.tight_layout()
        
    except Exception as e:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating velocity plot:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_chain_traces_custom(figure, fitter):
    """
    Plot MCMC chain traces for convergence diagnostics.
    
    Parameters
    ----------
    figure : matplotlib.Figure
        The Qt figure to plot on
    fitter : vfit
        MCMC fitter object with samples
    """
    figure.clear()
    
    try:
        if not hasattr(fitter, 'samples') or len(fitter.samples) == 0:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No MCMC samples available for trace plots',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return
            
        samples = fitter.samples
        n_params = samples.shape[1]
        n_comp = n_params // 3
        
        # Parameter names
        param_names = []
        for i in range(n_comp):
            param_names.extend([f'N_{i+1}', f'b_{i+1}', f'v_{i+1}'])
            
        # Create subplot grid (max 6 parameters per plot)
        n_show = min(6, n_params)
        n_cols = 2
        n_rows = (n_show + n_cols - 1) // n_cols
        
        for i in range(n_show):
            ax = figure.add_subplot(n_rows, n_cols, i + 1)
            
            # Plot trace
            ax.plot(samples[:, i], alpha=0.7, linewidth=0.5)
            ax.set_ylabel(param_names[i])
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = np.mean(samples[:, i])
            ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7)
            
        figure.suptitle('MCMC Chain Traces', fontsize=14)
        figure.tight_layout()
        
    except Exception as e:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating trace plots:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)


def update_results_plots_custom(results_tab, results, fitter, model):
    """
    Update all plots in results tab with custom plotting functions.
    
    Parameters
    ----------
    results_tab : ResultsTab
        The results tab widget
    results : FitResults or None
        Results object
    fitter : vfit
        MCMC fitter object
    model : VoigtModel
        Model object
    """
    try:
        # Update corner plot
        param_filter = results_tab.param_selector.currentText()
        if "N parameters" in param_filter:
            filter_type = "N"
        elif "b parameters" in param_filter:
            filter_type = "b"
        elif "v parameters" in param_filter:
            filter_type = "v"
        else:
            filter_type = "all"
            
        plot_corner_custom(results_tab.corner_figure, results, fitter, filter_type)
        results_tab.corner_canvas.draw()
        
        # Update model comparison
        show_components = results_tab.show_components_check.isChecked()
        show_residuals = results_tab.show_residuals_check.isChecked()
        
        plot_model_comparison_custom(results_tab.comparison_figure, results, fitter, model,
                                   show_components, show_residuals)
        results_tab.comparison_canvas.draw()
        
        # Update velocity plot
        vel_range_text = results_tab.vel_range_combo.currentText()
        if vel_range_text == "±200 km/s":
            velocity_range = (-200, 200)
        elif vel_range_text == "±500 km/s":
            velocity_range = (-500, 500)
        elif vel_range_text == "±1000 km/s":
            velocity_range = (-1000, 1000)
        else:
            velocity_range = None
            
        plot_velocity_space_custom(results_tab.velocity_figure, results, fitter, model,
                                 velocity_range, show_components, True)
        results_tab.velocity_canvas.draw()
        
    except Exception as e:
        print(f"Error updating custom plots: {e}")


# Convenience function for external use
def create_quick_plots(fitter, model, save_dir=None):
    """
    Create quick standalone plots for command-line use.
    
    Parameters
    ----------
    fitter : vfit
        MCMC fitter object
    model : VoigtModel
        Model object
    save_dir : str, optional
        Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        
        # Model comparison plot
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        wave = fitter.wave_obs
        flux = fitter.fnorm
        error = fitter.enorm
        model_flux = model.evaluate(fitter.best_theta, wave)
        
        # Data and model
        ax1.step(wave, flux, 'k-', where='mid', linewidth=1, label='Data')
        ax1.fill_between(wave, flux - error, flux + error, alpha=0.3, color='gray')
        ax1.plot(wave, model_flux, 'r-', linewidth=2, label='Best Fit')
        ax1.set_ylabel('Normalized Flux')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = (flux - model_flux) / error
        ax2.step(wave, residuals, 'k-', where='mid', linewidth=1)
        ax2.axhline(0, color='r', linestyle='-')
        ax2.fill_between(wave, -1, 1, alpha=0.3, color='gray')
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Residuals (σ)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        # Corner plot if available
        if HAS_CORNER and hasattr(fitter, 'samples') and len(fitter.samples) > 0:
            n_params = fitter.samples.shape[1]
            n_comp = n_params // 3
            param_names = []
            for i in range(n_comp):
                param_names.extend([f'N_{i+1}', f'b_{i+1}', f'v_{i+1}'])
                
            fig2 = corner.corner(fitter.samples, labels=param_names, show_titles=True)
            
            if save_dir:
                fig2.savefig(f'{save_dir}/corner_plot.png', dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
    except Exception as e:
        print(f"Error creating quick plots: {e}")
        
        
# Function to extract parameter info for table updates
def extract_parameter_info(fitter, model=None):
    """
    Extract parameter information for table display.
    
    Parameters
    ----------
    fitter : vfit
        MCMC fitter object
    model : VoigtModel, optional
        Model object for parameter names
        
    Returns
    -------
    param_names : list
        Parameter names
    best_values : array
        Best-fit parameter values
    errors : array
        Parameter uncertainties
    """
    try:
        best_values = fitter.best_theta
        
        # Calculate uncertainties from samples
        if hasattr(fitter, 'samples') and len(fitter.samples) > 0:
            errors = np.std(fitter.samples, axis=0)
        else:
            errors = np.zeros_like(best_values)
            
        # Generate parameter names
        n_params = len(best_values)
        n_comp = n_params // 3
        
        param_names = []
        for i in range(n_comp):
            param_names.extend([f'N_c{i+1}', f'b_c{i+1}', f'v_c{i+1}'])
            
        # Try to get better names from model if available
        if model and hasattr(model, 'param_manager'):
            try:
                param_names = model.param_manager.get_parameter_names()
            except Exception:
                pass  # Use generic names
                
        return param_names, best_values, errors
        
    except Exception as e:
        print(f"Error extracting parameter info: {e}")
        return [], np.array([]), np.array([])