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
        if hasattr(results, 'samples') and len(results.samples) > 0:
            samples = results.samples
            param_names = results.parameter_names
        else:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No MCMC samples available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return
            
        # Get parameter names
        #n_params = samples.shape[1]
        #n_comp = n_params // 3
        
            
        # Apply parameter filter
        #if param_filter == "N":
        #    indices = [i for i in range(0, n_comp)]
        #    filtered_names = [f'N_{i+1}' for i in range(n_comp)]
        #elif param_filter == "b":
        #    indices = [i for i in range(n_comp, 2*n_comp)]
        #    filtered_names = [f'b_{i+1}' for i in range(n_comp)]
        #elif param_filter == "v":
        #    indices = [i for i in range(2*n_comp, 3*n_comp)]
        #    filtered_names = [f'v_{i+1}' for i in range(n_comp)]
        #else:  # all
        #    indices = list(range(n_params))
        #   filtered_names = param_names
        
        # Fix the parameter filtering logic
        if param_filter == "N":
            # Find parameters that start with 'N' or 'logN'
            indices = [i for i, name in enumerate(param_names) 
                      if name.startswith('N_') or name.startswith('logN')]
            filtered_names = [param_names[i] for i in indices]
        elif param_filter == "b":
            indices = [i for i, name in enumerate(param_names) 
                      if name.startswith('b_')]
            filtered_names = [param_names[i] for i in indices]
        elif param_filter == "v":
            indices = [i for i, name in enumerate(param_names) 
                      if name.startswith('v_')]
            filtered_names = [param_names[i] for i in indices]
        else:  # all
            indices = list(range(len(param_names)))
            filtered_names = param_names

        # Only proceed if we found matching parameters
        if not indices:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, f'No {param_filter} parameters found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return


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


def plot_model_comparison_custom(figure, results, show_components=True, show_residuals=True,plot_ranges=None, instrument_name=None):
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
        # Use selected instrument or default to first
        if instrument_name and instrument_name in results.instrument_data:
            inst_data = results.instrument_data[instrument_name]
        else:
            # Default to first instrument
            primary_inst = results.instrument_names[0]
            inst_data = results.instrument_data[primary_inst]
            instrument_name = primary_inst
        
        wave = inst_data['wave']
        flux = inst_data['flux']
        error = inst_data['error']
        best_theta = results.best_fit
        
        # Get model for this instrument
        if results.config_metadata:
            if results.is_multi_instrument:
                model = results.reconstruct_model(instrument_name)
            else:
                model = results.reconstruct_model()
            model_flux = model.evaluate(best_theta, wave)
        else:
            model_flux = np.ones_like(flux)
                
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

    if plot_ranges:
        if hasattr(figure, 'get_axes') and figure.get_axes():
            ax = figure.get_axes()[0]  # Main plot axis
            
            # Apply x-axis limits
            if plot_ranges.get('x_min') is not None and plot_ranges.get('x_max') is not None:
                ax.set_xlim(plot_ranges['x_min'], plot_ranges['x_max'])
            
            # Apply y-axis limits  
            if plot_ranges.get('y_min') is not None and plot_ranges.get('y_max') is not None:
                ax.set_ylim(plot_ranges['y_min'], plot_ranges['y_max'])




def plot_velocity_space_custom(figure, results, velocity_range=None, show_components=True, show_rail=True, instrument_name=None,yrange=None):
    """Simplified velocity space plotting"""
    figure.clear()
    
    try:
        # Get instrument data
        if instrument_name and instrument_name in results.instrument_data:
            inst_data = results.instrument_data[instrument_name]
        else:
            primary_inst = results.instrument_names[0]
            inst_data = results.instrument_data[primary_inst]
            instrument_name = primary_inst
        
        wave_obs = inst_data['wave']
        flux_obs = inst_data['flux']
        error_obs = inst_data['error']
        best_theta = results.best_fit
        
        # Try to get model
        if results.config_metadata:
            if results.is_multi_instrument:
                model = results.reconstruct_model(instrument_name)
            else:
                model = results.reconstruct_model()
            
            # Simple velocity conversion using center wavelength
            center_wave = np.median(wave_obs)
            c_kms = 299792.458
            velocity = c_kms * (wave_obs / center_wave - 1)
            
            # Evaluate model
            model_flux = model.evaluate(best_theta, wave_obs)
            
            # Create plot
            ax = figure.add_subplot(111)
            ax.step(velocity, flux_obs, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
            ax.fill_between(velocity, flux_obs - error_obs, flux_obs + error_obs,
                           alpha=0.3, color='gray', label='1σ Error')
            ax.plot(velocity, model_flux, 'r-', linewidth=2, label='Best Fit')
            
            # Set velocity range
            if velocity_range:
                ax.set_xlim(velocity_range)
            if yrange:
                ax.set_ylim(yrange)
            
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Normalized Flux')
            ax.set_title(f'Velocity Space - {instrument_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Model reconstruction not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
    except Exception as e:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating velocity plot:\n{str(e)}',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)

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


def update_results_plots_custom(results_tab, results):
    """
    Update all plots in results tab with custom plotting functions.
    
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
            
        plot_corner_custom(results_tab.corner_figure, results, filter_type)
        results_tab.corner_canvas.draw()
        
        # Update model comparison
        show_components = results_tab.show_components_check.isChecked()
        show_residuals = results_tab.show_residuals_check.isChecked()
        
        plot_model_comparison_custom(results_tab.comparison_figure, results, show_components, show_residuals)
        results_tab.comparison_canvas.draw()
        
        # Update velocity plot with custom ranges
        if hasattr(results_tab, 'velocity_range_min') and hasattr(results_tab, 'velocity_range_max'):
            if results_tab.velocity_range_min is not None and results_tab.velocity_range_max is not None:
                velocity_range = (results_tab.velocity_range_min, results_tab.velocity_range_max)
            else:
                velocity_range = None  # Auto range
        else:
            velocity_range = (-600, 600)  # Default range
            
        plot_velocity_space_custom(results_tab.velocity_figure, results, velocity_range, show_components, True)
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