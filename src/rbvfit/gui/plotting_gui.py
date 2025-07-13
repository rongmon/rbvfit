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


def plot_corner_custom(figure, results, param_filter="all"):
    """
    Custom corner plot that works with Qt canvas.
    
    Parameters
    ----------
    figure : matplotlib.Figure
        The Qt figure to plot on
    results : FitResults or None
        Results object 

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




def plot_velocity_space_custom(figure, results, velocity_range=None, show_components=True, show_rail=True, instrument_name=None, yrange=None):
    """
    Create velocity space plot for GUI showing all transitions in a single row.
    
    Parameters
    ----------
    figure : matplotlib.Figure
        The Qt figure to plot on
    results : UnifiedResults
        Results object with config metadata
    velocity_range : tuple, optional
        Velocity range to plot in km/s (default: (-500, 500))
    show_components : bool
        Whether to show individual components
    show_rail : bool
        Whether to show transition markers (currently unused)
    instrument_name : str, optional
        Specific instrument to plot. If None, uses first instrument.
    yrange : tuple, optional
        Y-axis range for flux (default: (0, 1.2))
    """
    figure.clear()
    
    # Set defaults
    if velocity_range is None:
        velocity_range = (-500, 500)
    if yrange is None:
        yrange = (0, 1.2)
    
    try:
        # Check if we have config metadata for transition info
        if not results.config_metadata:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Model reconstruction not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Get instrument data
        if instrument_name and instrument_name in results.instrument_data:
            inst_data = results.instrument_data[instrument_name]
        else:
            primary_inst = results.instrument_names[0]
            inst_data = results.instrument_data[primary_inst]
            instrument_name = primary_inst
        
        wave_data = inst_data['wave']
        flux_data = inst_data['flux']
        error_data = inst_data['error']
        
        # Build list of all individual transitions
        all_transitions = []
        for system in results.config_metadata['systems']:
            system_z = system['redshift']
            for ion_group in system['ion_groups']:
                ion_name = ion_group['ion_name']
                for transition_wave in ion_group['transitions']:
                    all_transitions.append({
                        'system_z': system_z,
                        'ion_name': ion_name,
                        'rest_wavelength': transition_wave,
                        'obs_wavelength': transition_wave * (1 + system_z)
                    })
        
        if not all_transitions:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No transitions found in config',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        n_transitions = len(all_transitions)
        c_kms = 299792.458  # Speed of light in km/s
        
        # Determine grid layout
        if n_transitions == 1:
            n_rows, n_cols = 1, 1
        elif n_transitions == 2:
            n_rows, n_cols = 2, 1  # Vertical stack for 2 transitions
        else:
            n_cols = int(np.ceil(np.sqrt(n_transitions)))
            n_rows = int(np.ceil(n_transitions / n_cols))
        
        # Create subplot grid
        import math
        axes = []
        for i in range(n_transitions):
            if i == 0:
                ax = figure.add_subplot(n_rows, n_cols, i + 1)
            else:
                ax = figure.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0])
            axes.append(ax)
        
        # Get model for this instrument
        if results.is_multi_instrument:
            model = results.reconstruct_model(instrument_name)
        else:
            model = results.reconstruct_model()
        
        # Define colors for components
        component_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Plot each transition
        for trans_idx, transition in enumerate(all_transitions):
            ax = axes[trans_idx]
            system_z = transition['system_z']
            ion_name = transition['ion_name']
            rest_wavelength = transition['rest_wavelength']
            obs_wavelength = transition['obs_wavelength']
            
            # Check if this transition is covered by this instrument
            if not (wave_data.min() <= obs_wavelength <= wave_data.max()):
                # Transition not covered
                ax.text(0.5, 0.5, f'{ion_name}\n{rest_wavelength:.1f}\nnot covered', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                ax.set_xlim(velocity_range)
                ax.set_ylim(yrange)
                ax.set_title(f'{ion_name} {rest_wavelength:.1f}', fontsize=10)
                continue
            
            # Use this specific transition as velocity reference
            lambda_ref = obs_wavelength
            
            # Convert to velocity space
            velocity = c_kms * (wave_data / lambda_ref - 1)
            
            # Filter to velocity range
            vel_mask = (velocity >= velocity_range[0]) & (velocity <= velocity_range[1])
            if np.sum(vel_mask) == 0:
                ax.text(0.5, 0.5, f'No data in\nvelocity range\n{velocity_range}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_xlim(velocity_range)
                ax.set_ylim(yrange)
                ax.set_title(f'{ion_name} {rest_wavelength:.1f}', fontsize=10)
                continue
            
            vel_plot = velocity[vel_mask]
            flux_plot = flux_data[vel_mask]
            error_plot = error_data[vel_mask]
            
            # Evaluate model
            model_output = model.evaluate(results.best_fit, wave_data, return_components=True)
            model_flux = model_output['flux']
            model_plot = model_flux[vel_mask]
            
            # Plot data
            ax.step(vel_plot, flux_plot, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
            ax.fill_between(vel_plot, flux_plot - error_plot, flux_plot + error_plot, 
                           alpha=0.3, color='gray', step='mid', label='1σ error')
            
            # Plot model
            ax.plot(vel_plot, model_plot, 'r-', linewidth=2, label='Best-fit model')
            
            # Plot individual components if requested
            if show_components and 'components' in model_output and len(model_output['components']) > 0:
                model_components = model_output['components']
                model_comp_info = model_output['component_info']
                
                for j, (component_flux, comp_info) in enumerate(zip(model_components, model_comp_info)):
                    # Only plot components that match this transition
                    comp_lambda0 = comp_info.get('lambda0', 0)
                    if abs(comp_lambda0 - rest_wavelength) < 0.1:  # Match transition
                        component_plot = component_flux[vel_mask]
                        color = component_colors[j % len(component_colors)]
                        
                        # Create label with component info
                        v_value = comp_info.get('v_value', 0)
                        label = f'comp {j+1} (v={v_value:.1f})'
                        
                        ax.plot(vel_plot, component_plot, '--', color=color, linewidth=0.5, 
                               alpha=0.7, label=label)
            
            # Mark the reference transition at v=0
            ax.axvline(0, color='blue', linestyle=':', alpha=0.8, linewidth=2)
            
            # Format subplot
            ax.grid(True, alpha=0.3)
            ax.set_ylim(yrange)
            ax.set_xlim(velocity_range)
            
            # Hide tick labels for cleaner layout
            # Hide y-tick labels for all but leftmost column
            if not ((n_cols > 1 and trans_idx % n_cols == 0) or (n_cols == 1 and trans_idx == 0)):
                ax.set_yticklabels([])
            
            # Hide x-tick labels for panels that have a panel below them
            #if trans_idx + n_cols < n_transitions:
            #    ax.set_xticklabels([])
            
            # Add transition info as text box inside panel instead of title
            textstr = f'{ion_name} {rest_wavelength:.1f}\nz={system_z:.4f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
            
            # Only leftmost subplot gets y-label (for horizontal grids) or top subplot (for vertical)
            if (n_cols > 1 and trans_idx % n_cols == 0) or (n_cols == 1 and trans_idx == 0):
                ax.set_ylabel('Normalized Flux')
            
            # Bottom row gets x-label
            if trans_idx >= n_transitions - (n_transitions % n_cols if n_transitions % n_cols != 0 else n_cols):
                ax.set_xlabel('Velocity (km/s)')
        
        # Overall title
        title = f'Velocity Space - {instrument_name} ({n_transitions} transitions)'
        if show_components:
            title += ' (showing components)'
        figure.suptitle(title, fontsize=12)
        
        # Adjust layout - no spacing between panels, no legends
        figure.tight_layout()
        figure.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)
            
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