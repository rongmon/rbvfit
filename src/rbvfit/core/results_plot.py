"""
Plotting functions for UnifiedResults.

This module provides plotting capabilities for rbvfit 2.0 UnifiedResults objects.
All functions work with the self-contained UnifiedResults architecture.

Key Features:
- No dependencies on fitter objects
- Works with both fresh and loaded results
- Unified treatment of single/multi-instrument cases
- Clean separation of plotting from analysis
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpecFromSubplotSpec

# Optional dependencies with fallbacks
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False




def chain_trace_plot(results, figure=None, show=True, **kwargs):
    """
    Create trace plots for visual convergence assessment.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object (from zeus or emcee)
    save_path : str, optional
        Path to save the trace plot
    n_cols : int, optional
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to display the plot
        
    Returns
    -------
    plt.Figure
        The trace plot figure
    """
    # Get parameter names
    param_names = results.parameter_names
    n_params = len(param_names)

    # Handle sampler-specific chain shape
    chain = results.chain
    if results.sampler_name.lower() == 'emcee':
        # emcee: (n_walkers, n_steps, n_params) → (n_steps, n_walkers, n_params)
        chain = np.transpose(chain, (1, 0, 2))

    n_steps, n_walkers, _ = chain.shape

    # Get plotting options
    n_cols = kwargs.get('n_cols', 2)
    n_rows = (n_params + n_cols - 1) // n_cols
    figsize = kwargs.get('figsize')
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    save_path = kwargs.get('save_path')

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each parameter
    for i in range(n_params):
        ax = axes[i]
        for walker in range(n_walkers):
            ax.plot(chain[:, walker, i], alpha=0.7, linewidth=0.5)

        # Add best-fit line if available
        if hasattr(results, 'best_fit'):
            ax.axhline(results.best_fit[i], color='red', linestyle='--',
                       linewidth=2, alpha=0.8, label='Best fit')

        ax.set_title(param_names[i])
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Add convergence status
        try:
            convergence = results.convergence_diagnostics(verbose=False)
            status = convergence['overall_status']
            status_colors = {
                "GOOD": "green",
                "MARGINAL": "orange",
                "POOR": "red",
                "UNKNOWN": "purple"
            }
            ax.text(0.02, 0.98, status, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10, weight='bold',
                    color=status_colors.get(status, 'black'),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # Overall figure title
    try:
        convergence = results.convergence_diagnostics(verbose=False)
        status = convergence['overall_status']
        status_symbols = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}
        fig.suptitle(
            f'Chain Trace Plots - {status_symbols.get(status, "?")} {status} Convergence\n'
            f'{results.sampler_name} sampler: {n_walkers} walkers × {n_steps} steps',
            fontsize=14, y=0.98)
    except:
        fig.suptitle(
            f'Chain Trace Plots\n{results.sampler_name} sampler: {n_walkers} walkers × {n_steps} steps',
            fontsize=14, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Add guide text
    fig.text(0.02, 0.02,
             'Good traces: stable mixing around best-fit, no trends or jumps\n'
             'Poor traces: trending, stuck walkers, large jumps, non-stationary behavior',
             fontsize=10, style='italic', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved trace plot to {save_path}")
    elif show:
        plt.show()

    return fig



def corner_plot(results,save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Create corner plot of parameter posteriors.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    **kwargs
        Additional arguments passed to corner.corner()
        
    Returns
    -------
    matplotlib.Figure
        Corner plot figure
    """
    if not HAS_CORNER:
        raise ImportError("corner package required for corner plots. Install with: pip install corner")
    
    # Default corner plot settings
    corner_kwargs = {
        'labels': results.parameter_names,
        'truths': results.best_fit,
        'truth_color': 'red',
        'show_titles': True,
        'title_kwargs': {"fontsize": 12},
        'label_kwargs': {"fontsize": 14},
    }
    corner_kwargs.update(kwargs)
    
    fig = corner.corner(results.samples, **corner_kwargs)
    
    # Add overall title
    conv_status = "Unknown"
    try:
        conv_diagnostics = results.convergence_diagnostics(verbose=False)
        conv_status = conv_diagnostics['overall_status']
    except:
        pass
    
    status_symbols = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}

    fig.suptitle(f'Parameter Posterior Distributions - {status_symbols.get(conv_status, "?")} {conv_status} Convergence\n'
                 f'{results.sampler_name} sampler, {results.n_walkers} walkers, {results.n_steps} steps', 
                 fontsize=14, y=0.96)
    
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Leave room at top for suptitle

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved corner plot to {save_path}")
    else:
        plt.show()


    return fig


def correlation_plot(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot parameter correlation matrix heatmap.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.Figure
        Correlation plot figure
    """
    correlation = results.correlation_matrix()
    param_names = results.parameter_names
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add parameter names as labels
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Add correlation values as text
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                         ha="center", va="center", 
                         color="black" if abs(correlation[i, j]) < 0.5 else "white")
    
    ax.set_title("Parameter Correlation Matrix")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved correlation plot to {save_path}")
    else:
        plt.show()
    
    return fig


def velocity_plot(results, instrument_name: str = None, velocity_range: Tuple[float, float] = (-500, 500),
                 y_range: Tuple[float, float] = (0, 1.2), show_components: bool = False, 
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create velocity space plot of absorption line fits organized by systems and instruments.
    
    Layout: Systems (rows) x Instruments (columns)
    - Each row represents a different absorption system (redshift)
    - Each column represents a different instrument
    - Velocity axes are shared across instruments (columns)
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    instrument_name : str, optional
        Specific instrument to plot. If None, plots all instruments.
    velocity_range : tuple
        Velocity range to plot in km/s
    y_range : tuple
        Y-axis range for flux (default: (0, 1.2))
    show_components : bool
        Whether to show individual components
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.Figure
        Velocity plot figure
    """
    if results.config_metadata is None:
        raise ValueError("Model reconstruction required for velocity plots")
    
    # Determine which instruments to plot
    if instrument_name:
        instruments = [instrument_name]
    else:
        instruments = results.instrument_names
    
    # Build list of all individual transitions (system, ion, wavelength combinations)
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
    
    n_transitions = len(all_transitions)
    n_instruments = len(instruments)
    
    # Create subplot grid: transitions (rows) x instruments (columns)
    fig, axes = plt.subplots(n_transitions, n_instruments, 
                            figsize=(8*n_instruments, 3*n_transitions), 
                            sharex='col')  # Share x-axis within columns (instruments)
    
    # Handle single transition or single instrument cases
    if n_transitions == 1 and n_instruments == 1:
        axes = [[axes]]
    elif n_transitions == 1:
        axes = [axes]
    elif n_instruments == 1:
        axes = [[ax] for ax in axes]
    
    # Define colors for components
    component_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    c_kms = 299792.458  # Speed of light in km/s
    
    # Loop over individual transitions (rows) and instruments (columns)
    for trans_idx, transition in enumerate(all_transitions):
        system_z = transition['system_z']
        ion_name = transition['ion_name']
        rest_wavelength = transition['rest_wavelength']
        obs_wavelength = transition['obs_wavelength']
        
        for inst_idx, inst_name in enumerate(instruments):
            ax = axes[trans_idx][inst_idx]
            
            # Get instrument data
            data = results.instrument_data[inst_name]
            wave_data = data['wave']
            flux_data = data['flux']
            error_data = data['error']
            
            # Check if this transition is covered by this instrument
            if not (wave_data.min() <= obs_wavelength <= wave_data.max()):
                print(wave_data.min(),obs_wavelength,wave_data.max())
                # Transition not covered by this instrument
                ax.text(0.5, 0.5, f'{ion_name} {rest_wavelength:.1f}\nnot covered', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                ax.set_xlim(velocity_range)
                ax.set_ylim(y_range)
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
                ax.set_ylim(y_range)
                continue
            
            vel_plot = velocity[vel_mask]
            flux_plot = flux_data[vel_mask]
            error_plot = error_data[vel_mask]
            
            # Get model
            model = results.reconstruct_model(inst_name if results.is_multi_instrument else None)
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
            ax.text(0, y_range[1]*0.95, f'{ion_name}\n{rest_wavelength:.1f}', 
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
            
            # Format subplot
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_range)
            ax.set_xlim(velocity_range)
            
            # Add labels
            if trans_idx == 0:  # Top row - instrument names
                ax.set_title(f'{inst_name}', fontsize=12, fontweight='bold')
            
            if inst_idx == 0:  # Left column - transition info
                ax.set_ylabel(f'z = {system_z:.4f}\n{ion_name} {rest_wavelength:.1f}\nNormalized Flux', fontsize=9)
            
            if trans_idx == n_transitions - 1:  # Bottom row - velocity label
                ax.set_xlabel('Velocity (km/s)')
            
            # Legend only for first subplot to avoid clutter
            if trans_idx == 0 and inst_idx == 0:
                if show_components:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                else:
                    ax.legend(fontsize=8)
    
    # Overall title
    title = f'Velocity Space Fit - {n_transitions} Transition(s) × {n_instruments} Instrument(s)'
    if show_components:
        title += f' (showing components)'
    fig.suptitle(title, fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    if show_components:
        plt.subplots_adjust(right=0.85, top=0.93)
    else:
        plt.subplots_adjust(top=0.93)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved velocity plot to {save_path}")
    else:
        plt.show()
    
    return fig

def residuals_plot(results, instrument_name: str = None, x_range: Optional[Tuple[float, float]] = None,
                  y_range: Optional[Tuple[float, float]] = None, show_components: bool = False,
                  show_residuals: bool = True, show_legends: bool = False,save_path: Optional[str] = None) -> plt.Figure:
    """
    Create residuals plot showing model vs data comparison.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    instrument_name : str, optional
        Specific instrument to plot. If None, plots all instruments.
    x_range : tuple, optional
        Wavelength range to plot (min_wave, max_wave). If None, uses full range.
    y_range : tuple, optional
        Y-axis range for flux (min_flux, max_flux). If None, auto-scales.
    show_components : bool
        Whether to show individual components
    show_residuals : bool
        Whether to show residuals panel (default: True)
    show_legends : bool
        Whether to show legends (default: False)        
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.Figure
        Residuals plot figure
    """
    if results.config_metadata is None:
        raise ValueError("Model reconstruction required for residuals plots")
    
    # Determine which instruments to plot
    if instrument_name:
        instruments = [instrument_name]
    else:
        instruments = results.instrument_names
    
    n_instruments = len(instruments)
    
    # Determine panel configuration
    if show_residuals:
        n_panels = 2
        height_ratios = [3, 1]
        figsize = (10 * n_instruments, 4)
    else:
        n_panels = 1
        height_ratios = [1]
        figsize = (10 * n_instruments, 3)
    
    fig, axes = plt.subplots(
        nrows=n_panels,
        ncols=n_instruments,
        figsize=figsize,
        sharex='col',
        gridspec_kw={'height_ratios': height_ratios}
    )
    
    # Ensure axes is always 2D
    if n_panels == 1 and n_instruments == 1:
        axes = np.array([[axes]])
    elif n_panels == 1:
        axes = axes.reshape(1, n_instruments)
    elif n_instruments == 1:
        axes = axes.reshape(n_panels, 1)
    
    # Define colors for components
    component_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, inst_name in enumerate(instruments):
        ax_data = axes[0, i]  # row 0: data panel
        
        # Get data
        data = results.instrument_data[inst_name]
        wave_data = data['wave']
        flux_data = data['flux']
        error_data = data['error']
        
        # Apply x-range filter if specified
        if x_range is not None:
            wave_mask = (wave_data >= x_range[0]) & (wave_data <= x_range[1])
            if np.sum(wave_mask) == 0:
                print(f"Warning: No data in wavelength range {x_range} for {inst_name}")
                continue
            wave_plot = wave_data[wave_mask]
            flux_plot = flux_data[wave_mask]
            error_plot = error_data[wave_mask]
        else:
            wave_plot = wave_data
            flux_plot = flux_data
            error_plot = error_data
            wave_mask = slice(None)
        
        # Get model
        model = results.reconstruct_model(inst_name if results.is_multi_instrument else None)
        
        # Get model output with components if requested
        if show_components:
            model_output = model.evaluate(results.best_fit, wave_data, return_components=True)
            model_flux = model_output['flux']
            model_components = model_output['components']
            model_comp_info = model_output['component_info']
        else:
            model_flux = model.evaluate(results.best_fit, wave_data)
        
        model_plot = model_flux[wave_mask]
        
        # Data vs model plot
        ax_data.step(wave_plot, flux_plot, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax_data.fill_between(wave_plot, flux_plot - error_plot, flux_plot + error_plot, 
                           alpha=0.3, color='gray', step='mid')
        ax_data.plot(wave_plot, model_plot, 'r-', linewidth=2, label='Model')
        
        # Plot individual components if requested
        if show_components and 'model_components' in locals():
            for j, (component_flux, comp_info) in enumerate(zip(model_components, model_comp_info)):
                component_plot = component_flux[wave_mask]
                color = component_colors[j % len(component_colors)]
                
                # Create label with component info
                lambda0 = comp_info['lambda0']
                z_total = comp_info['z_total']
                v_value = comp_info['v_value']
                label = f'λ{lambda0:.1f} (z={z_total:.4f}, v={v_value:.1f})'
                
                ax_data.plot(wave_plot, component_plot, '--', color=color, linewidth=0.5, 
                           alpha=0.7, label=label)
        
        # Add rail system
        ax_data = _add_rail_system(ax_data, results, wave_plot)
        
        ax_data.set_ylabel('Normalized Flux')
        ax_data.set_title(f'{inst_name} - Data vs Model')
        ax_data.grid(True, alpha=0.3)
        
        # Set y-range if specified
        if y_range is not None:
            ax_data.set_ylim(y_range)
        
        if show_legends:
            # Handle legend - components can make it crowded
            if show_components and 'model_components' in locals() and len(model_components) > 5:
                # If many components, put legend outside plot
                ax_data.legend(bbox_to_anchor=(1.05, 1), loc='lower right')
            else:
                ax_data.legend()
        
        # Residuals plot (only if show_residuals is True)
        if show_residuals:
            ax_res = axes[1, i]  # row 1: residuals panel
            
            residuals = (flux_plot - model_plot) / error_plot
            ax_res.step(wave_plot, residuals, 'k-', where='mid', linewidth=1)
            ax_res.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax_res.axhline(2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
            ax_res.axhline(-2, color='orange', linestyle=':', alpha=0.5)
            ax_res.axhline(3, color='red', linestyle=':', alpha=0.5, label='±3σ')
            ax_res.axhline(-3, color='red', linestyle=':', alpha=0.5)
            
            ax_res.set_ylabel('Residuals/σ')
            if show_legends:
                ax_res.legend()
            ax_res.grid(True, alpha=0.3)
            
            # Set x-range for residuals panel
            if x_range is not None:
                ax_res.set_xlim(x_range)
    
    # Set x-labels
    if show_residuals:
        for i in range(n_instruments):
            axes[1, i].set_xlabel('Wavelength (Å)')
    else:
        for i in range(n_instruments):
            axes[0, i].set_xlabel('Wavelength (Å)')
    
    # Adjust layout
    if show_components and any('model_components' in locals() and len(model_components) > 5 for _ in instruments):
        plt.tight_layout()
        plt.subplots_adjust(right=0.75, hspace=0.05 if show_residuals else 0)
    else:
        plt.tight_layout()
        if show_residuals:
            plt.subplots_adjust(hspace=0.05)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved residuals plot to {save_path}")
    else:
        plt.show()
    
    return fig



def _add_rail_system(ax,results, wave_data):
    """
    Add rail system visualization for v2.0 ion groups.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    best_theta : array-like
        The flat parameter vector (N, b, v for each component).
    wave_data : np.ndarray
        Observed wavelength array (to restrict tick display).
    """
    try:
        all_systems = results.config_metadata['systems']
        best_theta=results.best_fit
        n_comp = len(best_theta) // 3

        N_list = best_theta[0:n_comp]
        b_list = best_theta[n_comp:2*n_comp]
        v_list = best_theta[2*n_comp:3*n_comp]

        # Speed of light in km/s
        c_kms = 299792.458

        # Plotting config
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
        y_rail_base = 1.2
        rail_spacing = 0.1
        tick_len = 0.2

        # Track global component index
        glob_comp_idx = 0
        rail_idx = 0

        N_sys_all, b_sys_all, v_sys_all = [], [], []

        for system in all_systems:
            z = system['redshift']
            N_sys, b_sys, v_sys = [], [], []
            transition_list = []

            for ion_group in system['ion_groups']:
                ion_name = ion_group['ion_name']
                transitions = ion_group['transitions']
                n_components = ion_group['components']

                # Expand transition and logN arrays
                transition_list_group = np.tile(transitions, n_components)

                for _ in range(n_components):
                    N_sys.extend([N_list[glob_comp_idx]] * len(transitions))
                    b_sys.extend([b_list[glob_comp_idx]] * len(transitions))
                    v_sys.extend([v_list[glob_comp_idx]] * len(transitions))
                    glob_comp_idx += 1

                transition_list.extend(transition_list_group)

                # Plot rail line
                y_rail = y_rail_base + rail_idx * rail_spacing
                wave_obs_sys = np.array(transition_list_group) * (1 + z) * \
                               (1 + np.array(v_sys[-len(transition_list_group):]) / c_kms)

                rail_start = wave_obs_sys.min()
                rail_end = wave_obs_sys.max()

                ax.plot([rail_start, rail_end], [y_rail, y_rail], 
                        color='gray', linewidth=2, alpha=0.7)

                # Ion label
                rail_center = (rail_start + rail_end) / 2
                #ax.text(rail_center, y_rail + 0.015, f'{ion_name} z={z:.3f}', 
                #        ha='center', va='bottom', fontsize=10, weight='bold',
                #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
                ax.text(rail_center,
                        y_rail + 0.02,
                        f"{ion_name}\nz = {z:.3f}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        weight='medium',
                        bbox=dict(
                            boxstyle='round,pad=0.25',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.8
                        ),
                        zorder=10
                    )


                # Component ticks and velocity labels
                for comp_local_idx in range(n_components):
                    color = colors[comp_local_idx % len(colors)]
                    v_comp = v_list[glob_comp_idx - n_components + comp_local_idx]
                    
                    for rest_wave in transitions:
                        obs_wave = rest_wave * (1 + z) * (1 + v_comp / c_kms)

                        if wave_data.min() <= obs_wave <= wave_data.max():
                            ax.plot([obs_wave, obs_wave], [y_rail, y_rail - tick_len],
                                    color=color, linewidth=3, alpha=0.8)

                            # Label only once per component
                            if rest_wave == transitions[0]:
                                #ax.text(obs_wave, y_rail + 3.5*tick_len , 
                                #        f'c{comp_local_idx+1}\n{v_comp:.0f} km/s', 
                                #        ha='center', va='top', fontsize=9, 
                                #        color=color, rotation=90)
                                ax.text(obs_wave,
                                        y_rail + 0.25 * tick_len,
                                        f'c{comp_local_idx+1} ({v_comp:.0f} km/s)',
                                        ha='center',
                                        va='bottom',
                                        fontsize=8,
                                        color=color,
                                        rotation=90,
                                        bbox=dict(
                                            boxstyle='round,pad=0.2',
                                            facecolor='white',
                                            edgecolor='none',
                                            alpha=0.7
                                        ),
                                        zorder=10
                                    )


                rail_idx += 1

            # Save N/b/v per system
            N_sys_all.append(np.array(N_sys))
            b_sys_all.append(np.array(b_sys))
            v_sys_all.append(np.array(v_sys))

        # Adjust plot limits after all rails
        current_ylim = ax.get_ylim()
        new_top = max(current_ylim[1], y_rail_base + rail_idx * rail_spacing + 0.08)
        ax.set_ylim(current_ylim[0], new_top)

        print(f"Added rail system for {rail_idx} ion groups.")
        return ax 
    except Exception as e:
        print(f"Warning: Could not add rail system: {e}")


def diagnostic_summary_plot(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create diagnostic summary visualization.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.Figure
        Diagnostic summary figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Parameter values with uncertainties
    ax1 = fig.add_subplot(gs[0, 0])
    param_names = results.parameter_names
    param_values = results.best_fit
    param_errors = (results.bounds_84th - results.bounds_16th) / 2
    
    y_pos = np.arange(len(param_names))
    ax1.errorbar(param_values, y_pos, xerr=param_errors, fmt='o', capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([name.replace('_', ' ') for name in param_names])
    ax1.set_xlabel('Parameter Value')
    ax1.set_title('Best-fit Parameters')
    ax1.grid(True, alpha=0.3)
    
    # 2. Sample distribution (last few parameters)
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(3, len(param_names))
    for i in range(n_show):
        idx = -(i+1)  # Show last few parameters
        ax2.hist(results.samples[:, idx], bins=50, alpha=0.7, 
                label=param_names[idx].replace('_', ' '))
        ax2.axvline(param_values[idx], color=f'C{i}', linestyle='--')
    
    ax2.set_xlabel('Parameter Value')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Parameter Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence info
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    try:
        conv_diag = results.convergence_diagnostics(verbose=False)
        status = conv_diag['overall_status']
        
        info_text = f"""Convergence Assessment
        
Status: {status}
Sampler: {results.sampler_name}
Walkers: {results.n_walkers}
Steps: {results.n_steps}
Parameters: {len(param_names)}

"""
        
        if results.acceptance_fraction is not None:
            info_text += f"Acceptance: {results.acceptance_fraction:.3f}\n"
        
        if results.rhat is not None:
            info_text += f"Max R-hat: {np.max(results.rhat):.3f}\n"
        
        if results.autocorr_time is not None:
            info_text += f"Mean τ: {np.nanmean(results.autocorr_time):.1f}\n"
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, 
                verticalalignment='top', fontfamily='monospace')
    except:
        ax3.text(0.05, 0.95, "Convergence info unavailable", 
                transform=ax3.transAxes, verticalalignment='top')
    
    # 4. Chain traces (subset)
    ax4 = fig.add_subplot(gs[1, :])
    n_show_trace = min(3, len(param_names))
    
    for i in range(n_show_trace):
        # Offset each trace vertically for visibility
        offset = i * 0.1
        for walker in range(min(10, results.n_walkers)):  # Show max 10 walkers
            ax4.plot(results.chain[:, walker, i] + offset, alpha=0.5, linewidth=0.5)
        
        # Add parameter name
        ax4.text(0.02, offset, param_names[i].replace('_', ' '), 
                transform=ax4.get_yaxis_transform())
    
    ax4.set_xlabel('MCMC Step')
    ax4.set_title('Chain Traces (Sample)')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'MCMC Diagnostic Summary - {results.sampler_name} Results', fontsize=16)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved diagnostic summary to {save_path}")
    else:
        plt.show()
    
    return fig