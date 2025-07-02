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
        status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}
        fig.suptitle(
            f'Chain Trace Plots - {status_symbol.get(status, "?")} {status} Convergence\n'
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



def corner_plot(results, **kwargs) -> plt.Figure:
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
    
    status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}

    fig.suptitle(f'Parameter Posterior Distributions - {status_symbols.get(conv_status, "?")} {conv_status} Convergence\n'
                 f'{results.sampler_name} sampler, {results.n_walkers} walkers, {results.n_steps} steps', 
                 fontsize=14, y=0.96)
    
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Leave room at top for suptitle
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
                 show_components: bool = False, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create velocity space plot of absorption line fits.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    instrument_name : str, optional
        Specific instrument to plot. If None, plots all instruments.
    velocity_range : tuple
        Velocity range to plot in km/s
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
    
    n_instruments = len(instruments)
    fig, axes = plt.subplots(n_instruments, 1, figsize=(12, 4*n_instruments), sharex=True)
    if n_instruments == 1:
        axes = [axes]
    
    for i, inst_name in enumerate(instruments):
        ax = axes[i]
        
        # Get data
        data = results.instrument_data[inst_name]
        wave_data = data['wave']
        flux_data = data['flux']
        error_data = data['error']
        
        # Get model
        model = results.reconstruct_model(inst_name if results.is_multi_instrument else None)
        model_flux = model.evaluate(results.best_fit, wave_data)
        
        # Convert to velocity space (use absorption minimum as reference)
        min_flux_idx = np.argmin(flux_data)
        lambda_ref = wave_data[min_flux_idx]
        c_kms = 299792.458
        velocity = c_kms * (wave_data / lambda_ref - 1)
        
        # Filter to velocity range
        vel_mask = (velocity >= velocity_range[0]) & (velocity <= velocity_range[1])
        if np.sum(vel_mask) == 0:
            print(f"Warning: No data in velocity range {velocity_range} for {inst_name}")
            continue
        
        vel_plot = velocity[vel_mask]
        flux_plot = flux_data[vel_mask]
        error_plot = error_data[vel_mask]
        model_plot = model_flux[vel_mask]
        
        # Plot data
        ax.step(vel_plot, flux_plot, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax.fill_between(vel_plot, flux_plot - error_plot, flux_plot + error_plot, 
                       alpha=0.3, color='gray', step='mid', label='1σ error')
        
        # Plot model
        ax.plot(vel_plot, model_plot, 'r-', linewidth=2, label='Best-fit model')
        
        # Add components if requested
        if show_components:
            try:
                # This would require component evaluation capability
                # For now, just plot the total model
                pass
            except:
                pass
        
        # Format
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'{inst_name}')
        ax.legend()
        
        # Add residuals in small panel
        if i == len(instruments) - 1:  # Only for last subplot
            divider = GridSpecFromSubplotSpec(
                2, 1, axes[i].get_subplotspec(), height_ratios=[3, 1], hspace=0)
            ax.set_subplotspec(divider[0])
            ax_res = fig.add_subplot(divider[1], sharex=ax)
            
            residuals = (flux_plot - model_plot) / error_plot
            ax_res.step(vel_plot, residuals, 'k-', where='mid', linewidth=1)
            ax_res.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax_res.set_ylabel('Residuals/σ')
            ax_res.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Velocity (km/s)')
    
    # Overall title
    fig.suptitle(f'Velocity Space Fit - {len(instruments)} Instrument(s)', fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved velocity plot to {save_path}")
    else:
        plt.show()
    
    return fig


def residuals_plot(results, instrument_name: str = None, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create residuals plot showing model vs data comparison.
    
    Parameters
    ----------
    results : UnifiedResults
        The fit results object
    instrument_name : str, optional
        Specific instrument to plot. If None, plots all instruments.
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
    fig, axes = plt.subplots(n_instruments, 2, figsize=(15, 4*n_instruments))
    if n_instruments == 1:
        axes = axes.reshape(1, -1)
    
    for i, inst_name in enumerate(instruments):
        ax_data = axes[i, 0]
        ax_res = axes[i, 1]
        
        # Get data
        data = results.instrument_data[inst_name]
        wave_data = data['wave']
        flux_data = data['flux']
        error_data = data['error']
        
        # Get model
        model = results.reconstruct_model(inst_name if results.is_multi_instrument else None)
        model_flux = model.evaluate(results.best_fit, wave_data)
        
        # Data vs model plot
        ax_data.step(wave_data, flux_data, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax_data.fill_between(wave_data, flux_data - error_data, flux_data + error_data, 
                           alpha=0.3, color='gray', step='mid')
        ax_data.plot(wave_data, model_flux, 'r-', linewidth=2, label='Model')
        
        ax_data.set_ylabel('Normalized Flux')
        ax_data.set_title(f'{inst_name} - Data vs Model')
        ax_data.legend()
        ax_data.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = (flux_data - model_flux) / error_data
        ax_res.step(wave_data, residuals, 'k-', where='mid', linewidth=1)
        ax_res.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax_res.axhline(2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
        ax_res.axhline(-2, color='orange', linestyle=':', alpha=0.5)
        ax_res.axhline(3, color='red', linestyle=':', alpha=0.5, label='±3σ')
        ax_res.axhline(-3, color='red', linestyle=':', alpha=0.5)
        
        ax_res.set_ylabel('Residuals/σ')
        ax_res.set_title(f'{inst_name} - Residuals')
        ax_res.legend()
        ax_res.grid(True, alpha=0.3)
        
        # Add chi-squared info
        try:
            chi2_stats = results.chi_squared(inst_name)
            chi2_reduced = chi2_stats[f'reduced_chi2_{inst_name}']
            ax_res.text(0.02, 0.98, f'χ²/ν = {chi2_reduced:.2f}', 
                       transform=ax_res.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    for ax in axes.flat:
        ax.set_xlabel('Wavelength (Å)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved residuals plot to {save_path}")
    else:
        plt.show()
    
    return fig


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