"""
Dedicated plotting module for rbvfit 2.0 FitResults.

This module contains all plotting functionality for fit results analysis,
separated from the core data management in fit_results.py.

Functions are designed to work with FitResults objects and provide
enhanced plotting capabilities with pre-computed model evaluations
and fallback to live model evaluation when needed.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Optional dependencies with fallbacks
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    corner = None


def plot_velocity_fits(results, show_components: bool = True, 
                      show_rail_system: bool = True,
                      figsize_per_panel: Tuple[float, float] = (4, 3),
                      save_path: Optional[str] = None,
                      velocity_range: Optional[Tuple[float, float]] = None,
                      **kwargs) -> Dict[str, plt.Figure]:
    """
    Create velocity space plots for each ion group with multi-instrument support.
    
    This is the main plotting function that creates separate figures for each ion,
    with transitions as rows and instruments as columns.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
    show_components : bool
        Whether to show individual velocity components
    show_rail_system : bool
        Whether to show rail system with component markers
    figsize_per_panel : tuple
        Size of each subplot panel (width, height)
    save_path : str, optional
        Base path for saving figures (will append ion names)
    velocity_range : tuple, optional
        Velocity range (vmin, vmax) in km/s for all plots
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    dict
        Dictionary mapping ion names to their figure objects
    """
    # Detect ions and instruments from model and data
    ion_info = _detect_ions_and_instruments(results)
    
    if not ion_info:
        print("❌ No ion information could be extracted from model")
        # Fall back to simple velocity plot
        return {"simple": _plot_simple_velocity_fallback(results, show_components, show_rail_system, velocity_range, save_path)}
    
    print(f"📊 Creating velocity plots for {len(ion_info)} ion group(s)")
    
    figures = {}
    
    for ion_key, ion_data in ion_info.items():
        print(f"  📈 Plotting {ion_data['ion_name']} at z={ion_data['redshift']:.6f}")
        
        # Create figure for this ion
        fig = _create_ion_velocity_figure(
            results, ion_data, show_components, show_rail_system, 
            figsize_per_panel, velocity_range, **kwargs
        )
        
        figures[ion_key] = fig
        
        # Save individual figure if requested
        if save_path:
            # Parse the save_path to get directory, base name, and extension
            save_path_obj = Path(save_path)
            base_dir = save_path_obj.parent
            base_name = save_path_obj.stem
            file_ext = save_path_obj.suffix or '.jpg'  # Default to .jpg if no extension
            
            # Create filename with ion info
            ion_filename = base_dir / f"{base_name}_{ion_data['ion_name']}_z{ion_data['redshift']:.3f}{file_ext}"
            
            # Save with appropriate DPI for the format
            dpi = 150 if file_ext.lower() in ['.jpg', '.jpeg', '.png'] else 300
            fig.savefig(ion_filename, dpi=dpi, bbox_inches='tight')
            print(f"  ✅ Saved {ion_data['ion_name']} plot to {ion_filename}")
    
    if not save_path:
        plt.show()
    
    return figures


def _detect_ions_and_instruments(results) -> Dict[str, Dict]:
    """
    Detect ion groups and instruments from model configuration and data.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
        
    Returns
    -------
    dict
        Dictionary mapping ion keys to ion information
    """
    ion_info = {}
    
    # Try to get ion information from saved config metadata first
    if hasattr(results, 'config_metadata') and results.config_metadata is not None:
        try:
            if 'systems' in results.config_metadata:
                print("Using saved configuration metadata for ion detection")
                for sys_idx, system in enumerate(results.config_metadata['systems']):
                    for ion_group in system['ion_groups']:
                        ion_key = f"{ion_group['ion']}_z{system['redshift']:.6f}"
                        
                        ion_info[ion_key] = {
                            'ion_name': ion_group['ion'],
                            'redshift': system['redshift'],
                            'transitions': ion_group['transitions'],
                            'components': ion_group['components'],
                            'system_idx': sys_idx
                        }
        except Exception as e:
            print(f"⚠️ Could not extract ion info from saved config metadata: {e}")
    
    # Fall back to live model configuration if available
    if not ion_info and results.model is not None:
        if hasattr(results.model, 'config') and results.model.config is not None:
            try:
                print("Using live model configuration for ion detection")
                for sys_idx, system in enumerate(results.model.config.systems):
                    for ion_group in system.ion_groups:
                        ion_key = f"{ion_group.ion_name}_z{system.redshift:.6f}"
                        
                        ion_info[ion_key] = {
                            'ion_name': ion_group.ion_name,
                            'redshift': system.redshift,
                            'transitions': ion_group.transitions,
                            'components': ion_group.components,
                            'system_idx': sys_idx
                        }
                        
            except Exception as e:
                print(f"⚠️ Could not extract ion info from live model config: {e}")
    
    # Detect instruments
    instruments = ['Primary']
    if results.is_multi_instrument and results.instrument_data:
        instruments = [name for name in results.instrument_data.keys() if name != 'main']
        if 'main' in results.instrument_data:
            instruments = ['Primary'] + [name for name in instruments if name != 'Primary']
    
    # Add instrument info to each ion
    for ion_key in ion_info:
        ion_info[ion_key]['instruments'] = instruments
    
    return ion_info


def _create_ion_velocity_figure(results, ion_data: Dict, show_components: bool,
                               show_rail_system: bool, figsize_per_panel: Tuple[float, float],
                               velocity_range: Optional[Tuple[float, float]], **kwargs) -> plt.Figure:
    """
    Create velocity space figure for a single ion group.
    
    Layout: transitions (rows) × instruments (columns)
    """
    transitions = ion_data['transitions']
    instruments = ion_data['instruments']
    ion_name = ion_data['ion_name']
    redshift = ion_data['redshift']
    
    n_transitions = len(transitions)
    n_instruments = len(instruments)
    
    # Calculate figure size
    fig_width = figsize_per_panel[0] * n_instruments
    fig_height = figsize_per_panel[1] * n_transitions
    
    # Create subplot grid
    fig, axes = plt.subplots(n_transitions, n_instruments, 
                            figsize=(fig_width, fig_height))
    
    # Handle single subplot cases
    if n_transitions == 1 and n_instruments == 1:
        axes = [[axes]]
    elif n_transitions == 1:
        axes = [axes]
    elif n_instruments == 1:
        axes = [[ax] for ax in axes]
    
    # Get model parameters for this ion
    summary = results.parameter_summary(verbose=False)
    ion_params = _extract_ion_parameters(results, ion_data, summary)
    
    # Plot each transition × instrument combination
    for i, transition in enumerate(transitions):
        for j, instrument in enumerate(instruments):
            ax = axes[i][j]
            
            # Get data for this instrument
            if instrument == 'Primary':
                wave_data = results.fitter.wave_obs
                flux_data = results.fitter.fnorm
                error_data = results.fitter.enorm
            else:
                inst_data = results.instrument_data[instrument]
                wave_data = inst_data['wave']
                flux_data = inst_data['flux']
                error_data = inst_data['error']
            
            # Convert to velocity space for this transition
            velocity = _wavelength_to_velocity(wave_data, transition, redshift)
            
            # Plot data and model for this panel
            _plot_velocity_panel(
                results, ax, velocity, flux_data, error_data,
                ion_data, transition, instrument, ion_params,
                show_components, show_rail_system and (i == 0),  # Rail only on top row
                velocity_range, **kwargs
            )
            
            # Panel labeling
            if i == 0:  # Top row
                ax.set_title(f'{instrument}', fontsize=12, weight='bold')
            if j == 0:  # Left column
                ax.set_ylabel(f'{transition:.1f} Å\nNormalized Flux', fontsize=10)
            if i == n_transitions - 1:  # Bottom row
                ax.set_xlabel('Velocity (km/s)', fontsize=10)
    
    # Overall figure title
    convergence = results.convergence_diagnostics(verbose=False)
    status = convergence['overall_status']
    status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}
    
    fig.suptitle(f'{status_symbol.get(status, "?")} {ion_name} at z = {redshift:.6f}\n'
                f'rbvfit 2.0: {ion_data["components"]} component(s), {status} convergence',
                fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room at top for suptitle
    
    return fig


def _extract_ion_parameters(results, ion_data: Dict, summary) -> Dict:
    """Extract parameters for specific ion group."""
    ion_name = ion_data['ion_name']
    redshift = ion_data['redshift']
    components = ion_data['components']
    
    # Find parameters matching this ion
    ion_params = {
        'N': [], 'b': [], 'v': [],
        'N_err': [], 'b_err': [], 'v_err': []
    }
    
    for i, name in enumerate(summary.names):
        # Check if parameter belongs to this ion
        if (ion_name in name and 
            f"z{redshift:.3f}" in name and
            any(f"c{c}" in name for c in range(components))):
            
            if name.startswith('N_'):
                ion_params['N'].append(summary.best_fit[i])
                ion_params['N_err'].append(summary.errors[i])
            elif name.startswith('b_'):
                ion_params['b'].append(summary.best_fit[i])
                ion_params['b_err'].append(summary.errors[i])
            elif name.startswith('v_'):
                ion_params['v'].append(summary.best_fit[i])
                ion_params['v_err'].append(summary.errors[i])
    
    # Convert to arrays and sort by component index
    for key in ion_params:
        ion_params[key] = np.array(ion_params[key])
    
    return ion_params


def _wavelength_to_velocity(wavelength: np.ndarray, rest_wavelength: float, 
                           redshift: float) -> np.ndarray:
    """Convert wavelength to velocity space relative to transition."""
    c_kms = 299792.458  # km/s
    
    # Expected observed wavelength at systemic redshift
    lambda_sys = rest_wavelength * (1 + redshift)
    
    # Convert to velocity relative to systemic
    velocity = c_kms * (wavelength / lambda_sys - 1)
    
    return velocity


def _plot_velocity_panel(results, ax, velocity: np.ndarray, flux: np.ndarray, 
                       error: np.ndarray, ion_data: Dict, transition: float,
                       instrument: str, ion_params: Dict, show_components: bool,
                       show_rail: bool, velocity_range: Optional[Tuple[float, float]], **kwargs):
    """Plot data and model for a single velocity panel."""
    
    # Plot data
    ax.step(velocity, flux, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
    ax.step(velocity, error, 'gray', where='mid', alpha=0.3, linewidth=0.5)
    
    # Plot model - try pre-computed first, fall back to live evaluation
    try:
        # Method 1: Use pre-computed model evaluations if available
        if results.has_model_evaluations():
            instrument_key = 'main' if instrument == 'Primary' else instrument
            if instrument_key in results.model_evaluations:
                # We have pre-computed total model flux
                model_data = results.get_model_flux(instrument_key)
                model_wave = model_data['wave']
                model_flux = model_data['flux']
                
                # Interpolate to velocity grid
                model_flux_interp = np.interp(velocity, 
                                            _wavelength_to_velocity(model_wave, transition, ion_data['redshift']),
                                            model_flux)
                
                ax.plot(velocity, model_flux_interp, 'r-', linewidth=2, label='Best Fit')
                
                # Try to add individual components if available
                if show_components and results.has_component_evaluations():
                    if instrument_key in results.component_evaluations:
                        _add_precomputed_components(results, ax, velocity, transition, ion_data, instrument_key)
            else:
                raise ValueError("Pre-computed model not available for this instrument")
        else:
            raise ValueError("No pre-computed model evaluations")
            
    except Exception:
        # Method 2: Fall back to live model evaluation
        try:
            # Get corresponding wavelength array
            rest_wavelength = transition
            redshift = ion_data['redshift']
            c_kms = 299792.458
            lambda_sys = rest_wavelength * (1 + redshift)
            wavelength = lambda_sys * (1 + velocity / c_kms)
            
            # Evaluate model live
            summary = results.parameter_summary(verbose=False)
            model_flux = results.model.evaluate(summary.best_fit, wavelength)
            
            ax.plot(velocity, model_flux, 'r-', linewidth=2, label='Best Fit')
            
            # Plot individual components if requested
            if show_components and len(ion_params['v']) > 0:
                _add_component_profiles(ax, velocity, wavelength, ion_data, 
                                       transition, ion_params)
        except Exception as e:
            print(f"⚠️ Could not evaluate model for {instrument} {transition:.1f}Å: {e}")
    
    # Add rail system for component positions
    if show_rail and len(ion_params['v']) > 0:
        _add_rail_system(ax, ion_params, velocity_range)
    
    # Format panel
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    if velocity_range:
        ax.set_xlim(velocity_range)
    else:
        # Auto-range around components
        if len(ion_params['v']) > 0:
            v_center = np.mean(ion_params['v'])
            v_range = max(200, np.ptp(ion_params['v']) * 2)
            ax.set_xlim(v_center - v_range, v_center + v_range)
    
    # Add zero velocity reference
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)


def _add_precomputed_components(results, ax, velocity: np.ndarray, transition: float, 
                              ion_data: Dict, instrument_key: str):
    """Add individual components using pre-computed component evaluations."""
    try:
        comp_data = results.get_component_flux(instrument_key)
        if 'components' in comp_data and comp_data['components']:
            components = comp_data['components']
            model_wave = comp_data['wave']
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
            
            for i, comp_flux in enumerate(components):
                # Interpolate component to velocity grid
                comp_flux_interp = np.interp(velocity,
                                           _wavelength_to_velocity(model_wave, transition, ion_data['redshift']),
                                           comp_flux)
                
                ax.plot(velocity, comp_flux_interp, '--', color=colors[i], 
                       linewidth=1.5, alpha=0.7, label=f'Comp {i+1}')
                       
    except Exception as e:
        print(f"⚠️ Could not add pre-computed components: {e}")


def _add_component_profiles(ax, velocity: np.ndarray, wavelength: np.ndarray,
                          ion_data: Dict, transition: float, ion_params: Dict):
    """Add individual component Voigt profiles to plot."""
    try:
        # This would require access to individual component evaluation
        # For now, just mark component positions
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, (v_comp, N_comp, b_comp) in enumerate(zip(
            ion_params['v'], ion_params['N'], ion_params['b']
        )):
            color = colors[i % len(colors)]
            
            # Add vertical line at component velocity
            ax.axvline(v_comp, color=color, linestyle='--', alpha=0.7, 
                      linewidth=2, label=f'Comp {i+1}')
            
            # Add component info text
            if i < 3:  # Only label first 3 components to avoid clutter
                y_pos = 0.9 - i * 0.15
                ax.text(0.02, y_pos, f'C{i+1}: N={N_comp:.2f}, b={b_comp:.0f}', 
                       transform=ax.transAxes, fontsize=8, color=color,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                       
    except Exception as e:
        print(f"⚠️ Could not add component profiles: {e}")


def _add_rail_system(ax, ion_params: Dict, velocity_range: Optional[Tuple[float, float]]):
    """Add rail system showing component velocity positions."""
    if len(ion_params['v']) == 0:
        return
        
    # Rail positioning
    y_rail = 1.05
    rail_height = 0.03
    tick_height = 0.02
    
    # Component colors
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Determine rail extent
    if velocity_range:
        rail_start, rail_end = velocity_range
    else:
        v_min, v_max = np.min(ion_params['v']), np.max(ion_params['v'])
        v_range = max(100, v_max - v_min)
        rail_start = v_min - v_range * 0.2
        rail_end = v_max + v_range * 0.2
    
    # Draw horizontal rail
    ax.plot([rail_start, rail_end], [y_rail, y_rail], 
           color='gray', linewidth=3, alpha=0.7)
    
    # Add component ticks and labels
    for i, (v_comp, v_err) in enumerate(zip(ion_params['v'], ion_params['v_err'])):
        color = colors[i % len(colors)]
        
        # Vertical tick at component position
        ax.plot([v_comp, v_comp], [y_rail - tick_height, y_rail + tick_height],
               color=color, linewidth=3, alpha=0.8)
        
        # Error bar if available
        if v_err > 0:
            ax.plot([v_comp - v_err, v_comp + v_err], [y_rail, y_rail],
                   color=color, linewidth=2, alpha=0.5)
        
        # Component label
        ax.text(v_comp, y_rail + tick_height + 0.01, f'C{i+1}', 
               ha='center', va='bottom', fontsize=9, color=color, weight='bold')
    
    # Adjust y-limits to accommodate rail
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max(current_ylim[1], y_rail + 0.08))


def _plot_simple_velocity_fallback(results, show_components: bool, show_rail_system: bool,
                                  velocity_range: Optional[Tuple[float, float]], save_path: Optional[str]) -> plt.Figure:
    """
    Simple fallback velocity plot when ion configuration is not available.
    """
    print("Using simple velocity plot fallback...")
    
    instruments = results.list_instruments()
    n_instruments = len(instruments)
    
    fig, axes = plt.subplots(n_instruments, 1, figsize=(12, 4*n_instruments), sharex=True)
    if n_instruments == 1:
        axes = [axes]
    
    # Get component velocities from parameter structure
    try:
        summary = results.parameter_summary(verbose=False)
        n_params = len(summary.best_fit)
        if n_params % 3 == 0:
            n_components = n_params // 3
            v_components = summary.best_fit[2*n_components:]
        else:
            v_components = []
    except:
        v_components = []
    
    for i, instrument in enumerate(instruments):
        ax = axes[i]
        
        # Get data
        if instrument == 'main':
            wave_data = results.fitter.wave_obs
            flux_data = results.fitter.fnorm
            error_data = results.fitter.enorm
        elif results.instrument_data and instrument in results.instrument_data:
            inst_data = results.instrument_data[instrument]
            wave_data = inst_data['wave']
            flux_data = inst_data['flux']
            error_data = inst_data['error']
        else:
            continue
        
        # Simple velocity conversion (use absorption minimum as reference)
        min_flux_idx = np.argmin(flux_data)
        lambda_ref = wave_data[min_flux_idx]
        c_kms = 299792.458
        velocity = c_kms * (wave_data / lambda_ref - 1)
        
        # Plot data
        ax.step(velocity, flux_data, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax.fill_between(velocity, flux_data - error_data, flux_data + error_data, 
                       alpha=0.3, color='gray', step='mid')
        
        # Plot model if available
        if results.has_model_evaluations():
            instrument_key = 'main' if instrument == 'Primary' else instrument
            if instrument_key in results.model_evaluations:
                model_data = results.get_model_flux(instrument_key)
                model_flux = model_data['flux']
                ax.plot(velocity, model_flux, 'r-', linewidth=2, label='Model')
        
        # Format
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f"Simple Velocity View ({instrument})")
        
        if velocity_range:
            ax.set_xlim(velocity_range)
        
        if i == 0:
            ax.legend()
    
    axes[-1].set_xlabel('Velocity (km/s)')
    fig.suptitle('Simple Velocity Analysis (Fallback Mode)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        # Respect user's file format choice
        save_path_obj = Path(save_path)
        file_ext = save_path_obj.suffix or '.jpg'  # Default to .jpg if no extension
        
        if save_path_obj.suffix:
            # User provided extension, use as-is
            final_path = save_path
        else:
            # No extension provided, add .jpg
            final_path = f"{save_path}.jpg"
        
        # Save with appropriate DPI
        dpi = 150 if file_ext.lower() in ['.jpg', '.jpeg', '.png'] else 300
        fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved simple velocity plot to {final_path}")
    
    return fig


def corner_plot(results, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Create corner plot of parameter posterior distributions.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
    save_path : str, optional
        Path to save the corner plot
    **kwargs
        Additional arguments passed to corner.corner()
        
    Returns
    -------
    plt.Figure
        The corner plot figure
    """
    if not HAS_CORNER:
        raise ImportError(
            "Corner plots require the 'corner' package. "
            "Install with: pip install corner"
        )
    
    # Get samples and parameter info
    samples = results._get_samples()
    summary = results.parameter_summary(verbose=False)
    
    # Default corner plot arguments
    corner_kwargs = {
        'labels': summary.names,
        'truths': summary.best_fit,
        'show_titles': True,
        'title_fmt': '.3f',
        'quantiles': [0.16, 0.5, 0.84],
        'levels': (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)),
        'plot_density': False,
        'plot_datapoints': True,
        'fill_contours': True,
        'max_n_ticks': 3
    }
    
    # Update with user-provided kwargs
    corner_kwargs.update(kwargs)
    
    # Create corner plot
    fig = corner.corner(samples, **corner_kwargs)
    
    # Add title with convergence status
    convergence = results.convergence_diagnostics(verbose=False)
    status = convergence['overall_status']
    status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗"}
    
    fig.suptitle(f'{status_symbol.get(status, "?")} MCMC Results - {status} Convergence\n'
                 f'{results.sampler_name} sampler, {results.n_walkers} walkers, {results.n_steps} steps', 
                 fontsize=14, y=0.96)
    
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Leave room at top for suptitle
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved corner plot to {save_path}")
    else:
        plt.show()
    
    return fig


def chain_trace_plot(results, save_path: Optional[str] = None, 
                    n_cols: int = 3, figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Create trace plots for visual convergence assessment.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
    save_path : str, optional
        Path to save the trace plot
    n_cols : int, optional
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns
    -------
    plt.Figure
        The trace plot figure
    """
    # Get samples and chain
    try:
        if hasattr(results.fitter.sampler, 'get_chain'):
            chain = results.fitter.sampler.get_chain()
        else:
            # Fallback: create artificial chain from samples
            samples = results._get_samples()
            chain = samples.reshape(results.n_walkers, -1, samples.shape[1])
    except Exception as e:
        print(f"Could not extract chain for trace plots: {e}")
        return None
    
    # Get parameter names
    summary = results.parameter_summary(verbose=False)
    param_names = summary.names
    n_params = len(param_names)
    
    # Calculate subplot layout
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Set figure size
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Normalize axes to a flat list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]        
    # Plot each parameter
    for i in range(n_params):
        ax = axes[i]
        
        # Plot all walkers for this parameter
        for walker in range(results.n_walkers):
            ax.plot(chain[walker, :, i], alpha=0.7, linewidth=0.5)
        
        # Add best-fit line
        ax.axhline(summary.best_fit[i], color='red', linestyle='--', 
                  linewidth=2, alpha=0.8, label='Best fit')
        
        # Format subplot
        ax.set_title(param_names[i])
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add convergence assessment text
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
    
    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    # Overall title
    convergence = results.convergence_diagnostics(verbose=False)
    status = convergence['overall_status']
    status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}
    
    fig.suptitle(f'Chain Trace Plots - {status_symbol.get(status, "?")} {status} Convergence\n'
                f'{results.sampler_name} sampler: {results.n_walkers} walkers × {results.n_steps} steps', 
                fontsize=14, y=0.98)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room at top for suptitle

    
    # Add interpretation guide
    fig.text(0.02, 0.02, 
            'Good traces: stable mixing around best-fit, no trends or jumps\n'
            'Poor traces: trending, stuck walkers, large jumps, non-stationary behavior',
            fontsize=10, style='italic', alpha=0.7,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved trace plot to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_components_breakdown(results, instrument_name: str = None, save_path: str = None) -> plt.Figure:
    """
    Plot component breakdown for detailed analysis.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
    instrument_name : str, optional
        Instrument to plot. If None, uses first available.
    save_path : str, optional
        Save figure to this path
        
    Returns
    -------
    plt.Figure
        Component breakdown figure
    """
    if not results.has_component_evaluations():
        raise ValueError("Component evaluations not available")
    
    # Get component data
    comp_data = results.get_component_flux(instrument_name)
    wave = comp_data['wave']
    components = comp_data['components']
    
    # Get corresponding observed data
    if instrument_name is None or instrument_name == 'main':
        flux_obs = results.fitter.fnorm
        error_obs = results.fitter.enorm
    elif results.instrument_data and instrument_name in results.instrument_data:
        flux_obs = results.instrument_data[instrument_name]['flux']
        error_obs = results.instrument_data[instrument_name]['error']
    else:
        raise ValueError(f"No data found for instrument {instrument_name}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Top panel: data + total model + components
    ax1.step(wave, flux_obs, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
    ax1.fill_between(wave, flux_obs - error_obs, flux_obs + error_obs,
                    alpha=0.3, color='gray', step='mid')
    
    # Total model
    total_flux = results.get_model_flux(instrument_name)['flux']
    ax1.plot(wave, total_flux, 'r-', linewidth=2, label='Total Model')
    
    # Individual components
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    for i, comp_flux in enumerate(components):
        ax1.plot(wave, comp_flux, '--', color=colors[i], 
                linewidth=1.5, alpha=0.7, label=f'Component {i+1}')
    
    ax1.set_ylabel('Normalized Flux')
    ax1.set_ylim(0, 1.2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: residuals
    residuals = flux_obs - total_flux
    ax2.step(wave, residuals, 'k-', where='mid', linewidth=1)
    ax2.axhline(0, color='r', linestyle='--', alpha=0.7)
    ax2.fill_between(wave, -error_obs, error_obs, alpha=0.3, color='gray', step='mid')
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Title
    title = "Component Breakdown"
    if instrument_name:
        title += f" ({instrument_name})"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Component breakdown saved to {save_path}")
    
    return fig


def plot_model_comparison(results, save_path: Optional[str] = None, 
                         show_residuals: bool = True, **kwargs) -> plt.Figure:
    """
    Plot model vs data comparison for all instruments.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
    save_path : str, optional
        Save figure to this path
    show_residuals : bool
        Whether to show residuals subplot
    **kwargs
        Additional plotting parameters
        
    Returns
    -------
    plt.Figure
        Model comparison figure
    """
    instruments = results.list_instruments()
    n_instruments = len(instruments)
    
    # Create figure layout
    if show_residuals:
        fig, axes = plt.subplots(2 * n_instruments, 1, figsize=(12, 4 * n_instruments),
                                gridspec_kw={'height_ratios': [3, 1] * n_instruments})
    else:
        fig, axes = plt.subplots(n_instruments, 1, figsize=(12, 4 * n_instruments))
    
    if n_instruments == 1:
        axes = [axes] if not show_residuals else axes
    
    for i, instrument in enumerate(instruments):
        # Get data
        if instrument == 'main':
            wave_data = results.fitter.wave_obs
            flux_data = results.fitter.fnorm
            error_data = results.fitter.enorm
        elif results.instrument_data and instrument in results.instrument_data:
            inst_data = results.instrument_data[instrument]
            wave_data = inst_data['wave']
            flux_data = inst_data['flux']
            error_data = inst_data['error']
        else:
            continue
        
        # Main plot
        ax_main = axes[2*i] if show_residuals else axes[i]
        
        # Plot data
        ax_main.step(wave_data, flux_data, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
        ax_main.fill_between(wave_data, flux_data - error_data, flux_data + error_data,
                           alpha=0.3, color='gray', step='mid')
        
        # Plot model if available
        if results.has_model_evaluations():
            instrument_key = 'main' if instrument == 'Primary' else instrument
            if instrument_key in results.model_evaluations:
                model_data = results.get_model_flux(instrument_key)
                model_flux = model_data['flux']
                ax_main.plot(wave_data, model_flux, 'r-', linewidth=2, label='Model')
        
        # Format main plot
        ax_main.set_ylabel('Normalized Flux')
        ax_main.set_ylim(0, 1.2)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        ax_main.set_title(f"{instrument}")
        
        # Residuals plot
        if show_residuals:
            ax_resid = axes[2*i + 1]
            
            if results.has_model_evaluations():
                instrument_key = 'main' if instrument == 'Primary' else instrument
                if instrument_key in results.model_evaluations:
                    model_data = results.get_model_flux(instrument_key)
                    model_flux = model_data['flux']
                    residuals = flux_data - model_flux
                    
                    ax_resid.step(wave_data, residuals, 'k-', where='mid', linewidth=1)
                    ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
                    ax_resid.fill_between(wave_data, -error_data, error_data, 
                                        alpha=0.3, color='gray', step='mid')
            
            ax_resid.set_ylabel('Residuals')
            ax_resid.grid(True, alpha=0.3)
            
            # Only add x-label to bottom residual plot
            if i == n_instruments - 1:
                ax_resid.set_xlabel('Wavelength (Å)')
        else:
            # Only add x-label to bottom main plot
            if i == n_instruments - 1:
                ax_main.set_xlabel('Wavelength (Å)')
    
    # Overall title
    convergence = results.convergence_diagnostics(verbose=False)
    status = convergence['overall_status']
    status_symbol = {"GOOD": "✓", "MARGINAL": "⚠", "POOR": "✗", "UNKNOWN": "?"}
    
    fig.suptitle(f'{status_symbol.get(status, "?")} Model vs Data Comparison - {status} Convergence\n'
                f'rbvfit 2.0: {results.sampler_name} sampler', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {save_path}")
    
    return fig


def plot_correlation_matrix(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot parameter correlation matrix heatmap.
    
    Parameters
    ----------
    results : FitResults
        The fit results object
    save_path : str, optional
        Path to save correlation plot
        
    Returns
    -------
    plt.Figure
        Correlation matrix figure
    """
    correlation = results.correlation_matrix(plot=False)
    summary = results.parameter_summary(verbose=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    n_params = len(summary.names)
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(summary.names, rotation=45, ha='right')
    ax.set_yticklabels(summary.names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            if abs(correlation[i, j]) > 0.3:  # Only show significant correlations
                text = ax.text(j, i, f'{correlation[i, j]:.2f}', 
                             ha="center", va="center", 
                             color="white" if abs(correlation[i, j]) > 0.7 else "black",
                             fontsize=8)
    
    ax.set_title('Parameter Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation plot to {save_path}")
    else:
        plt.show()
    
    return fig