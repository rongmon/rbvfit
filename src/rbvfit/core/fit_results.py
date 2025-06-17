"""
Comprehensive results management for rbvfit 2.0 with ion-aware analysis.

This module provides the FitResults class with enhanced analysis capabilities,
ion-specific parameter organization, multi-instrument support, correlation analysis
with parameter tying awareness, and publication-ready plotting methods.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import h5py
import json
from scipy.stats import chi2

# Core rbvfit 2.0 imports
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager, ParameterBounds
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.voigt_fitter import Dataset, MCMCSettings

# Try to import corner for plotting
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    corner = None


@dataclass
class ParameterSummary:
    """
    Summary statistics for a single parameter with ion-aware metadata.
    
    Attributes
    ----------
    name : str
        Parameter name (e.g., 'N_MgII_z0.348_c0')
    best_fit : float
        Best-fit value (median of posterior)
    lower_error : float
        Lower 1-sigma error
    upper_error : float
        Upper 1-sigma error
    percentile_16 : float
        16th percentile value
    percentile_84 : float
        84th percentile value
    mean : float
        Mean of posterior
    std : float
        Standard deviation of posterior
    ion_name : str, optional
        Ion species (e.g., 'MgII')
    redshift : float, optional
        System redshift
    component_idx : int, optional
        Component index within ion group
    param_type : str, optional
        Parameter type ('N', 'b', 'v')
    units : str, optional
        Physical units
    """
    name: str
    best_fit: float
    lower_error: float
    upper_error: float
    percentile_16: float
    percentile_84: float
    mean: float
    std: float
    ion_name: Optional[str] = None
    redshift: Optional[float] = None
    component_idx: Optional[int] = None
    param_type: Optional[str] = None
    units: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation for display."""
        unit_str = f" {self.units}" if self.units else ""
        return f"{self.name}: {self.best_fit:.3f} +{self.upper_error:.3f} -{self.lower_error:.3f}{unit_str}"
    
    def latex_string(self) -> str:
        """LaTeX representation for publication."""
        unit_str = f"\\,\\mathrm{{{self.units}}}" if self.units else ""
        return f"{self.name} = {self.best_fit:.3f}_{{-{self.lower_error:.3f}}}^{{+{self.upper_error:.3f}}}{unit_str}"
    
    def physical_interpretation(self) -> str:
        """Get physical interpretation of parameter value."""
        if self.param_type == 'N':
            linear_N = 10**self.best_fit
            return f"Linear column density: {linear_N:.2e} cm^-2"
        elif self.param_type == 'b':
            thermal_b_oi = 12.9  # km/s for OI at 10^4 K
            thermal_b_mgii = 6.1  # km/s for MgII at 10^4 K
            if 'OI' in str(self.ion_name):
                thermal_component = thermal_b_oi
            elif 'MgII' in str(self.ion_name):
                thermal_component = thermal_b_mgii
            else:
                thermal_component = 10.0  # Generic estimate
            
            if self.best_fit > thermal_component:
                turbulent = np.sqrt(self.best_fit**2 - thermal_component**2)
                return f"Thermal: ~{thermal_component:.1f} km/s, Turbulent: ~{turbulent:.1f} km/s"
            else:
                return f"Dominated by thermal motion (~{thermal_component:.1f} km/s)"
        elif self.param_type == 'v':
            return f"Velocity offset from systemic redshift"
        else:
            return ""


@dataclass
class DerivedQuantity:
    """
    Container for derived physical quantities.
    
    Attributes
    ----------
    name : str
        Quantity name
    value : float
        Best-fit value
    error : float
        Uncertainty
    units : str
        Physical units
    description : str
        Physical meaning
    """
    name: str
    value: float
    error: float
    units: str
    description: str
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value:.3e} ± {self.error:.3e} {self.units}"


@dataclass
class IonGroupResults:
    """
    Results for a specific ion group (tied parameters).
    
    Attributes
    ----------
    ion_name : str
        Ion species
    redshift : float
        System redshift
    transitions : List[float]
        Rest wavelengths included
    components : int
        Number of velocity components
    parameters : Dict[str, ParameterSummary]
        Parameter summaries by type
    derived_quantities : List[DerivedQuantity]
        Derived physical quantities
    """
    ion_name: str
    redshift: float
    transitions: List[float]
    components: int
    parameters: Dict[str, ParameterSummary]
    derived_quantities: List[DerivedQuantity]
    
    def get_total_column_density(self) -> DerivedQuantity:
        """Calculate total column density for this ion."""
        N_params = [p for p in self.parameters.values() if p.param_type == 'N']
        total_linear = sum(10**p.best_fit for p in N_params)
        total_log = np.log10(total_linear)
        
        # Error propagation (simplified)
        error_linear = np.sqrt(sum((10**p.best_fit * np.log(10) * p.std)**2 for p in N_params))
        error_log = error_linear / (total_linear * np.log(10))
        
        return DerivedQuantity(
            name=f"N_total_{self.ion_name}",
            value=total_log,
            error=error_log,
            units="log cm^-2",
            description=f"Total {self.ion_name} column density"
        )


class FitResults:
    """
    Enhanced container for single-instrument MCMC fitting results with ion-aware analysis.
    
    This class provides detailed analysis capabilities including parameter
    summaries, correlation analysis, model comparison metrics, ion-specific
    organization, and publication-ready plotting methods.
    
    Attributes
    ----------
    fitter : vfit
        The MCMC fitter object (contains sampler, data, bounds, etc.)
    model : VoigtModel
        The fitted model with configuration
    dataset : Dataset, optional
        Override dataset for visualization (if None, uses fitter's data)
    param_manager : ParameterManager
        Parameter management object (created from model.config)
    ion_groups : Dict[Tuple[float, str], IonGroupResults]
        Results organized by ion groups
    """
    
    def __init__(self, fitter, model: VoigtModel, dataset: Optional[Dataset] = None):
        """
        Initialize results container from fitter and model.
        
        Parameters
        ----------
        fitter : vfit
            MCMC fitter object after running fit
        model : VoigtModel
            The model object with configuration
        dataset : Dataset, optional
            Override dataset for visualization. If None, extracts from fitter.
        """
        self.fitter = fitter
        self.model = model
        
        # Create parameter manager from model configuration
        self.param_manager = ParameterManager(model.config)
        
        # Set up datasets
        if dataset is not None:
            # Use provided dataset for visualization
            self.datasets = [dataset]
        else:
            # Extract from fitter
            self.datasets = [Dataset(
                fitter.wave_obs, fitter.fnorm, fitter.enorm, 
                name="Fitted Data"
            )]
        
        # Extract key information from fitter
        self.sampler = fitter.sampler
        self.fit_time = getattr(fitter, 'fit_time', 0.0)
        
        # Simplify bounds storage
        self.bounds = {
            'lower': getattr(fitter, 'lb', None),
            'upper': getattr(fitter, 'ub', None)
        }
        
        # Auto-detect MCMC settings from fitter
        self.mcmc_settings = self._extract_mcmc_settings()
        
        # Extract and analyze results
        self._extract_samples()
        self._calculate_parameter_summaries()
        self._organize_ion_group_results()
        self._calculate_model_statistics()
        self._calculate_derived_quantities()
        
        # Cache for expensive operations
        self._correlation_matrix = None
        self._covariance_matrix = None
        self._ion_correlation_matrices = None
    
    def _extract_mcmc_settings(self):
        """Extract MCMC settings from fitter object."""
        # Simple object to hold settings
        class MCMCSettings:
            def __init__(self, sampler, n_walkers, n_steps, n_burn, thin=1):
                self.sampler = sampler
                self.n_walkers = n_walkers
                self.n_steps = n_steps
                self.n_burn = n_burn
                self.thin = thin
        
        # Extract from fitter attributes
        sampler_name = getattr(self.fitter, 'sampler_name', 'emcee')
        n_walkers = getattr(self.fitter, 'no_of_Chain', 30)
        n_steps = getattr(self.fitter, 'no_of_steps', 1000)
        
        # Estimate burn-in (fitter might have this, or use 20% default)
        n_burn = getattr(self.fitter, 'burnin', int(0.2 * n_steps))
        
        return MCMCSettings(sampler_name, n_walkers, n_steps, n_burn)
        
    def _extract_samples(self):
        """Extract samples from the sampler."""
        # Get samples (discard burn-in)
        if hasattr(self.sampler, 'get_chain'):  # emcee
            self.samples = self.sampler.get_chain(
                discard=self.mcmc_settings.n_burn,
                thin=self.mcmc_settings.thin,
                flat=True
            )
            self.chain = self.sampler.get_chain()
        else:  # zeus
            chain = self.sampler.get_chain(flat=True)
            n_total = len(chain)
            start_idx = self.mcmc_settings.n_burn * self.mcmc_settings.n_walkers
            self.samples = chain[start_idx::self.mcmc_settings.thin]
            self.chain = self.sampler.get_chain()
            
        self.n_samples = len(self.samples)
        self.n_params = self.samples.shape[1]
        
    def _calculate_parameter_summaries(self):
        """Calculate detailed parameter summaries with ion-aware metadata."""
        param_names = self.param_manager.get_parameter_names()
        
        self.parameter_summaries = []
        
        for i, name in enumerate(param_names):
            samples_i = self.samples[:, i]
            
            # Calculate percentiles
            p16, p50, p84 = np.percentile(samples_i, [16, 50, 84])
            
            # Parse parameter metadata from name
            ion_name, redshift, component_idx, param_type, units = self._parse_parameter_name(name)
            
            summary = ParameterSummary(
                name=name,
                best_fit=p50,
                lower_error=p50 - p16,
                upper_error=p84 - p50,
                percentile_16=p16,
                percentile_84=p84,
                mean=np.mean(samples_i),
                std=np.std(samples_i),
                ion_name=ion_name,
                redshift=redshift,
                component_idx=component_idx,
                param_type=param_type,
                units=units
            )
            
            self.parameter_summaries.append(summary)
        
        # Convenience arrays
        self.best_fit = np.array([s.best_fit for s in self.parameter_summaries])
        self.uncertainties = np.array([s.std for s in self.parameter_summaries])
        self.lower_errors = np.array([s.lower_error for s in self.parameter_summaries])
        self.upper_errors = np.array([s.upper_error for s in self.parameter_summaries])
    
    def _parse_parameter_name(self, name: str) -> Tuple[str, float, int, str, str]:
        """
        Parse parameter name to extract metadata.
        
        Parameters
        ----------
        name : str
            Parameter name like 'N_MgII_z0.348_c0'
            
        Returns
        -------
        tuple
            (ion_name, redshift, component_idx, param_type, units)
        """
        try:
            # Expected format: param_type_ion_zredshift_ccomponent
            parts = name.split('_')
            param_type = parts[0]  # N, b, or v
            ion_name = parts[1]    # MgII, FeII, etc.
            z_part = parts[2]      # z0.348
            c_part = parts[3]      # c0
            
            redshift = float(z_part[1:])  # Remove 'z' prefix
            component_idx = int(c_part[1:])  # Remove 'c' prefix
            
            # Determine units
            if param_type == 'N':
                units = 'log cm^-2'
            elif param_type == 'b':
                units = 'km/s'
            elif param_type == 'v':
                units = 'km/s'
            else:
                units = ''
            
            return ion_name, redshift, component_idx, param_type, units
            
        except (IndexError, ValueError):
            # Fallback for non-standard naming
            return None, None, None, None, ''
    
    def _organize_ion_group_results(self):
        """Organize results by ion groups for easier analysis."""
        self.ion_groups = {}
        
        # Group parameters by (redshift, ion_name)
        param_groups = {}
        for summary in self.parameter_summaries:
            if summary.ion_name and summary.redshift is not None:
                key = (summary.redshift, summary.ion_name)
                if key not in param_groups:
                    param_groups[key] = []
                param_groups[key].append(summary)
        
        # Create IonGroupResults for each group
        for (redshift, ion_name), params in param_groups.items():
            # Get transitions for this ion group from model config
            transitions = []
            components = 0
            for system in self.model.config.systems:
                if abs(system.redshift - redshift) < 1e-6:
                    for ion_group in system.ion_groups:
                        if ion_group.ion_name == ion_name:
                            transitions = ion_group.transitions
                            components = ion_group.components
                            break
            
            # Organize parameters by type
            param_dict = {}
            for param in params:
                param_dict[f"{param.param_type}_{param.component_idx}"] = param
            
            self.ion_groups[key] = IonGroupResults(
                ion_name=ion_name,
                redshift=redshift,
                transitions=transitions,
                components=components,
                parameters=param_dict,
                derived_quantities=[]
            )
    
    def _calculate_derived_quantities(self):
        """Calculate derived physical quantities for each ion group."""
        for ion_group in self.ion_groups.values():
            # Total column density
            total_N = ion_group.get_total_column_density()
            ion_group.derived_quantities.append(total_N)
            
            # Velocity dispersion
            v_params = [p for p in ion_group.parameters.values() if p.param_type == 'v']
            if len(v_params) > 1:
                velocities = [p.best_fit for p in v_params]
                v_dispersion = np.std(velocities)
                v_range = max(velocities) - min(velocities)
                
                ion_group.derived_quantities.extend([
                    DerivedQuantity(
                        name=f"v_dispersion_{ion_group.ion_name}",
                        value=v_dispersion,
                        error=0.0,  # TODO: Proper error propagation
                        units="km/s",
                        description=f"{ion_group.ion_name} velocity dispersion"
                    ),
                    DerivedQuantity(
                        name=f"v_range_{ion_group.ion_name}",
                        value=v_range,
                        error=0.0,
                        units="km/s", 
                        description=f"{ion_group.ion_name} velocity range"
                    )
                ])
    
    def _calculate_model_statistics(self):
        """Calculate model comparison statistics with ion tying corrections."""
        # Calculate chi-squared for best fit using the fitted data (from fitter)
        self.chi2_best = 0.0
        self.n_data_points = 0
        
        # Use the data that was actually fitted (from fitter object)
        fitted_dataset = Dataset(
            self.fitter.wave_obs, self.fitter.fnorm, self.fitter.enorm, 
            name="Fitted Data"
        )
        
        # Evaluate model on fitted data
        model_flux = self.model.evaluate(self.best_fit, fitted_dataset.wavelength)
        chi2_dataset = np.sum((fitted_dataset.flux - model_flux)**2 / fitted_dataset.error**2)
        self.chi2_best += chi2_dataset
        self.n_data_points += len(fitted_dataset.wavelength)
        
        # Handle multi-instrument case if present
        if hasattr(self.fitter, 'instrument_data') and self.fitter.instrument_data:
            for instrument_name, data in self.fitter.instrument_data.items():
                inst_dataset = Dataset(data['wave'], data['flux'], data['error'], name=instrument_name)
                
                # Need to evaluate model for this specific instrument
                if hasattr(data, 'model'):
                    model_flux = data['model'](self.best_fit, inst_dataset.wavelength)
                else:
                    # Fallback: use main model
                    model_flux = self.model.evaluate(self.best_fit, inst_dataset.wavelength)
                
                chi2_dataset = np.sum((inst_dataset.flux - model_flux)**2 / inst_dataset.error**2)
                self.chi2_best += chi2_dataset
                self.n_data_points += len(inst_dataset.wavelength)
        
        # Rest of statistics calculation remains the same
        self.dof = self.n_data_points - self.n_params
        self.reduced_chi2 = self.chi2_best / self.dof if self.dof > 0 else np.inf
        
        # Calculate effective number of parameters accounting for ion tying
        n_transitions = sum(
            len(ion_group.transitions) 
            for ion_group in self.ion_groups.values()
        )
        n_untied_params = n_transitions * 3  # If all transitions were independent
        tying_reduction = n_untied_params - self.n_params
        
        # Store tying information
        self.n_transitions = n_transitions
        self.n_untied_params = n_untied_params
        self.tying_reduction = tying_reduction
        self.tying_efficiency = tying_reduction / n_untied_params if n_untied_params > 0 else 0
        
        # Calculate AIC and BIC with ion tying awareness
        self.aic = self.chi2_best + 2 * self.n_params
        self.bic = self.chi2_best + self.n_params * np.log(self.n_data_points)
        
        # Alternative AIC/BIC using effective parameter count
        self.aic_effective = self.chi2_best + 2 * self.n_params * (1 + self.tying_efficiency)
        self.bic_effective = self.chi2_best + self.n_params * np.log(self.n_data_points) * (1 + self.tying_efficiency)
        
        # P-value from chi-squared distribution
        if self.dof > 0:
            self.p_value = 1.0 - chi2.cdf(self.chi2_best, self.dof)
        else:
            self.p_value = np.nan


    def get_correlation_matrix(self) -> np.ndarray:
        """
        Calculate parameter correlation matrix with caching.
        
        Returns
        -------
        np.ndarray
            Correlation matrix (n_params x n_params)
        """
        if self._correlation_matrix is None:
            self._correlation_matrix = np.corrcoef(self.samples.T)
        return self._correlation_matrix
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Calculate parameter covariance matrix with caching.
        
        Returns
        -------
        np.ndarray
            Covariance matrix (n_params x n_params)
        """
        if self._covariance_matrix is None:
            self._covariance_matrix = np.cov(self.samples.T)
        return self._covariance_matrix
    
    def summary(self, verbose: bool = False, show_tying: bool = True) -> str:
        """
        Generate a comprehensive summary of fit results with ion-aware organization.
        
        Parameters
        ----------
        verbose : bool
            Whether to include detailed parameter information
        show_tying : bool
            Whether to show ion parameter tying information
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = ["MCMC Fit Results", "=" * 60]
        
        # Basic fit information
        lines.append(f"Sampler: {self.mcmc_settings.sampler}")
        lines.append(f"Walkers: {self.mcmc_settings.n_walkers}")
        lines.append(f"Steps: {self.mcmc_settings.n_steps}")
        lines.append(f"Burn-in: {self.mcmc_settings.n_burn}")
        lines.append(f"Samples: {self.n_samples}")
        lines.append(f"Fit time: {self.fit_time:.1f} seconds")
        lines.append(f"Datasets: {len(self.datasets)}")
        
        # Ion tying information
        if show_tying and hasattr(self, 'n_transitions'):
            lines.append(f"\nIon Parameter Tying:")
            lines.append(f"Total transitions: {self.n_transitions}")
            lines.append(f"Fitted parameters: {self.n_params}")
            lines.append(f"Untied parameters: {self.n_untied_params}")
            lines.append(f"Parameter reduction: {self.tying_reduction} ({self.tying_efficiency*100:.1f}%)")
        
        # Model statistics
        lines.append(f"\nModel Statistics:")
        lines.append(f"Chi-squared: {self.chi2_best:.2f}")
        lines.append(f"Degrees of freedom: {self.dof}")
        lines.append(f"Reduced chi-squared: {self.reduced_chi2:.3f}")
        lines.append(f"AIC: {self.aic:.2f}")
        lines.append(f"BIC: {self.bic:.2f}")
        if hasattr(self, 'aic_effective'):
            lines.append(f"AIC (tying-aware): {self.aic_effective:.2f}")
            lines.append(f"BIC (tying-aware): {self.bic_effective:.2f}")
        lines.append(f"P-value: {self.p_value:.4f}")
        
        # Data points information
        lines.append(f"\nData Information:")
        for i, dataset in enumerate(self.datasets):
            lines.append(f"  Dataset {i+1} ({dataset.name}): {len(dataset.wavelength)} points")
        
        # Ion group summaries
        if hasattr(self, 'ion_groups') and self.ion_groups:
            lines.append(f"\nIon Groups Summary:")
            for key, ion_group in self.ion_groups.items():
                redshift, ion_name = key
                lines.append(f"  {ion_name} at z={redshift:.6f}: {ion_group.components} components, "
                            f"{len(ion_group.transitions)} transitions")
                
                # Show derived quantities
                for derived in ion_group.derived_quantities:
                    lines.append(f"    {derived}")
        
        # Parameter summary
        if verbose:
            lines.append(f"\nDetailed Parameter Summary:")
            lines.append("-" * 50)
            for summary in self.parameter_summaries:
                lines.append(str(summary))
                if hasattr(summary, 'physical_interpretation') and summary.physical_interpretation():
                    lines.append(f"    → {summary.physical_interpretation()}")
        else:
            lines.append(f"\nParameter Summary by Ion:")
            lines.append("-" * 40)
            
            if hasattr(self, 'ion_groups') and self.ion_groups:
                for key, ion_group in self.ion_groups.items():
                    redshift, ion_name = key
                    lines.append(f"\n{ion_name} at z={redshift:.6f}:")
                    
                    for comp_idx in range(ion_group.components):
                        lines.append(f"  Component {comp_idx+1}:")
                        
                        # Find parameters for this component
                        for param_type in ['N', 'b', 'v']:
                            param_key = f"{param_type}_{comp_idx}"
                            if param_key in ion_group.parameters:
                                param = ion_group.parameters[param_key]
                                lines.append(f"    {param_type} = {param.best_fit:.3f} ± {param.std:.3f} {param.units or ''}")
            else:
                # Fallback if no ion groups
                lines.append("  Basic parameter summary:")
                for summary in self.parameter_summaries[:6]:  # Show first 6
                    lines.append(f"  {summary.name}: {summary.best_fit:.3f} ± {summary.std:.3f}")
    
        return "\n".join(lines)
    

    def plot_corner(self, figsize: Tuple[float, float] = (12, 12), 
                   show_titles: bool = True, title_fmt: str = ".3f",
                   color: str = "blue", truth_color: str = "red",
                   save_path: Optional[str] = None, 
                   group_by_ion: bool = False) -> plt.Figure:
        """
        Create corner plot of parameter posterior distributions.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        show_titles : bool
            Whether to show parameter values as titles
        title_fmt : str
            Format string for title values
        color : str
            Color for histograms and contours
        truth_color : str
            Color for truth values (if provided)
        save_path : str, optional
            Path to save the figure
        group_by_ion : bool
            Whether to create separate corner plots for each ion group
            
        Returns
        -------
        plt.Figure
            The corner plot figure
        """
        if not HAS_CORNER:
            raise ImportError("Corner plots require the 'corner' package. "
                            "Install with: pip install corner")
        
        # Get parameter names for labels
        param_names = self.param_manager.get_parameter_latex_names()
        
        # Create corner plot
        fig = corner.corner(
            self.samples,
            labels=param_names,
            show_titles=show_titles,
            title_fmt=title_fmt,
            color=color,
            figsize=figsize,
            truths=self.best_fit,
            truth_color=truth_color
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_model_comparison(self, figsize: Tuple[float, float] = (15, 10),
                             show_residuals: bool = True,
                             show_components: bool = False,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data vs model comparison for all datasets.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        show_residuals : bool
            Whether to show residual plots
        show_components : bool
            Whether to show individual velocity components (if available)
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The model comparison figure
        """
        n_datasets = len(self.datasets)
        n_rows = n_datasets * (2 if show_residuals else 1)
        
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        
        for i, dataset in enumerate(self.datasets):
            # Calculate model
            try:
                model_flux = self.model.evaluate(self.best_fit, dataset.wavelength)
            except Exception as e:
                print(f"Warning: Could not evaluate model for dataset {i}: {e}")
                continue
            
            # Main plot
            ax_main = axes[i * (2 if show_residuals else 1)]
            
            # Plot data
            ax_main.step(dataset.wavelength, dataset.flux, 'k-', where='mid',
                        label='Observed', alpha=0.7, linewidth=1)
            ax_main.step(dataset.wavelength, dataset.error, 'gray', where='mid',
                        alpha=0.3, linewidth=0.5, label='Error')
            
            # Plot model
            ax_main.plot(dataset.wavelength, model_flux, 'r-', 
                        label='Best fit', linewidth=2)
            
            # Mark transition locations if we have ion groups
            if hasattr(self, 'ion_groups') and self.ion_groups:
                for ion_group in self.ion_groups.values():
                    for trans_wave in ion_group.transitions:
                        obs_wave = trans_wave * (1 + ion_group.redshift)
                        if min(dataset.wavelength) <= obs_wave <= max(dataset.wavelength):
                            ax_main.axvline(obs_wave, color='red', linestyle=':', alpha=0.5, linewidth=1)
                            ax_main.text(obs_wave, 1.05, f'{trans_wave:.0f}', ha='center', fontsize=8, rotation=90)
            
            ax_main.set_ylabel('Normalized Flux')
            ax_main.set_title(f'Dataset {i+1}: {dataset.name}')
            ax_main.legend()
            ax_main.grid(True, alpha=0.3)
            ax_main.set_ylim(0, 1.2)
            
            # Residuals plot
            if show_residuals:
                ax_resid = axes[i * 2 + 1]
                residuals = (dataset.flux - model_flux) / dataset.error
                
                ax_resid.step(dataset.wavelength, residuals, 'k-', where='mid',
                             alpha=0.7, linewidth=1)
                ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
                ax_resid.axhline(1, color='gray', linestyle=':', alpha=0.5)
                ax_resid.axhline(-1, color='gray', linestyle=':', alpha=0.5)
                
                ax_resid.set_ylabel('Residuals (σ)')
                ax_resid.grid(True, alpha=0.3)
                
                # Calculate residual statistics
                rms = np.sqrt(np.mean(residuals**2))
                mean_resid = np.mean(residuals)
                ax_resid.text(0.02, 0.95, f'RMS = {rms:.2f}\nMean = {mean_resid:.2f}', 
                             transform=ax_resid.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel('Wavelength (Å)')
        
        # Add summary title
        if hasattr(self, 'ion_groups') and self.ion_groups:
            n_ions = len(self.ion_groups)
            n_components = sum(ig.components for ig in self.ion_groups.values())
            plt.suptitle(f'Fit Results: {n_ions} ion groups, {n_components} components, '
                        f'χ²/ν = {self.reduced_chi2:.2f}', fontsize=12, y=0.98)
        else:
            plt.suptitle(f'Fit Results: χ²/ν = {self.reduced_chi2:.2f}', fontsize=12, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter samples
        """
        param_names = self.param_manager.get_parameter_names()
        return pd.DataFrame(self.samples, columns=param_names)
    
    def export_csv(self, filepath: str, include_samples: bool = False):
        """
        Export results to CSV file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        include_samples : bool
            Whether to export full sample chains
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if include_samples:
            # Export full sample chain
            df = self.to_pandas()
            df.to_csv(filepath, index=False)
        else:
            # Export parameter summary
            data = []
            for summary in self.parameter_summaries:
                data.append({
                    'parameter': summary.name,
                    'best_fit': summary.best_fit,
                    'lower_error': summary.lower_error,
                    'upper_error': summary.upper_error,
                    'std': summary.std,
                    'mean': summary.mean
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)    

class MultiInstrumentFitResults(FitResults):
    """
    Enhanced container for multi-instrument MCMC fitting results.
    
    Extends FitResults to handle multiple instruments with shared parameters
    but different instrumental responses and wavelength coverage.
    
    Attributes
    ----------
    fitter : vfit
        Multi-instrument MCMC fitter object
    master_model : VoigtModel
        Master model compiled with all instrument configurations
    datasets : Dict[str, Dataset] or None
        Override datasets for visualization, keyed by instrument name
    """
    
    def __init__(self, fitter, master_model: VoigtModel, 
                 datasets: Optional[Union[Dict[str, Dataset], Dataset]] = None):
        """
        Initialize multi-instrument results container.
        
        Parameters
        ----------
        fitter : vfit
            Multi-instrument MCMC fitter object
        master_model : VoigtModel
            Master model compiled with instrument configurations
        datasets : dict, Dataset, or None
            Override datasets for visualization:
            - None: Use fitter's internal data
            - Dict: {'instrument_name': Dataset} for specific overrides
            - Single Dataset: Override primary instrument only
        """
        # Set the master model
        self.model = master_model
        self.fitter = fitter
        
        # Create parameter manager from master model configuration
        self.param_manager = ParameterManager(master_model.config)
        
        # Handle datasets setup for multi-instrument
        self._setup_multi_instrument_datasets(datasets)
        
        # Extract key information from fitter
        self.sampler = fitter.sampler
        self.fit_time = getattr(fitter, 'fit_time', 0.0)
        
        # Simplify bounds storage
        self.bounds = {
            'lower': getattr(fitter, 'lb', None),
            'upper': getattr(fitter, 'ub', None)
        }
        
        # Auto-detect MCMC settings
        self.mcmc_settings = self._extract_mcmc_settings()
        
        # Extract and analyze results
        self._extract_samples()
        self._calculate_parameter_summaries()
        self._calculate_multi_instrument_statistics()
        self._organize_ion_group_results()
        self._calculate_derived_quantities()
        
        # Cache for expensive operations
        self._correlation_matrix = None
        self._covariance_matrix = None
        self._ion_correlation_matrices = None
    
    def _setup_multi_instrument_datasets(self, datasets):
        """Set up datasets for multi-instrument analysis."""
        # Always start with primary dataset
        self.datasets = [Dataset(
            self.fitter.wave_obs, self.fitter.fnorm, self.fitter.enorm,
            name="Primary"
        )]
        
        # Track instrument datasets separately for multi-instrument operations
        self.instrument_datasets = {
            "Primary": self.datasets[0]
        }
        
        # Add additional instruments from fitter
        if hasattr(self.fitter, 'instrument_data') and self.fitter.instrument_data:
            for instrument_name, data in self.fitter.instrument_data.items():
                fitted_dataset = Dataset(
                    data['wave'], data['flux'], data['error'], 
                    name=instrument_name
                )
                self.datasets.append(fitted_dataset)
                self.instrument_datasets[instrument_name] = fitted_dataset
        
        # Handle override datasets
        if datasets is not None:
            if isinstance(datasets, dict):
                # Dictionary: override specific instruments
                for instrument_name, override_dataset in datasets.items():
                    if override_dataset is not None:
                        self.instrument_datasets[instrument_name] = override_dataset
                        # Update main datasets list
                        for i, ds in enumerate(self.datasets):
                            if ds.name == instrument_name:
                                self.datasets[i] = override_dataset
                                break
                        
            elif hasattr(datasets, 'name'):
                # Single Dataset: override primary only
                self.instrument_datasets["Primary"] = datasets
                self.datasets[0] = datasets
            else:
                raise ValueError("datasets must be None, dict, or single Dataset object")
    
    def _calculate_multi_instrument_statistics(self):
        """Calculate model statistics for multi-instrument fit."""
        self.chi2_best = 0.0
        self.n_data_points = 0
        self.chi2_by_instrument = {}
        
        # Calculate chi-squared for each instrument using FITTED data (not override datasets)
        for instrument_name in self.instrument_datasets.keys():
            # Always use the data that was actually fitted
            if instrument_name == "Primary":
                fitted_data = Dataset(
                    self.fitter.wave_obs, self.fitter.fnorm, self.fitter.enorm,
                    name="Primary"
                )
            else:
                data = self.fitter.instrument_data[instrument_name]
                fitted_data = Dataset(
                    data['wave'], data['flux'], data['error'],
                    name=instrument_name
                )
            
            # Evaluate model for this instrument
            if hasattr(data, 'model') and instrument_name != "Primary":
                # Use instrument-specific model function if available
                model_flux = data['model'](self.best_fit, fitted_data.wavelength)
            else:
                # Use master model (assumes it can handle instrument specification)
                try:
                    # Try instrument-specific evaluation
                    model_flux = self.model.evaluate(
                        self.best_fit, fitted_data.wavelength, 
                        instrument=instrument_name if instrument_name != "Primary" else None
                    )
                except TypeError:
                    # Fallback to standard evaluation
                    model_flux = self.model.evaluate(self.best_fit, fitted_data.wavelength)
            
            # Calculate chi-squared for this instrument
            chi2_instrument = np.sum((fitted_data.flux - model_flux)**2 / fitted_data.error**2)
            self.chi2_by_instrument[instrument_name] = chi2_instrument
            self.chi2_best += chi2_instrument
            self.n_data_points += len(fitted_data.wavelength)
        
        # Continue with standard statistics
        self.dof = self.n_data_points - self.n_params
        self.reduced_chi2 = self.chi2_best / self.dof if self.dof > 0 else np.inf
        
        # Ion tying calculations (same as parent class)
        n_transitions = sum(
            len(ion_group.transitions) 
            for ion_group in self.ion_groups.values()
        )
        n_untied_params = n_transitions * 3
        tying_reduction = n_untied_params - self.n_params
        
        self.n_transitions = n_transitions
        self.n_untied_params = n_untied_params  
        self.tying_reduction = tying_reduction
        self.tying_efficiency = tying_reduction / n_untied_params if n_untied_params > 0 else 0
        
        # Information criteria
        self.aic = self.chi2_best + 2 * self.n_params
        self.bic = self.chi2_best + self.n_params * np.log(self.n_data_points)
        self.aic_effective = self.chi2_best + 2 * self.n_params * (1 + self.tying_efficiency)
        self.bic_effective = self.chi2_best + self.n_params * np.log(self.n_data_points) * (1 + self.tying_efficiency)
        
        if self.dof > 0:
            self.p_value = 1.0 - chi2.cdf(self.chi2_best, self.dof)
        else:
            self.p_value = np.nan
    
    def summary(self, verbose: bool = False, show_tying: bool = True) -> str:
        """
        Generate multi-instrument fit summary with per-instrument statistics.
        """
        lines = ["Multi-Instrument MCMC Fit Results", "=" * 60]
        
        # Basic fit information
        lines.append(f"Sampler: {self.mcmc_settings.sampler}")
        lines.append(f"Walkers: {self.mcmc_settings.n_walkers}")
        lines.append(f"Steps: {self.mcmc_settings.n_steps}")
        lines.append(f"Burn-in: {self.mcmc_settings.n_burn}")
        lines.append(f"Samples: {self.n_samples}")
        lines.append(f"Fit time: {self.fit_time:.1f} seconds")
        lines.append(f"Instruments: {len(self.instrument_datasets)}")
        
        # Per-instrument chi-squared
        lines.append(f"\nPer-Instrument Statistics:")
        for instrument, chi2 in self.chi2_by_instrument.items():
            dataset = self.instrument_datasets[instrument]
            n_points = len(dataset.wavelength)
            lines.append(f"  {instrument}: χ² = {chi2:.2f} ({n_points} points)")
        
        # Continue with standard summary
        if show_tying:
            lines.append(f"\nIon Parameter Tying:")
            lines.append(f"Total transitions: {self.n_transitions}")
            lines.append(f"Fitted parameters: {self.n_params}")
            lines.append(f"Untied parameters: {self.n_untied_params}")
            lines.append(f"Parameter reduction: {self.tying_reduction} ({self.tying_efficiency*100:.1f}%)")
        
        # Model statistics
        lines.append(f"\nOverall Model Statistics:")
        lines.append(f"Combined χ²: {self.chi2_best:.2f}")
        lines.append(f"Degrees of freedom: {self.dof}")
        lines.append(f"Reduced χ²: {self.reduced_chi2:.3f}")
        lines.append(f"AIC: {self.aic:.2f}")
        lines.append(f"BIC: {self.bic:.2f}")
        if hasattr(self, 'aic_effective'):
            lines.append(f"AIC (tying-aware): {self.aic_effective:.2f}")
        lines.append(f"P-value: {self.p_value:.4f}")
        
        # Ion group summaries (same as parent)
        lines.append(f"\nIon Groups Summary:")
        for key, ion_group in self.ion_groups.items():
            redshift, ion_name = key
            lines.append(f"  {ion_name} at z={redshift:.6f}: {ion_group.components} components, "
                        f"{len(ion_group.transitions)} transitions")
        
        return "\n".join(lines)
            
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Calculate parameter correlation matrix with caching.
        
        Returns
        -------
        np.ndarray
            Correlation matrix (n_params x n_params)
        """
        if self._correlation_matrix is None:
            self._correlation_matrix = np.corrcoef(self.samples.T)
        return self._correlation_matrix
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Calculate parameter covariance matrix with caching.
        
        Returns
        -------
        np.ndarray
            Covariance matrix (n_params x n_params)
        """
        if self._covariance_matrix is None:
            self._covariance_matrix = np.cov(self.samples.T)
        return self._covariance_matrix
    
    def get_ion_correlation_matrices(self) -> Dict[Tuple[float, str], np.ndarray]:
        """
        Calculate correlation matrices for individual ion groups.
        
        Returns
        -------
        dict
            Mapping from (redshift, ion_name) to correlation matrix for that ion's parameters
        """
        if self._ion_correlation_matrices is None:
            self._ion_correlation_matrices = {}
            
            for key, ion_group in self.ion_groups.items():
                # Find parameter indices for this ion group
                param_indices = []
                for param_name, param_summary in ion_group.parameters.items():
                    param_idx = next(i for i, p in enumerate(self.parameter_summaries) 
                                   if p.name == param_summary.name)
                    param_indices.append(param_idx)
                
                if param_indices:
                    ion_samples = self.samples[:, param_indices]
                    self._ion_correlation_matrices[key] = np.corrcoef(ion_samples.T)
        
        return self._ion_correlation_matrices
    
    def get_parameter_degeneracies(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated (degenerate) parameter pairs.
        
        Parameters
        ----------
        threshold : float
            Correlation threshold for flagging degeneracies
            
        Returns
        -------
        list
            List of (param1, param2, correlation) for degenerate pairs
        """
        corr_matrix = self.get_correlation_matrix()
        param_names = [p.name for p in self.parameter_summaries]
        
        degeneracies = []
        n_params = len(param_names)
        
        for i in range(n_params):
            for j in range(i + 1, n_params):
                corr = abs(corr_matrix[i, j])
                if corr > threshold:
                    degeneracies.append((param_names[i], param_names[j], corr_matrix[i, j]))
        
        return sorted(degeneracies, key=lambda x: abs(x[2]), reverse=True)
    
    def summary(self, verbose: bool = False, show_tying: bool = True) -> str:
        """
        Generate a comprehensive summary of fit results with ion-aware organization.
        
        Parameters
        ----------
        verbose : bool
            Whether to include detailed parameter information
        show_tying : bool
            Whether to show ion parameter tying information
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = ["MCMC Fit Results", "=" * 60]
        
        # Basic fit information
        lines.append(f"Sampler: {self.mcmc_settings.sampler}")
        lines.append(f"Walkers: {self.mcmc_settings.n_walkers}")
        lines.append(f"Steps: {self.mcmc_settings.n_steps}")
        lines.append(f"Burn-in: {self.mcmc_settings.n_burn}")
        lines.append(f"Samples: {self.n_samples}")
        lines.append(f"Fit time: {self.fit_time:.1f} seconds")
        lines.append(f"Datasets: {len(self.datasets)}")
        
        # Ion tying information
        if show_tying:
            lines.append(f"\nIon Parameter Tying:")
            lines.append(f"Total transitions: {self.n_transitions}")
            lines.append(f"Fitted parameters: {self.n_params}")
            lines.append(f"Untied parameters: {self.n_untied_params}")
            lines.append(f"Parameter reduction: {self.tying_reduction} ({self.tying_efficiency*100:.1f}%)")
        
        # Model statistics
        lines.append(f"\nModel Statistics:")
        lines.append(f"Chi-squared: {self.chi2_best:.2f}")
        lines.append(f"Degrees of freedom: {self.dof}")
        lines.append(f"Reduced chi-squared: {self.reduced_chi2:.3f}")
        lines.append(f"AIC: {self.aic:.2f}")
        lines.append(f"BIC: {self.bic:.2f}")
        if hasattr(self, 'aic_effective'):
            lines.append(f"AIC (tying-aware): {self.aic_effective:.2f}")
            lines.append(f"BIC (tying-aware): {self.bic_effective:.2f}")
        lines.append(f"P-value: {self.p_value:.4f}")
        
        # Data points information
        lines.append(f"\nData Information:")
        for i, dataset in enumerate(self.datasets):
            lines.append(f"  Dataset {i+1} ({dataset.name}): {len(dataset.wavelength)} points")
        
        # Ion group summaries
        lines.append(f"\nIon Groups Summary:")
        for key, ion_group in self.ion_groups.items():
            redshift, ion_name = key
            lines.append(f"  {ion_name} at z={redshift:.6f}: {ion_group.components} components, "
                        f"{len(ion_group.transitions)} transitions")
            
            # Show derived quantities
            for derived in ion_group.derived_quantities:
                lines.append(f"    {derived}")
        
        # Parameter summary
        if verbose:
            lines.append(f"\nDetailed Parameter Summary:")
            lines.append("-" * 50)
            for summary in self.parameter_summaries:
                lines.append(str(summary))
                if hasattr(summary, 'physical_interpretation') and summary.physical_interpretation():
                    lines.append(f"    → {summary.physical_interpretation()}")
        else:
            lines.append(f"\nParameter Summary by Ion:")
            lines.append("-" * 40)
            
            for key, ion_group in self.ion_groups.items():
                redshift, ion_name = key
                lines.append(f"\n{ion_name} at z={redshift:.6f}:")
                
                for comp_idx in range(ion_group.components):
                    lines.append(f"  Component {comp_idx+1}:")
                    
                    # Find parameters for this component
                    for param_type in ['N', 'b', 'v']:
                        param_key = f"{param_type}_{comp_idx}"
                        if param_key in ion_group.parameters:
                            param = ion_group.parameters[param_key]
                            lines.append(f"    {param_type} = {param.best_fit:.3f} ± {param.std:.3f} {param.units or ''}")
        
        # Degeneracy warnings
        degeneracies = self.get_parameter_degeneracies(threshold=0.9)
        if degeneracies:
            lines.append(f"\n⚠ Parameter Degeneracy Warnings:")
            for param1, param2, corr in degeneracies[:3]:  # Show top 3
                lines.append(f"  {param1} ↔ {param2}: correlation = {corr:.3f}")
        
        return "\n".join(lines)
    
    def get_ion_summary_table(self, ion_name: str = None, redshift: float = None) -> str:
        """
        Generate detailed summary table for specific ion or all ions.
        
        Parameters
        ----------
        ion_name : str, optional
            Specific ion to summarize (if None, summarize all)
        redshift : float, optional
            Specific redshift to summarize (if None, summarize all)
            
        Returns
        -------
        str
            Formatted table string
        """
        lines = []
        
        if ion_name or redshift is not None:
            # Filter ion groups
            filtered_groups = {}
            for key, ion_group in self.ion_groups.items():
                z, ion = key
                if (ion_name is None or ion == ion_name) and (redshift is None or abs(z - redshift) < 1e-6):
                    filtered_groups[key] = ion_group
        else:
            filtered_groups = self.ion_groups
        
        if not filtered_groups:
            return "No ion groups match the specified criteria."
        
        lines.append("Ion Group Summary Table")
        lines.append("=" * 80)
        
        for key, ion_group in filtered_groups.items():
            redshift, ion_name = key
            lines.append(f"\n{ion_name} at z = {redshift:.6f}")
            lines.append(f"Transitions: {', '.join(f'{w:.1f}' for w in ion_group.transitions)} Å")
            lines.append("-" * 60)
            lines.append("Comp |    N     |   b    |    v    | Physical Interpretation")
            lines.append("-" * 60)
            
            for comp_idx in range(ion_group.components):
                N_key = f"N_{comp_idx}"
                b_key = f"b_{comp_idx}"
                v_key = f"v_{comp_idx}"
                
                N_param = ion_group.parameters.get(N_key)
                b_param = ion_group.parameters.get(b_key)
                v_param = ion_group.parameters.get(v_key)
                
                if N_param and b_param and v_param:
                    # Physical interpretation
                    linear_N = 10**N_param.best_fit
                    if 'MgII' in ion_name:
                        thermal_b = 6.1
                    elif 'OI' in ion_name:
                        thermal_b = 12.9
                    else:
                        thermal_b = 10.0
                    
                    if b_param.best_fit > thermal_b:
                        turbulent_b = np.sqrt(b_param.best_fit**2 - thermal_b**2)
                        b_interp = f"T+{turbulent_b:.1f}"
                    else:
                        b_interp = "Thermal"
                    
                    lines.append(
                        f" {comp_idx+1:2d}  | {N_param.best_fit:8.2f} | {b_param.best_fit:6.1f} | "
                        f"{v_param.best_fit:7.1f} | N={linear_N:.1e}, {b_interp}"
                    )
            
            # Derived quantities
            if ion_group.derived_quantities:
                lines.append("")
                lines.append("Derived Quantities:")
                for derived in ion_group.derived_quantities:
                    lines.append(f"  {derived.description}: {derived.value:.3f} ± {derived.error:.3f} {derived.units}")
        
        return "\n".join(lines)
    
    def plot_corner(self, figsize: Tuple[float, float] = (12, 12), 
                   show_titles: bool = True, title_fmt: str = ".3f",
                   color: str = "blue", truth_color: str = "red",
                   save_path: Optional[str] = None, 
                   group_by_ion: bool = True) -> Union[plt.Figure, Dict[str, plt.Figure]]:
        """
        Create corner plot of parameter posterior distributions with ion grouping.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        show_titles : bool
            Whether to show parameter values as titles
        title_fmt : str
            Format string for title values
        color : str
            Color for histograms and contours
        truth_color : str
            Color for truth values (if provided)
        save_path : str, optional
            Path to save the figure
        group_by_ion : bool
            Whether to create separate corner plots for each ion group
            
        Returns
        -------
        plt.Figure or dict
            The corner plot figure(s)
        """
        if not HAS_CORNER:
            raise ImportError("Corner plots require the 'corner' package. "
                            "Install with: pip install corner")
        
        if group_by_ion and len(self.ion_groups) > 1:
            # Create separate corner plots for each ion group
            figures = {}
            
            for key, ion_group in self.ion_groups.items():
                redshift, ion_name = key
                
                # Get parameter indices for this ion
                param_indices = []
                param_labels = []
                param_truths = []
                
                for comp_idx in range(ion_group.components):
                    for param_type in ['N', 'b', 'v']:
                        param_key = f"{param_type}_{comp_idx}"
                        if param_key in ion_group.parameters:
                            param_summary = ion_group.parameters[param_key]
                            param_idx = next(i for i, p in enumerate(self.parameter_summaries) 
                                           if p.name == param_summary.name)
                            param_indices.append(param_idx)
                            
                            # Create clean labels
                            if param_type == 'N':
                                label = f"$\\log N_{{{ion_name}}}^{{({comp_idx+1})}}$"
                            elif param_type == 'b':
                                label = f"$b_{{{ion_name}}}^{{({comp_idx+1})}}$"
                            else:  # v
                                label = f"$v_{{{ion_name}}}^{{({comp_idx+1})}}$"
                            param_labels.append(label)
                            param_truths.append(param_summary.best_fit)
                
                if param_indices:
                    ion_samples = self.samples[:, param_indices]
                    
                    fig = corner.corner(
                        ion_samples,
                        labels=param_labels,
                        show_titles=show_titles,
                        title_fmt=title_fmt,
                        color=color,
                        figsize=figsize,
                        truths=param_truths,
                        truth_color=truth_color
                    )
                    
                    fig.suptitle(f'{ion_name} at z = {redshift:.6f}', fontsize=14, y=0.98)
                    figures[f"{ion_name}_z{redshift:.3f}"] = fig
            
            if save_path:
                for name, fig in figures.items():
                    path_parts = save_path.split('.')
                    ion_path = f"{path_parts[0]}_{name}.{path_parts[1]}"
                    fig.savefig(ion_path, dpi=300, bbox_inches='tight')
            
            return figures
        
        else:
            # Standard corner plot for all parameters
            param_names = self.param_manager.get_parameter_latex_names()
            
            fig = corner.corner(
                self.samples,
                labels=param_names,
                show_titles=show_titles,
                title_fmt=title_fmt,
                color=color,
                figsize=figsize,
                truths=self.best_fit,
                truth_color=truth_color
            )
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_model_comparison(self, figsize: Tuple[float, float] = (15, 10),
                             show_residuals: bool = True,
                             show_components: bool = False,
                             velocity_space: bool = False,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data vs model comparison for all datasets with enhanced options.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        show_residuals : bool
            Whether to show residual plots
        show_components : bool
            Whether to show individual velocity components
        velocity_space : bool
            Whether to plot in velocity space (requires transition info)
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The model comparison figure
        """
        n_datasets = len(self.datasets)
        n_rows = n_datasets * (2 if show_residuals else 1)
        
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        
        for i, dataset in enumerate(self.datasets):
            # Calculate model
            if dataset.lsf_params:
                model_flux = self.model.evaluate(
                    self.best_fit, dataset.wavelength,
                    FWHM=dataset.lsf_params.get('FWHM', '6.5')
                )
            else:
                model_flux = self.model.evaluate(self.best_fit, dataset.wavelength)
            
            # Main plot
            ax_main = axes[i * (2 if show_residuals else 1)]
            
            # Determine x-axis (wavelength or velocity)
            if velocity_space and len(self.ion_groups) == 1:
                # Convert to velocity space for single ion
                ion_group = list(self.ion_groups.values())[0]
                rest_wave = ion_group.transitions[0]  # Use first transition
                z_abs = ion_group.redshift
                
                # Convert to velocity
                c = 299792.458  # km/s
                x_data = c * (dataset.wavelength / (rest_wave * (1 + z_abs)) - 1)
                x_label = 'Velocity (km/s)'
                
                # Mark component velocities
                for comp_idx in range(ion_group.components):
                    v_key = f"v_{comp_idx}"
                    if v_key in ion_group.parameters:
                        v_comp = ion_group.parameters[v_key].best_fit
                        ax_main.axvline(v_comp, color='orange', linestyle='--', alpha=0.7, linewidth=1)
                        ax_main.text(v_comp, 1.05, f'C{comp_idx+1}', ha='center', fontsize=8)
            else:
                x_data = dataset.wavelength
                x_label = 'Wavelength (Å)'
                
                # Mark transition locations
                for ion_group in self.ion_groups.values():
                    for trans_wave in ion_group.transitions:
                        obs_wave = trans_wave * (1 + ion_group.redshift)
                        if min(x_data) <= obs_wave <= max(x_data):
                            ax_main.axvline(obs_wave, color='red', linestyle=':', alpha=0.5, linewidth=1)
                            ax_main.text(obs_wave, 1.05, f'{trans_wave:.0f}', ha='center', fontsize=8, rotation=90)
            
            # Plot data
            ax_main.step(x_data, dataset.flux, 'k-', where='mid',
                        label='Observed', alpha=0.7, linewidth=1)
            ax_main.step(x_data, dataset.error, 'gray', where='mid',
                        alpha=0.3, linewidth=0.5, label='Error')
            
            # Plot model
            ax_main.plot(x_data, model_flux, 'r-', 
                        label='Best fit', linewidth=2)
            
            # Plot individual components if requested and available
            if show_components:
                try:
                    # This requires model to support component evaluation
                    component_fluxes = self.model.evaluate(
                        self.best_fit, dataset.wavelength, return_components=True
                    )
                    if isinstance(component_fluxes, dict) and 'components' in component_fluxes:
                        for j, comp_flux in enumerate(component_fluxes['components']):
                            ax_main.plot(x_data, comp_flux, ':', alpha=0.7, linewidth=1,
                                       label=f'Component {j+1}')
                except:
                    pass  # Skip if model doesn't support component decomposition
            
            ax_main.set_ylabel('Normalized Flux')
            ax_main.set_title(f'Dataset {i+1}: {dataset.name}')
            ax_main.legend()
            ax_main.grid(True, alpha=0.3)
            ax_main.set_ylim(0, 1.2)
            
            # Residuals plot
            if show_residuals:
                ax_resid = axes[i * 2 + 1]
                residuals = (dataset.flux - model_flux) / dataset.error
                
                ax_resid.step(x_data, residuals, 'k-', where='mid',
                             alpha=0.7, linewidth=1)
                ax_resid.axhline(0, color='r', linestyle='--', alpha=0.7)
                ax_resid.axhline(1, color='gray', linestyle=':', alpha=0.5)
                ax_resid.axhline(-1, color='gray', linestyle=':', alpha=0.5)
                
                ax_resid.set_ylabel('Residuals (σ)')
                ax_resid.grid(True, alpha=0.3)
                
                # Calculate residual statistics
                rms = np.sqrt(np.mean(residuals**2))
                mean_resid = np.mean(residuals)
                ax_resid.text(0.02, 0.95, f'RMS = {rms:.2f}\nMean = {mean_resid:.2f}', 
                             transform=ax_resid.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel(x_label)
        
        # Add summary title
        n_ions = len(self.ion_groups)
        n_components = sum(ig.components for ig in self.ion_groups.values())
        plt.suptitle(f'Multi-Ion Fit: {n_ions} ion groups, {n_components} components, '
                    f'χ²/ν = {self.reduced_chi2:.2f}', fontsize=12, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def compare_models(self, other_result: 'FitResults') -> Dict[str, float]:
        """
        Compare this fit with another fit result with ion tying awareness.
        
        Parameters
        ----------
        other_result : FitResults
            Other fit result to compare with
            
        Returns
        -------
        dict
            Comparison metrics including tying-aware statistics
        """
        comparison = {
            'delta_aic': other_result.aic - self.aic,
            'delta_bic': other_result.bic - self.bic,
            'delta_chi2': other_result.chi2_best - self.chi2_best,
            'delta_reduced_chi2': other_result.reduced_chi2 - self.reduced_chi2
        }
        
        # Add tying-aware comparisons if available
        if hasattr(self, 'aic_effective') and hasattr(other_result, 'aic_effective'):
            comparison['delta_aic_effective'] = other_result.aic_effective - self.aic_effective
            comparison['delta_bic_effective'] = other_result.bic_effective - self.bic_effective
        
        # Interpretation
        if comparison['delta_aic'] > 10:
            comparison['aic_preference'] = 'strong evidence for this model'
        elif comparison['delta_aic'] > 2:
            comparison['aic_preference'] = 'moderate evidence for this model'
        elif comparison['delta_aic'] < -10:
            comparison['aic_preference'] = 'strong evidence for other model'
        elif comparison['delta_aic'] < -2:
            comparison['aic_preference'] = 'moderate evidence for other model'
        else:
            comparison['aic_preference'] = 'models comparable'
        
        return comparison
    
    def export_ion_table(self, format: str = 'latex', save_path: Optional[str] = None) -> str:
        """
        Export ion-organized parameter table for publication.
        
        Parameters
        ----------
        format : str
            Output format ('latex', 'ascii', 'csv')
        save_path : str, optional
            Path to save the table
            
        Returns
        -------
        str
            Formatted table string
        """
        lines = []
        
        if format == 'latex':
            lines.extend([
                r"\begin{table}[h]",
                r"\centering",
                r"\begin{tabular}{lcccccc}",
                r"\hline",
                r"Ion & $z$ & Comp & $\log N$ & $b$ & $v$ & Transitions \\",
                r"    &     &      & [cm$^{-2}$] & [km s$^{-1}$] & [km s$^{-1}$] & [\AA] \\",
                r"\hline"
            ])
            
            for key, ion_group in self.ion_groups.items():
                redshift, ion_name = key
                trans_str = ", ".join(f"{w:.0f}" for w in ion_group.transitions)
                
                for comp_idx in range(ion_group.components):
                    N_key = f"N_{comp_idx}"
                    b_key = f"b_{comp_idx}"
                    v_key = f"v_{comp_idx}"
                    
                    N_param = ion_group.parameters.get(N_key)
                    b_param = ion_group.parameters.get(b_key)
                    v_param = ion_group.parameters.get(v_key)
                    
                    if N_param and b_param and v_param:
                        ion_col = ion_name if comp_idx == 0 else ""
                        z_col = f"{redshift:.6f}" if comp_idx == 0 else ""
                        trans_col = trans_str if comp_idx == 0 else ""
                        
                        lines.append(
                            f"{ion_col} & {z_col} & {comp_idx+1} & "
                            f"{N_param.best_fit:.2f}$_{{-{N_param.lower_error:.2f}}}^{{+{N_param.upper_error:.2f}}}$ & "
                            f"{b_param.best_fit:.1f}$_{{-{b_param.lower_error:.1f}}}^{{+{b_param.upper_error:.1f}}}$ & "
                            f"{v_param.best_fit:.1f}$_{{-{v_param.lower_error:.1f}}}^{{+{v_param.upper_error:.1f}}}$ & "
                            f"{trans_col} \\\\"
                        )
            
            lines.extend([
                r"\hline",
                r"\end{tabular}",
                r"\caption{MCMC fit results for absorption line systems}",
                r"\label{tab:absorption_results}",
                r"\end{table}"
            ])
        
        elif format == 'csv':
            lines.append("Ion,Redshift,Component,logN,logN_err_low,logN_err_high,b,b_err_low,b_err_high,v,v_err_low,v_err_high,Transitions")
            
            for key, ion_group in self.ion_groups.items():
                redshift, ion_name = key
                trans_str = ";".join(f"{w:.1f}" for w in ion_group.transitions)
                
                for comp_idx in range(ion_group.components):
                    N_key = f"N_{comp_idx}"
                    b_key = f"b_{comp_idx}"
                    v_key = f"v_{comp_idx}"
                    
                    N_param = ion_group.parameters.get(N_key)
                    b_param = ion_group.parameters.get(b_key)
                    v_param = ion_group.parameters.get(v_key)
                    
                    if N_param and b_param and v_param:
                        lines.append(
                            f"{ion_name},{redshift:.6f},{comp_idx+1},"
                            f"{N_param.best_fit:.3f},{N_param.lower_error:.3f},{N_param.upper_error:.3f},"
                            f"{b_param.best_fit:.1f},{b_param.lower_error:.1f},{b_param.upper_error:.1f},"
                            f"{v_param.best_fit:.1f},{v_param.lower_error:.1f},{v_param.upper_error:.1f},"
                            f"{trans_str}"
                        )
        
        result = "\n".join(lines)
        
        if save_path:
            Path(save_path).write_text(result)
        
        return result