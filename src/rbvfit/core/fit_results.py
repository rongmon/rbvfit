"""
Comprehensive results management for rbvfit 2.0.

This module provides the FitResults class with analysis capabilities,
parameter summary with proper uncertainties, correlation analysis,
and publication-ready plotting methods.
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
    Summary statistics for a single parameter.
    
    Attributes
    ----------
    name : str
        Parameter name
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
    """
    name: str
    best_fit: float
    lower_error: float
    upper_error: float
    percentile_16: float
    percentile_84: float
    mean: float
    std: float
    
    def __str__(self) -> str:
        """String representation for display."""
        return f"{self.name}: {self.best_fit:.3f} +{self.upper_error:.3f} -{self.lower_error:.3f}"
    
    def latex_string(self) -> str:
        """LaTeX representation for publication."""
        return f"{self.name} = {self.best_fit:.3f}_{{-{self.lower_error:.3f}}}^{{+{self.upper_error:.3f}}}"


class FitResults:
    """
    Comprehensive container for MCMC fitting results.
    
    This class provides detailed analysis capabilities including parameter
    summaries, correlation analysis, model comparison metrics, and 
    publication-ready plotting methods.
    
    Attributes
    ----------
    sampler : object
        The MCMC sampler object (emcee or zeus)
    param_manager : ParameterManager
        Parameter management object
    datasets : List[Dataset]
        List of fitted datasets
    mcmc_settings : MCMCSettings
        MCMC configuration used
    bounds : ParameterBounds
        Parameter bounds used in fitting
    model : VoigtModel
        The fitted model
    fit_time : float
        Total fitting time in seconds
    """
    
    def __init__(self, sampler, param_manager: ParameterManager, datasets: List[Dataset],
                 mcmc_settings: MCMCSettings, bounds: ParameterBounds, model: VoigtModel,
                 fit_time: float):
        """Initialize results container."""
        self.sampler = sampler
        self.param_manager = param_manager
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.mcmc_settings = mcmc_settings
        self.bounds = bounds
        self.model = model
        self.fit_time = fit_time
        
        # Extract and analyze results
        self._extract_samples()
        self._calculate_parameter_summaries()
        self._calculate_model_statistics()
        
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
        """Calculate detailed parameter summaries."""
        param_names = self.param_manager.get_parameter_names()
        
        self.parameter_summaries = []
        
        for i, name in enumerate(param_names):
            samples_i = self.samples[:, i]
            
            # Calculate percentiles
            p16, p50, p84 = np.percentile(samples_i, [16, 50, 84])
            
            summary = ParameterSummary(
                name=name,
                best_fit=p50,
                lower_error=p50 - p16,
                upper_error=p84 - p50,
                percentile_16=p16,
                percentile_84=p84,
                mean=np.mean(samples_i),
                std=np.std(samples_i)
            )
            
            self.parameter_summaries.append(summary)
        
        # Convenience arrays
        self.best_fit = np.array([s.best_fit for s in self.parameter_summaries])
        self.uncertainties = np.array([s.std for s in self.parameter_summaries])
        self.lower_errors = np.array([s.lower_error for s in self.parameter_summaries])
        self.upper_errors = np.array([s.upper_error for s in self.parameter_summaries])
        
    def _calculate_model_statistics(self):
        """Calculate model comparison statistics."""
        # Calculate chi-squared for best fit
        self.chi2_best = 0.0
        self.n_data_points = 0
        
        for dataset in self.datasets:
            if dataset.lsf_params:
                model_flux = self.model.evaluate(
                    self.best_fit, dataset.wavelength,
                    FWHM=dataset.lsf_params.get('FWHM', '6.5')
                )
            else:
                model_flux = self.model.evaluate(self.best_fit, dataset.wavelength)
            
            chi2_dataset = np.sum((dataset.flux - model_flux)**2 / dataset.error**2)
            self.chi2_best += chi2_dataset
            self.n_data_points += len(dataset.wavelength)
        
        # Degrees of freedom
        self.dof = self.n_data_points - self.n_params
        self.reduced_chi2 = self.chi2_best / self.dof if self.dof > 0 else np.inf
        
        # Calculate AIC and BIC
        self.aic = self.chi2_best + 2 * self.n_params
        self.bic = self.chi2_best + self.n_params * np.log(self.n_data_points)
        
        # P-value from chi-squared distribution
        if self.dof > 0:
            self.p_value = 1.0 - chi2.cdf(self.chi2_best, self.dof)
        else:
            self.p_value = np.nan
            
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Calculate parameter correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix (n_params x n_params)
        """
        return np.corrcoef(self.samples.T)
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Calculate parameter covariance matrix.
        
        Returns
        -------
        np.ndarray
            Covariance matrix (n_params x n_params)
        """
        return np.cov(self.samples.T)
    
    def summary(self, verbose: bool = False) -> str:
        """
        Generate a comprehensive summary of fit results.
        
        Parameters
        ----------
        verbose : bool
            Whether to include detailed parameter information
            
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
        
        # Model statistics
        lines.append(f"\nModel Statistics:")
        lines.append(f"Chi-squared: {self.chi2_best:.2f}")
        lines.append(f"Degrees of freedom: {self.dof}")
        lines.append(f"Reduced chi-squared: {self.reduced_chi2:.3f}")
        lines.append(f"AIC: {self.aic:.2f}")
        lines.append(f"BIC: {self.bic:.2f}")
        lines.append(f"P-value: {self.p_value:.4f}")
        
        # Data points information
        lines.append(f"\nData Information:")
        for i, dataset in enumerate(self.datasets):
            lines.append(f"  Dataset {i+1} ({dataset.name}): {len(dataset.wavelength)} points")
        
        # Parameter summary
        if verbose:
            lines.append(f"\nDetailed Parameter Summary:")
            lines.append("-" * 50)
            for summary in self.parameter_summaries:
                lines.append(str(summary))
        else:
            lines.append(f"\nParameter Summary:")
            lines.append("-" * 40)
            
            # Group by ion and component
            param_dict = self.param_manager.theta_to_parameters(self.best_fit)
            
            for key, params in param_dict.items():
                sys_idx, ion_name, comp_idx = key
                lines.append(f"{ion_name} component {comp_idx+1}:")
                lines.append(f"  N = {params.N:.3f} ± {self.uncertainties[self._get_param_index(key, 'N')]:.3f}")
                lines.append(f"  b = {params.b:.1f} ± {self.uncertainties[self._get_param_index(key, 'b')]:.1f} km/s")
                lines.append(f"  v = {params.v:.1f} ± {self.uncertainties[self._get_param_index(key, 'v')]:.1f} km/s")
        
        return "\n".join(lines)
    
    def _get_param_index(self, key: Tuple[int, str, int], param_type: str) -> int:
        """Get the index in theta array for a specific parameter."""
        sys_idx, ion_name, comp_idx = key
        
        # Calculate global component index
        global_comp_idx = 0
        for s_idx, system in enumerate(self.model.config.systems):
            if s_idx == sys_idx:
                for ion_group in system.ion_groups:
                    if ion_group.ion_name == ion_name:
                        global_comp_idx += comp_idx
                        break
                    else:
                        global_comp_idx += ion_group.components
                break
            else:
                for ion_group in system.ion_groups:
                    global_comp_idx += ion_group.components
        
        # Calculate total components
        total_components = sum(
            ion_group.components 
            for system in self.model.config.systems 
            for ion_group in system.ion_groups
        )
        
        # Return index based on parameter type
        if param_type == 'N':
            return global_comp_idx
        elif param_type == 'b':
            return total_components + global_comp_idx
        elif param_type == 'v':
            return 2 * total_components + global_comp_idx
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def plot_corner(self, figsize: Tuple[float, float] = (12, 12), 
                   show_titles: bool = True, title_fmt: str = ".3f",
                   color: str = "blue", truth_color: str = "red",
                   save_path: Optional[str] = None) -> plt.Figure:
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
    
    def plot_chains(self, params_to_plot: Optional[List[int]] = None,
                   figsize: Tuple[float, float] = (12, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot MCMC chain evolution.
        
        Parameters
        ----------
        params_to_plot : list of int, optional
            Parameter indices to plot (default: first 6 parameters)
        figsize : tuple
            Figure size in inches
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The chain plot figure
        """
        if params_to_plot is None:
            params_to_plot = list(range(min(6, self.n_params)))
        
        n_plots = len(params_to_plot)
        param_names = self.param_manager.get_parameter_names()
        
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        for i, param_idx in enumerate(params_to_plot):
            ax = axes[i]
            
            # Plot chains for subset of walkers to avoid overcrowding
            n_walkers_to_plot = min(20, self.mcmc_settings.n_walkers)
            walker_indices = np.linspace(0, self.mcmc_settings.n_walkers-1, 
                                       n_walkers_to_plot, dtype=int)
            
            for walker_idx in walker_indices:
                ax.plot(self.chain[:, walker_idx, param_idx], 
                       alpha=0.3, color='blue', linewidth=0.5)
            
            # Mark burn-in
            ax.axvline(self.mcmc_settings.n_burn, color='red', linestyle='--', 
                      label='Burn-in end' if i == 0 else '', alpha=0.7)
            
            # Mark best-fit value
            ax.axhline(self.best_fit[param_idx], color='green', linestyle='-', 
                      label='Best fit' if i == 0 else '', alpha=0.7)
            
            ax.set_ylabel(param_names[param_idx])
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
        
        axes[-1].set_xlabel('Step')
        plt.suptitle('MCMC Chain Evolution')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_model_comparison(self, figsize: Tuple[float, float] = (15, 10),
                             show_residuals: bool = True,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data vs model comparison for all datasets.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        show_residuals : bool
            Whether to show residual plots
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
            
            ax_main.step(dataset.wavelength, dataset.flux, 'k-', where='mid',
                        label='Observed', alpha=0.7, linewidth=1)
            ax_main.step(dataset.wavelength, dataset.error, 'gray', where='mid',
                        alpha=0.3, linewidth=0.5, label='Error')
            ax_main.plot(dataset.wavelength, model_flux, 'r-', 
                        label='Best fit', linewidth=2)
            
            ax_main.set_ylabel('Normalized Flux')
            ax_main.set_title(f'Dataset {i+1}: {dataset.name}')
            ax_main.legend()
            ax_main.grid(True, alpha=0.3)
            
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
                ax_resid.text(0.02, 0.95, f'RMS = {rms:.2f}', 
                             transform=ax_resid.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel('Wavelength (Å)')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_matrix(self, figsize: Tuple[float, float] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot parameter correlation matrix.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The correlation matrix figure
        """
        corr_matrix = self.get_correlation_matrix()
        param_names = self.param_manager.get_parameter_names()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        
        # Set ticks and labels
        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.set_yticklabels(param_names)
        
        # Add correlation values as text
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation coefficient')
        
        ax.set_title('Parameter Correlation Matrix')
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
    
    def to_table(self, format: str = 'ascii') -> str:
        """
        Generate parameter summary table.
        
        Parameters
        ----------
        format : str
            Table format ('ascii', 'latex', 'html')
            
        Returns
        -------
        str
            Formatted table string
        """
        # Create data for table
        data = []
        for summary in self.parameter_summaries:
            data.append([
                summary.name,
                f"{summary.best_fit:.4f}",
                f"{summary.lower_error:.4f}",
                f"{summary.upper_error:.4f}",
                f"{summary.std:.4f}"
            ])
        
        if format == 'latex':
            lines = [
                r"\begin{table}[h]",
                r"\centering", 
                r"\begin{tabular}{lcccc}",
                r"\hline",
                r"Parameter & Best Fit & Lower Error & Upper Error & Std Dev \\",
                r"\hline"
            ]
            
            for row in data:
                lines.append(" & ".join(row) + r" \\")
            
            lines.extend([
                r"\hline",
                r"\end{tabular}",
                r"\caption{MCMC fit results}",
                r"\end{table}"
            ])
            
            return "\n".join(lines)
            
        elif format == 'html':
            lines = [
                "<table>",
                "<thead>",
                "<tr><th>Parameter</th><th>Best Fit</th><th>Lower Error</th><th>Upper Error</th><th>Std Dev</th></tr>",
                "</thead>",
                "<tbody>"
            ]
            
            for row in data:
                lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
            
            lines.extend([
                "</tbody>",
                "</table>"
            ])
            
            return "\n".join(lines)
            
        else:  # ASCII format
            # Calculate column widths
            headers = ["Parameter", "Best Fit", "Lower Error", "Upper Error", "Std Dev"]
            widths = [max(len(headers[i]), max(len(row[i]) for row in data)) + 2 
                     for i in range(len(headers))]
            
            # Create table
            lines = []
            
            # Header
            header_line = "|".join(f"{headers[i]:^{widths[i]}}" for i in range(len(headers)))
            separator = "+".join("-" * widths[i] for i in range(len(headers)))
            
            lines.append(separator)
            lines.append(header_line)
            lines.append(separator)
            
            # Data rows
            for row in data:
                data_line = "|".join(f"{row[i]:^{widths[i]}}" for i in range(len(row)))
                lines.append(data_line)
            
            lines.append(separator)
            
            return "\n".join(lines)
    
    def save_hdf5(self, filepath: str, include_samples: bool = True):
        """
        Save results to HDF5 file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        include_samples : bool
            Whether to include full sample chains
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['sampler'] = self.mcmc_settings.sampler
            meta_group.attrs['n_walkers'] = self.mcmc_settings.n_walkers
            meta_group.attrs['n_steps'] = self.mcmc_settings.n_steps
            meta_group.attrs['n_burn'] = self.mcmc_settings.n_burn
            meta_group.attrs['fit_time'] = self.fit_time
            meta_group.attrs['n_samples'] = self.n_samples
            meta_group.attrs['n_params'] = self.n_params
            
            # Configuration
            config_group = f.create_group('configuration')
            config_group.attrs['config_json'] = self.model.config.serialize()
            
            # Parameter names
            param_names = self.param_manager.get_parameter_names()
            param_group = f.create_group('parameters')
            param_group.create_dataset('names', data=[name.encode() for name in param_names])
            param_group.create_dataset('best_fit', data=self.best_fit)
            param_group.create_dataset('uncertainties', data=self.uncertainties)
            param_group.create_dataset('lower_errors', data=self.lower_errors)
            param_group.create_dataset('upper_errors', data=self.upper_errors)
            
            # Model statistics
            stats_group = f.create_group('statistics')
            stats_group.attrs['chi2_best'] = self.chi2_best
            stats_group.attrs['dof'] = self.dof
            stats_group.attrs['reduced_chi2'] = self.reduced_chi2
            stats_group.attrs['aic'] = self.aic
            stats_group.attrs['bic'] = self.bic
            stats_group.attrs['p_value'] = self.p_value
            
            # Correlation matrix
            corr_group = f.create_group('correlations')
            corr_group.create_dataset('correlation_matrix', data=self.get_correlation_matrix())
            corr_group.create_dataset('covariance_matrix', data=self.get_covariance_matrix())
            
            # Datasets information
            data_group = f.create_group('datasets')
            for i, dataset in enumerate(self.datasets):
                ds_group = data_group.create_group(f'dataset_{i}')
                ds_group.attrs['name'] = dataset.name
                ds_group.create_dataset('wavelength', data=dataset.wavelength)
                ds_group.create_dataset('flux', data=dataset.flux)
                ds_group.create_dataset('error', data=dataset.error)
                if dataset.lsf_params:
                    ds_group.attrs['lsf_params'] = json.dumps(dataset.lsf_params)
            
            # Sample chains
            if include_samples:
                samples_group = f.create_group('samples')
                samples_group.create_dataset('flat_samples', data=self.samples)
                samples_group.create_dataset('chain', data=self.chain)
    
    @classmethod
    def load_hdf5(cls, filepath: str) -> 'FitResults':
        """
        Load results from HDF5 file.
        
        Parameters
        ----------
        filepath : str
            Input file path
            
        Returns
        -------
        FitResults
            Loaded results object
        """
        # This would be a complex reconstruction process
        # For now, just raise NotImplementedError
        raise NotImplementedError("HDF5 loading not yet implemented. "
                                "This will be added in a future update.")
    
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
    
    def get_best_fit_model(self, dataset_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get best-fit model for a specific dataset.
        
        Parameters
        ----------
        dataset_index : int
            Index of dataset to evaluate
            
        Returns
        -------
        tuple
            (wavelength, model_flux) arrays
        """
        if dataset_index >= len(self.datasets):
            raise ValueError(f"Dataset index {dataset_index} out of range")
        
        dataset = self.datasets[dataset_index]
        
        if dataset.lsf_params:
            model_flux = self.model.evaluate(
                self.best_fit, dataset.wavelength,
                FWHM=dataset.lsf_params.get('FWHM', '6.5')
            )
        else:
            model_flux = self.model.evaluate(self.best_fit, dataset.wavelength)
        
        return dataset.wavelength, model_flux
    
    def get_model_uncertainties(self, dataset_index: int = 0, 
                               n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate model uncertainties from posterior samples.
        
        Parameters
        ----------
        dataset_index : int
            Index of dataset to evaluate
        n_samples : int
            Number of posterior samples to use
            
        Returns
        -------
        tuple
            (wavelength, model_mean, model_std) arrays
        """
        if dataset_index >= len(self.datasets):
            raise ValueError(f"Dataset index {dataset_index} out of range")
        
        dataset = self.datasets[dataset_index]
        
        # Randomly select samples
        sample_indices = np.random.choice(len(self.samples), size=n_samples, replace=False)
        model_samples = []
        
        for idx in sample_indices:
            theta = self.samples[idx]
            
            if dataset.lsf_params:
                model_flux = self.model.evaluate(
                    theta, dataset.wavelength,
                    FWHM=dataset.lsf_params.get('FWHM', '6.5')
                )
            else:
                model_flux = self.model.evaluate(theta, dataset.wavelength)
            
            model_samples.append(model_flux)
        
        model_samples = np.array(model_samples)
        model_mean = np.mean(model_samples, axis=0)
        model_std = np.std(model_samples, axis=0)
        
        return dataset.wavelength, model_mean, model_std
    
    def compare_models(self, other_result: 'FitResults') -> Dict[str, float]:
        """
        Compare this fit with another fit result.
        
        Parameters
        ----------
        other_result : FitResults
            Other fit result to compare with
            
        Returns
        -------
        dict
            Comparison metrics
        """
        comparison = {
            'delta_aic': other_result.aic - self.aic,
            'delta_bic': other_result.bic - self.bic,
            'delta_chi2': other_result.chi2_best - self.chi2_best,
            'delta_reduced_chi2': other_result.reduced_chi2 - self.reduced_chi2
        }
        
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