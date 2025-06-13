#!/usr/bin/env python
"""
Examples demonstrating comprehensive results analysis in rbvfit 2.0.

This script shows:
1. Parameter summary and uncertainty analysis
2. Correlation analysis and plotting
3. Model comparison and residual analysis
4. Publication-ready plots and tables
5. Data export capabilities
6. Model selection and comparison
7. Advanced MCMC diagnostics
8. Posterior predictive checks
9. Sensitivity analysis
10. Information content analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.voigt_fitter import VoigtFitter, Dataset, MCMCSettings
from rbvfit.core.fit_results import FitResults


def generate_synthetic_data(wave, theta, model, noise_level=0.02):
    """Generate synthetic data for testing."""
    flux_true = model.evaluate(theta, wave)
    noise = np.random.normal(0, noise_level, len(wave))
    flux_obs = flux_true + noise
    error = np.full_like(wave, noise_level)
    return flux_obs, error, flux_true


def run_example_fit():
    """Run a sample fit to demonstrate results analysis."""
    print("Running example fit for results analysis...")
    
    # Create multi-component MgII system
    config = FitConfiguration()
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=3)
    
    model = VoigtModel(config, FWHM='6.5')
    
    # True parameters for 3-component system
    theta_true = np.array([
        13.8, 13.5, 13.2,    # N values
        25.0, 15.0, 35.0,    # b values  
        -80.0, -20.0, 40.0   # v values
    ])
    
    # Generate data
    wave = np.linspace(3760, 3820, 2000)
    flux_obs, error, flux_true = generate_synthetic_data(wave, theta_true, model, 0.025)
    
    dataset = Dataset(wave, flux_obs, error, name="Complex_MgII_System")
    
    # Quick MCMC run for demonstration
    mcmc_settings = MCMCSettings(n_walkers=40, n_steps=500, n_burn=100)
    fitter = VoigtFitter(model, dataset, mcmc_settings)
    
    # Add some noise to initial guess
    initial_guess = theta_true + 0.1 * np.random.randn(len(theta_true))
    
    print("Running MCMC...")
    result = fitter.fit(initial_guess, optimize_first=True)
    print("MCMC complete.")
    
    return result, theta_true


def example_parameter_analysis(result, theta_true):
    """Example 1: Detailed parameter analysis."""
    print("\n" + "=" * 60)
    print("Example 1: Parameter Analysis")
    print("=" * 60)
    
    # Basic summary
    print("\nBasic Summary:")
    print(result.summary())
    
    # Detailed parameter analysis
    print("\nDetailed Parameter Analysis:")
    print("-" * 50)
    
    for i, summary in enumerate(result.parameter_summaries):
        true_val = theta_true[i] if i < len(theta_true) else None
        
        print(f"\n{summary.name}:")
        print(f"  Best fit: {summary.best_fit:.4f}")
        print(f"  +/- errors: +{summary.upper_error:.4f} -{summary.lower_error:.4f}")
        print(f"  Standard dev: {summary.std:.4f}")
        print(f"  Mean: {summary.mean:.4f}")
        
        if true_val is not None:
            deviation = abs(summary.best_fit - true_val) / summary.std
            print(f"  True value: {true_val:.4f}")
            print(f"  Deviation: {deviation:.1f}σ")
            
            if deviation < 1.0:
                print(f"  ✓ Good agreement (< 1σ)")
            elif deviation < 2.0:
                print(f"  ⚠ Moderate deviation (1-2σ)")
            else:
                print(f"  ✗ Poor agreement (> 2σ)")
    
    # Statistical tests
    print(f"\nModel Statistics:")
    print(f"  χ² = {result.chi2_best:.2f}")
    print(f"  DOF = {result.dof}")
    print(f"  χ²/DOF = {result.reduced_chi2:.3f}")
    print(f"  P-value = {result.p_value:.4f}")
    
    if 0.05 < result.p_value < 0.95:
        print("  ✓ Good fit (p-value suggests reasonable model)")
    elif result.p_value <= 0.05:
        print("  ⚠ Poor fit (p-value suggests model problems)")
    else:
        print("  ⚠ Suspiciously good fit (p-value very high)")
    
    return result


def example_correlation_analysis(result):
    """Example 2: Parameter correlation analysis."""
    print("\n" + "=" * 60)
    print("Example 2: Correlation Analysis")
    print("=" * 60)
    
    # Get correlation matrix
    corr_matrix = result.get_correlation_matrix()
    param_names = result.param_manager.get_parameter_names()
    
    print("\nParameter Correlations (|r| > 0.5):")
    print("-" * 40)
    
    strong_correlations = []
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > 0.5:
                strong_correlations.append((i, j, corr_val))
                print(f"{param_names[i]} ↔ {param_names[j]}: r = {corr_val:.3f}")
    
    if not strong_correlations:
        print("No strong correlations found (all |r| < 0.5)")
    
    # Plot correlation matrix
    print("\nGenerating correlation matrix plot...")
    fig_corr = result.plot_correlation_matrix(figsize=(10, 8))
    plt.show()
    
    # Identify parameter groups by correlation
    print(f"\nCorrelation Analysis:")
    print(f"  Found {len(strong_correlations)} strong correlations")
    
    if strong_correlations:
        # Analyze correlation patterns
        print("  Correlation patterns:")
        for i, j, corr_val in strong_correlations:
            param1 = param_names[i]
            param2 = param_names[j]
            
            if corr_val > 0:
                relationship = "positively correlated"
            else:
                relationship = "negatively correlated"
            
            print(f"    {param1} and {param2} are {relationship} (r={corr_val:.3f})")
            
            # Physical interpretation
            if 'N_' in param1 and 'b_' in param2:
                print(f"      → Column density vs Doppler parameter correlation")
            elif 'b_' in param1 and 'v_' in param2:
                print(f"      → Doppler parameter vs velocity correlation")
            elif 'N_' in param1 and 'N_' in param2:
                print(f"      → Inter-component column density correlation")
    
    return corr_matrix


def example_model_comparison_plots(result):
    """Example 3: Model comparison and residual analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Model Comparison and Residuals")
    print("=" * 60)
    
    # Generate model comparison plot
    print("Generating model comparison plot...")
    fig_model = result.plot_model_comparison(figsize=(12, 8), show_residuals=True)
    plt.show()
    
    # Detailed residual analysis
    print("\nResidual Analysis:")
    for i, dataset in enumerate(result.datasets):
        wave, model_flux = result.get_best_fit_model(dataset_index=i)
        residuals = (dataset.flux - model_flux) / dataset.error
        
        print(f"\nDataset {i+1} ({dataset.name}):")
        print(f"  RMS residuals: {np.sqrt(np.mean(residuals**2)):.3f}")
        print(f"  Mean residuals: {np.mean(residuals):.3f}")
        print(f"  Std residuals: {np.std(residuals):.3f}")
        print(f"  Max |residual|: {np.max(np.abs(residuals)):.3f}")
        
        # Check for systematic trends
        if abs(np.mean(residuals)) > 0.1:
            print(f"  ⚠ Systematic offset detected (mean = {np.mean(residuals):.3f})")
        
        if np.max(np.abs(residuals)) > 3.0:
            outlier_frac = np.sum(np.abs(residuals) > 3.0) / len(residuals) * 100
            print(f"  ⚠ Outliers detected: {outlier_frac:.1f}% of points > 3σ")
        
        # Reduced chi-squared for this dataset
        chi2_dataset = np.sum(residuals**2)
        n_points = len(residuals)
        n_params = result.n_params
        reduced_chi2_dataset = chi2_dataset / (n_points - n_params)
        print(f"  χ²/DOF for this dataset: {reduced_chi2_dataset:.3f}")
    
    return fig_model


def example_uncertainty_analysis(result):
    """Example 4: Model uncertainty analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Model Uncertainty Analysis")
    print("=" * 60)
    
    # Get model uncertainties from posterior
    print("Calculating model uncertainties from posterior samples...")
    
    dataset_idx = 0  # Use first dataset
    wave, model_mean, model_std = result.get_model_uncertainties(
        dataset_index=dataset_idx, n_samples=100
    )
    
    dataset = result.datasets[dataset_idx]
    
    # Plot uncertainties
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Main plot with uncertainty bands
    ax1.step(dataset.wavelength, dataset.flux, 'k-', where='mid', 
             label='Observed', alpha=0.7)
    ax1.plot(wave, model_mean, 'r-', label='Best fit model', linewidth=2)
    
    # Uncertainty bands
    ax1.fill_between(wave, model_mean - model_std, model_mean + model_std,
                     alpha=0.3, color='red', label='1σ model uncertainty')
    ax1.fill_between(wave, model_mean - 2*model_std, model_mean + 2*model_std,
                     alpha=0.2, color='red', label='2σ model uncertainty')
    
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Model with Posterior Uncertainties')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Uncertainty magnitude plot
    ax2.plot(wave, model_std, 'b-', linewidth=2, label='Model uncertainty (1σ)')
    ax2.plot(wave, dataset.error, 'g--', linewidth=2, label='Data uncertainty')
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Uncertainty')
    ax2.set_title('Uncertainty Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Uncertainty statistics
    print(f"\nUncertainty Statistics:")
    print(f"  Mean model uncertainty: {np.mean(model_std):.4f}")
    print(f"  Max model uncertainty: {np.max(model_std):.4f}")
    print(f"  Mean data uncertainty: {np.mean(dataset.error):.4f}")
    
    uncertainty_ratio = np.mean(model_std) / np.mean(dataset.error)
    print(f"  Model/Data uncertainty ratio: {uncertainty_ratio:.2f}")
    
    if uncertainty_ratio < 0.1:
        print("  → Model uncertainties much smaller than data uncertainties")
    elif uncertainty_ratio > 1.0:
        print("  → Model uncertainties comparable to or larger than data uncertainties")
    else:
        print("  → Model uncertainties moderate compared to data uncertainties")
    
    return wave, model_mean, model_std


def example_chain_diagnostics(result):
    """Example 5: MCMC chain diagnostics."""
    print("\n" + "=" * 60)
    print("Example 5: MCMC Chain Diagnostics")
    print("=" * 60)
    
    # Plot chains
    print("Generating chain evolution plots...")
    fig_chains = result.plot_chains(params_to_plot=[0, 1, 2, 3, 4, 5], figsize=(12, 10))
    plt.show()
    
    # Chain diagnostics
    print("\nChain Diagnostics:")
    
    # Acceptance fraction
    if hasattr(result.sampler, 'acceptance_fraction'):
        acc_frac = np.mean(result.sampler.acceptance_fraction)
        print(f"  Mean acceptance fraction: {acc_frac:.3f}")
        
        if acc_frac < 0.2:
            print("  ⚠ Low acceptance fraction - consider adjusting proposal scale")
        elif acc_frac > 0.5:
            print("  ⚠ High acceptance fraction - consider increasing proposal scale")
        else:
            print("  ✓ Good acceptance fraction")
    
    # Autocorrelation analysis
    try:
        if hasattr(result.sampler, 'get_autocorr_time'):
            autocorr_times = result.sampler.get_autocorr_time()
            max_autocorr = np.max(autocorr_times)
            
            print(f"  Autocorrelation times: {autocorr_times}")
            print(f"  Maximum autocorr time: {max_autocorr:.1f} steps")
            
            # Effective sample size
            effective_samples = result.n_samples / (2 * max_autocorr)
            print(f"  Effective samples: {effective_samples:.0f}")
            
            if effective_samples < 100:
                print("  ⚠ Low effective sample size - consider longer chains")
            else:
                print("  ✓ Good effective sample size")
                
            # Check convergence criterion
            chain_length = result.mcmc_settings.n_steps - result.mcmc_settings.n_burn
            if chain_length > 50 * max_autocorr:
                print("  ✓ Chain is long enough (> 50 × τ)")
            else:
                recommended_length = int(50 * max_autocorr) + result.mcmc_settings.n_burn
                print(f"  ⚠ Chain may be too short, recommend {recommended_length} total steps")
                
    except Exception as e:
        print(f"  Could not compute autocorrelation: {e}")
    
    # Geweke convergence test (simplified)
    print(f"\nConvergence Assessment:")
    first_half = result.samples[:len(result.samples)//2]
    second_half = result.samples[len(result.samples)//2:]
    
    converged_params = 0
    for i in range(result.n_params):
        mean1 = np.mean(first_half[:, i])
        mean2 = np.mean(second_half[:, i])
        std1 = np.std(first_half[:, i])
        std2 = np.std(second_half[:, i])
        
        # Simple convergence test
        z_score = abs(mean1 - mean2) / np.sqrt(std1**2 + std2**2)
        
        if z_score < 2.0:  # Within 2 sigma
            converged_params += 1
    
    convergence_fraction = converged_params / result.n_params
    print(f"  Parameters showing convergence: {converged_params}/{result.n_params} "
          f"({convergence_fraction*100:.0f}%)")
    
    if convergence_fraction > 0.9:
        print("  ✓ Good convergence")
    elif convergence_fraction > 0.7:
        print("  ⚠ Moderate convergence - some parameters may need longer chains")
    else:
        print("  ✗ Poor convergence - longer chains recommended")
    
    return fig_chains


def example_publication_plots(result):
    """Example 6: Publication-ready plots and tables."""
    print("\n" + "=" * 60)
    print("Example 6: Publication-Ready Outputs")
    print("=" * 60)
    
    # Corner plot
    print("Generating corner plot...")
    try:
        fig_corner = result.plot_corner(
            figsize=(12, 12),
            show_titles=True,
            title_fmt=".3f",
            color="blue"
        )
        plt.show()
        print("  ✓ Corner plot generated")
    except ImportError:
        print("  ⚠ Corner plot requires 'corner' package")
        fig_corner = None
    
    # Generate publication tables
    print("\nGenerating publication tables...")
    
    # ASCII table
    ascii_table = result.to_table(format='ascii')
    print("\nASCII Table:")
    print(ascii_table)
    
    # LaTeX table
    latex_table = result.to_table(format='latex')
    print("\nLaTeX Table:")
    print(latex_table)
    
    # HTML table
    html_table = result.to_table(format='html')
    print("\nHTML Table (first few lines):")
    print("\n".join(html_table.split('\n')[:10]) + "\n...")
    
    # Parameter summary in different formats
    print("\nParameter Summary Formats:")
    for i, summary in enumerate(result.parameter_summaries[:3]):  # First 3 parameters
        print(f"  {summary.name}:")
        print(f"    Standard: {summary}")
        print(f"    LaTeX: ${summary.latex_string()}$")
    
    return fig_corner, ascii_table, latex_table


def example_data_export(result):
    """Example 7: Data export capabilities."""
    print("\n" + "=" * 60)
    print("Example 7: Data Export")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("rbvfit_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Exporting results to {output_dir}/")
    
    # Export parameter summary to CSV
    csv_path = output_dir / "parameter_summary.csv"
    result.export_csv(csv_path, include_samples=False)
    print(f"  ✓ Parameter summary: {csv_path}")
    
    # Export full samples to CSV  
    samples_path = output_dir / "mcmc_samples.csv"
    result.export_csv(samples_path, include_samples=True)
    print(f"  ✓ MCMC samples: {samples_path}")
    
    # Export to HDF5
    try:
        hdf5_path = output_dir / "fit_results.h5"
        result.save_hdf5(hdf5_path, include_samples=True)
        print(f"  ✓ HDF5 results: {hdf5_path}")
    except Exception as e:
        print(f"  ⚠ HDF5 export failed: {e}")
    
    # Export tables
    table_formats = ['ascii', 'latex', 'html']
    for fmt in table_formats:
        table_path = output_dir / f"parameter_table.{fmt}"
        table_content = result.to_table(format=fmt)
        table_path.write_text(table_content)
        print(f"  ✓ {fmt.upper()} table: {table_path}")
    
    # Export plots
    plot_paths = {}
    
    # Model comparison plot
    fig_model = result.plot_model_comparison(figsize=(12, 8))
    model_path = output_dir / "model_comparison.png"
    fig_model.savefig(model_path, dpi=300, bbox_inches='tight')
    plot_paths['model'] = model_path
    plt.close(fig_model)
    
    # Correlation matrix
    fig_corr = result.plot_correlation_matrix(figsize=(10, 8))
    corr_path = output_dir / "correlation_matrix.png"
    fig_corr.savefig(corr_path, dpi=300, bbox_inches='tight')
    plot_paths['correlation'] = corr_path
    plt.close(fig_corr)
    
    # Chain evolution
    fig_chains = result.plot_chains(figsize=(12, 8))
    chains_path = output_dir / "chain_evolution.png"
    fig_chains.savefig(chains_path, dpi=300, bbox_inches='tight')
    plot_paths['chains'] = chains_path
    plt.close(fig_chains)
    
    # Corner plot (if available)
    try:
        fig_corner = result.plot_corner(figsize=(12, 12))
        corner_path = output_dir / "corner_plot.png"
        fig_corner.savefig(corner_path, dpi=300, bbox_inches='tight')
        plot_paths['corner'] = corner_path
        plt.close(fig_corner)
        print(f"  ✓ Corner plot: {corner_path}")
    except ImportError:
        print("  ⚠ Corner plot skipped (requires corner package)")
    
    for plot_type, path in plot_paths.items():
        if plot_type != 'corner':
            print(f"  ✓ {plot_type.title()} plot: {path}")
    
    # Create summary report
    report_path = output_dir / "fit_report.txt"
    report_content = result.summary(verbose=True)
    report_path.write_text(report_content)
    print(f"  ✓ Fit report: {report_path}")
    
    print(f"\nAll results exported to {output_dir}/")
    return output_dir


def example_model_selection():
    """Example 8: Model selection and comparison."""
    print("\n" + "=" * 60)
    print("Example 8: Model Selection")
    print("=" * 60)
    
    print("Comparing different component numbers for MgII system...")
    
    # Generate common dataset
    wave = np.linspace(3760, 3820, 1500)
    
    # True model with 2 components
    config_true = FitConfiguration()
    config_true.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    model_true = VoigtModel(config_true)
    
    theta_true = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
    flux_obs, error, _ = generate_synthetic_data(wave, theta_true, model_true, 0.03)
    
    dataset = Dataset(wave, flux_obs, error, name="MgII_Test")
    
    # Test different numbers of components
    component_numbers = [1, 2, 3]
    results = {}
    
    for n_comp in component_numbers:
        print(f"\nFitting {n_comp}-component model...")
        
        config = FitConfiguration()
        config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], 
                         components=n_comp)
        model = VoigtModel(config)
        
        # Quick MCMC for comparison
        mcmc_settings = MCMCSettings(n_walkers=30, n_steps=300, n_burn=60)
        fitter = VoigtFitter(model, dataset, mcmc_settings)
        
        # Generate initial guess
        if n_comp == 1:
            initial_guess = np.array([13.5, 25.0, 0.0])
        elif n_comp == 2:
            initial_guess = np.array([13.8, 13.3, 20.0, 30.0, -40.0, 20.0])
        else:  # n_comp == 3
            initial_guess = np.array([13.8, 13.3, 13.0, 20.0, 30.0, 15.0, 
                                    -40.0, 20.0, 60.0])
        
        result = fitter.fit(initial_guess, optimize_first=True)
        results[n_comp] = result
        
        print(f"  χ²/DOF = {result.reduced_chi2:.3f}")
        print(f"  AIC = {result.aic:.1f}")
        print(f"  BIC = {result.bic:.1f}")
    
    # Compare models
    print(f"\nModel Comparison Summary:")
    print(f"{'Components':<12} {'χ²/DOF':<10} {'AIC':<10} {'BIC':<10} {'ΔAIC':<10} {'ΔBIC':<10}")
    print("-" * 60)
    
    # Use 1-component as reference
    ref_aic = results[1].aic
    ref_bic = results[1].bic
    
    for n_comp in component_numbers:
        result = results[n_comp]
        delta_aic = result.aic - ref_aic
        delta_bic = result.bic - ref_bic
        
        print(f"{n_comp:<12} {result.reduced_chi2:<10.3f} {result.aic:<10.1f} "
              f"{result.bic:<10.1f} {delta_aic:<10.1f} {delta_bic:<10.1f}")
    
    # Determine best model
    print(f"\nModel Selection Results:")
    
    best_aic = min(results.values(), key=lambda r: r.aic)
    best_bic = min(results.values(), key=lambda r: r.bic)
    
    aic_components = [k for k, v in results.items() if v.aic == best_aic.aic][0]
    bic_components = [k for k, v in results.items() if v.bic == best_bic.bic][0]
    
    print(f"  Best model by AIC: {aic_components} components")
    print(f"  Best model by BIC: {bic_components} components")
    print(f"  True model: 2 components")
    
    # Detailed comparison between top models
    if aic_components != bic_components:
        comparison = results[aic_components].compare_models(results[bic_components])
        print(f"\nDetailed comparison (AIC vs BIC preference):")
        print(f"  ΔAIC = {comparison['delta_aic']:.1f}")
        print(f"  ΔBIC = {comparison['delta_bic']:.1f}")
        print(f"  Interpretation: {comparison['aic_preference']}")
    
    return results


def example_advanced_diagnostics(result):
    """Example 9: Advanced MCMC diagnostics and quality checks."""
    print("\n" + "=" * 60)
    print("Example 9: Advanced MCMC Diagnostics")
    print("=" * 60)
    
    # Check for parameter degeneracies
    print("Checking for parameter degeneracies...")
    corr_matrix = result.get_correlation_matrix()
    
    # Find highly correlated parameter pairs
    high_corr_pairs = []
    param_names = result.param_manager.get_parameter_names()
    
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    if high_corr_pairs:
        print("  ⚠ High correlations found (|r| > 0.9):")
        for i, j, corr in high_corr_pairs:
            print(f"    {param_names[i]} ↔ {param_names[j]}: r = {corr:.3f}")
        print("  → Consider reparameterization or additional constraints")
    else:
        print("  ✓ No extreme parameter correlations found")
    
    # Check parameter ranges vs bounds
    print(f"\nParameter range analysis:")
    bounds = result.bounds
    
    for i, (summary, lower, upper) in enumerate(zip(result.parameter_summaries, 
                                                   bounds.lower, bounds.upper)):
        param_range = summary.percentile_84 - summary.percentile_16
        bound_range = upper - lower
        range_fraction = param_range / bound_range
        
        print(f"  {summary.name}:")
        print(f"    Used range: {range_fraction*100:.1f}% of allowed bounds")
        
        if range_fraction < 0.1:
            print(f"    ⚠ Very narrow range - bounds may be too loose")
        elif range_fraction > 0.8:
            print(f"    ⚠ Near bounds - may need wider limits")
        else:
            print(f"    ✓ Good range usage")
        
        # Check if hitting bounds
        if abs(summary.best_fit - lower) < 0.01 * bound_range:
            print(f"    ⚠ Near lower bound!")
        elif abs(summary.best_fit - upper) < 0.01 * bound_range:
            print(f"    ⚠ Near upper bound!")
    
    # Sample quality metrics
    print(f"\nSample quality metrics:")
    print(f"  Total samples: {result.n_samples}")
    print(f"  Parameters: {result.n_params}")
    print(f"  Samples per parameter: {result.n_samples / result.n_params:.0f}")
    
    if result.n_samples / result.n_params < 100:
        print("  ⚠ Low samples per parameter (<100)")
    else:
        print("  ✓ Good samples per parameter")
    
    return high_corr_pairs


def example_predictive_checks(result):
    """Example 10: Posterior predictive checks."""
    print("\n" + "=" * 60)
    print("Example 10: Posterior Predictive Checks")
    print("=" * 60)
    
    print("Performing posterior predictive checks...")
    
    # Generate model predictions from posterior samples
    dataset = result.datasets[0]  # Use first dataset
    n_pred_samples = 50
    
    # Randomly select samples for predictions
    sample_indices = np.random.choice(len(result.samples), size=n_pred_samples, replace=False)
    
    predicted_fluxes = []
    for idx in sample_indices:
        theta = result.samples[idx]
        
        if dataset.lsf_params:
            pred_flux = result.model.evaluate(
                theta, dataset.wavelength,
                FWHM=dataset.lsf_params.get('FWHM', '6.5')
            )
        else:
            pred_flux = result.model.evaluate(theta, dataset.wavelength)
        
        predicted_fluxes.append(pred_flux)
    
    predicted_fluxes = np.array(predicted_fluxes)
    
    # Calculate prediction statistics
    pred_mean = np.mean(predicted_fluxes, axis=0)
    pred_std = np.std(predicted_fluxes, axis=0)
    pred_5 = np.percentile(predicted_fluxes, 5, axis=0)
    pred_95 = np.percentile(predicted_fluxes, 95, axis=0)
    
    # Plot predictive distribution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main predictive plot
    ax1.step(dataset.wavelength, dataset.flux, 'k-', where='mid', 
             label='Observed data', linewidth=2, alpha=0.8)
    
    # Plot sample of predictions
    for i in range(min(10, n_pred_samples)):
        ax1.plot(dataset.wavelength, predicted_fluxes[i], 'b-', 
                alpha=0.1, linewidth=0.5)
    
    # Plot prediction bands
    ax1.fill_between(dataset.wavelength, pred_5, pred_95, 
                     alpha=0.3, color='blue', label='90% prediction interval')
    ax1.plot(dataset.wavelength, pred_mean, 'r-', 
             label='Prediction mean', linewidth=2)
    
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Posterior Predictive Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals from prediction mean
    pred_residuals = (dataset.flux - pred_mean) / dataset.error
    
    ax2.step(dataset.wavelength, pred_residuals, 'k-', where='mid', 
             alpha=0.7, linewidth=1)
    ax2.axhline(0, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Prediction Residuals (σ)')
    ax2.set_title('Residuals from Prediction Mean')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Predictive check statistics
    print(f"\nPredictive Check Results:")
    
    # Coverage probability
    in_90_interval = np.sum((dataset.flux >= pred_5) & (dataset.flux <= pred_95))
    coverage_90 = in_90_interval / len(dataset.flux) * 100
    print(f"  90% interval coverage: {coverage_90:.1f}% (expected: 90%)")
    
    if 85 <= coverage_90 <= 95:
        print("  ✓ Good coverage probability")
    elif coverage_90 < 85:
        print("  ⚠ Under-coverage - model may be overconfident")
    else:
        print("  ⚠ Over-coverage - model may be underconfident")
    
    # Prediction accuracy
    pred_rms = np.sqrt(np.mean(pred_residuals**2))
    print(f"  Prediction RMS: {pred_rms:.3f}")
    
    if pred_rms < 1.2:
        print("  ✓ Good prediction accuracy")
    else:
        print("  ⚠ Poor prediction accuracy - model systematic errors")
    
    return predicted_fluxes, pred_mean, pred_std


def example_sensitivity_analysis(result):
    """Example 11: Parameter sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Example 11: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    print("Analyzing parameter sensitivity...")
    
    # Calculate parameter sensitivities by finite differences
    dataset = result.datasets[0]
    base_model = result.model.evaluate(result.best_fit, dataset.wavelength)
    
    sensitivities = []
    param_names = result.param_manager.get_parameter_names()
    
    for i, param_name in enumerate(param_names):
        # Perturb parameter by 1% or 1-sigma, whichever is smaller
        perturbation = min(0.01 * abs(result.best_fit[i]), 
                          result.uncertainties[i])
        
        if perturbation == 0:
            perturbation = 0.01  # Fallback for zero parameters
        
        # Calculate finite difference
        theta_plus = result.best_fit.copy()
        theta_plus[i] += perturbation
        
        theta_minus = result.best_fit.copy()
        theta_minus[i] -= perturbation
        
        try:
            model_plus = result.model.evaluate(theta_plus, dataset.wavelength)
            model_minus = result.model.evaluate(theta_minus, dataset.wavelength)
            
            # Sensitivity = change in model / change in parameter
            sensitivity = (model_plus - model_minus) / (2 * perturbation)
            
            # Integrated sensitivity (sum of absolute changes)
            integrated_sens = np.sum(np.abs(sensitivity))
            
            sensitivities.append({
                'parameter': param_name,
                'index': i,
                'sensitivity_curve': sensitivity,
                'integrated_sensitivity': integrated_sens,
                'max_sensitivity': np.max(np.abs(sensitivity)),
                'perturbation': perturbation
            })
            
        except Exception as e:
            print(f"  Warning: Could not calculate sensitivity for {param_name}: {e}")
            sensitivities.append({
                'parameter': param_name,
                'index': i,
                'sensitivity_curve': np.zeros_like(dataset.wavelength),
                'integrated_sensitivity': 0.0,
                'max_sensitivity': 0.0,
                'perturbation': perturbation
            })
    
    # Sort by integrated sensitivity
    sensitivities.sort(key=lambda x: x['integrated_sensitivity'], reverse=True)
    
    print(f"\nParameter Sensitivity Ranking:")
    print(f"{'Rank':<6} {'Parameter':<20} {'Integrated':<12} {'Max':<12}")
    print("-" * 50)
    
    for rank, sens in enumerate(sensitivities, 1):
        print(f"{rank:<6} {sens['parameter']:<20} "
              f"{sens['integrated_sensitivity']:<12.3e} "
              f"{sens['max_sensitivity']:<12.3e}")
    
    # Plot top sensitivity curves
    n_plot = min(6, len(sensitivities))
    fig, axes = plt.subplots((n_plot + 1) // 2, 2, figsize=(15, 3 * ((n_plot + 1) // 2)))
    if n_plot <= 2:
        axes = axes.reshape(1, -1)
    
    for i in range(n_plot):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        sens = sensitivities[i]
        ax.plot(dataset.wavelength, sens['sensitivity_curve'], 'b-', linewidth=2)
        ax.set_title(f"{sens['parameter']} (Rank {i+1})")
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('dFlux/dParam')
        ax.grid(True, alpha=0.3)
        
        # Mark regions of high sensitivity
        high_sens_mask = np.abs(sens['sensitivity_curve']) > 0.5 * sens['max_sensitivity']
        if np.any(high_sens_mask):
            ax.fill_between(dataset.wavelength, 
                           sens['sensitivity_curve'],
                           0, where=high_sens_mask, 
                           alpha=0.3, color='red',
                           label='High sensitivity')
            ax.legend()
    
    # Hide extra subplots
    for i in range(n_plot, len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis insights
    print(f"\nSensitivity Analysis Insights:")
    most_sensitive = sensitivities[0]
    least_sensitive = sensitivities[-1]
    
    print(f"  Most sensitive parameter: {most_sensitive['parameter']}")
    print(f"  Least sensitive parameter: {least_sensitive['parameter']}")
    
    sensitivity_ratio = (most_sensitive['integrated_sensitivity'] / 
                        least_sensitive['integrated_sensitivity'])
    print(f"  Sensitivity ratio (max/min): {sensitivity_ratio:.1f}")
    
    if sensitivity_ratio > 1000:
        print("  ⚠ Very large sensitivity range - some parameters poorly constrained")
    elif sensitivity_ratio > 100:
        print("  ⚠ Large sensitivity range - check parameter correlations")
    else:
        print("  ✓ Reasonable sensitivity range")
    
    return sensitivities


def example_information_content_analysis(result):
    """Example 12: Information content and Fisher matrix analysis."""
    print("\n" + "=" * 60)
    print("Example 12: Information Content Analysis")
    print("=" * 60)
    
    print("Analyzing information content...")
    
    # Calculate Fisher Information Matrix
    # F_ij = sum_k (1/σ_k²) * (∂μ_k/∂θ_i) * (∂μ_k/∂θ_j)
    
    dataset = result.datasets[0]
    n_params = len(result.best_fit)
    
    # Calculate gradients (same as sensitivity analysis but for Fisher matrix)
    gradients = np.zeros((len(dataset.wavelength), n_params))
    
    for i in range(n_params):
        perturbation = min(0.001 * abs(result.best_fit[i]), 
                          0.1 * result.uncertainties[i])
        if perturbation == 0:
            perturbation = 0.001
        
        theta_plus = result.best_fit.copy()
        theta_plus[i] += perturbation
        
        theta_minus = result.best_fit.copy()  
        theta_minus[i] -= perturbation
        
        try:
            model_plus = result.model.evaluate(theta_plus, dataset.wavelength)
            model_minus = result.model.evaluate(theta_minus, dataset.wavelength)
            gradients[:, i] = (model_plus - model_minus) / (2 * perturbation)
        except:
            gradients[:, i] = 0.0
    
    # Calculate Fisher Information Matrix
    inv_var = 1.0 / dataset.error**2
    fisher_matrix = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            fisher_matrix[i, j] = np.sum(inv_var * gradients[:, i] * gradients[:, j])
    
    # Calculate theoretical uncertainties from Fisher matrix
    try:
        covariance_matrix = np.linalg.inv(fisher_matrix)
        theoretical_uncertainties = np.sqrt(np.diag(covariance_matrix))
        
        print(f"\nFisher Matrix Analysis:")
        print(f"{'Parameter':<20} {'MCMC σ':<12} {'Fisher σ':<12} {'Ratio':<10}")
        print("-" * 54)
        
        param_names = result.param_manager.get_parameter_names()
        
        for i, name in enumerate(param_names):
            mcmc_err = result.uncertainties[i]
            fisher_err = theoretical_uncertainties[i]
            ratio = mcmc_err / fisher_err if fisher_err > 0 else np.inf
            
            print(f"{name:<20} {mcmc_err:<12.4f} {fisher_err:<12.4f} {ratio:<10.2f}")
        
        # Overall comparison
        mean_ratio = np.mean([result.uncertainties[i] / theoretical_uncertainties[i] 
                             for i in range(n_params) 
                             if theoretical_uncertainties[i] > 0])
        
        print(f"\nMean MCMC/Fisher ratio: {mean_ratio:.2f}")
        
        if 0.9 <= mean_ratio <= 1.1:
            print("  ✓ Excellent agreement - MCMC sampling is efficient")
        elif 0.8 <= mean_ratio <= 1.2:
            print("  ✓ Good agreement - minor sampling inefficiencies")
        elif mean_ratio > 1.2:
            print("  ⚠ MCMC uncertainties larger - sampling inefficient or correlations")
        else:
            print("  ⚠ MCMC uncertainties smaller - possible undersampling")
        
        # Condition number analysis
        eigenvals = np.linalg.eigvals(fisher_matrix)
        condition_number = np.max(eigenvals) / np.min(eigenvals)
        
        print(f"\nFisher Matrix Conditioning:")
        print(f"  Condition number: {condition_number:.2e}")
        
        if condition_number < 1e6:
            print("  ✓ Well-conditioned problem")
        elif condition_number < 1e12:
            print("  ⚠ Moderately ill-conditioned")
        else:
            print("  ✗ Severely ill-conditioned - numerical issues likely")
        
        return fisher_matrix, covariance_matrix, theoretical_uncertainties
        
    except np.linalg.LinAlgError:
        print("  ✗ Fisher matrix is singular - parameters are degenerate")
        return fisher_matrix, None, None


if __name__ == "__main__":
    """Run comprehensive results analysis examples."""
    
    print("rbvfit 2.0 Results Analysis Examples")
    print("=" * 60)
    
    try:
        # Run example fit
        print("Setting up example fit...")
        result, theta_true = run_example_fit()
        
        # Run analysis examples
        print("\n" + "🔍 Running comprehensive analysis examples...")
        
        # Basic analysis
        example_parameter_analysis(result, theta_true)
        example_correlation_analysis(result)
        example_model_comparison_plots(result)
        example_uncertainty_analysis(result)
        example_chain_diagnostics(result)
        
        # Publication outputs  
        example_publication_plots(result)
        example_data_export(result)
        
        # Advanced analysis
        example_advanced_diagnostics(result)
        example_predictive_checks(result)
        example_sensitivity_analysis(result)
        example_information_content_analysis(result)
        
        # Model selection
        example_model_selection()
        
        print("\n" + "=" * 60)
        print("✅ All results analysis examples completed successfully!")
        print("📁 Results exported to ./rbvfit_results/ directory")
        print("🎯 Comprehensive analysis pipeline demonstrated")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error in results analysis examples: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("⚠️  Some examples may require additional dependencies:")
        print("   - corner: pip install corner")
        print("   - h5py: pip install h5py") 
        print("   - pandas: pip install pandas")
        print("=" * 60)