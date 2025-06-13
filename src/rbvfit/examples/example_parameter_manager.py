#!/usr/bin/env python
"""
Example usage of the ParameterManager class in rbvfit 2.0.

This example demonstrates:
1. Creating parameter mappings from configurations
2. Converting between theta arrays and structured parameters
3. Generating parameter bounds
4. Getting parameter names for output
"""

import numpy as np
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager, ParameterSet


def example_basic_parameter_mapping():
    """Example 1: Basic parameter mapping."""
    print("=" * 60)
    print("Example 1: Basic Parameter Mapping")
    print("=" * 60)
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.348, 'FeII', [2600.2], components=1)
    
    # Create parameter manager
    pm = ParameterManager(config)
    
    # Show parameter structure
    structure = pm.config_to_theta_structure()
    print(f"\nTotal parameters: {structure['total_parameters']}")
    
    # Create example theta array
    theta = np.array([
        # MgII: N1, N2
        13.5, 13.2,
        # FeII: N1
        12.8,
        # MgII: b1, b2
        15.0, 20.0,
        # FeII: b1
        25.0,
        # MgII: v1, v2
        -50.0, 0.0,
        # FeII: v1
        10.0
    ])
    
    print(f"\nTheta array: {theta}")
    
    # Convert to structured parameters
    params = pm.theta_to_parameters(theta)
    
    print("\nStructured parameters:")
    for key, param_set in sorted(params.items()):
        sys_idx, ion_name, comp_idx = key
        print(f"  System {sys_idx}, {ion_name} component {comp_idx}: "
              f"N={param_set.N:.1f}, b={param_set.b:.1f}, v={param_set.v:.1f}")
    
    # Convert back to theta
    theta_reconstructed = pm.parameters_to_theta(params)
    print(f"\nReconstructed theta: {theta_reconstructed}")
    print(f"Arrays match: {np.allclose(theta, theta_reconstructed)}")
    
    return config, pm, theta


def example_line_parameters():
    """Example 2: Getting parameters for individual lines."""
    print("\n" + "=" * 60)
    print("Example 2: Line Parameters")
    print("=" * 60)
    
    # Create configuration with tied transitions
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    
    pm = ParameterManager(config)
    
    # Create theta array
    theta = np.array([
        # N values
        13.5, 13.2,  # MgII
        14.0,        # OVI
        # b values
        15.0, 20.0,  # MgII
        30.0,        # OVI
        # v values
        -50.0, 0.0,  # MgII
        25.0         # OVI
    ])
    
    # Get line parameters
    line_params = pm.theta_to_line_parameters(theta)
    
    print(f"\nTotal lines: {len(line_params)}")
    print("\nLine parameters:")
    for i, lp in enumerate(line_params):
        print(f"  Line {i+1}: {lp['ion']} {lp['wavelength']:.1f}Å at z={lp['z']:.3f}, "
              f"comp {lp['component_idx']+1}: "
              f"N={lp['N']:.1f}, b={lp['b']:.1f}, v={lp['v']:.1f}")
    
    # Show how MgII transitions share parameters
    print("\nNote: MgII 2796.3 and 2803.5 share the same parameters (ion tying)")
    
    return config, pm, line_params


def example_parameter_bounds():
    """Example 3: Generating parameter bounds."""
    print("\n" + "=" * 60)
    print("Example 3: Parameter Bounds")
    print("=" * 60)
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.348, 'HI', [1215.67], components=1)
    config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    
    pm = ParameterManager(config)
    
    # Generate default bounds
    bounds = pm.generate_parameter_bounds()
    
    print("\nDefault bounds:")
    names = pm.get_parameter_names()
    for i, name in enumerate(names):
        print(f"  {name}: [{bounds.lower[i]:.1f}, {bounds.upper[i]:.1f}]")
    
    # Generate custom bounds
    custom = {
        'MgII': {
            'N': (12.0, 15.0),
            'b': (5.0, 50.0),
            'v': (-100.0, 100.0)
        }
    }
    
    custom_bounds = pm.generate_parameter_bounds(custom_bounds=custom)
    
    print("\nCustom bounds for MgII:")
    for i, name in enumerate(names):
        if 'MgII' in name:
            print(f"  {name}: [{custom_bounds.lower[i]:.1f}, {custom_bounds.upper[i]:.1f}]")
    
    return config, pm, bounds


def example_parameter_names():
    """Example 4: Parameter naming for output."""
    print("\n" + "=" * 60)
    print("Example 4: Parameter Names")
    print("=" * 60)
    
    # Create configuration
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    
    pm = ParameterManager(config)
    
    # Get regular names
    names = pm.get_parameter_names()
    print("\nParameter names:")
    for i, name in enumerate(names):
        print(f"  theta[{i}] = {name}")
    
    # Get LaTeX names
    latex_names = pm.get_parameter_latex_names()
    print("\nLaTeX names (for plots):")
    for i, name in enumerate(latex_names):
        print(f"  theta[{i}] = {name}")
    
    return config, pm


def example_parameter_summary():
    """Example 5: Parameter summary table."""
    print("\n" + "=" * 60)
    print("Example 5: Parameter Summary Table")
    print("=" * 60)
    
    # Create complex configuration
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=3)
    config.add_system(0.348, 'FeII', [2600.2, 2382.8], components=2)
    config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=2)
    
    pm = ParameterManager(config)
    
    # Create theta with some reasonable values
    theta = np.array([
        # N values
        13.5, 13.2, 12.8,  # MgII
        13.0, 12.5,        # FeII
        14.2, 13.8,        # OVI
        # b values
        15.0, 20.0, 25.0,  # MgII
        18.0, 22.0,        # FeII
        35.0, 40.0,        # OVI
        # v values
        -100.0, -50.0, 0.0,  # MgII
        -80.0, -20.0,        # FeII
        -30.0, 20.0,         # OVI
    ])
    
    # Get summary table
    summary = pm.get_summary_table(theta)
    print("\n" + summary)
    
    return config, pm


def example_validation():
    """Example 6: Parameter validation."""
    print("\n" + "=" * 60)
    print("Example 6: Parameter Validation")
    print("=" * 60)
    
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    
    pm = ParameterManager(config)
    
    # Valid theta
    theta_valid = np.array([13.5, 13.2, 15.0, 20.0, -50.0, 0.0])
    print(f"Valid theta: {pm.validate_theta(theta_valid)}")
    
    # Invalid length
    theta_short = np.array([13.5, 13.2, 15.0])
    try:
        pm.validate_theta(theta_short)
    except ValueError as e:
        print(f"Invalid length caught: {e}")
    
    # Invalid values
    theta_nan = np.array([13.5, np.nan, 15.0, 20.0, -50.0, 0.0])
    try:
        pm.validate_theta(theta_nan)
    except ValueError as e:
        print(f"Invalid values caught: {e}")
    
    return config, pm


def example_complex_mapping():
    """Example 7: Complex multi-system mapping."""
    print("\n" + "=" * 60)
    print("Example 7: Complex Multi-System Mapping")
    print("=" * 60)
    
    # Create a complex sightline
    config = FitConfiguration()
    
    # DLA system
    config.add_system(2.456, 'HI', [1215.67], components=5)
    config.add_system(2.456, 'SiII', [1190.42, 1193.29, 1260.42, 1526.71], components=3)
    
    # MgII system
    config.add_system(1.234, 'MgII', [2796.3, 2803.5], components=2)
    
    pm = ParameterManager(config)
    
    # Show how parameters are organized
    structure = pm.config_to_theta_structure()
    print(f"\nTotal parameters: {structure['total_parameters']}")
    
    # Show parameter organization
    print("\nParameter organization in theta array:")
    print("  Indices 0-4:   HI N values (5 components)")
    print("  Indices 5-7:   SiII N values (3 components)")
    print("  Indices 8-9:   MgII N values (2 components)")
    print("  Indices 10-14: HI b values")
    print("  Indices 15-17: SiII b values")
    print("  Indices 18-19: MgII b values")
    print("  Indices 20-24: HI v values")
    print("  Indices 25-27: SiII v values")
    print("  Indices 28-29: MgII v values")
    
    # Show efficiency of parameter tying
    n_transitions = sum(
        len(ig.transitions) for sys in config.systems for ig in sys.ion_groups
    )
    untied_params = n_transitions * 10 * 3  # if each transition had separate params
    
    print(f"\nParameter efficiency:")
    print(f"  Transitions: {n_transitions}")
    print(f"  With tying: {structure['total_parameters']} parameters")
    print(f"  Without tying: {untied_params} parameters")
    print(f"  Reduction: {100 * (1 - structure['total_parameters']/untied_params):.1f}%")
    
    return config, pm


if __name__ == "__main__":
    # Run all examples
    example_basic_parameter_mapping()
    example_line_parameters()
    example_parameter_bounds()
    example_parameter_names()
    example_parameter_summary()
    example_validation()
    example_complex_mapping()
    
    print("\n" + "=" * 60)
    print("All parameter manager examples completed successfully!")
    print("=" * 60)