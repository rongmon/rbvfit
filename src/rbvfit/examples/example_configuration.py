#!/usr/bin/env python
"""
Example usage of rbvfit 2.0 FitConfiguration system.

This example demonstrates:
1. Creating configurations for different scenarios
2. Automatic ion parameter tying
3. Parameter structure inspection
4. Serialization/deserialization
"""

from rbvfit.core.fit_configuration import FitConfiguration


def example_simple_single_ion():
    """Example 1: Simple single ion system."""
    print("=" * 60)
    print("Example 1: Simple MgII doublet")
    print("=" * 60)
    
    config = FitConfiguration()
    
    # Add MgII doublet at z=0.348 with 2 velocity components
    config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
    
    print(config)
    print("\n" + config.summary())
    
    # Inspect parameter structure
    structure = config.get_parameter_structure()
    print(f"\nTotal parameters: {structure['total_parameters']}")
    print("\nParameter mapping:")
    for key, idx in sorted(structure['parameter_map'].items()):
        print(f"  {key}: theta[{idx}]")
    
    return config


def example_multi_ion_same_system():
    """Example 2: Multiple ions at same redshift."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple ions at same redshift")
    print("=" * 60)
    
    config = FitConfiguration()
    
    # Add multiple ions to the same system
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.348, 'FeII', [2600.2, 2382.8], components=2)
    config.add_system(0.348, 'HI', [1215.67], components=1)
    
    print(config)
    print("\n" + config.summary())
    
    # Show how parameters are organized
    structure = config.get_parameter_structure()
    print(f"\nTotal parameters: {structure['total_parameters']}")
    
    print("\nIon groups in system 1:")
    for ion_info in structure['systems'][0]['ion_groups']:
        print(f"  {ion_info['ion']}: {len(ion_info['transitions'])} transitions, "
              f"{ion_info['components']} components, "
              f"parameters {ion_info['parameter_slice'][0]}-{ion_info['parameter_slice'][1]-1}")
    
    return config


def example_multi_system():
    """Example 3: Multiple absorption systems."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple absorption systems (contamination)")
    print("=" * 60)
    
    config = FitConfiguration()
    
    # System 1: Main absorber at z=0.348
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.348, 'FeII', [2600.2], components=1)
    
    # System 2: Contaminating absorber at z=0.524
    config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    
    # System 3: Another contaminator at z=0.712
    config.add_system(0.712, 'HI', [1215.67], components=3)
    
    print(config)
    print("\n" + config.summary())
    
    # Show parameter organization across systems
    structure = config.get_parameter_structure()
    print(f"\nTotal parameters: {structure['total_parameters']}")
    
    for i, sys_info in enumerate(structure['systems']):
        total_params = sum(
            group['parameter_slice'][1] - group['parameter_slice'][0]
            for group in sys_info['ion_groups']
        )
        print(f"\nSystem {i+1} (z={sys_info['redshift']:.3f}): {total_params} parameters")
        for ion_info in sys_info['ion_groups']:
            print(f"  - {ion_info['ion']}: parameters "
                  f"{ion_info['parameter_slice'][0]}-{ion_info['parameter_slice'][1]-1}")
    
    return config


def example_complex_sightline():
    """Example 4: Complex sightline with many ions."""
    print("\n" + "=" * 60)
    print("Example 4: Complex sightline analysis")
    print("=" * 60)
    
    config = FitConfiguration()
    
    # DLA system at z=2.456
    config.add_system(2.456, 'HI', [1215.67], components=5)
    config.add_system(2.456, 'SiII', [1190.42, 1193.29, 1260.42, 1526.71], components=3)
    config.add_system(2.456, 'CII', [1334.53], components=3)
    config.add_system(2.456, 'AlII', [1670.79], components=2)
    
    # Metal-strong system at z=1.234
    config.add_system(1.234, 'MgII', [2796.3, 2803.5], components=4)
    config.add_system(1.234, 'FeII', [2600.2, 2382.8, 2344.2], components=4)
    
    print(config)
    print("\n" + config.summary())
    
    # Show efficiency of parameter tying
    structure = config.get_parameter_structure()
    
    # Calculate what parameters would be without tying
    untied_params = 0
    for sys_info in structure['systems']:
        for ion_info in sys_info['ion_groups']:
            # Each transition would need separate params
            untied_params += (
                len(ion_info['transitions']) * ion_info['components'] * 3
            )
    
    print(f"\nParameter efficiency:")
    print(f"  With ion tying: {structure['total_parameters']} parameters")
    print(f"  Without tying: {untied_params} parameters")
    print(f"  Reduction: {untied_params - structure['total_parameters']} parameters "
          f"({100*(1 - structure['total_parameters']/untied_params):.1f}%)")
    
    return config


def example_serialization():
    """Example 5: Saving and loading configurations."""
    print("\n" + "=" * 60)
    print("Example 5: Configuration serialization")
    print("=" * 60)
    
    # Create a configuration
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.348, 'FeII', [2600.2], components=1)
    config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    
    print("Original configuration:")
    print(config.summary())
    
    # Serialize to JSON
    json_str = config.serialize()
    print("\nJSON representation:")
    print(json_str)
    
    # Save to file
    config.save('example_config.json')
    print("\nSaved to 'example_config.json'")
    
    # Load from file
    loaded_config = FitConfiguration.load('example_config.json')
    print("\nLoaded configuration:")
    print(loaded_config.summary())
    
    # Verify they're the same
    assert config.serialize() == loaded_config.serialize()
    print("\n✓ Round-trip serialization successful!")
    
    # Clean up
    import os
    os.remove('example_config.json')
    
    return config


def example_theta_array_usage():
    """Example 6: Understanding theta array organization."""
    print("\n" + "=" * 60)
    print("Example 6: Theta array organization")
    print("=" * 60)
    
    config = FitConfiguration()
    config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    config.add_system(0.348, 'FeII', [2600.2], components=1)
    
    structure = config.get_parameter_structure()
    
    print(config.summary())
    print(f"\nTotal parameters: {structure['total_parameters']}")
    
    # Create example theta array
    import numpy as np
    theta = np.zeros(structure['total_parameters'])
    
    # Set some example values
    # MgII: N=[13.5, 13.2], b=[15, 20], v=[-50, 0]
    theta[0] = 13.5  # N1
    theta[1] = 13.2  # N2
    theta[2] = 15.0  # b1
    theta[3] = 20.0  # b2
    theta[4] = -50.0 # v1
    theta[5] = 0.0   # v2
    
    # FeII: N=[12.8], b=[25], v=[10]
    theta[6] = 12.8  # N1
    theta[7] = 25.0  # b1
    theta[8] = 10.0  # v1
    
    print("\nTheta array:")
    print(f"theta = {theta}")
    
    print("\nParameter values:")
    pmap = structure['parameter_map']
    for param_name in ['N', 'b', 'v']:
        print(f"\n{param_name} values:")
        for key, idx in sorted(pmap.items()):
            if key.endswith(f'_{param_name}'):
                print(f"  {key}: {theta[idx]}")
    
    return config, theta


if __name__ == "__main__":
    # Run all examples
    example_simple_single_ion()
    example_multi_ion_same_system()
    example_multi_system()
    example_complex_sightline()
    example_serialization()
    example_theta_array_usage()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)