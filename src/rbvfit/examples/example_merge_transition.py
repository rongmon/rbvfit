#!/usr/bin/env python
"""
Example demonstrating the transition merging functionality in rbvfit 2.0
"""

from rbvfit.core.fit_configuration import FitConfiguration


def example_merge_transitions():
    """Example showing different ways to add transitions."""
    
    print("=" * 60)
    print("Example: Adding transitions to existing ions")
    print("=" * 60)
    
    # Method 1: Using merge=True
    print("\nMethod 1: Using merge=True parameter")
    config1 = FitConfiguration()
    
    # Add MgII with just one transition
    config1.add_system(z=0.1, ion='MgII', transitions=[2796.3], components=2)
    print("After adding 2796.3:")
    print(config1.summary())
    
    # Add the second transition using merge=True
    config1.add_system(z=0.1, ion='MgII', transitions=[2803.5], components=2, merge=True)
    print("\nAfter merging 2803.5:")
    print(config1.summary())
    
    # Method 2: Using append_transitions
    print("\n" + "=" * 60)
    print("Method 2: Using append_transitions")
    config2 = FitConfiguration()
    
    # Start with FeII with two transitions
    config2.add_system(z=0.2, ion='FeII', transitions=[2344.2, 2382.8], components=3)
    print("Initial FeII:")
    print(config2.summary())
    
    # Append more transitions
    config2.append_transitions(z=0.2, ion='FeII', transitions=[2600.2, 2586.7])
    print("\nAfter appending more transitions:")
    print(config2.summary())
    
    # Method 3: Building up SiII multiplet gradually
    print("\n" + "=" * 60)
    print("Method 3: Building SiII multiplet gradually")
    config3 = FitConfiguration()
    
    # Start with one SiII transition
    config3.add_system(z=0.5, ion='SiII', transitions=[1260.42], components=2)
    
    # Add more one by one
    config3.append_transitions(z=0.5, ion='SiII', transitions=[1190.42])
    config3.append_transitions(z=0.5, ion='SiII', transitions=[1193.29])
    config3.append_transitions(z=0.5, ion='SiII', transitions=[1526.71])
    
    print("Final SiII configuration:")
    print(config3.summary())
    
    # Show parameter structure remains consistent
    structure = config3.get_parameter_structure()
    print(f"\nTotal parameters: {structure['total_parameters']} (still just 6 for 2 components)")
    
    # Error handling examples
    print("\n" + "=" * 60)
    print("Error handling examples")
    
    # Example 1: Trying to add with different components
    print("\nExample 1: Component mismatch")
    config4 = FitConfiguration()
    config4.add_system(z=0.3, ion='OVI', transitions=[1031.9], components=2)
    
    try:
        # This will fail - different number of components
        config4.add_system(z=0.3, ion='OVI', transitions=[1037.6], components=3, merge=True)
    except ValueError as e:
        print(f"Error caught: {e}")
    
    # Example 2: Trying to append to non-existent ion
    print("\nExample 2: Non-existent ion")
    try:
        config4.append_transitions(z=0.3, ion='NV', transitions=[1238.8])
    except ValueError as e:
        print(f"Error caught: {e}")
    
    # Example 3: Forgetting merge=True
    print("\nExample 3: Forgetting merge=True")
    try:
        config4.add_system(z=0.3, ion='OVI', transitions=[1037.6], components=2)  # No merge=True
    except ValueError as e:
        print(f"Error caught: {e}")
    
    return config1, config2, config3


if __name__ == "__main__":
    example_merge_transitions()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)