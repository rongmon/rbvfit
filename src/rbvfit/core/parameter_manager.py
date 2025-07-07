"""
Parameter management system for rbvfit 2.0 - Clean elegant approach.

This module handles the mapping between the hierarchical configuration structure
and the flat theta parameter array used for optimization. No multi-instrument
complexity - focuses purely on physics parameter organization.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
from rbvfit.core.fit_configuration import FitConfiguration, IonGroup


@dataclass
class ParameterBounds:
    """
    Container for parameter bounds.
    
    Attributes
    ----------
    lower : np.ndarray
        Lower bounds for each parameter
    upper : np.ndarray
        Upper bounds for each parameter
    """
    lower: np.ndarray
    upper: np.ndarray
    
    def __post_init__(self):
        """Validate bounds."""
        if len(self.lower) != len(self.upper):
            raise ValueError("Lower and upper bounds must have same length")
        if np.any(self.lower > self.upper):
            raise ValueError("Lower bounds must be <= upper bounds")


@dataclass
class ParameterSet:
    """
    Container for parameters of a single velocity component.
    
    Attributes
    ----------
    N : float
        Column density (log10 cm^-2)
    b : float
        Doppler parameter (km/s)
    v : float
        Velocity (km/s)
    """
    N: float
    b: float
    v: float


class ParameterManager:
    """
    Manages parameter mapping between configuration and theta arrays.
    
    This class handles:
    - Converting configurations to theta array structure
    - Mapping theta values back to individual line parameters
    - Generating appropriate bounds for parameters
    - Providing parameter names for output
    
    Focus: Pure physics parameter management with no instrument awareness.
    """
    
    def __init__(self, config: FitConfiguration):
        """
        Initialize parameter manager.
        
        Parameters
        ----------
        config : FitConfiguration
            The fitting configuration
        """
        self.config = config
        self.structure = config.get_parameter_structure()
        self._cached_line_params = None
        
    def config_to_theta_structure(self) -> Dict[str, Any]:
        """
        Get the theta array structure from configuration.
        
        Returns
        -------
        dict
            Structure information including total parameters and mapping
        """
        return self.structure
    
    def theta_to_parameters(self, theta: np.ndarray) -> Dict[Tuple[int, str, int], ParameterSet]:
        """
        Map theta array to individual line parameters.
        
        Parameters
        ----------
        theta : np.ndarray
            Flat parameter array [N1,N2,...,b1,b2,...,v1,v2,...]
            
        Returns
        -------
        dict
            Dictionary mapping (system_idx, ion_name, component_idx) to ParameterSet
        """
        if len(theta) != self.structure['total_parameters']:
            raise ValueError(
                f"Theta array length ({len(theta)}) doesn't match "
                f"expected parameters ({self.structure['total_parameters']})"
            )
        
        parameters = {}
        
        # Calculate total components across all ion groups
        total_components = 0
        for sys_info in self.structure['systems']:
            for ion_info in sys_info['ion_groups']:
                total_components += ion_info['components']
        
        # Split theta into global N, b, v sections
        N_section = theta[0:total_components]
        b_section = theta[total_components:2*total_components]
        v_section = theta[2*total_components:3*total_components]
        
        # Track global component index
        global_comp_idx = 0
        
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Extract parameters for this ion group from global sections
                N_values = N_section[global_comp_idx:global_comp_idx + n_comp]
                b_values = b_section[global_comp_idx:global_comp_idx + n_comp]
                v_values = v_section[global_comp_idx:global_comp_idx + n_comp]
                
                # Create ParameterSet for each component
                for comp_idx in range(n_comp):
                    key = (sys_idx, ion_name, comp_idx)
                    parameters[key] = ParameterSet(
                        N=N_values[comp_idx],
                        b=b_values[comp_idx],
                        v=v_values[comp_idx]
                    )
                
                # Advance global component index
                global_comp_idx += n_comp
        
        return parameters
    
    def parameters_to_theta(self, parameters: Dict[Tuple[int, str, int], ParameterSet]) -> np.ndarray:
        """
        Convert parameter dictionary back to theta array.
        
        Parameters
        ----------
        parameters : dict
            Dictionary mapping (system_idx, ion_name, component_idx) to ParameterSet
            
        Returns
        -------
        np.ndarray
            Flat theta array
        """
        # Collect all N, b, v values globally
        all_N_values = []
        all_b_values = []
        all_v_values = []
        
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Collect parameters for this ion group
                for comp_idx in range(n_comp):
                    key = (sys_idx, ion_name, comp_idx)
                    if key not in parameters:
                        raise ValueError(f"Missing parameters for {key}")
                    
                    param_set = parameters[key]
                    all_N_values.append(param_set.N)
                    all_b_values.append(param_set.b)
                    all_v_values.append(param_set.v)
        
        # Concatenate into theta array: [all_Ns, all_bs, all_vs]
        theta = np.concatenate([all_N_values, all_b_values, all_v_values])
        
        return theta
    
    def theta_to_line_parameters(self, theta: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert theta to parameters for each individual line.
        
        This method expands ion groups so each transition gets its own
        parameter set, respecting the ion tying.
        
        Parameters
        ----------
        theta : np.ndarray
            Flat parameter array
            
        Returns
        -------
        list
            List of dictionaries, one per transition, containing:
            - 'N': column density
            - 'b': Doppler parameter
            - 'v': velocity
            - 'z': redshift
            - 'wavelength': rest wavelength
        """
        if len(theta) != self.structure['total_parameters']:
            raise ValueError(
                f"Theta array length ({len(theta)}) doesn't match "
                f"expected parameters ({self.structure['total_parameters']})"
            )
        
        # First convert to parameter dictionary
        parameters = self.theta_to_parameters(theta)
        
        line_params = []
        
        # Now expand to individual lines
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Expand for each transition and component
                for wavelength in ion_info['transitions']:
                    for comp_idx in range(n_comp):
                        key = (sys_idx, ion_name, comp_idx)
                        param_set = parameters[key]
                        
                        line_params.append({
                            'N': param_set.N,
                            'b': param_set.b,
                            'v': param_set.v,
                            'z': z,
                            'wavelength': wavelength,
                            'ion': ion_name,
                            'system_idx': sys_idx,
                            'component_idx': comp_idx
                        })
        
        return line_params
    
    def get_parameter_names(self) -> List[str]:
        """
        Get human-readable names for each parameter.
        
        Returns
        -------
        list
            List of parameter names matching theta array order
        """
        names = []
        
        # Collect all N names first
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                for comp in range(n_comp):
                    names.append(f"N_{ion_name}_z{z:.3f}_c{comp+1}")
        
        # Then all b names
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                for comp in range(n_comp):
                    names.append(f"b_{ion_name}_z{z:.3f}_c{comp+1}")
        
        # Finally all v names
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                for comp in range(n_comp):
                    names.append(f"v_{ion_name}_z{z:.3f}_c{comp+1}")
        
        return names
    
    def get_parameter_latex_names(self) -> List[str]:
        """
        Get LaTeX-formatted names for each parameter.
        
        Returns
        -------
        list
            List of LaTeX parameter names for plotting
        """
        names = []
        
        # Collect all N names first
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Format ion name for LaTeX (e.g., MgII -> Mg II)
                latex_ion = ion_name
                if len(ion_name) > 2:
                    # Insert space before roman numerals
                    for i, char in enumerate(ion_name):
                        if char in 'IVX' and i > 0 and ion_name[i-1] not in 'IVX':
                            latex_ion = ion_name[:i] + r'\,' + ion_name[i:]
                            break
                
                for comp in range(n_comp):
                    names.append(rf"$\log N_{{{latex_ion}}}^{{({comp+1})}}$")
        
        # Then all b names
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Format ion name for LaTeX
                latex_ion = ion_name
                if len(ion_name) > 2:
                    for i, char in enumerate(ion_name):
                        if char in 'IVX' and i > 0 and ion_name[i-1] not in 'IVX':
                            latex_ion = ion_name[:i] + r'\,' + ion_name[i:]
                            break
                
                for comp in range(n_comp):
                    names.append(rf"$b_{{{latex_ion}}}^{{({comp+1})}}$")
        
        # Finally all v names
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Format ion name for LaTeX
                latex_ion = ion_name
                if len(ion_name) > 2:
                    for i, char in enumerate(ion_name):
                        if char in 'IVX' and i > 0 and ion_name[i-1] not in 'IVX':
                            latex_ion = ion_name[:i] + r'\,' + ion_name[i:]
                            break
                
                for comp in range(n_comp):
                    names.append(rf"$v_{{{latex_ion}}}^{{({comp+1})}}$")
        
        return names
    
    def generate_theta_bounds(self, ion_specific: bool = True) -> ParameterBounds:
        """
        Generate reasonable parameter bounds based on configuration.
        
        Parameters
        ----------
        ion_specific : bool, optional
            Whether to use ion-specific bounds (default: True)
            
        Returns
        -------
        ParameterBounds
            Parameter bounds object
        """
        n_params = self.structure['total_parameters']
        total_components = n_params // 3
        
        lower_bounds = np.zeros(n_params)
        upper_bounds = np.zeros(n_params)
        
        # Ion-specific bounds lookup table
        ion_bounds = {
            'HI': {'N': (12.0, 22.0), 'b': (5.0, 100.0), 'v': (-500.0, 500.0)},
            'CIV': {'N': (12.0, 16.0), 'b': (5.0, 80.0), 'v': (-200.0, 200.0)},
            'OVI': {'N': (13.0, 16.0), 'b': (10.0, 100.0), 'v': (-300.0, 300.0)},
            'SiIV': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-150.0, 150.0)},
            'MgII': {'N': (11.0, 16.0), 'b': (5.0, 80.0), 'v': (-100.0, 100.0)},
            'FeII': {'N': (11.0, 16.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
            'AlIII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
            'NV': {'N': (12.0, 15.0), 'b': (10.0, 80.0), 'v': (-200.0, 200.0)},
            'OI': {'N': (13.0, 16.0), 'b': (5.0, 50.0), 'v': (-100.0, 100.0)},
            'SiII': {'N': (11.0, 16.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
            'AlII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
            'CII': {'N': (13.0, 17.0), 'b': (5.0, 50.0), 'v': (-100.0, 100.0)},
            'NII': {'N': (13.0, 16.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
            'SiIII': {'N': (11.0, 15.0), 'b': (5.0, 60.0), 'v': (-100.0, 100.0)},
            'CIII': {'N': (13.0, 16.0), 'b': (5.0, 80.0), 'v': (-150.0, 150.0)},
        }
        
        # Default bounds for unknown ions
        default_bounds = {'N': (10.0, 18.0), 'b': (5.0, 100.0), 'v': (-300.0, 300.0)}
        
        # Build bounds for each parameter type
        comp_idx = 0
        
        for sys_info in self.structure['systems']:
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Get bounds for this ion
                if ion_specific and ion_name in ion_bounds:
                    bounds = ion_bounds[ion_name]
                else:
                    bounds = default_bounds
                
                # Set bounds for each component of this ion
                for i in range(n_comp):
                    # N bounds
                    lower_bounds[comp_idx + i] = bounds['N'][0]
                    upper_bounds[comp_idx + i] = bounds['N'][1]
                    
                    # b bounds
                    lower_bounds[total_components + comp_idx + i] = bounds['b'][0]
                    upper_bounds[total_components + comp_idx + i] = bounds['b'][1]
                    
                    # v bounds
                    lower_bounds[2 * total_components + comp_idx + i] = bounds['v'][0]
                    upper_bounds[2 * total_components + comp_idx + i] = bounds['v'][1]
                
                comp_idx += n_comp
        
        return ParameterBounds(lower_bounds, upper_bounds)
    
    def validate_theta(self, theta: np.ndarray) -> bool:
        """
        Validate that theta array matches expected structure.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter array to validate
            
        Returns
        -------
        bool
            True if valid, raises ValueError if not
        """
        expected = self.structure['total_parameters']
        if len(theta) != expected:
            raise ValueError(
                f"Invalid theta length: got {len(theta)}, expected {expected}"
            )
        
        if not np.all(np.isfinite(theta)):
            raise ValueError("Theta contains non-finite values")
        
        return True
    
    def get_summary_table(self, theta: np.ndarray) -> str:
        """
        Generate a formatted summary table of parameters.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameter array
            
        Returns
        -------
        str
            Formatted table string
        """
        params = self.theta_to_parameters(theta)
        
        lines = ["Parameter Summary", "=" * 60]
        
        for sys_idx, system in enumerate(self.config.systems):
            lines.append(f"\nSystem {sys_idx+1} (z={system.redshift:.6f}):")
            
            for ion_group in system.ion_groups:
                lines.append(f"\n  {ion_group.ion_name}:")
                lines.append("  " + "-" * 40)
                lines.append("  Comp |    N    |   b    |    v")
                lines.append("  " + "-" * 40)
                
                for comp in range(ion_group.components):
                    key = (sys_idx, ion_group.ion_name, comp)
                    p = params[key]
                    lines.append(
                        f"   {comp+1:2d}  | {p.N:7.2f} | {p.b:6.1f} | {p.v:7.1f}"
                    )
        
        return "\n".join(lines)
    
    def get_component_count(self) -> int:
        """
        Get total number of velocity components across all systems.
        
        Returns
        -------
        int
            Total number of components
        """
        return self.structure['total_parameters'] // 3
    
    def get_parameter_info(self, param_idx: int) -> Dict[str, Any]:
        """
        Get information about a specific parameter.
        
        Parameters
        ----------
        param_idx : int
            Index in theta array
            
        Returns
        -------
        dict
            Parameter information including type, system, ion, component
        """
        total_params = self.structure['total_parameters']
        total_components = total_params // 3
        
        if param_idx < 0 or param_idx >= total_params:
            raise ValueError(f"Parameter index {param_idx} out of range [0, {total_params})")
        
        # Determine parameter type
        if param_idx < total_components:
            param_type = 'N'
            comp_idx = param_idx
        elif param_idx < 2 * total_components:
            param_type = 'b'
            comp_idx = param_idx - total_components
        else:
            param_type = 'v'
            comp_idx = param_idx - 2 * total_components
        
        # Find which system/ion/component this corresponds to
        global_comp_idx = 0
        
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            for ion_info in sys_info['ion_groups']:
                n_comp = ion_info['components']
                
                if comp_idx < global_comp_idx + n_comp:
                    # This is the ion group
                    local_comp_idx = comp_idx - global_comp_idx
                    
                    return {
                        'type': param_type,
                        'system_idx': sys_idx,
                        'redshift': sys_info['redshift'],
                        'ion': ion_info['ion'],
                        'component_idx': local_comp_idx,
                        'global_component_idx': comp_idx,
                        'theta_idx': param_idx
                    }
                
                global_comp_idx += n_comp
        
        raise ValueError(f"Could not find parameter info for index {param_idx}")


if __name__ == "__main__":
    print("rbvfit 2.0 ParameterManager - Clean Physics-Only Implementation")
    print("Usage:")
    print("  config = FitConfiguration()")
    print("  config.add_system(z=0.5, ion='MgII', transitions=[2796.35, 2803.53], components=1)")
    print("  param_manager = ParameterManager(config)")
    print("  names = param_manager.get_parameter_names()")
    print("Happy fitting! 🎉")