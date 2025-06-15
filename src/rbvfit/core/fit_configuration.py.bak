"""
Parameter management system for rbvfit 2.0.

This module handles the mapping between the hierarchical configuration structure
and the flat theta parameter array used for optimization.
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
    
    def generate_parameter_bounds(self, custom_bounds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None) -> ParameterBounds:
        """
        Generate parameter bounds based on configuration.
        
        Parameters
        ----------
        custom_bounds : dict, optional
            Custom bounds per ion and parameter type. Format:
            {
                'ion_name': {
                    'N': (lower, upper),
                    'b': (lower, upper),
                    'v': (lower, upper)
                }
            }
            
        Returns
        -------
        ParameterBounds
            Object containing lower and upper bound arrays
        """
        n_params = self.structure['total_parameters']
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)
        
        # Default bounds
        default_bounds = {
            'N': (10.0, 22.0),      # log column density
            'b': (2.0, 200.0),      # Doppler parameter km/s
            'v': (-500.0, 500.0)    # velocity km/s
        }
        
        # Ion-specific default bounds
        ion_defaults = {
            'HI': {'N': (12.0, 22.0), 'b': (10.0, 200.0)},
            'MgII': {'N': (11.0, 18.0), 'b': (3.0, 100.0)},
            'FeII': {'N': (11.0, 17.0), 'b': (3.0, 100.0)},
            'OVI': {'N': (12.0, 16.0), 'b': (10.0, 150.0)},
            'CIV': {'N': (12.0, 16.0), 'b': (5.0, 150.0)},
            'SiII': {'N': (11.0, 17.0), 'b': (3.0, 100.0)},
            'CII': {'N': (12.0, 18.0), 'b': (3.0, 100.0)},
            'AlII': {'N': (10.0, 15.0), 'b': (3.0, 100.0)},
            'NV': {'N': (12.0, 16.0), 'b': (10.0, 150.0)},
            'SiIV': {'N': (12.0, 16.0), 'b': (5.0, 150.0)},
        }
        
        # Calculate total components across all ion groups
        total_components = 0
        for sys_info in self.structure['systems']:
            for ion_info in sys_info['ion_groups']:
                total_components += ion_info['components']
        
        # Track global component index
        global_comp_idx = 0
        
        # Apply bounds for each ion group using global indexing
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                
                # Get bounds for this ion
                if custom_bounds and ion_name in custom_bounds:
                    ion_bounds = custom_bounds[ion_name]
                elif ion_name in ion_defaults:
                    ion_bounds = ion_defaults[ion_name]
                else:
                    ion_bounds = default_bounds
                
                # Set bounds using global indexing
                # N bounds
                if 'N' in ion_bounds:
                    lo, hi = ion_bounds['N']
                else:
                    lo, hi = default_bounds['N']
                lower[global_comp_idx:global_comp_idx+n_comp] = lo
                upper[global_comp_idx:global_comp_idx+n_comp] = hi
                
                # b bounds
                if 'b' in ion_bounds:
                    lo, hi = ion_bounds['b']
                else:
                    lo, hi = default_bounds['b']
                lower[total_components+global_comp_idx:total_components+global_comp_idx+n_comp] = lo
                upper[total_components+global_comp_idx:total_components+global_comp_idx+n_comp] = hi
                
                # v bounds
                if 'v' in ion_bounds:
                    lo, hi = ion_bounds['v']
                else:
                    lo, hi = default_bounds['v']
                lower[2*total_components+global_comp_idx:2*total_components+global_comp_idx+n_comp] = lo
                upper[2*total_components+global_comp_idx:2*total_components+global_comp_idx+n_comp] = hi
                
                # Advance global component index
                global_comp_idx += n_comp
        
        return ParameterBounds(lower, upper)
    
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
                    names.append(f"N_{ion_name}_z{z:.3f}_c{comp}")
        
        # Then all b names
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                for comp in range(n_comp):
                    names.append(f"b_{ion_name}_z{z:.3f}_c{comp}")
        
        # Finally all v names
        for sys_idx, sys_info in enumerate(self.structure['systems']):
            z = sys_info['redshift']
            for ion_info in sys_info['ion_groups']:
                ion_name = ion_info['ion']
                n_comp = ion_info['components']
                for comp in range(n_comp):
                    names.append(f"v_{ion_name}_z{z:.3f}_c{comp}")
        
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