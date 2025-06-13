"""
Configuration system for rbvfit 2.0 with automatic ion parameter tying.

This module provides the core configuration classes for setting up multi-group
absorption line fitting with automatic ion detection and parameter tying.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path

# Import the line database from rbvfit v1
try:
    from rbvfit import rb_setline as rb
except ImportError:
    # Fallback for development
    rb = None


@dataclass
class IonGroup:
    """
    Represents a group of transitions from the same ion at the same redshift.
    
    Parameters within an ion group are automatically tied together, meaning
    all transitions share the same N, b, and v parameters for each component.
    
    Attributes
    ----------
    ion_name : str
        Name of the ion (e.g., 'MgII', 'HI', 'OVI')
    transitions : List[float]
        Rest wavelengths of transitions in Angstroms
    components : int
        Number of velocity components for this ion
    redshift : float
        Redshift of the absorption system
    """
    ion_name: str
    transitions: List[float]
    components: int
    redshift: float
    
    def __post_init__(self):
        """Validate the ion group after initialization."""
        if self.components <= 0:
            raise ValueError(f"Number of components must be positive, got {self.components}")
        self.validate_transitions()
        
    def get_parameter_count(self) -> int:
        """
        Get the total number of parameters for this ion group.
        
        Returns
        -------
        int
            Number of parameters (3 * components for N, b, v)
        """
        return 3 * self.components  # N, b, v for each component
    
    def validate_transitions(self) -> None:
        """
        Validate that all transitions belong to the same ion.
        
        Raises
        ------
        ValueError
            If transitions don't belong to the same ion or are invalid
        """
        if not self.transitions:
            raise ValueError(f"No transitions provided for ion {self.ion_name}")
            
        if rb is not None:
            # Validate each transition belongs to the declared ion
            for wave in self.transitions:
                try:
                    line_info = rb.rb_setline(wave, 'closest')
                    detected_ion = self._extract_ion_name(line_info['name'][0])
                    if detected_ion != self.ion_name:
                        raise ValueError(
                            f"Transition {wave}Å belongs to {detected_ion}, "
                            f"not {self.ion_name}"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid transition {wave}Å: {str(e)}")
    
    @staticmethod
    def _extract_ion_name(line_name: str) -> str:
        """
        Extract ion name from line database format.
        
        Parameters
        ----------
        line_name : str
            Line name from database (e.g., "MgII 2796")
            
        Returns
        -------
        str
            Ion name (e.g., "MgII")
        """
        # Handle format "IonName Wavelength"
        parts = line_name.split()
        if len(parts) >= 2:
            return parts[0]
        return line_name
    
    def __repr__(self) -> str:
        return (f"IonGroup({self.ion_name}, transitions={self.transitions}, "
                f"components={self.components}, z={self.redshift})")


@dataclass
class AbsorptionSystem:
    """
    Represents an absorption system at a specific redshift.
    
    An absorption system can contain multiple ions, each with their own
    velocity components. Parameters are tied within each ion group.
    
    Attributes
    ----------
    redshift : float
        Redshift of the absorption system
    ion_groups : List[IonGroup]
        List of ion groups in this system
    """
    redshift: float
    ion_groups: List[IonGroup] = field(default_factory=list)
    
    def add_ion(self, ion_name: str, transitions: List[float], 
                components: int, merge: bool = False) -> None:
        """
        Add an ion to this absorption system.
        
        Parameters
        ----------
        ion_name : str
            Name of the ion (e.g., 'MgII')
        transitions : List[float]
            Rest wavelengths of transitions
        components : int
            Number of velocity components
        merge : bool, optional
            If True, merge transitions with existing ion if present.
            Components must match for merging.
        """
        # Check if ion already exists
        for group in self.ion_groups:
            if group.ion_name == ion_name:
                if merge:
                    # Merge transitions
                    if group.components != components:
                        raise ValueError(
                            f"Cannot merge ion {ion_name}: component mismatch "
                            f"(existing: {group.components}, new: {components})"
                        )
                    # Add new transitions, avoiding duplicates
                    for trans in transitions:
                        if trans not in group.transitions:
                            group.transitions.append(trans)
                    # Re-validate after adding transitions
                    group.validate_transitions()
                    return
                else:
                    raise ValueError(
                        f"Ion {ion_name} already exists in system at z={self.redshift}. "
                        f"Use merge=True to add transitions to existing ion."
                    )
        
        ion_group = IonGroup(ion_name, transitions, components, self.redshift)
        self.ion_groups.append(ion_group)
    
    def get_parameter_count(self) -> int:
        """Get total parameter count for this system."""
        return sum(group.get_parameter_count() for group in self.ion_groups)
    
    def get_ion_names(self) -> List[str]:
        """Get list of ion names in this system."""
        return [group.ion_name for group in self.ion_groups]


class FitConfiguration:
    """
    Main configuration class for rbvfit 2.0 fitting.
    
    This class manages the entire fitting configuration including multiple
    absorption systems, automatic ion detection, and parameter organization.
    
    Examples
    --------
    >>> config = FitConfiguration()
    >>> config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    >>> config.add_system(0.348, 'FeII', [2600.2], components=1)
    >>> config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    >>> structure = config.get_parameter_structure()
    """
    
    def __init__(self):
        """Initialize empty configuration."""
        self.systems: List[AbsorptionSystem] = []
        self._validated: bool = False
        
    def add_system(self, z: float, ion: str, transitions: List[float], 
                   components: int, merge: bool = False) -> None:
        """
        Add an absorption system or ion to the configuration.
        
        If a system at the given redshift already exists, the ion is added
        to that system. Otherwise, a new system is created.
        
        Parameters
        ----------
        z : float
            Redshift of the absorption system
        ion : str
            Ion name (e.g., 'MgII', 'HI', 'OVI')
        transitions : List[float]
            Rest wavelengths of transitions in Angstroms
        components : int
            Number of velocity components
        merge : bool, optional
            If True and ion already exists, merge transitions instead of raising error.
            Default is False for backward compatibility.
            
        Examples
        --------
        >>> config = FitConfiguration()
        >>> config.add_system(0.348, 'MgII', [2796.3], 2)
        >>> config.add_system(0.348, 'MgII', [2803.5], 2, merge=True)  # Adds to existing
        """
        # Check if system at this redshift already exists
        system = self._get_or_create_system(z)
        
        # Validate ion if automatic detection is available
        if ion == 'auto' and rb is not None:
            ion = self._detect_ion(transitions[0])
        
        system.add_ion(ion, transitions, components, merge=merge)
        self._validated = False
        
    def _get_or_create_system(self, z: float) -> AbsorptionSystem:
        """Get existing system at redshift z or create new one."""
        # Use small tolerance for redshift comparison
        z_tol = 1e-6
        for system in self.systems:
            if abs(system.redshift - z) < z_tol:
                return system
        
        # Create new system
        new_system = AbsorptionSystem(z)
        self.systems.append(new_system)
        return new_system
    
    def _detect_ion(self, wavelength: float) -> str:
        """
        Automatically detect ion from wavelength.
        
        Parameters
        ----------
        wavelength : float
            Rest wavelength in Angstroms
            
        Returns
        -------
        str
            Detected ion name
        """
        if rb is None:
            raise RuntimeError("Ion detection requires rbvfit v1 line database")
            
        line_info = rb.rb_setline(wavelength, 'closest')
        return IonGroup._extract_ion_name(line_info['name'][0])
    
    def get_parameter_structure(self) -> Dict[str, Any]:
        """
        Get the parameter structure for theta array organization.
        
        Returns
        -------
        dict
            Parameter structure including:
            - total_parameters: Total number of parameters
            - systems: List of system structures
            - parameter_map: Mapping of parameters to theta indices
        """
        self.validate()
        
        structure = {
            'total_parameters': 0,
            'systems': [],
            'parameter_map': {}
        }
        
        param_idx = 0
        
        for sys_idx, system in enumerate(self.systems):
            sys_info = {
                'redshift': system.redshift,
                'ion_groups': [],
                'parameter_indices': {}
            }
            
            for ion_group in system.ion_groups:
                n_params = ion_group.get_parameter_count()
                ion_info = {
                    'ion': ion_group.ion_name,
                    'transitions': ion_group.transitions,
                    'components': ion_group.components,
                    'parameter_slice': (param_idx, param_idx + n_params)
                }
                
                # Create parameter mapping
                for comp in range(ion_group.components):
                    base_idx = param_idx + comp
                    key = f"sys{sys_idx}_{ion_group.ion_name}_comp{comp}"
                    
                    structure['parameter_map'][f"{key}_N"] = base_idx
                    structure['parameter_map'][f"{key}_b"] = (
                        base_idx + ion_group.components
                    )
                    structure['parameter_map'][f"{key}_v"] = (
                        base_idx + 2 * ion_group.components
                    )
                
                sys_info['ion_groups'].append(ion_info)
                param_idx += n_params
            
            structure['systems'].append(sys_info)
        
        structure['total_parameters'] = param_idx
        return structure
    
    def validate(self) -> None:
        """
        Validate the configuration.
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        if self._validated:
            return
            
        if not self.systems:
            raise ValueError("No absorption systems defined")
        
        # Validate each system
        for system in self.systems:
            if not system.ion_groups:
                raise ValueError(
                    f"System at z={system.redshift} has no ions defined"
                )
        
        self._validated = True
    
    def serialize(self) -> str:
        """
        Serialize configuration to JSON string.
        
        Returns
        -------
        str
            JSON representation of configuration
        """
        self.validate()
        
        data = {
            'version': '2.0',
            'systems': []
        }
        
        for system in self.systems:
            sys_data = {
                'redshift': system.redshift,
                'ions': []
            }
            
            for ion_group in system.ion_groups:
                ion_data = {
                    'name': ion_group.ion_name,
                    'transitions': ion_group.transitions,
                    'components': ion_group.components
                }
                sys_data['ions'].append(ion_data)
            
            data['systems'].append(sys_data)
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def deserialize(cls, json_str: str) -> FitConfiguration:
        """
        Create configuration from JSON string.
        
        Parameters
        ----------
        json_str : str
            JSON representation of configuration
            
        Returns
        -------
        FitConfiguration
            Reconstructed configuration object
        """
        data = json.loads(json_str)
        
        if data.get('version') != '2.0':
            raise ValueError(f"Unsupported configuration version: {data.get('version')}")
        
        config = cls()
        
        for sys_data in data['systems']:
            z = sys_data['redshift']
            for ion_data in sys_data['ions']:
                config.add_system(
                    z,
                    ion_data['name'],
                    ion_data['transitions'],
                    ion_data['components']
                )
        
        return config
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.write_text(self.serialize())
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> FitConfiguration:
        """Load configuration from file."""
        filepath = Path(filepath)
        return cls.deserialize(filepath.read_text())
    
    def __repr__(self) -> str:
        n_systems = len(self.systems)
        n_ions = sum(len(s.ion_groups) for s in self.systems)
        n_params = sum(s.get_parameter_count() for s in self.systems)
        return (f"FitConfiguration({n_systems} systems, {n_ions} ion groups, "
                f"{n_params} parameters)")
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the configuration.
        
        Returns
        -------
        str
            Formatted summary string
        """
        self.validate()
        
        lines = ["FitConfiguration Summary", "=" * 50]
        
        for i, system in enumerate(self.systems):
            lines.append(f"\nSystem {i+1} (z={system.redshift:.6f}):")
            
            for ion_group in system.ion_groups:
                trans_str = ", ".join(f"{w:.1f}" for w in ion_group.transitions)
                lines.append(
                    f"  {ion_group.ion_name}: [{trans_str}] Å, "
                    f"{ion_group.components} components"
                )
        
        lines.append(f"\nTotal parameters: {sum(s.get_parameter_count() for s in self.systems)}")
        
        return "\n".join(lines)
    
    def append_transitions(self, z: float, ion: str, transitions: List[float]) -> None:
        """
        Append transitions to an existing ion in a system.
        
        This is a convenience method that calls add_system with merge=True.
        The ion must already exist in the system.
        
        Parameters
        ----------
        z : float
            Redshift of the absorption system
        ion : str
            Ion name that already exists in the system
        transitions : List[float]
            Additional rest wavelengths to add
            
        Examples
        --------
        >>> config = FitConfiguration()
        >>> config.add_system(0.348, 'MgII', [2796.3], 2)
        >>> config.append_transitions(0.348, 'MgII', [2803.5])  # Adds 2803.5
        
        Raises
        ------
        ValueError
            If the ion doesn't exist in the system
        """
        # Find the system
        system = None
        z_tol = 1e-6
        for sys in self.systems:
            if abs(sys.redshift - z) < z_tol:
                system = sys
                break
        
        if system is None:
            raise ValueError(f"No system found at redshift z={z}")
        
        # Find the ion
        ion_group = None
        for group in system.ion_groups:
            if group.ion_name == ion:
                ion_group = group
                break
        
        if ion_group is None:
            raise ValueError(f"Ion {ion} not found in system at z={z}")
        
        # Use add_system with merge=True
        self.add_system(z, ion, transitions, ion_group.components, merge=True)