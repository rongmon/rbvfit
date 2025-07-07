"""
Configuration system for rbvfit 2.0 - Clean elegant approach.

This defines the physics of absorption systems with no multi-instrument complexity.
For multi-instrument fitting, the same FitConfiguration is shared across
multiple VoigtModel instances with different FWHM values.
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
    validate_ion: bool = True
    
    def __post_init__(self):
        """Validate and correct the ion group after initialization."""
        if self.components <= 0:
            raise ValueError(f"Number of components must be positive, got {self.components}")
        self.validate_transitions()
    
    def set_components(self, new_components: int) -> None:
        """
        Update the number of components for this ion group.
        
        Parameters
        ----------
        new_components : int
            New number of velocity components
        """
        if new_components <= 0:
            raise ValueError(f"Number of components must be positive, got {new_components}")
        self.components = new_components
        
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
        Validate and correct transitions to use exact database wavelengths.
        
        This method:
        1. Always corrects wavelengths to exact database values
        2. Optionally validates that all transitions belong to the same ion
        
        Raises
        ------
        ValueError
            If transitions don't belong to the same ion (when validate_ion=True)
            or are invalid
        """
        if not self.transitions:
            raise ValueError(f"No transitions provided for ion {self.ion_name}")
        
        if rb is None:
            # Can't validate or correct without line database
            return
            
        corrected_transitions = []
        detected_ions = []
        
        # Process each transition
        for i, wave in enumerate(self.transitions):
            try:
                # Get exact wavelength and ion info from database
                line_info = rb.rb_setline(wave, 'closest')
                exact_wavelength = line_info['wave'][0]
                detected_ion = self._extract_ion_name(line_info['name'][0])
                
                # Always use the exact database wavelength
                corrected_transitions.append(exact_wavelength)
                detected_ions.append(detected_ion)
                
                # Optional validation that transition belongs to declared ion
                if self.validate_ion and detected_ion != self.ion_name:
                    raise ValueError(
                        f"Transition {wave}Å (corrected to {exact_wavelength}Å) "
                        f"belongs to {detected_ion}, not {self.ion_name}"
                    )
                    
            except Exception as e:
                raise ValueError(f"Invalid transition {wave}Å: {str(e)}")
        
        # Update with corrected wavelengths
        self.transitions = corrected_transitions
        
        # If validation is enabled, check all transitions belong to same ion
        if self.validate_ion and len(set(detected_ions)) > 1:
            raise ValueError(
                f"Mixed ion transitions detected: {set(detected_ions)}. "
                f"All transitions in an ion group must belong to the same ion."
            )
    
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
    
    def merge_transitions(self, other_transitions: List[float], validate_ion: bool = True) -> None:
        """
        Merge additional transitions into this ion group.
        
        Parameters
        ----------
        other_transitions : List[float]
            Additional transitions to add
        validate_ion : bool, optional
            Whether to validate new transitions belong to this ion
        """
        if rb is not None:
            # Correct and validate new transitions
            corrected_transitions = []
            for wave in other_transitions:
                try:
                    line_info = rb.rb_setline(wave, 'closest')
                    exact_wavelength = line_info['wave'][0]
                    
                    if validate_ion:
                        detected_ion = self._extract_ion_name(line_info['name'][0])
                        if detected_ion != self.ion_name:
                            raise ValueError(
                                f"Transition {wave}Å belongs to {detected_ion}, "
                                f"not {self.ion_name}"
                            )
                    
                    corrected_transitions.append(exact_wavelength)
                except Exception as e:
                    raise ValueError(f"Invalid transition {wave}Å: {str(e)}")
            
            # Add only new transitions (avoid duplicates)
            for trans in corrected_transitions:
                if trans not in self.transitions:
                    self.transitions.append(trans)
        else:
            # Without rb database, just add as-is
            for trans in other_transitions:
                if trans not in self.transitions:
                    self.transitions.append(trans)

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
                components: int, merge: bool = False, validate_ion: bool = True) -> None:
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
        validate_ion : bool, optional
            Whether to validate that all transitions belong to the same ion
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
                    group.merge_transitions(transitions, validate_ion)
                    return
                else:
                    raise ValueError(
                        f"Ion {ion_name} already exists in system at z={self.redshift}. "
                        f"Use merge=True to add transitions to existing ion."
                    )
        
        ion_group = IonGroup(ion_name, transitions, components, self.redshift, validate_ion)
        self.ion_groups.append(ion_group)
    
    def update_ion_components(self, ion_name: str, new_components: int) -> None:
        """
        Update the number of components for an existing ion.
        
        Parameters
        ----------
        ion_name : str
            Name of the ion to update
        new_components : int
            New number of velocity components
        """
        for group in self.ion_groups:
            if group.ion_name == ion_name:
                group.set_components(new_components)
                return
        
        raise ValueError(f"Ion {ion_name} not found in system at z={self.redshift}")
    
    def get_parameter_count(self) -> int:
        """Get total parameter count for this system."""
        return sum(group.get_parameter_count() for group in self.ion_groups)
    
    def get_ion_names(self) -> List[str]:
        """Get list of ion names in this system."""
        return [group.ion_name for group in self.ion_groups]


class FitConfiguration:
    """
    Main configuration class for rbvfit 2.0 fitting.
    
    This class defines the physics of absorption systems only.
    For multi-instrument fitting, the same FitConfiguration is shared
    across multiple VoigtModel instances with different FWHM values.
    
    Examples
    --------
    >>> config = FitConfiguration()
    >>> config.add_system(0.348, 'MgII', [2796.3, 2803.5], components=2)
    >>> config.add_system(0.348, 'FeII', [2600.2], components=1)
    >>> config.add_system(0.524, 'OVI', [1031.9, 1037.6], components=1)
    >>> structure = config.get_parameter_structure()
    """
    
    def __init__(self, FWHM=None, grating=None, life_position=None, cen_wave=None):
        """Initialize empty configuration."""
        self.systems: List[AbsorptionSystem] = []
        self._validated: bool = False
        self.instrumental_params = {}
        if FWHM is not None:
            self.instrumental_params['FWHM'] = FWHM
        if grating is not None:
            self.instrumental_params['grating'] = grating
        if life_position is not None:
            self.instrumental_params['life_position'] = life_position
        if cen_wave is not None:
            self.instrumental_params['cen_wave'] = cen_wave
        
    def add_system(self, z: float, ion: str = 'auto', transitions: List[float] = None, 
                   components: int = 1, merge: bool = False, validate_ion: bool = True) -> None:
        """
        Add an absorption system or ion to the configuration.
        
        Parameters
        ----------
        z : float
            Redshift of the absorption system
        ion : str, optional
            Ion name (e.g., 'MgII', 'HI', 'OVI'). Default 'auto' for automatic detection.
        transitions : List[float], optional
            Rest wavelengths of transitions in Angstroms
        components : int, optional
            Number of velocity components, default 1
        merge : bool, optional
            If True and ion already exists, merge transitions instead of raising error.
            Default is False.
        validate_ion : bool, optional
            Whether to validate that all transitions belong to the same ion.
            Default is True.
        """
        if transitions is None:
            raise ValueError("transitions list cannot be None")
            
        # Check if system at this redshift already exists
        system = self._get_or_create_system(z)
        
        # Validate ion if automatic detection is available
        if ion == 'auto' and rb is not None:
            ion = self._detect_ion(transitions[0])
        
        system.add_ion(ion, transitions, components, merge, validate_ion)
        self._validated = False
    
    def update_system_components(self, z: float, ion: str, new_components: int) -> None:
        """
        Update the number of components for an existing ion in a system.
        
        Parameters
        ----------
        z : float
            Redshift of the absorption system
        ion : str
            Name of the ion to update
        new_components : int
            New number of velocity components
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
        
        system.update_ion_components(ion, new_components)
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
            raise RuntimeError("Ion detection requires rbvfit line database")
            
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
    def deserialize(cls, json_str: str) -> 'FitConfiguration':
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
    def load(cls, filepath: Union[str, Path]) -> 'FitConfiguration':
        """Load configuration from file."""
        filepath = Path(filepath)
        return cls.deserialize(filepath.read_text())
    
    def append_transitions(self, z: float, ion: str, transitions: List[float], 
                          validate_ion: bool = True) -> None:
        """
        Append transitions to an existing ion in a system.
        
        Parameters
        ----------
        z : float
            Redshift of the absorption system
        ion : str
            Ion name that already exists in the system
        transitions : List[float]
            Additional rest wavelengths to add
        validate_ion : bool, optional
            Whether to validate new transitions belong to the specified ion
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
        self.add_system(z, ion, transitions, ion_group.components, 
                       merge=True, validate_ion=validate_ion)
    
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


if __name__ == "__main__":
    print("rbvfit 2.0 FitConfiguration - Clean Physics-Only Implementation")
    print("Usage:")
    print("  config = FitConfiguration()")
    print("  config.add_system(z=0.5, ion='MgII', transitions=[2796.35, 2803.53], components=1)")
    print("  model1 = VoigtModel(config, FWHM='2.2')  # Instrument 1")
    print("  model2 = VoigtModel(config, FWHM='4.0')  # Instrument 2")
    print("Happy fitting! 🎉")