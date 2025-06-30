#!/usr/bin/env python
"""
Multi-Instrument Parameter Manager for rbvfit 2.0

This module handles the sequential collection and merging of parameters across
multiple instruments, avoiding the double-counting bug when the same physical
system appears in multiple configurations.

Author: rbvfit team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from rbvfit.core.fit_configuration import FitConfiguration


@dataclass
class SystemInfo:
    """Information about a physical absorption system."""
    z: float
    ion: str
    transitions: List[float]
    components: int
    source_instrument: str
    source_config: str
    parameters: pd.DataFrame  # DataFrame with columns: Component, N, b, v


@dataclass
class InstrumentMapping:
    """Mapping information for an instrument to master theta."""
    name: str
    config_name: str
    theta_indices: List[int]  # Which indices from master theta this instrument uses
    systems_used: List[Tuple[float, str]]  # List of (z, ion) tuples used by this instrument


@dataclass
class ParameterCollectionResult:
    """Result of parameter collection process."""
    master_theta: np.ndarray
    master_systems: List[SystemInfo]
    instrument_mappings: List[InstrumentMapping]
    master_config: FitConfiguration
    collection_log: List[str]  # Log messages for GUI display


class MultiInstrumentParameterManager:
    """
    Manages parameter collection and merging across multiple instruments.
    
    Uses sequential approach: process instruments one by one, adding only
    new systems to avoid duplication when the same physical system appears
    in multiple configurations.
    """
    
    def __init__(self):
        self.tolerance_z = 1e-6  # Redshift tolerance for matching systems
        
    def collect_and_merge_parameters(self, 
                                   config_systems: Dict[str, List[Dict]], 
                                   config_parameters: Dict[Tuple[str, str], pd.DataFrame],
                                   config_fit_configs: Dict[str, FitConfiguration]) -> ParameterCollectionResult:
        """
        Collect and merge parameters from multiple instruments sequentially.
        
        Parameters
        ----------
        config_systems : Dict[str, List[Dict]]
            Dictionary mapping config names to lists of system dictionaries
        config_parameters : Dict[Tuple[str, str], pd.DataFrame]  
            Dictionary mapping (config_name, system_id) to parameter DataFrames
        config_fit_configs : Dict[str, FitConfiguration]
            Dictionary mapping config names to FitConfiguration objects
            
        Returns
        -------
        ParameterCollectionResult
            Complete result with master theta, mappings, and logs
        """
        master_systems = []
        instrument_mappings = []
        collection_log = []
        
        # Sort configurations for consistent processing order
        sorted_configs = sorted(config_fit_configs.keys())
        
        collection_log.append(f"Starting parameter collection for {len(sorted_configs)} instruments")
        collection_log.append("=" * 60)
        
        # Process each instrument sequentially
        for config_idx, config_name in enumerate(sorted_configs):
            collection_log.append(f"\nProcessing Instrument {config_idx + 1}: {config_name}")
            
            systems = config_systems.get(config_name, [])
            config_obj = config_fit_configs[config_name]
            
            # Track which systems this instrument uses
            instrument_systems = []
            instrument_theta_indices = []
            
            collection_log.append(f"  Found {len(systems)} systems defined")
            
            # Process each system in this configuration
            for system in systems:
                z = system['z']
                ion = system['ion']
                system_id = system['id']
                
                # Check if this physical system already exists in master
                existing_system = self._find_existing_system(master_systems, z, ion)
                
                if existing_system is not None:
                    # System already exists - use existing parameters
                    collection_log.append(f"    {ion} z={z:.6f}: Already exists (from {existing_system.source_instrument}) - reusing parameters")
                    
                    # Add to instrument mapping
                    instrument_systems.append((z, ion))
                    system_theta_indices = self._get_system_theta_indices(existing_system, master_systems)
                    instrument_theta_indices.extend(system_theta_indices)
                    
                else:
                    # New system - check if we have parameters for it
                    param_key = (config_name, system_id)
                    
                    if param_key in config_parameters and not config_parameters[param_key].empty:
                        # We have parameters - add new system
                        df = config_parameters[param_key].copy()
                        df_sorted = df.sort_values('Component')
                        
                        new_system = SystemInfo(
                            z=z,
                            ion=ion,
                            transitions=system['transitions'],
                            components=len(df_sorted),
                            source_instrument=config_name,
                            source_config=config_name,
                            parameters=df_sorted
                        )
                        
                        master_systems.append(new_system)
                        collection_log.append(f"    {ion} z={z:.6f}: Added new system ({len(df_sorted)} components) from {config_name}")
                        
                        # Add to instrument mapping
                        instrument_systems.append((z, ion))
                        system_theta_indices = self._get_system_theta_indices(new_system, master_systems)
                        instrument_theta_indices.extend(system_theta_indices)
                        
                    else:
                        # No parameters available
                        collection_log.append(f"    {ion} z={z:.6f}: ⚠️  No parameters found - skipping")
                        # Note: This will cause compilation to fail later with clear error
            
            # Create instrument mapping
            mapping = InstrumentMapping(
                name=config_name,
                config_name=config_name,
                theta_indices=sorted(instrument_theta_indices),
                systems_used=instrument_systems
            )
            instrument_mappings.append(mapping)
            
            collection_log.append(f"    {config_name} uses {len(instrument_theta_indices)} parameters from {len(instrument_systems)} systems")
        
        # Build master theta array and configuration
        collection_log.append("\n" + "=" * 60)
        collection_log.append("BUILDING MASTER THETA ARRAY")
        
        master_theta, master_config = self._build_master_theta_and_config(master_systems, collection_log)
        
        # Validation
        self._validate_result(master_systems, instrument_mappings, collection_log)
        
        # Final summary
        collection_log.append("\n" + "=" * 60)
        collection_log.append("PARAMETER COLLECTION SUMMARY")
        collection_log.append(f"  Unique physical systems: {len(master_systems)}")
        collection_log.append(f"  Total parameters: {len(master_theta)}")
        collection_log.append(f"  Instruments: {len(instrument_mappings)}")
        
        for system in master_systems:
            collection_log.append(f"    {system.ion} z={system.z:.6f}: {system.components} components (from {system.source_instrument})")
        
        return ParameterCollectionResult(
            master_theta=master_theta,
            master_systems=master_systems,
            instrument_mappings=instrument_mappings,
            master_config=master_config,
            collection_log=collection_log
        )
    
    def _find_existing_system(self, master_systems: List[SystemInfo], z: float, ion: str) -> Optional[SystemInfo]:
        """Find if a physical system already exists in master list."""
        for system in master_systems:
            if (abs(system.z - z) < self.tolerance_z and 
                system.ion.upper() == ion.upper()):
                return system
        return None
    
    def _get_system_theta_indices(self, system: SystemInfo, master_systems: List[SystemInfo]) -> List[int]:
        """Get theta indices for a specific system within the master theta array."""
        # Find position of this system in master list
        system_idx = master_systems.index(system)
        
        # Calculate total components before this system
        components_before = sum(sys.components for sys in master_systems[:system_idx])
        
        # Calculate total components across all systems
        total_components = sum(sys.components for sys in master_systems)
        
        # Generate indices for this system's parameters
        indices = []
        
        # N parameters (first third of theta)
        for i in range(system.components):
            indices.append(components_before + i)
        
        # b parameters (second third of theta)
        for i in range(system.components):
            indices.append(total_components + components_before + i)
        
        # v parameters (final third of theta)
        for i in range(system.components):
            indices.append(2 * total_components + components_before + i)
        
        return indices
    
    def _build_master_theta_and_config(self, master_systems: List[SystemInfo], log: List[str]) -> Tuple[np.ndarray, FitConfiguration]:
        """Build master theta array and master configuration."""
        all_nguess = []
        all_bguess = []
        all_vguess = []
        
        # Create master configuration
        master_config = FitConfiguration()
        
        # Process systems in order
        for system in master_systems:
            # Add to master config
            master_config.add_system(
                z=system.z,
                ion=system.ion,
                transitions=system.transitions,
                components=system.components
            )
            
            # Extract parameters
            df_sorted = system.parameters.sort_values('Component')
            for _, row in df_sorted.iterrows():
                all_nguess.append(row['N'])
                all_bguess.append(row['b'])
                all_vguess.append(row['v'])
        
        # Build master theta following rbvfit structure
        master_theta = np.concatenate([all_nguess, all_bguess, all_vguess])
        
        log.append(f"  Master theta structure: [{len(all_nguess)} N, {len(all_bguess)} b, {len(all_vguess)} v] = {len(master_theta)} total")
        
        return master_theta, master_config
    
    def _validate_result(self, master_systems: List[SystemInfo], instrument_mappings: List[InstrumentMapping], log: List[str]):
        """Validate the parameter collection result."""
        log.append("\nVALIDATION:")
        
        # Check that all systems have parameters
        if not master_systems:
            raise ValueError("No systems with parameters found - please estimate parameters first")
        
        # Check that all instruments have at least one system
        for mapping in instrument_mappings:
            if not mapping.systems_used:
                log.append(f"  ⚠️  Warning: {mapping.name} has no systems with parameters")
            else:
                log.append(f"  ✓ {mapping.name}: {len(mapping.systems_used)} systems, {len(mapping.theta_indices)} parameters")
        
        log.append("  ✓ Validation complete")
    
    def update_master_theta(self, result: ParameterCollectionResult, new_theta: np.ndarray) -> ParameterCollectionResult:
        """Update master theta with user-edited values."""
        if len(new_theta) != len(result.master_theta):
            raise ValueError(f"New theta length ({len(new_theta)}) doesn't match expected ({len(result.master_theta)})")
        
        # Create updated result
        updated_result = ParameterCollectionResult(
            master_theta=new_theta.copy(),
            master_systems=result.master_systems,
            instrument_mappings=result.instrument_mappings,
            master_config=result.master_config,
            collection_log=result.collection_log + [f"Master theta updated by user ({len(new_theta)} parameters)"]
        )
        
        return updated_result
    
    def get_instrument_theta(self, result: ParameterCollectionResult, instrument_name: str) -> np.ndarray:
        """Extract theta subset for a specific instrument."""
        # Find instrument mapping
        mapping = None
        for m in result.instrument_mappings:
            if m.name == instrument_name:
                mapping = m
                break
        
        if mapping is None:
            raise ValueError(f"Instrument {instrument_name} not found in mappings")
        
        # Extract subset
        return result.master_theta[mapping.theta_indices]
    
    def get_parameter_info_table(self, result: ParameterCollectionResult) -> pd.DataFrame:
        """Create a table with parameter information for GUI display."""
        rows = []
        
        total_components = sum(sys.components for sys in result.master_systems)
        
        # Track parameter index
        param_idx = 0
        
        for sys_idx, system in enumerate(result.master_systems):
            for comp_idx in range(system.components):
                # N parameter
                rows.append({
                    'Parameter': f'N_{system.ion}_z{system.z:.6f}_c{comp_idx+1}',
                    'Type': 'N',
                    'System': f'{system.ion} z={system.z:.6f}',
                    'Component': comp_idx + 1,
                    'Value': result.master_theta[param_idx],
                    'Source': system.source_instrument,
                    'Theta_Index': param_idx,
                    'Instruments': ', '.join([m.name for m in result.instrument_mappings if param_idx in m.theta_indices])
                })
                param_idx += 1
        
        for sys_idx, system in enumerate(result.master_systems):
            for comp_idx in range(system.components):
                # b parameter  
                rows.append({
                    'Parameter': f'b_{system.ion}_z{system.z:.6f}_c{comp_idx+1}',
                    'Type': 'b',
                    'System': f'{system.ion} z={system.z:.6f}',
                    'Component': comp_idx + 1,
                    'Value': result.master_theta[param_idx],
                    'Source': system.source_instrument,
                    'Theta_Index': param_idx,
                    'Instruments': ', '.join([m.name for m in result.instrument_mappings if param_idx in m.theta_indices])
                })
                param_idx += 1
        
        for sys_idx, system in enumerate(result.master_systems):
            for comp_idx in range(system.components):
                # v parameter
                rows.append({
                    'Parameter': f'v_{system.ion}_z{system.z:.6f}_c{comp_idx+1}',
                    'Type': 'v', 
                    'System': f'{system.ion} z={system.z:.6f}',
                    'Component': comp_idx + 1,
                    'Value': result.master_theta[param_idx],
                    'Source': system.source_instrument,
                    'Theta_Index': param_idx,
                    'Instruments': ', '.join([m.name for m in result.instrument_mappings if param_idx in m.theta_indices])
                })
                param_idx += 1
        
        return pd.DataFrame(rows)