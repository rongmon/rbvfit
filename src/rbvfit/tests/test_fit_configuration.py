"""
Comprehensive test suite for FitConfiguration with ion tying logic.
"""

import pytest
import json
from pathlib import Path
import tempfile

# Import the configuration classes
from rbvfit.core.fit_configuration import FitConfiguration, IonGroup, AbsorptionSystem


class TestIonGroup:
    """Test the IonGroup class."""
    
    def test_basic_creation(self):
        """Test basic IonGroup creation."""
        ion = IonGroup('MgII', [2796.3, 2803.5], 2, 0.348)
        
        assert ion.ion_name == 'MgII'
        assert ion.transitions == [2796.3, 2803.5]
        assert ion.components == 2
        assert ion.redshift == 0.348
    
    def test_parameter_count(self):
        """Test parameter counting."""
        # Single component = 3 parameters (N, b, v)
        ion1 = IonGroup('HI', [1215.67], 1, 0.0)
        assert ion1.get_parameter_count() == 3
        
        # Two components = 6 parameters
        ion2 = IonGroup('MgII', [2796.3, 2803.5], 2, 0.348)
        assert ion2.get_parameter_count() == 6
        
        # Three components = 9 parameters
        ion3 = IonGroup('OVI', [1031.9, 1037.6], 3, 0.524)
        assert ion3.get_parameter_count() == 9
    
    def test_empty_transitions_error(self):
        """Test that empty transitions raise error."""
        with pytest.raises(ValueError, match="No transitions provided"):
            IonGroup('MgII', [], 2, 0.348)
    
    def test_extract_ion_name(self):
        """Test ion name extraction from line database format."""
        assert IonGroup._extract_ion_name("MgII 2796") == "MgII"
        assert IonGroup._extract_ion_name("HI 1215") == "HI"
        assert IonGroup._extract_ion_name("OVI 1031") == "OVI"
        assert IonGroup._extract_ion_name("SiII 1260") == "SiII"
        assert IonGroup._extract_ion_name("SingleName") == "SingleName"


class TestAbsorptionSystem:
    """Test the AbsorptionSystem class."""
    
    def test_basic_creation(self):
        """Test basic system creation."""
        system = AbsorptionSystem(0.348)
        assert system.redshift == 0.348
        assert system.ion_groups == []
        assert system.get_parameter_count() == 0
    
    def test_add_single_ion(self):
        """Test adding a single ion."""
        system = AbsorptionSystem(0.348)
        system.add_ion('MgII', [2796.3, 2803.5], 2)
        
        assert len(system.ion_groups) == 1
        assert system.ion_groups[0].ion_name == 'MgII'
        assert system.get_parameter_count() == 6
        assert system.get_ion_names() == ['MgII']
    
    def test_add_multiple_ions(self):
        """Test adding multiple ions to same system."""
        system = AbsorptionSystem(0.348)
        system.add_ion('MgII', [2796.3, 2803.5], 2)
        system.add_ion('FeII', [2600.2, 2382.8], 2)
        system.add_ion('HI', [1215.67], 1)
        
        assert len(system.ion_groups) == 3
        assert system.get_parameter_count() == 15  # 6 + 6 + 3
        assert system.get_ion_names() == ['MgII', 'FeII', 'HI']
    
    def test_duplicate_ion_error(self):
        """Test that adding duplicate ion raises error."""
        system = AbsorptionSystem(0.348)
        system.add_ion('MgII', [2796.3, 2803.5], 2)
        
        with pytest.raises(ValueError, match="Ion MgII already exists"):
            system.add_ion('MgII', [2796.3], 1)


class TestFitConfiguration:
    """Test the main FitConfiguration class."""
    
    def test_basic_creation(self):
        """Test basic configuration creation."""
        config = FitConfiguration()
        assert config.systems == []
        assert not config._validated
    
    def test_add_single_system_single_ion(self):
        """Test adding single system with single ion."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        
        assert len(config.systems) == 1
        assert config.systems[0].redshift == 0.348
        assert len(config.systems[0].ion_groups) == 1
        assert config.systems[0].ion_groups[0].ion_name == 'MgII'
    
    def test_add_multiple_ions_same_system(self):
        """Test adding multiple ions to same redshift creates single system."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 1)
        
        # Should have only one system
        assert len(config.systems) == 1
        assert len(config.systems[0].ion_groups) == 2
        assert config.systems[0].get_ion_names() == ['MgII', 'FeII']
    
    def test_add_multiple_systems(self):
        """Test adding multiple systems at different redshifts."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 1)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 1)
        
        assert len(config.systems) == 2
        assert config.systems[0].redshift == 0.348
        assert config.systems[1].redshift == 0.524
        assert len(config.systems[0].ion_groups) == 2
        assert len(config.systems[1].ion_groups) == 1
    
    def test_parameter_structure(self):
        """Test parameter structure generation."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 1)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 1)
        
        structure = config.get_parameter_structure()
        
        # Both transitions should use same 6 parameters
        assert structure['total_parameters'] == 6
        mgii_group = structure['systems'][0]['ion_groups'][0]
        assert mgii_group['transitions'] == [2796.3, 2803.5]
        assert mgii_group['components'] == 2
        assert mgii_group['parameter_slice'] == (0, 6)
    
    def test_independent_parameters_different_ions(self):
        """Test that different ions have independent parameters."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 2)
        
        structure = config.get_parameter_structure()
        
        # Should have 12 total parameters (6 for MgII, 6 for FeII)
        assert structure['total_parameters'] == 12
        
        # Check they have separate parameter slices
        mgii_slice = structure['systems'][0]['ion_groups'][0]['parameter_slice']
        feii_slice = structure['systems'][0]['ion_groups'][1]['parameter_slice']
        
        assert mgii_slice == (0, 6)
        assert feii_slice == (6, 12)
    
    def test_independent_parameters_same_ion_different_z(self):
        """Test that same ion at different z has independent parameters."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 1)
        config.add_system(0.524, 'MgII', [2796.3, 2803.5], 1)
        
        structure = config.get_parameter_structure()
        
        # Should have 6 total parameters (3 for each system)
        assert structure['total_parameters'] == 6
        
        # Check they have separate parameter slices
        sys0_mgii = structure['systems'][0]['ion_groups'][0]['parameter_slice']
        sys1_mgii = structure['systems'][1]['ion_groups'][0]['parameter_slice']
        
        assert sys0_mgii == (0, 3)
        assert sys1_mgii == (3, 6)
    
    def test_complex_multi_ion_system(self):
        """Test complex system with multiple ions and transitions."""
        config = FitConfiguration()
        # System 1: MgII (tied) + FeII (tied) + HI
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2, 2382.8], 2)
        config.add_system(0.348, 'HI', [1215.67], 1)
        # System 2: OVI (tied)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 3)
        
        structure = config.get_parameter_structure()
        
        # Total: MgII(6) + FeII(6) + HI(3) + OVI(9) = 24
        assert structure['total_parameters'] == 24
        
        # Verify each ion group
        sys0_ions = structure['systems'][0]['ion_groups']
        sys1_ions = structure['systems'][1]['ion_groups']
        
        assert len(sys0_ions) == 3
        assert len(sys1_ions) == 1
        
        # Check MgII tying
        assert sys0_ions[0]['ion'] == 'MgII'
        assert len(sys0_ions[0]['transitions']) == 2
        assert sys0_ions[0]['parameter_slice'] == (0, 6)
        
        # Check FeII tying
        assert sys0_ions[1]['ion'] == 'FeII'
        assert len(sys0_ions[1]['transitions']) == 2
        assert sys0_ions[1]['parameter_slice'] == (6, 12)
        
        # Check OVI tying
        assert sys1_ions[0]['ion'] == 'OVI'
        assert len(sys1_ions[0]['transitions']) == 2
        assert sys1_ions[0]['parameter_slice'] == (15, 24)
    
    def test_parameter_savings_with_multiplets(self):
        """Test parameter savings for ions with many transitions."""
        config = FitConfiguration()
        # SiII with 4 transitions
        config.add_system(0.5, 'SiII', [1190.42, 1193.29, 1260.42, 1526.71], 3)
        
        structure = config.get_parameter_structure()
        
        # With tying: 3 components * 3 params = 9 total
        assert structure['total_parameters'] == 9
        
        # Without tying: 4 transitions * 3 components * 3 params = 36
        # Savings: 27 parameters (75% reduction)
        
        siII_group = structure['systems'][0]['ion_groups'][0]
        assert len(siII_group['transitions']) == 4
        assert siII_group['components'] == 3
        assert siII_group['parameter_slice'] == (0, 9)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_transition_ion(self):
        """Test ion with single transition."""
        config = FitConfiguration()
        config.add_system(0.348, 'HI', [1215.67], 2)
        
        structure = config.get_parameter_structure()
        assert structure['total_parameters'] == 6
        
        hi_group = structure['systems'][0]['ion_groups'][0]
        assert len(hi_group['transitions']) == 1
        assert hi_group['components'] == 2
    
    def test_many_component_system(self):
        """Test system with many velocity components."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 10)
        
        structure = config.get_parameter_structure()
        # 10 components * 3 parameters = 30
        assert structure['total_parameters'] == 30
    
    def test_very_close_redshifts_same_system(self):
        """Test that very close redshifts are treated as same system."""
        config = FitConfiguration()
        config.add_system(0.348000, 'MgII', [2796.3, 2803.5], 1)
        config.add_system(0.348000001, 'FeII', [2600.2], 1)  # Within tolerance
        
        # Should have single system
        assert len(config.systems) == 1
        assert len(config.systems[0].ion_groups) == 2
    
    def test_zero_components_error(self):
        """Test that zero components raises error during validation."""
        config = FitConfiguration()
        # This should work initially
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 0)
        
        # But parameter structure should catch it
        with pytest.raises(ValueError):
            config.get_parameter_structure()
    
    def test_negative_components_error(self):
        """Test that negative components raises error."""
        config = FitConfiguration()
        with pytest.raises(ValueError):
            config.add_system(0.348, 'MgII', [2796.3, 2803.5], -1)
    
    def test_empty_transitions_list(self):
        """Test that empty transitions list raises error."""
        config = FitConfiguration()
        with pytest.raises(ValueError, match="No transitions provided"):
            config.add_system(0.348, 'MgII', [], 2)
    
    def test_invalid_version_deserialization(self):
        """Test deserialization with wrong version."""
        json_str = '{"version": "1.0", "systems": []}'
        
        with pytest.raises(ValueError, match="Unsupported configuration version"):
            FitConfiguration.deserialize(json_str)


class TestParameterMapping:
    """Test parameter mapping functionality."""
    
    def test_parameter_map_indexing(self):
        """Test that parameter map provides correct indices."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 1)
        
        structure = config.get_parameter_structure()
        pmap = structure['parameter_map']
        
        # Check all MgII parameters
        assert pmap['sys0_MgII_comp0_N'] == 0
        assert pmap['sys0_MgII_comp1_N'] == 1
        assert pmap['sys0_MgII_comp0_b'] == 2
        assert pmap['sys0_MgII_comp1_b'] == 3
        assert pmap['sys0_MgII_comp0_v'] == 4
        assert pmap['sys0_MgII_comp1_v'] == 5
        
        # Check OVI parameters
        assert pmap['sys1_OVI_comp0_N'] == 6
        assert pmap['sys1_OVI_comp0_b'] == 7
        assert pmap['sys1_OVI_comp0_v'] == 8
    
    def test_parameter_organization_nbv(self):
        """Test that parameters are organized as [N1,N2,...,b1,b2,...,v1,v2,...]."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 3)
        
        structure = config.get_parameter_structure()
        pmap = structure['parameter_map']
        
        # N parameters should be 0,1,2
        assert pmap['sys0_MgII_comp0_N'] == 0
        assert pmap['sys0_MgII_comp1_N'] == 1
        assert pmap['sys0_MgII_comp2_N'] == 2
        
        # b parameters should be 3,4,5
        assert pmap['sys0_MgII_comp0_b'] == 3
        assert pmap['sys0_MgII_comp1_b'] == 4
        assert pmap['sys0_MgII_comp2_b'] == 5
        
        # v parameters should be 6,7,8
        assert pmap['sys0_MgII_comp0_v'] == 6
        assert pmap['sys0_MgII_comp1_v'] == 7
        assert pmap['sys0_MgII_comp2_v'] == 8


class TestRealWorldScenarios:
    """Test realistic fitting scenarios."""
    
    def test_typical_cos_observation(self):
        """Test typical COS observation setup."""
        config = FitConfiguration()
        
        # Main system - strong MgII absorber
        config.add_system(0.7251, 'MgII', [2796.3, 2803.5], components=3)
        config.add_system(0.7251, 'FeII', [2344.2, 2374.5, 2382.8], components=2)
        
        # Contaminating Lyman alpha
        config.add_system(1.234, 'HI', [1215.67], components=1)
        
        structure = config.get_parameter_structure()
        
        # MgII: 3*3=9, FeII: 2*3=6, HI: 1*3=3, Total: 18
        assert structure['total_parameters'] == 18
        
        # Without tying would be: MgII(2*3*3=18) + FeII(3*2*3=18) + HI(3) = 39
        # Savings: 21 parameters (54% reduction)
    
    def test_dla_system(self):
        """Test DLA system with many ions."""
        config = FitConfiguration()
        z_dla = 2.456
        
        # Multiple ions typical in DLA
        ions_and_transitions = [
            ('HI', [1215.67], 5),  # Broad HI with multiple components
            ('SiII', [1190.42, 1193.29, 1260.42, 1526.71], 3),
            ('CII', [1334.53], 3),
            ('OI', [1302.17], 2),
            ('AlII', [1670.79], 2),
            ('FeII', [1608.45, 2344.2, 2382.8], 2),
        ]
        
        for ion, transitions, ncomp in ions_and_transitions:
            config.add_system(z_dla, ion, transitions, ncomp)
        
        structure = config.get_parameter_structure()
        
        # Calculate expected parameters
        expected = sum(ncomp * 3 for _, _, ncomp in ions_and_transitions)
        assert structure['total_parameters'] == expected
        
        # Verify all ions present
        ion_names = [g['ion'] for g in structure['systems'][0]['ion_groups']]
        assert set(ion_names) == {'HI', 'SiII', 'CII', 'OI', 'AlII', 'FeII'}
    
    def test_blended_systems(self):
        """Test heavily blended absorption systems."""
        config = FitConfiguration()
        
        # Three systems with overlapping velocity structure
        config.add_system(1.5432, 'CIV', [1548.2, 1550.8], components=4)
        config.add_system(1.5440, 'CIV', [1548.2, 1550.8], components=2)  # Close by
        config.add_system(1.5445, 'SiIV', [1393.8, 1402.8], components=3)
        
        structure = config.get_parameter_structure()
        
        # Should have 3 separate systems despite close redshifts
        assert len(structure['systems']) == 3
        assert structure['total_parameters'] == (4 + 2 + 3) * 3
    
    def test_associated_absorber(self):
        """Test associated absorber with high ionization."""
        config = FitConfiguration()
        z_qso = 2.134
        
        # High ionization associated absorber
        config.add_system(z_qso, 'CIV', [1548.2, 1550.8], components=5)
        config.add_system(z_qso, 'NV', [1238.8, 1242.8], components=5)
        config.add_system(z_qso, 'OVI', [1031.9, 1037.6], components=3)
        
        # Intervening system
        config.add_system(1.876, 'MgII', [2796.3, 2803.5], components=2)
        
        structure = config.get_parameter_structure()
        
        assert len(structure['systems']) == 2
        assert structure['total_parameters'] == (5 + 5 + 3 + 2) * 3


class TestConfigurationUtilities:
    """Test utility methods and edge cases."""
    
    def test_get_total_transitions(self):
        """Test counting total transitions across configuration."""
        config = FitConfiguration()
        config.add_system(0.5, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.5, 'SiII', [1190.42, 1193.29, 1260.42, 1526.71], 1)
        config.add_system(0.7, 'OVI', [1031.9, 1037.6], 1)
        
        # Count total transitions
        total_transitions = 0
        for system in config.systems:
            for ion_group in system.ion_groups:
                total_transitions += len(ion_group.transitions)
        
        assert total_transitions == 8  # 2 + 4 + 2
    
    def test_configuration_copying(self):
        """Test that configurations can be properly copied."""
        config1 = FitConfiguration()
        config1.add_system(0.5, 'MgII', [2796.3, 2803.5], 2)
        
        # Create from serialization (effective copy)
        config2 = FitConfiguration.deserialize(config1.serialize())
        
        # Modify config2
        config2.add_system(0.7, 'OVI', [1031.9, 1037.6], 1)
        
        # Original should be unchanged
        assert len(config1.systems) == 1
        assert len(config2.systems) == 2
    
    def test_parameter_map_completeness(self):
        """Test that parameter map covers all parameters."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 3)
        
        structure = config.get_parameter_structure()
        pmap = structure['parameter_map']
        
        # Collect all indices from parameter map
        all_indices = set(pmap.values())
        
        # Should cover 0 to total_parameters-1
        expected_indices = set(range(structure['total_parameters']))
        assert all_indices == expected_indices
    
    def test_summary_formatting(self):
        """Test that summary is properly formatted."""
        config = FitConfiguration()
        config.add_system(0.348123456, 'MgII', [2796.352, 2803.531], 2)
        
        summary = config.summary()
        
        # Check formatting
        assert "z=0.348123" in summary  # 6 decimal places
        assert "2796.4" in summary  # 1 decimal for wavelengths
        assert "2803.5" in summary
        assert "2 components" in summary
        assert "Total parameters: 6" in summary


class TestFutureCompatibility:
    """Test features for future extensions."""
    
    def test_metadata_storage(self):
        """Test that configuration can be extended with metadata."""
        config = FitConfiguration()
        config.add_system(0.5, 'MgII', [2796.3, 2803.5], 2)
        
        # Add custom metadata (for future use)
        config.metadata = {
            'instrument': 'HST/COS',
            'grating': 'G130M',
            'dataset': 'test_spectrum.fits',
            'date': '2024-01-15'
        }
        
        # Should still serialize/deserialize basic config
        json_str = config.serialize()
        config2 = FitConfiguration.deserialize(json_str)
        
        # Basic config preserved
        assert len(config2.systems) == 1
        assert config2.systems[0].ion_groups[0].ion_name == 'MgII'
    
    def test_custom_ion_properties(self):
        """Test that ions can have custom properties in future."""
        # This tests the extensibility of the design
        ion = IonGroup('MgII', [2796.3, 2803.5], 2, 0.348)
        
        # Add custom properties (for future use)
        ion.custom_bounds = {'b_min': 5.0, 'b_max': 100.0}
        ion.tied_to_other_ion = None
        ion.fixed_parameters = []
        
        # Core functionality still works
        assert ion.get_parameter_count() == 6
        assert ion.ion_name == 'MgII'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) Check total parameters: MgII(6) + FeII(3) + OVI(3) = 12
        assert structure['total_parameters'] == 12
        assert len(structure['systems']) == 2
        
        # Check first system (MgII + FeII)
        sys0 = structure['systems'][0]
        assert sys0['redshift'] == 0.348
        assert len(sys0['ion_groups']) == 2
        
        # Check MgII parameters
        mgii = sys0['ion_groups'][0]
        assert mgii['ion'] == 'MgII'
        assert mgii['components'] == 2
        assert mgii['parameter_slice'] == (0, 6)
        
        # Check FeII parameters
        feii = sys0['ion_groups'][1]
        assert feii['ion'] == 'FeII'
        assert feii['components'] == 1
        assert feii['parameter_slice'] == (6, 9)
        
        # Check parameter map
        pmap = structure['parameter_map']
        # MgII component 0
        assert pmap['sys0_MgII_comp0_N'] == 0
        assert pmap['sys0_MgII_comp0_b'] == 2
        assert pmap['sys0_MgII_comp0_v'] == 4
        # MgII component 1
        assert pmap['sys0_MgII_comp1_N'] == 1
        assert pmap['sys0_MgII_comp1_b'] == 3
        assert pmap['sys0_MgII_comp1_v'] == 5
        # FeII component 0
        assert pmap['sys0_FeII_comp0_N'] == 6
        assert pmap['sys0_FeII_comp0_b'] == 7
        assert pmap['sys0_FeII_comp0_v'] == 8
    
    def test_validation_empty_config(self):
        """Test validation fails for empty configuration."""
        config = FitConfiguration()
        with pytest.raises(ValueError, match="No absorption systems defined"):
            config.validate()
    
    def test_validation_empty_system(self):
        """Test validation fails for system with no ions."""
        config = FitConfiguration()
        # Manually add empty system
        config.systems.append(AbsorptionSystem(0.348))
        
        with pytest.raises(ValueError, match="has no ions defined"):
            config.validate()
    
    def test_serialization(self):
        """Test configuration serialization to JSON."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 1)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 1)
        
        json_str = config.serialize()
        data = json.loads(json_str)
        
        assert data['version'] == '2.0'
        assert len(data['systems']) == 2
        
        # Check first system
        sys0 = data['systems'][0]
        assert sys0['redshift'] == 0.348
        assert len(sys0['ions']) == 2
        assert sys0['ions'][0]['name'] == 'MgII'
        assert sys0['ions'][0]['transitions'] == [2796.3, 2803.5]
        assert sys0['ions'][0]['components'] == 2
    
    def test_deserialization(self):
        """Test configuration deserialization from JSON."""
        json_str = '''
        {
            "version": "2.0",
            "systems": [
                {
                    "redshift": 0.348,
                    "ions": [
                        {
                            "name": "MgII",
                            "transitions": [2796.3, 2803.5],
                            "components": 2
                        },
                        {
                            "name": "FeII",
                            "transitions": [2600.2],
                            "components": 1
                        }
                    ]
                }
            ]
        }
        '''
        
        config = FitConfiguration.deserialize(json_str)
        
        assert len(config.systems) == 1
        assert config.systems[0].redshift == 0.348
        assert len(config.systems[0].ion_groups) == 2
        assert config.systems[0].ion_groups[0].ion_name == 'MgII'
        assert config.systems[0].ion_groups[1].ion_name == 'FeII'
    
    def test_round_trip_serialization(self):
        """Test that serialization/deserialization preserves configuration."""
        config1 = FitConfiguration()
        config1.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config1.add_system(0.348, 'FeII', [2600.2], 1)
        config1.add_system(0.524, 'OVI', [1031.9, 1037.6], 3)
        
        # Serialize and deserialize
        json_str = config1.serialize()
        config2 = FitConfiguration.deserialize(json_str)
        
        # Check they're equivalent
        assert len(config1.systems) == len(config2.systems)
        
        struct1 = config1.get_parameter_structure()
        struct2 = config2.get_parameter_structure()
        
        assert struct1['total_parameters'] == struct2['total_parameters']
        assert len(struct1['systems']) == len(struct2['systems'])
    
    def test_save_load_file(self):
        """Test saving and loading configuration to/from file."""
        config1 = FitConfiguration()
        config1.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config1.add_system(0.524, 'OVI', [1031.9, 1037.6], 1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config1.save(f.name)
            temp_path = f.name
        
        try:
            config2 = FitConfiguration.load(temp_path)
            
            assert len(config2.systems) == 2
            assert config2.systems[0].redshift == 0.348
            assert config2.systems[1].redshift == 0.524
        finally:
            Path(temp_path).unlink()
    
    def test_summary(self):
        """Test configuration summary generation."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 1)
        config.add_system(0.524, 'OVI', [1031.9, 1037.6], 1)
        
        summary = config.summary()
        
        assert "FitConfiguration Summary" in summary
        assert "System 1 (z=0.348000)" in summary
        assert "MgII: [2796.3, 2803.5] Å, 2 components" in summary
        assert "FeII: [2600.2] Å, 1 components" in summary
        assert "System 2 (z=0.524000)" in summary
        assert "OVI: [1031.9, 1037.6] Å, 1 components" in summary
        assert "Total parameters: 12" in summary
    
    def test_repr(self):
        """Test string representation."""
        config = FitConfiguration()
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        config.add_system(0.348, 'FeII', [2600.2], 1)
        
        repr_str = repr(config)
        assert "FitConfiguration" in repr_str
        assert "1 systems" in repr_str
        assert "2 ion groups" in repr_str
        assert "9 parameters" in repr_str


class TestIonTyingLogic:
    """Test the automatic ion parameter tying logic."""
    
    def test_tied_parameters_same_ion_same_z(self):
        """Test that same ion at same z shares parameters."""
        config = FitConfiguration()
        # MgII doublet - should share parameters
        config.add_system(0.348, 'MgII', [2796.3, 2803.5], 2)
        
        structure = config.get_parameter_structure()
        
        #