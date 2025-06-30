#!/usr/bin/env python
"""
Updated Model Setup Tab for rbvfit 2.0 GUI

This tab handles ion system assignment to configurations and parameter estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                            QGroupBox, QTreeWidget, QTreeWidgetItem, QPushButton,
                            QLabel, QComboBox, QTableWidget, QTableWidgetItem, 
                            QHeaderView, QMessageBox, QDialog, QFormLayout,
                            QDoubleSpinBox, QSpinBox, QLineEdit, QDialogButtonBox,
                            QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.parameter_manager import ParameterManager
from rbvfit import vfit_mcmc as mc
from rbvfit.gui.interactive_param_dialog import InteractiveParameterDialog

class ModelFunc:
    def __init__(self, compiled_model, instrument_name):
        self.compiled_model = compiled_model
        self.instrument_name = instrument_name
    
    def __call__(self, theta, wave):
        return self.compiled_model.model_flux(theta, wave, instrument=self.instrument_name)


class SystemDialog(QDialog):
    """Dialog for adding/editing ion systems"""
    
    def __init__(self, system_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ion System Configuration")
        self.setModal(True)
        self.resize(400, 300)
        
        self.system_data = system_data or {}
        self.setup_ui()
        
    def setup_ui(self):
        """Create dialog interface"""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Redshift
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-0.1, 20.0)
        self.z_spin.setDecimals(6)
        self.z_spin.setValue(self.system_data.get('z', 0.0))
        layout.addRow("Redshift (z):", self.z_spin)
        
        # Ion name
        self.ion_edit = QLineEdit()
        self.ion_edit.setText(self.system_data.get('ion', ''))
        self.ion_edit.setPlaceholderText("e.g., CIV, OI, SiII")
        layout.addRow("Ion:", self.ion_edit)
        
        # Transitions
        self.transitions_edit = QLineEdit()
        transitions = self.system_data.get('transitions', [])
        if transitions:
            self.transitions_edit.setText(', '.join(map(str, transitions)))
        self.transitions_edit.setPlaceholderText("e.g., 1548.2, 1550.3")
        layout.addRow("Transitions (Å):", self.transitions_edit)
        
        # Number of components
        #self.components_spin = QSpinBox()
        #self.components_spin.setRange(1, 10)
        #self.components_spin.setValue(self.system_data.get('components', 1))
        #layout.addRow("Components:", self.components_spin)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_system_data(self):
        """Return system data as dict"""
        transitions_text = self.transitions_edit.text().strip()
        transitions = []
        if transitions_text:
            try:
                transitions = [float(x.strip()) for x in transitions_text.split(',') if x.strip()]
            except ValueError:
                pass
        
        ion_name = self.ion_edit.text().strip()
        
        # Auto-detect ion name if empty (like command line version)
        if not ion_name and transitions:
            try:
                from rbvfit import rb_setline as rb
                # Use first transition to detect ion
                line_info = rb.rb_setline(transitions[0], 'closest')
                if 'name' in line_info and line_info['name']:
                    detected_name = line_info['name'][0]
                    # Extract ion part (e.g., "CIV 1548" -> "CIV")
                    ion_name = detected_name.split()[0] if ' ' in detected_name else detected_name
                    print(f"Auto-detected ion name: {ion_name} from transition {transitions[0]}")
            except Exception as e:
                print(f"Could not auto-detect ion name: {e}")
                ion_name = "Unknown"
        
        return {
            'z': self.z_spin.value(),
            'ion': ion_name,
            'transitions': transitions
        }


class ModelSetupTab(QWidget):
    """Updated tab for setting up absorption line models with configurations"""
    
    model_updated = pyqtSignal(dict, dict, dict)  # (instrument_data, theta_dict, bounds)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.configurations = {}  # From Tab 1
        self.config_systems = {}  # Dict: config_name -> list of system dicts
        self.config_parameters = {}  # Dict: (config_name, system_id) -> DataFrame
        self.config_fit_configs = {}  # Dict: config_name -> FitConfiguration
        self.master_config = None
        self.current_config = None
        self.current_system_id = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create the model setup interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Main splitter: configurations | systems | parameters
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        self.setup_config_panel(main_splitter)
        self.setup_systems_panel(main_splitter)
        self.setup_parameters_panel(main_splitter)
        
        main_splitter.setSizes([250, 300, 450])
        
        # Bottom controls
        self.setup_bottom_controls(layout)
        
    def setup_config_panel(self, parent):
        """Create configuration selection panel"""
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        parent.addWidget(config_widget)
        
        # Configuration group
        config_group = QGroupBox("Available Configurations")
        config_group_layout = QVBoxLayout()
        config_group.setLayout(config_group_layout)
        config_layout.addWidget(config_group)
        
        # Configuration list
        self.config_tree = QTreeWidget()
        self.config_tree.setHeaderLabels(['Configuration', 'Details'])
        self.config_tree.setColumnWidth(0, 120)
        config_group_layout.addWidget(self.config_tree)
        
        # Configuration info
        self.config_info = QTextEdit()
        self.config_info.setMaximumHeight(100)
        self.config_info.setReadOnly(True)
        config_group_layout.addWidget(self.config_info)
        
    def setup_systems_panel(self, parent):
        """Create systems management panel"""
        systems_widget = QWidget()
        systems_layout = QVBoxLayout()
        systems_widget.setLayout(systems_layout)
        parent.addWidget(systems_widget)
        
        # Current configuration selector
        current_config_layout = QHBoxLayout()
        systems_layout.addLayout(current_config_layout)
        
        current_config_layout.addWidget(QLabel("Current Config:"))
        self.current_config_combo = QComboBox()
        current_config_layout.addWidget(self.current_config_combo)
        
        # Systems group
        systems_group = QGroupBox("Ion Systems")
        systems_group_layout = QVBoxLayout()
        systems_group.setLayout(systems_group_layout)
        systems_layout.addWidget(systems_group)
        
        # Systems list
        self.systems_tree = QTreeWidget()
        self.systems_tree.setHeaderLabels(['System', 'Details'])
        self.systems_tree.setColumnWidth(0, 120)
        systems_group_layout.addWidget(self.systems_tree)
        
        # System controls
        sys_controls_layout = QHBoxLayout()
        systems_group_layout.addLayout(sys_controls_layout)
        
        self.add_system_btn = QPushButton("Add System")
        self.edit_system_btn = QPushButton("Edit")
        self.delete_system_btn = QPushButton("Delete")
        
        sys_controls_layout.addWidget(self.add_system_btn)
        sys_controls_layout.addWidget(self.edit_system_btn)
        sys_controls_layout.addWidget(self.delete_system_btn)
        
        self.edit_system_btn.setEnabled(False)
        self.delete_system_btn.setEnabled(False)
        
    def setup_parameters_panel(self, parent):
        """Create parameters panel"""
        params_widget = QWidget()
        params_layout = QVBoxLayout()
        params_widget.setLayout(params_layout)
        parent.addWidget(params_widget)
        
        # System selector
        selector_layout = QHBoxLayout()
        params_layout.addLayout(selector_layout)
        
        selector_layout.addWidget(QLabel("Current System:"))
        self.system_combo = QComboBox()
        selector_layout.addWidget(self.system_combo)
        selector_layout.addStretch()
        
        # Parameter input methods
        methods_layout = QHBoxLayout()
        params_layout.addLayout(methods_layout)
        
        self.interactive_btn = QPushButton("Interactive Parameters")
        self.manual_btn = QPushButton("Manual Entry")
        self.clear_params_btn = QPushButton("Clear")
        
        methods_layout.addWidget(self.interactive_btn)
        methods_layout.addWidget(self.manual_btn)
        methods_layout.addWidget(self.clear_params_btn)
        methods_layout.addStretch()
        
        # Parameter table
        params_group = QGroupBox("Parameters")
        params_group_layout = QVBoxLayout()
        params_group.setLayout(params_group_layout)
        params_layout.addWidget(params_group)
        
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels(['Component', 'N (log cm⁻²)', 'b (km/s)', 'v (km/s)'])
        
        header = self.params_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        params_group_layout.addWidget(self.params_table)
        
        # Parameter controls
        param_controls_layout = QHBoxLayout()
        params_group_layout.addLayout(param_controls_layout)
        
        self.add_component_btn = QPushButton("Add Component")
        self.delete_component_btn = QPushButton("Delete Component")
        
        param_controls_layout.addWidget(self.add_component_btn)
        param_controls_layout.addWidget(self.delete_component_btn)
        param_controls_layout.addStretch()
        
        # Progress indicator
        self.progress_label = QLabel("No configurations available")
        params_layout.addWidget(self.progress_label)
        
    def setup_bottom_controls(self, parent_layout):
        """Create bottom control buttons"""
        controls_layout = QHBoxLayout()
        parent_layout.addLayout(controls_layout)
        
        controls_layout.addStretch()
        
        self.compile_btn = QPushButton("Compile Models")
        self.compile_btn.setEnabled(False)
        controls_layout.addWidget(self.compile_btn)
        
    def setup_connections(self):
        """Connect signals and slots"""
        # Configuration selection
        self.config_tree.itemSelectionChanged.connect(self.on_config_tree_selection_changed)
        self.current_config_combo.currentTextChanged.connect(self.on_current_config_changed)
        
        # System management
        self.add_system_btn.clicked.connect(self.add_system)
        self.edit_system_btn.clicked.connect(self.edit_system)
        self.delete_system_btn.clicked.connect(self.delete_system)
        self.systems_tree.itemSelectionChanged.connect(self.on_system_selection_changed)
        
        # Parameter management
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        self.interactive_btn.clicked.connect(self.launch_interactive_params)
        self.manual_btn.clicked.connect(self.setup_manual_params)
        self.clear_params_btn.clicked.connect(self.clear_params)
        self.add_component_btn.clicked.connect(self.add_component)
        self.delete_component_btn.clicked.connect(self.delete_component)
        
        # Compilation
        self.compile_btn.clicked.connect(self.compile_models)
        
    def set_configurations(self, configurations):
        """Set configurations from Tab 1"""
        self.configurations = configurations
        
        # Initialize systems for new configurations
        for config_name in configurations:
            if config_name not in self.config_systems:
                self.config_systems[config_name] = []
                
        self.update_config_display()
        self.update_progress()
        
    def update_config_display(self):
        """Update configuration tree display"""
        self.config_tree.clear()
        self.current_config_combo.clear()
        
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is not None:  # Only show configs with data
                item = QTreeWidgetItem()
                item.setText(0, config_name)
                
                wave = config_data['wave']
                n_systems = len(self.config_systems.get(config_name, []))
                item.setText(1, f"FWHM={config_data['fwhm']}, {n_systems} systems")
                item.setData(0, Qt.UserRole, config_name)
                
                self.config_tree.addTopLevelItem(item)
                self.current_config_combo.addItem(config_name)
                
    def on_config_tree_selection_changed(self):
        """Handle configuration tree selection change"""
        current_item = self.config_tree.currentItem()
        
        if current_item:
            config_name = current_item.data(0, Qt.UserRole)
            config_data = self.configurations[config_name]
            
            info_text = f"Configuration: {config_name}\n"
            info_text += f"FWHM: {config_data['fwhm']} pixels\n"
            info_text += f"Data file: {config_data['filename']}\n"
            
            if config_data['wave'] is not None:
                wave = config_data['wave']
                info_text += f"λ range: {wave.min():.1f} - {wave.max():.1f} Å\n"
                info_text += f"Data points: {len(wave)}"
                
            self.config_info.setText(info_text)
        else:
            self.config_info.clear()
            
    def on_current_config_changed(self, config_name):
        """Handle current configuration change"""
        self.current_config = config_name
        self.update_systems_display()
        self.update_system_combo()
        self.update_progress()
        
    def add_system(self):
        """Add new ion system to current configuration"""
        if not self.current_config:
            QMessageBox.warning(self, "No Configuration", "Please select a configuration first")
            return
            
        dialog = SystemDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            system_data = dialog.get_system_data()
            
            if not system_data['transitions']:
                QMessageBox.warning(self, "Incomplete Data", 
                                  "Please specify at least one transition wavelength")
                return
                
            # Ion name is now auto-detected if empty, so we don't require it to be manually entered
            if not system_data['ion']:
                system_data['ion'] = "Unknown"  # Fallback
                
            # Generate unique system ID
            system_id = f"{system_data['ion']}_z{system_data['z']:.6f}"
            system_data['id'] = system_id
            
            # Add to current configuration
            if self.current_config not in self.config_systems:
                self.config_systems[self.current_config] = []
                
            self.config_systems[self.current_config].append(system_data)
            
            # Initialize EMPTY parameter table - components will be added dynamically
            key = (self.current_config, system_id)
            self.config_parameters[key] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_progress()
            
    def edit_system(self):
        """Edit selected system"""
        current_item = self.systems_tree.currentItem()
        if not current_item:
            return
            
        system_id = current_item.data(0, Qt.UserRole)
        systems = self.config_systems.get(self.current_config, [])
        system_data = next((s for s in systems if s['id'] == system_id), None)
        
        if not system_data:
            return
            
        dialog = SystemDialog(system_data, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            new_system_data = dialog.get_system_data()
            
            if not new_system_data['transitions']:
                QMessageBox.warning(self, "Incomplete Data", 
                                  "Please specify at least one transition wavelength")
                return
                
            # Ion name is auto-detected if empty
            if not new_system_data['ion']:
                new_system_data['ion'] = "Unknown"  # Fallback
                
            # Update system data
            new_system_id = f"{new_system_data['ion']}_z{new_system_data['z']:.6f}"
            new_system_data['id'] = new_system_id
            
            # Handle ID change
            if new_system_id != system_id:
                # Move parameter data
                old_key = (self.current_config, system_id)
                new_key = (self.current_config, new_system_id)
                if old_key in self.config_parameters:
                    self.config_parameters[new_key] = self.config_parameters.pop(old_key)
                    
            system_data.update(new_system_data)
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_progress()
            
    def delete_system(self):
        """Delete selected system"""
        current_item = self.systems_tree.currentItem()
        if not current_item:
            return
            
        system_id = current_item.data(0, Qt.UserRole)
        
        reply = QMessageBox.question(self, "Confirm Delete",
                                   f"Delete system '{system_id}'?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            systems = self.config_systems.get(self.current_config, [])
            self.config_systems[self.current_config] = [s for s in systems if s['id'] != system_id]
            
            # Remove parameter data
            key = (self.current_config, system_id)
            if key in self.config_parameters:
                del self.config_parameters[key]
                
            self.update_systems_display()
            self.update_system_combo()
            self.update_progress()
            

    def update_systems_display(self):
        """Update systems tree display"""
        self.systems_tree.clear()
        
        if not self.current_config:
            return
            
        systems = self.config_systems.get(self.current_config, [])
        
        for system in systems:
            item = QTreeWidgetItem()
            item.setText(0, f"{system['ion']} z={system['z']:.6f}")
            
            # Show actual component count from parameter table, not stored value
            key = (self.current_config, system['id'])
            actual_components = 0
            if key in self.config_parameters:
                actual_components = len(self.config_parameters[key])
            
            transitions_str = ', '.join(f"{t:.1f}" for t in system['transitions'])
            item.setText(1, f"{actual_components} comp, [{transitions_str}] Å")
            item.setData(0, Qt.UserRole, system['id'])
            
            self.systems_tree.addTopLevelItem(item)


    def update_system_combo(self):
        """Update system combo box"""
        self.system_combo.clear()
        
        if not self.current_config:
            return
            
        systems = self.config_systems.get(self.current_config, [])
        
        for system in systems:
            self.system_combo.addItem(f"{system['ion']} z={system['z']:.6f}", system['id'])            

    def on_system_selection_changed(self):
        """Handle system selection change"""
        current_item = self.systems_tree.currentItem()
        has_selection = current_item is not None
        
        self.edit_system_btn.setEnabled(has_selection)
        self.delete_system_btn.setEnabled(has_selection)
        
    def on_system_changed(self, text):
        """Handle system combo change"""
        if text:
            system_id = self.system_combo.currentData()
            self.current_system_id = system_id
            self.load_parameter_table()
            
    def launch_interactive_params(self):
        """Launch interactive parameter estimation"""
        if not self.current_config or not self.current_system_id:
            QMessageBox.warning(self, "No Selection", 
                              "Please select configuration and system first")
            return
            
        # Get system data
        systems = self.config_systems.get(self.current_config, [])
        system_data = next((s for s in systems if s['id'] == self.current_system_id), None)
        
        if not system_data:
            return
            
        # Get spectrum data
        config_data = self.configurations[self.current_config]
        spectrum_data = {
            'wave': config_data['wave'],
            'flux': config_data['flux'],
            'error': config_data['error']
        }
        
        # Launch interactive dialog
        dialog = InteractiveParameterDialog(system_data, spectrum_data, self)
        dialog.parameters_ready.connect(self.on_interactive_parameters_ready)
        dialog.exec_()
        
    def on_interactive_parameters_ready(self, df):
        """Handle interactive parameters result"""
        if self.current_config and self.current_system_id:
            key = (self.current_config, self.current_system_id)
            self.config_parameters[key] = df.copy()
            self.load_parameter_table()
            self.update_progress()
            
    def setup_manual_params(self):
        """Set up manual parameter entry"""
        if not self.current_config or not self.current_system_id:
            QMessageBox.warning(self, "No Selection", 
                              "Please select configuration and system first")
            return
            
        # Add default component if table is empty
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters or self.config_parameters[key].empty:
            self.add_component()
            
    def clear_params(self):
        """Clear parameters for current system"""
        if self.current_config and self.current_system_id:
            key = (self.current_config, self.current_system_id)
            self.config_parameters[key] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            self.load_parameter_table()
            self.update_progress()
            
    def add_component(self):
        """Add new component to current system"""
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            self.config_parameters[key] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            
        df = self.config_parameters[key]
        new_component = len(df) + 1
        
        new_row = pd.DataFrame({
            'Component': [new_component],
            'N': [13.5],  # Default log column density
            'b': [25.0],  # Default b parameter
            'v': [0.0]    # Default velocity
        })
        
        self.config_parameters[key] = pd.concat([df, new_row], ignore_index=True)
        self.load_parameter_table()
        self.update_progress()
        
    def delete_component(self):
        """Delete last component"""
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        if len(df) > 0:
            self.config_parameters[key] = df.iloc[:-1].reset_index(drop=True)
            self.load_parameter_table()
            self.update_progress()
            
    def load_parameter_table(self):
        """Load parameter table for current system"""
        self.params_table.setRowCount(0)
        
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        
        self.params_table.setRowCount(len(df))
        
        for i, row in df.iterrows():
            self.params_table.setItem(i, 0, QTableWidgetItem(str(row['Component'])))
            self.params_table.setItem(i, 1, QTableWidgetItem(f"{row['N']:.2f}"))
            self.params_table.setItem(i, 2, QTableWidgetItem(f"{row['b']:.1f}"))
            self.params_table.setItem(i, 3, QTableWidgetItem(f"{row['v']:.1f}"))
            
    def compile_models(self):
        """Compile models for all configurations"""
        if not self.configurations:
            QMessageBox.warning(self, "No Configurations", "No configurations available")
            return
            
        try:
            # Create FitConfiguration for each instrument configuration
            self.config_fit_configs = {}
            
            for config_name, config_data in self.configurations.items():
                if config_data['wave'] is None:
                    continue  # Skip configurations without data
                    
                # Create FitConfiguration
                fit_config = FitConfiguration()
                
                # Set instrumental parameters including FWHM
                fit_config.instrumental_params = {
                    'FWHM': str(config_data['fwhm'])
                }
                
                # Add systems for this configuration
                systems = self.config_systems.get(config_name, [])
                for system in systems:
                    # Check if system has parameters
                    key = (config_name, system['id'])
                    if key not in self.config_parameters or self.config_parameters[key].empty:
                        QMessageBox.warning(self, "Missing Parameters", 
                                          f"System {system['id']} in {config_name} has no parameters")
                        return
                    
                    # Get DYNAMIC component count from parameter table
                    df = self.config_parameters[key]
                    actual_components = len(df)
                    
                    # Add system to FitConfiguration with actual component count
                    fit_config.add_system(
                        z=system['z'],
                        ion=system['ion'],
                        transitions=system['transitions'],
                        components=actual_components  # Use dynamic count from parameter table!
                    )
                    
                self.config_fit_configs[config_name] = fit_config
                
            if not self.config_fit_configs:
                QMessageBox.warning(self, "No Valid Configurations", 
                                  "No configurations with both data and systems found")
                return
                
            # Create master configuration
            self.master_config = FitConfiguration.create_master_config(self.config_fit_configs)
            
            # Build global parameter arrays
            theta, lb, ub = self.build_global_parameters()
            
            # Create instrument data for vfit_mcmc
            instrument_data = self.build_instrument_data()
            
            # Emit signal with all necessary data
            theta_dict = {'theta': theta, 'length': len(theta)}
            bounds = {'lb': lb, 'ub': ub}
            self.model_updated.emit(instrument_data, theta_dict, bounds)
            
            QMessageBox.information(self, "Success", 
                                  f"Models compiled successfully!\n"
                                  f"Configurations: {len(instrument_data)}\n"
                                  f"Total parameters: {len(theta)}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Compilation Error", f"Failed to compile models:\n{str(e)}")    
            
    def build_global_parameters(self):
        """Build global parameter arrays following tutorial structure"""
        all_nguess = []
        all_bguess = []
        all_vguess = []
        
        # Collect all parameters in consistent order
        # Process by config, then by system, then by component
        for config_name in sorted(self.config_fit_configs.keys()):
            systems = self.config_systems.get(config_name, [])
            
            # Sort systems for consistency
            systems_sorted = sorted(systems, key=lambda s: (s['z'], s['ion']))
            
            for system in systems_sorted:
                key = (config_name, system['id'])
                if key in self.config_parameters:
                    df = self.config_parameters[key]
                    
                    # Sort components by component number
                    df_sorted = df.sort_values('Component')
                    
                    # Add parameters for each component
                    for _, row in df_sorted.iterrows():
                        all_nguess.append(row['N'])
                        all_bguess.append(row['b'])
                        all_vguess.append(row['v'])
                        
        if not all_nguess:
            raise ValueError("No parameters found - add systems and estimate parameters first")
            
        # Create global theta array following tutorial structure
        theta = np.concatenate([all_nguess, all_bguess, all_vguess])
        
        # Set bounds using rbvfit's set_bounds function
        bounds, lb, ub = mc.set_bounds(all_nguess, all_bguess, all_vguess)
        
        print(f"Built global parameters:")
        print(f"  Components: {len(all_nguess)}")
        print(f"  Theta structure: N[0:{len(all_nguess)}], b[{len(all_nguess)}:{len(all_nguess)+len(all_bguess)}], v[{len(all_nguess)+len(all_bguess)}:]")
        print(f"  Total parameters: {len(theta)}")
        
        return theta, lb, ub
        

    def build_instrument_data(self):
        """Build instrument_data dictionary for vfit_mcmc"""
        instrument_data = {}
        
        # Check we have configurations
        if not self.config_fit_configs:
            raise ValueError("No instrument configurations available")
            
        # Pick the first config to create VoigtModel (doesn't matter which)
        first_config_name = list(self.config_fit_configs.keys())[0]
        first_config = self.config_fit_configs[first_config_name]
        
        # Create VoigtModel with first config
        voigt_model = VoigtModel(first_config)
        # STORE the original VoigtModel for FitResults
        self.voigt_model = voigt_model 
        

        is_multi = len(self.config_fit_configs) > 1

        if is_multi:
            compiled_model = voigt_model.compile(instrument_configs=self.config_fit_configs, verbose=True)
        else:
            compiled_model = voigt_model.compile(verbose=True)
        
        # Store the compiled model
        self.compiled_model = compiled_model


        
        for config_name, fit_config in self.config_fit_configs.items():
            config_data = self.configurations[config_name]
        
            # Create picklable model function
            if is_multi:
                model_func = ModelFunc(compiled_model, config_name)
            else:
                model_func = compiled_model.model_flux
        
            instrument_data[config_name] = {
                'model': model_func,  
                'wave': config_data['wave'],
                'flux': config_data['flux'],
                'error': config_data['error']
            }
            
        print(f"Built instrument data:")
        for name, data in instrument_data.items():
            wave = data['wave']
            print(f"  {name}: {len(wave)} points, {wave.min():.1f}-{wave.max():.1f} Å")
            
        return instrument_data        
        
    def update_progress(self):
        """Update progress indicator"""
        if not self.configurations:
            self.progress_label.setText("No configurations available")
            self.compile_btn.setEnabled(False)
            return
            
        # Count configurations with data
        configs_with_data = sum(1 for config in self.configurations.values() 
                               if config['wave'] is not None)
        
        # Count systems and parameters
        total_systems = 0
        systems_with_params = 0
        
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is None:
                continue
                
            systems = self.config_systems.get(config_name, [])
            total_systems += len(systems)
            
            for system in systems:
                key = (config_name, system['id'])
                if key in self.config_parameters and not self.config_parameters[key].empty:
                    systems_with_params += 1
                    
        status = f"Configs with data: {configs_with_data}, "
        status += f"Systems: {total_systems}, "
        status += f"Systems with parameters: {systems_with_params}"
        
        self.progress_label.setText(status)
        
        # Enable compile button if ready
        ready = (configs_with_data > 0 and 
                total_systems > 0 and 
                systems_with_params == total_systems and
                systems_with_params > 0)  # Need at least one system with parameters
        self.compile_btn.setEnabled(ready)
        
    def get_current_model(self):
        """Return current model for main window"""
        return self.master_config
        
    def get_spectrum_data(self):
        """Return spectrum data for current configuration (for interactive dialog)"""
        if not self.current_config or self.current_config not in self.configurations:
            return None
            
        config_data = self.configurations[self.current_config]
        if config_data['wave'] is None:
            return None
            
        return {
            'wave': config_data['wave'],
            'flux': config_data['flux'],
            'error': config_data['error']
        }