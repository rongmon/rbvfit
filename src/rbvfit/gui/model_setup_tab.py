#!/usr/bin/env python
"""
Clean Model Setup Tab for rbvfit 2.0 GUI

This tab handles ion system assignment to configurations and parameter estimation.
Complete rewrite with clean parameter manager integration.
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
from rbvfit import vfit_mcmc as mc
from rbvfit.gui.interactive_param_dialog import InteractiveParameterDialog

# Import the parameter manager
from multi_instrument_parameter_manager import MultiInstrumentParameterManager


ION_TEMPLATES = {
    "Custom (manual entry)": {
        "ion": "",
        "transitions": []
    },
    "CIV": {
        "ion": "CIV", 
        "transitions": [1548.2, 1550.8]
    },
    "SiIV": {
        "ion": "SiIV",
        "transitions": [1393.8, 1402.8]
    },
    "OI": {
        "ion": "OI",
        "transitions": [1302.2]
    },
    "SiII": {
        "ion": "SiII", 
        "transitions": [1260.4, 1304.4, 1526.7]
    },
    "FeII": {
        "ion": "FeII",
        "transitions": [1608.5, 2374.5, 2382.8, 2586.7, 2600.2]
    },
    "MgII": {
        "ion": "MgII",
        "transitions": [2796.4, 2803.5]
    },
    "AlIII": {
        "ion": "AlIII",
        "transitions": [1854.7, 1862.8]
    },
    "CIII": {
        "ion": "CIII",
        "transitions": [977.0]
    },
    "NV": {
        "ion": "NV", 
        "transitions": [1238.8, 1242.8]
    },
    "OVI": {
        "ion": "OVI",
        "transitions": [1031.9, 1037.6]
    },
    "Lyman-α": {
        "ion": "HI",
        "transitions": [1215.7]
    },
    "Lyman-β": {
        "ion": "HI", 
        "transitions": [1025.7]
    }
}

class ModelFunc:
    def __init__(self, compiled_model, instrument_name):
        self.compiled_model = compiled_model
        self.instrument_name = instrument_name
    
    def __call__(self, theta, wave):
        return self.compiled_model.model_flux(theta, wave, instrument=self.instrument_name)





class SystemDialog(QDialog):
    """Dialog for adding/editing ion systems with templates"""
    
    def __init__(self, system_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ion System Configuration")
        self.setModal(True)
        self.resize(450, 350)  # Slightly larger for template dropdown
        
        self.system_data = system_data or {}
        self.setup_ui()
        
    def setup_ui(self):
        """Create dialog interface with ion templates"""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # NEW: Ion Template Dropdown
        self.template_combo = QComboBox()
        self.template_combo.setToolTip("Select predefined ion template or custom entry")
        for template_name in ION_TEMPLATES.keys():
            self.template_combo.addItem(template_name)
        self.template_combo.setCurrentText("Custom (manual entry)")
        layout.addRow("Ion Template:", self.template_combo)
        
        # Add a separator line
        layout.addRow("", QLabel("─" * 40))
        
        # Redshift (existing)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-0.1, 20.0)
        self.z_spin.setDecimals(6)
        self.z_spin.setValue(self.system_data.get('z', 0.0))
        layout.addRow("Redshift (z):", self.z_spin)
        
        # Ion name (existing, but will be populated by template)
        self.ion_edit = QLineEdit()
        self.ion_edit.setText(self.system_data.get('ion', ''))
        self.ion_edit.setPlaceholderText("e.g., CIV, OI, SiII")
        layout.addRow("Ion:", self.ion_edit)
        
        # Transitions (existing, but will be populated by template)
        self.transitions_edit = QLineEdit()
        transitions = self.system_data.get('transitions', [])
        if transitions:
            self.transitions_edit.setText(', '.join(map(str, transitions)))
        self.transitions_edit.setPlaceholderText("e.g., 1548.2, 1550.3")
        layout.addRow("Transitions (Å):", self.transitions_edit)
        
        # Components (existing)
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 10)
        self.components_spin.setValue(self.system_data.get('components', 1))
        layout.addRow("Components:", self.components_spin)
        
        # Buttons (existing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(buttons)
        
        # NEW: Connect template dropdown
        self.template_combo.currentTextChanged.connect(self.on_template_changed)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # If editing existing system, try to match template
        if self.system_data:
            self.match_existing_template()
    
    def on_template_changed(self, template_name):
        """Handle template selection change"""
        if template_name not in ION_TEMPLATES:
            return
            
        template = ION_TEMPLATES[template_name]
        
        if template_name == "Custom (manual entry)":
            # Clear fields for manual entry
            if not self.system_data:  # Only clear if not editing existing
                self.ion_edit.clear()
                self.transitions_edit.clear()
        else:
            # Populate fields from template
            self.ion_edit.setText(template["ion"])
            
            if template["transitions"]:
                transitions_str = ', '.join(map(str, template["transitions"]))
                self.transitions_edit.setText(transitions_str)
            else:
                self.transitions_edit.clear()
    
    def match_existing_template(self):
        """Try to match existing system data to a template"""
        if not self.system_data:
            return
            
        existing_ion = self.system_data.get('ion', '').strip()
        existing_transitions = set(self.system_data.get('transitions', []))
        
        # Look for matching template
        for template_name, template in ION_TEMPLATES.items():
            if template_name == "Custom (manual entry)":
                continue
                
            template_ion = template["ion"]
            template_transitions = set(template["transitions"])
            
            # Check for exact match (ion name and transitions)
            if (existing_ion.upper() == template_ion.upper() and 
                existing_transitions == template_transitions):
                self.template_combo.setCurrentText(template_name)
                return
                
            # Check for partial match (ion name and subset of transitions)
            if (existing_ion.upper() == template_ion.upper() and 
                existing_transitions.issubset(template_transitions)):
                self.template_combo.setCurrentText(template_name)
                return
        
        # No match found, keep "Custom"
        self.template_combo.setCurrentText("Custom (manual entry)")
    
    def get_system_data(self):
        """Get system data from dialog (existing method, unchanged)"""
        transitions_text = self.transitions_edit.text().strip()
        transitions = []
        
        if transitions_text:
            try:
                transitions = [float(w.strip()) for w in transitions_text.split(',')]
            except ValueError:
                pass
        
        return {
            'z': self.z_spin.value(),
            'ion': self.ion_edit.text().strip(),
            'transitions': transitions,
            'components': self.components_spin.value()
        }



class MasterThetaDialog(QDialog):
    """Dialog for viewing and editing master theta parameters"""
    
    def __init__(self, collection_result=None, parent=None):
        super().__init__(parent)
        self.collection_result = collection_result
        self.parent_tab = parent  # Store reference to ModelSetupTab

        self.setWindowTitle("Master Theta Parameters")
        self.setModal(True)
        self.resize(800, 600)
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """Create dialog interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Summary info
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        self.summary_label = QLabel()
        summary_layout.addWidget(self.summary_label)
        
        # Parameter table
        table_group = QGroupBox("Master Parameters")
        table_layout = QVBoxLayout()
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        self.table = QTableWidget()
        table_layout.addWidget(self.table)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def load_data(self):
        """Load parameter data into table"""
        if self.collection_result is None:
            # Empty case
            self.summary_label.setText("No parameter collection result available")
            self.table.setRowCount(0)
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(['Parameter', 'Type', 'System', 'Component', 'Value', 'Source', 'Instruments'])
            return
        
        # Update summary
        n_systems = len(self.collection_result.master_systems)
        n_params = len(self.collection_result.master_theta)
        n_instruments = len(self.collection_result.instrument_mappings)
        
        summary_text = f"Systems: {n_systems}, Parameters: {n_params}, Instruments: {n_instruments}\n"
        summary_text += f"Parameter structure: [N×{n_params//3}, b×{n_params//3}, v×{n_params//3}]"
        self.summary_label.setText(summary_text)
        
        # Get parameter info table
        manager = MultiInstrumentParameterManager()
        df = manager.get_parameter_info_table(self.collection_result)
        
        # Set up table
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Populate table
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                if col == 'Value':
                    # Make value column editable
                    item = QTableWidgetItem(f"{row[col]:.3f}")
                    item.setData(Qt.UserRole, row[col])  # Store original value
                else:
                    item = QTableWidgetItem(str(row[col]))
                    if col != 'Value':
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make non-editable
                
                self.table.setItem(i, j, item)
        
        # Resize columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Parameter name

    def accept(self):
        """Handle OK button - update master theta and re-emit"""
        if self.parent_tab and hasattr(self.parent_tab, 'current_collection_result'):
            updated_theta = self.get_updated_theta()
            
            # Update the collection result
            self.parent_tab.current_collection_result = self.parent_tab.param_manager.update_master_theta(
                self.parent_tab.current_collection_result, 
                updated_theta
            )
            
            # Re-emit the model signal
            self.parent_tab.recompile_and_emit()
            
        super().accept()

        
    def get_updated_theta(self):
        """Get updated theta array from table"""
        if self.collection_result is None:
            return np.array([])
        
        # Get the value column index
        value_col = None
        for i in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(i).text() == 'Value':
                value_col = i
                break
        
        if value_col is None:
            return self.collection_result.master_theta.copy()
        
        # Extract values from table
        updated_theta = np.zeros_like(self.collection_result.master_theta)
        
        for i in range(self.table.rowCount()):
            item = self.table.item(i, value_col)
            try:
                updated_theta[i] = float(item.text())
            except (ValueError, TypeError):
                # Use original value if parsing fails
                updated_theta[i] = item.data(Qt.UserRole)
        
        return updated_theta


class ModelSetupTab(QWidget):
    """Clean tab for setting up absorption line models with configurations"""
    
    model_updated = pyqtSignal(dict, dict, dict)  # (instrument_data, theta_dict, bounds)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.configurations = {}  # From Tab 1
        self.config_systems = {}  # Dict: config_name -> list of system dicts
        self.config_parameters = {}  # Dict: (config_name, system_id) -> DataFrame
        self.current_config = None
        self.current_system_id = None
        
        # Parameter manager and results
        self.param_manager = MultiInstrumentParameterManager()
        self.current_collection_result = None
        
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
        
        # Status display
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(100)
        self.status_display.setReadOnly(True)
        params_layout.addWidget(self.status_display)
        
    def setup_bottom_controls(self, parent_layout):
        """Create bottom control buttons"""
        controls_layout = QHBoxLayout()
        parent_layout.addLayout(controls_layout)
        
        controls_layout.addStretch()
        
        # Master theta button - always visible
        self.show_master_theta_btn = QPushButton("Master Theta")
        controls_layout.addWidget(self.show_master_theta_btn)
        
        self.compile_btn = QPushButton("Compile Models")
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
        self.params_table.itemChanged.connect(self.on_parameter_table_edited)

        
        # Compilation
        self.compile_btn.clicked.connect(self.compile_models)
        self.show_master_theta_btn.clicked.connect(self.show_master_theta_dialog)
        
    def set_configurations(self, configurations):
        """Set configurations from Tab 1"""
        self.configurations = configurations
        
        # Initialize systems for new configurations
        for config_name in configurations:
            if config_name not in self.config_systems:
                self.config_systems[config_name] = []
                
        self.update_config_display()
        self.update_status()
        
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

    def on_parameter_table_edited(self, item):
        """Handle parameter table cell edits"""
        if not self.current_config or not self.current_system_id:
            return
            
        # Update the stored DataFrame
        row = item.row()
        col = item.column()
        
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        if row >= len(df):
            return
            
        # Map column to DataFrame column
        col_names = ['Component', 'N', 'b', 'v']
        if col < len(col_names):
            try:
                if col == 0:  # Component (string)
                    df.iloc[row, col] = str(item.text())
                else:  # N, b, v (float)
                    df.iloc[row, col] = float(item.text())
                    
                # If we have a compiled model, re-emit the signal
                if self.current_collection_result is not None:
                    self.recompile_and_emit()
                else:
                    self.update_status("Parameter updated (compile models to propagate changes)")
        
                    
            except ValueError:
                # Invalid input - revert to original value
                self.load_parameter_table()
                self.update_status("Invalid parameter value - reverted to original")


    def recompile_and_emit(self):
        """Recompile and emit model_updated signal with current parameters"""
        if self.current_collection_result is None:
            return
            
        try:
            # Rebuild instrument data with updated parameters
            instrument_data = self.build_instrument_data(self.current_collection_result)
            
            # Rebuild bounds
            n_params = len(self.current_collection_result.master_theta)
            n_comp = n_params // 3
            
            nguess = self.current_collection_result.master_theta[:n_comp].tolist()
            bguess = self.current_collection_result.master_theta[n_comp:2*n_comp].tolist()
            vguess = self.current_collection_result.master_theta[2*n_comp:3*n_comp].tolist()
            
            bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
            
            # Re-emit signal
            theta_dict = {'theta': self.current_collection_result.master_theta}
            self.model_updated.emit(instrument_data, theta_dict, bounds)
            
            self.update_status("Parameters updated - changes propagated to fitting tab")
            
        except Exception as e:
            self.update_status(f"Error updating parameters: {str(e)}")
                
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
                
            if not system_data['ion']:
                system_data['ion'] = "Unknown"
                
            # Generate unique system ID
            system_id = f"{system_data['ion']}_z{system_data['z']:.6f}"
            system_data['id'] = system_id
            
            # Add to current configuration
            if self.current_config not in self.config_systems:
                self.config_systems[self.current_config] = []
                
            self.config_systems[self.current_config].append(system_data)
            
            # Initialize empty parameter table
            key = (self.current_config, system_id)
            self.config_parameters[key] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_status()
            
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
                
            if not new_system_data['ion']:
                new_system_data['ion'] = "Unknown"
                
            # Update system data
            new_system_id = f"{new_system_data['ion']}_z{new_system_data['z']:.6f}"
            new_system_data['id'] = new_system_id
            
            # Handle ID change
            if new_system_id != system_id:
                old_key = (self.current_config, system_id)
                new_key = (self.current_config, new_system_id)
                if old_key in self.config_parameters:
                    self.config_parameters[new_key] = self.config_parameters.pop(old_key)
                    
            system_data.update(new_system_data)
            
            self.update_systems_display()
            self.update_system_combo()
            
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
            
    def update_systems_display(self):
        """Update systems tree display"""
        self.systems_tree.clear()
        
        if not self.current_config:
            return
            
        systems = self.config_systems.get(self.current_config, [])
        
        for system in systems:
            item = QTreeWidgetItem()
            item.setText(0, f"{system['ion']} z={system['z']:.6f}")
            
            # Show actual component count from parameter table
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
            self.update_systems_display()  # Update component count display
            self.update_status()
            
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
            self.update_systems_display()
            
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
        self.update_systems_display()
        
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
            self.update_systems_display()
            
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
        """Clean compilation using parameter manager"""
        if not self.configurations:
            QMessageBox.warning(self, "No Configurations", "No configurations available")
            return
            
        try:
            # Simple delegation to parameter manager
            self.update_status("Starting compilation...")
            
            collection_result = self.param_manager.collect_and_merge_parameters(
                self.config_systems,
                self.config_parameters,
                self.configurations
            )
            
            # Store result
            self.current_collection_result = collection_result
            
            # Show collection log
            self.update_status('\n'.join(collection_result.collection_log))
            
            # Build instrument data
            instrument_data = self.build_instrument_data(collection_result)
            
            # Create bounds
            n_params = len(collection_result.master_theta)
            n_comp = n_params // 3
            
            nguess = collection_result.master_theta[:n_comp].tolist()
            bguess = collection_result.master_theta[n_comp:2*n_comp].tolist()
            vguess = collection_result.master_theta[2*n_comp:3*n_comp].tolist()
            
            bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
            
            # Emit to fitting tab
            theta_dict = {'theta': collection_result.master_theta, 'length': len(collection_result.master_theta)}
            bounds_dict = {'lb': lb, 'ub': ub}
            self.model_updated.emit(instrument_data, theta_dict, bounds_dict)
            
            QMessageBox.information(self, "Success", 
                                  f"Models compiled successfully!\n"
                                  f"Configurations: {len(instrument_data)}\n"
                                  f"Total parameters: {len(collection_result.master_theta)}\n"
                                  f"Unique systems: {len(collection_result.master_systems)}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Compilation Error", f"Failed to compile models:\n{str(e)}")
            self.update_status(f"Compilation failed: {str(e)}")
    
    def build_instrument_data(self, collection_result):
        """Build instrument data from collection result"""
        instrument_data = {}
        
        # Create instrument configs for VoigtModel compilation
        instrument_configs = {}
        
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is None:
                continue
                
            # Create FitConfiguration using parameter manager's master config
            # We'll use the master config and let VoigtModel handle the details
            instrument_configs[config_name] = collection_result.master_config
        
        if not instrument_configs:
            raise ValueError("No configurations with data found")
            
        # Create VoigtModel and compile
        first_config = list(instrument_configs.values())[0]
        voigt_model = VoigtModel(first_config)
        
        # Store for other uses
        self.voigt_model = voigt_model
        
        is_multi = len(instrument_configs) > 1
        
        if is_multi:
            # For multi-instrument, we need to create separate configs with correct FWHM
            fwhm_configs = {}
            for config_name, config_data in self.configurations.items():
                if config_data['wave'] is None:
                    continue
                
                # Create config with correct FWHM
                fit_config = FitConfiguration()
                fit_config.instrumental_params = {'FWHM': str(config_data['fwhm'])}
                
                # Add systems from master config
                for system in collection_result.master_systems:
                    fit_config.add_system(
                        z=system.z,
                        ion=system.ion,
                        transitions=system.transitions,
                        components=system.components
                    )
                
                fwhm_configs[config_name] = fit_config
            
            compiled_model = voigt_model.compile(instrument_configs=fwhm_configs, verbose=True)
        else:
            compiled_model = voigt_model.compile(verbose=True)
        
        # Store compiled model
        self.compiled_model = compiled_model
        
        # Build instrument data dictionary
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is None:
                continue
                
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
    
    def show_master_theta_dialog(self):
        """Show master theta dialog"""
        dialog = MasterThetaDialog(self.current_collection_result, self)
        if dialog.exec_() == QDialog.Accepted and self.current_collection_result is not None:
            # Update collection result with edited parameters
            new_theta = dialog.get_updated_theta()
            if len(new_theta) > 0:
                self.current_collection_result = self.param_manager.update_master_theta(
                    self.current_collection_result, new_theta
                )
                self.update_status("Master theta updated by user")
                QMessageBox.information(self, "Updated", 
                                      f"Master theta updated with {len(new_theta)} parameters")
    
    def update_status(self, message=None):
        """Update status display"""
        if message:
            self.status_display.setText(message)
        else:
            # Generate summary status
            n_configs = len([c for c in self.configurations.values() if c['wave'] is not None])
            n_systems = sum(len(systems) for systems in self.config_systems.values())
            n_with_params = sum(1 for df in self.config_parameters.values() if not df.empty)
            
            status = f"Configurations with data: {n_configs}\n"
            status += f"Systems defined: {n_systems}\n"
            status += f"Systems with parameters: {n_with_params}"
            
            if self.current_collection_result:
                status += f"\nLast compilation: {len(self.current_collection_result.master_systems)} unique systems"
            
            self.status_display.setText(status)
    
    def get_current_model(self):
        """Return current model for main window"""
        if self.current_collection_result:
            return self.current_collection_result.master_config
        return None
        
    def get_spectrum_data(self):
        """Return spectrum data for current configuration"""
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

    
    def _restore_master_theta(self, master_theta_list, collection_info):
        """Restore master theta and create minimal collection result"""
        import numpy as np
        
        # Convert list back to numpy array
        master_theta = np.array(master_theta_list)
        
        # Create a minimal collection result for GUI purposes
        # This won't be fully functional for compilation, but allows viewing/editing
        from types import SimpleNamespace
        
        # Create minimal master systems from saved info
        master_systems = []
        for sys_info in collection_info.get('system_info', []):
            system = SimpleNamespace(
                z=sys_info['z'],
                ion=sys_info['ion'], 
                transitions=sys_info['transitions'],
                components=sys_info['components'],
                source_instrument=sys_info['source_instrument']
            )
            master_systems.append(system)
        
        # Create minimal collection result
        minimal_result = SimpleNamespace(
            master_theta=master_theta,
            master_systems=master_systems,
            instrument_mappings=[],  # Empty for now
            master_config=None,
            collection_log=[f"Restored from project file ({len(master_theta)} parameters)"]
        )
        
        self.current_collection_result = minimal_result
        
        self.update_status(f"Master theta restored ({len(master_theta)} parameters)")


    def _restore_model_setup(self, config_systems, config_parameters, current_config, current_system_id):
        """Restore model setup state from project load"""
        self.config_systems = config_systems
        self.config_parameters = config_parameters
        self.current_config = current_config
        self.current_system_id = current_system_id
        
        # Update displays
        self.update_config_display()
        self.update_systems_display()
        self.update_system_combo()
        
        # Set current selections if they exist
        if current_config:
            # Find and select the current config in combo
            index = self.current_config_combo.findText(current_config)
            if index >= 0:
                self.current_config_combo.setCurrentIndex(index)
        
        if current_system_id:
            # Find and select the current system in combo
            index = self.system_combo.findData(current_system_id)
            if index >= 0:
                self.system_combo.setCurrentIndex(index)
        
        # Load parameter table for current system
        if current_config and current_system_id:
            self.load_parameter_table()
        
        self.update_status("Model setup restored from project")

    def _restore_master_theta(self, master_theta, collection_info):
        """Restore master theta and create minimal collection result"""
        import numpy as np
        from types import SimpleNamespace
        
        # Create minimal master systems from saved info
        master_systems = []
        for sys_info in collection_info.get('system_info', []):
            system = SimpleNamespace(
                z=sys_info['z'],
                ion=sys_info['ion'], 
                transitions=sys_info['transitions'],
                components=sys_info['components'],
                source_instrument=sys_info['source_instrument']
            )
            master_systems.append(system)
        
        # Create minimal collection result
        minimal_result = SimpleNamespace(
            master_theta=master_theta,
            master_systems=master_systems,
            instrument_mappings=[],  # Empty for now
            master_config=None,
            collection_log=[f"Restored from project file ({len(master_theta)} parameters)"]
        )
        
        self.current_collection_result = minimal_result
        
        self.update_status(f"Master theta restored ({len(master_theta)} parameters)")
    
    def clear_state(self):
        """Clear tab state for project loading"""
        # Clear data structures
        self.config_systems = {}
        self.config_parameters = {}
        self.current_config = None
        self.current_system_id = None
        self.current_collection_result = None
        
        # Clear UI elements
        self.config_tree.clear()
        self.current_config_combo.clear()
        self.systems_tree.clear()
        self.system_combo.clear()
        self.params_table.setRowCount(0)
        self.config_info.clear()
        self.status_display.clear()
        
        # Reset button states
        self.edit_system_btn.setEnabled(False)
        self.delete_system_btn.setEnabled(False)
            