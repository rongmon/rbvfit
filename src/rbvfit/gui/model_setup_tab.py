#!/usr/bin/env python
"""
Restored Model Setup Tab for rbvfit 2.0 - Full Functionality

Restored with proper:
- Custom ion support with templates
- Interactive parameter estimation with spectrum display
- Working parameter management
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QPushButton, QLabel, QComboBox, QTreeWidget,
                            QTreeWidgetItem, QHeaderView, QTableWidget, QTableWidgetItem,
                            QTextEdit, QFormLayout, QSpinBox, QDoubleSpinBox,
                            QMessageBox, QDialog, QDialogButtonBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

# Import rbvfit components
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager
import rbvfit.vfit_mcmc as mc

# Try to import interactive parameter dialog
try:
    from rbvfit.gui.interactive_param_dialog import InteractiveParameterDialog
    HAS_INTERACTIVE_DIALOG = True
except ImportError:
    HAS_INTERACTIVE_DIALOG = False


# Ion templates with full flexibility
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


class SystemDialog(QDialog):
    """Dialog for adding/editing ion systems with templates and custom support"""
    
    def __init__(self, system_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ion System Configuration")
        self.setModal(True)
        self.resize(450, 350)
        
        self.system_data = system_data or {}
        self.setup_ui()
        
    def setup_ui(self):
        """Create dialog interface with ion templates"""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Ion Template Dropdown
        self.template_combo = QComboBox()
        self.template_combo.setToolTip("Select predefined ion template or custom entry")
        for template_name in ION_TEMPLATES.keys():
            self.template_combo.addItem(template_name)
        self.template_combo.setCurrentText("Custom (manual entry)")
        layout.addRow("Ion Template:", self.template_combo)
        
        # Add a separator line
        layout.addRow("", QLabel("─" * 40))
        
        # Redshift
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-0.1, 20.0)
        self.z_spin.setDecimals(6)
        self.z_spin.setValue(self.system_data.get('z', 0.0))
        layout.addRow("Redshift (z):", self.z_spin)
        
        # Ion name (can be custom)
        self.ion_edit = QLineEdit()
        self.ion_edit.setText(self.system_data.get('ion', ''))
        self.ion_edit.setPlaceholderText("e.g., CIV, OI, SiII, or custom ion")
        layout.addRow("Ion:", self.ion_edit)
        
        # Transitions (can be custom wavelengths)
        self.transitions_edit = QLineEdit()
        transitions = self.system_data.get('transitions', [])
        if transitions:
            self.transitions_edit.setText(', '.join(map(str, transitions)))
        self.transitions_edit.setPlaceholderText("e.g., 1548.2, 1550.3 (rest wavelengths in Å)")
        layout.addRow("Transitions (Å):", self.transitions_edit)
        
        # Components
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 10)
        self.components_spin.setValue(self.system_data.get('components', 1))
        layout.addRow("Components:", self.components_spin)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(buttons)
        
        # Connect signals
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
        """Get system data from dialog"""
        transitions_text = self.transitions_edit.text().strip()
        transitions = []
        
        if transitions_text:
            for trans in transitions_text.split(','):
                try:
                    wavelength = float(trans.strip())
                    transitions.append(wavelength)
                except ValueError:
                    continue
        
        return {
            'z': self.z_spin.value(),
            'ion': self.ion_edit.text().strip(),
            'components': self.components_spin.value(),
            'transitions': transitions,
            'id': f"{self.z_spin.value():.6f}_{self.ion_edit.text().strip()}"  # Add ID for tracking
        }


class ModelSetupTab(QWidget):
    """Clean tab for setting up absorption line models with full functionality"""
    
    model_updated = pyqtSignal(dict, dict, dict)  # (instrument_data, theta_dict, bounds)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.configurations = {}  # From Tab 1
        self.config_systems = {}  # Dict: config_name -> list of system dicts
        self.config_parameters = {}  # Dict: (config_name, system_id) -> DataFrame
        self.current_config = None
        self.current_system_id = None
        
        # Simple unified approach
        self.master_config = None
        self.master_theta = None
        self.param_manager = None
        
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
        self.config_tree.setHeaderLabels(['Configuration', 'Info'])
        config_group_layout.addWidget(self.config_tree)
        
        # Current config selector
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Current:"))
        self.current_config_combo = QComboBox()
        current_layout.addWidget(self.current_config_combo)
        config_group_layout.addLayout(current_layout)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setReadOnly(True)
        config_layout.addWidget(self.status_text)
        
    def setup_systems_panel(self, parent):
        """Create systems management panel"""
        systems_widget = QWidget()
        systems_layout = QVBoxLayout()
        systems_widget.setLayout(systems_layout)
        parent.addWidget(systems_widget)
        
        # Systems group
        systems_group = QGroupBox("Ion Systems")
        systems_group_layout = QVBoxLayout()
        systems_group.setLayout(systems_group_layout)
        systems_layout.addWidget(systems_group)
        
        # Systems tree
        self.systems_tree = QTreeWidget()
        self.systems_tree.setHeaderLabels(['System', 'Details'])
        systems_group_layout.addWidget(self.systems_tree)
        
        # System controls
        system_controls = QHBoxLayout()
        self.add_system_btn = QPushButton("Add System")
        self.edit_system_btn = QPushButton("Edit")
        self.delete_system_btn = QPushButton("Delete")
        
        system_controls.addWidget(self.add_system_btn)
        system_controls.addWidget(self.edit_system_btn)
        system_controls.addWidget(self.delete_system_btn)
        systems_group_layout.addLayout(system_controls)
        
        # Initially disable edit/delete
        self.edit_system_btn.setEnabled(False)
        self.delete_system_btn.setEnabled(False)
        
    def setup_parameters_panel(self, parent):
        """Create parameters management panel"""
        params_widget = QWidget()
        params_layout = QVBoxLayout()
        params_widget.setLayout(params_layout)
        parent.addWidget(params_widget)
        
        # Parameters group
        params_group = QGroupBox("Parameter Estimation")
        params_group_layout = QVBoxLayout()
        params_group.setLayout(params_group_layout)
        params_layout.addWidget(params_group)
        
        # System selector
        system_layout = QHBoxLayout()
        system_layout.addWidget(QLabel("System:"))
        self.system_combo = QComboBox()
        system_layout.addWidget(self.system_combo)
        params_group_layout.addLayout(system_layout)
        
        # Parameter estimation controls
        param_controls = QHBoxLayout()
        
        # Interactive button (only if available)
        if HAS_INTERACTIVE_DIALOG:
            self.interactive_btn = QPushButton("Interactive Parameters")
            self.interactive_btn.setToolTip("Click on spectrum to estimate parameters")
            param_controls.addWidget(self.interactive_btn)
        else:
            self.interactive_btn = None
            
        self.manual_btn = QPushButton("Manual Setup")
        self.clear_params_btn = QPushButton("Clear")
        
        param_controls.addWidget(self.manual_btn)
        param_controls.addWidget(self.clear_params_btn)
        params_group_layout.addLayout(param_controls)
        
        # Component controls
        comp_controls = QHBoxLayout()
        self.add_component_btn = QPushButton("Add Component")
        self.delete_component_btn = QPushButton("Delete Component")
        
        comp_controls.addWidget(self.add_component_btn)
        comp_controls.addWidget(self.delete_component_btn)
        params_group_layout.addLayout(comp_controls)
        
        # Parameters table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels(['Component', 'N (log)', 'b (km/s)', 'v (km/s)'])
        header = self.params_table.horizontalHeader()
        header.setStretchLastSection(True)
        params_group_layout.addWidget(self.params_table)
        
    def setup_bottom_controls(self, layout):
        """Create bottom control buttons"""
        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)
        
        # Master theta button
        self.show_master_theta_btn = QPushButton("Show Master Parameters")
        self.show_master_theta_btn.setEnabled(False)
        controls_layout.addWidget(self.show_master_theta_btn)
        
        controls_layout.addStretch()
        
        # Compile button
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
        if self.interactive_btn:
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

    def launch_interactive_params(self):
        """Launch interactive parameter estimation with spectrum display"""
        if not self.current_config or not self.current_system_id:
            QMessageBox.warning(self, "Selection Required", 
                              "Please select a configuration and system")
            return
            
        config_data = self.configurations[self.current_config]
        if config_data['wave'] is None:
            QMessageBox.warning(self, "No Data", "Selected configuration has no data")
            return
            
        # Get system info
        systems = self.config_systems.get(self.current_config, [])
        current_system = None
        
        for system in systems:
            if system['id'] == self.current_system_id:
                current_system = system
                break
                
        if not current_system:
            QMessageBox.warning(self, "System Not Found", "Selected system not found")
            return
            
        try:
            # Prepare spectrum data
            spectrum_data = {
                'wave': config_data['wave'],
                'flux': config_data['flux'],
                'error': config_data['error']
            }
            
            # Launch interactive estimation dialog
            dialog = InteractiveParameterDialog(current_system, spectrum_data, parent=self)
            dialog.parameters_ready.connect(self.on_interactive_parameters_ready)
            dialog.exec_()
                
        except Exception as e:
            QMessageBox.critical(self, "Estimation Error", 
                               f"Interactive parameter estimation failed:\n{str(e)}")
            # Fallback to manual setup
            self.setup_manual_params()

    def on_interactive_parameters_ready(self, df):
        """Handle interactive parameters result"""
        if self.current_config and self.current_system_id:
            key = (self.current_config, self.current_system_id)
            self.config_parameters[key] = df.copy()
            self.load_parameter_table()
            self.update_systems_display()  # Update component count display
            self.update_status("Interactive parameter estimation completed")

    def compile_models(self):
        """Compile models for unified vfit interface"""
        if not self.configurations:
            QMessageBox.warning(self, "No Configurations", 
                              "No configurations available. Please set up configurations and data first.")
            return
            
        # Check if any systems are defined
        total_systems = sum(len(systems) for systems in self.config_systems.values())
        if total_systems == 0:
            QMessageBox.warning(self, "No Systems", 
                              "No ion systems defined. Please add systems before compiling.")
            return
            
        # Check if any parameters are estimated
        if not self.config_parameters:
            QMessageBox.warning(self, "No Parameters", 
                              "No parameters estimated. Please estimate parameters for your systems first.")
            return
            
        try:
            self.update_status("Creating master configuration...")


            # Update component counts in config_systems to match actual parameters
            #IMPORTANT BUG FIX
            for config_name, systems in self.config_systems.items():
                for system in systems:
                    key = (config_name, system['id'])
                    if key in self.config_parameters:
                        # Update component count to match actual parameter rows
                        actual_components = len(self.config_parameters[key])
                        system['components'] = actual_components
                    
            # Create a single master FitConfiguration from all systems
            self.master_config = FitConfiguration()
            
            # Collect all unique systems
            all_systems = []
            for config_name, systems in self.config_systems.items():
                for system in systems:
                    all_systems.append(system)
            
            # Add systems to master config
            for system in all_systems:
                self.master_config.add_system(
                    z=system['z'],
                    ion=system['ion'],
                    transitions=system['transitions'],
                    components=system['components']
                )
            
            self.update_status("Collecting parameters...")
            
            # Collect parameters from all systems
            all_N = []
            all_b = []
            all_v = []
            
            for (config_name, system_id), df in self.config_parameters.items():
                for _, row in df.iterrows():
                    all_N.append(row['N'])
                    all_b.append(row['b']) 
                    all_v.append(row['v'])
            
            # Create master theta array
            self.master_theta = np.array(all_N + all_b + all_v)
            
            self.update_status("Building instrument data...")
            
            # Build unified instrument data
            instrument_data = self.build_unified_instrument_data()
            
            self.update_status("Creating parameter bounds...")
            
            # Create bounds using vfit_mcmc
            n_comp = len(self.master_theta) // 3
            nguess = self.master_theta[:n_comp]
            bguess = self.master_theta[n_comp:2*n_comp]
            vguess = self.master_theta[2*n_comp:]
            
            bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
            
            # Emit to fitting tab
            theta_dict = {'theta': self.master_theta, 'length': len(self.master_theta)}
            bounds_dict = {'lb': lb, 'ub': ub}
            self.model_updated.emit(instrument_data, theta_dict, bounds_dict)
            
            # Enable master theta button
            self.show_master_theta_btn.setEnabled(True)
            
            self.update_status("Compilation completed successfully!")
            
            QMessageBox.information(self, "Success", 
                                  f"Models compiled successfully!\n"
                                  f"Configurations: {len(instrument_data)}\n"
                                  f"Total parameters: {len(self.master_theta)}\n"
                                  f"Systems: {total_systems}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Compilation Error", f"Failed to compile models:\n{str(e)}")
            self.update_status(f"Compilation failed: {str(e)}")
    
    def build_unified_instrument_data(self):
        """Build unified instrument data dictionary for vfit interface"""
        instrument_data = {}
        
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is None:
                continue
                
            # Create VoigtModel with instrument-specific FWHM
            model = VoigtModel(self.master_config, FWHM=str(config_data['fwhm']))
            
            # Compile model
            #compiled_model = model.compile(verbose=False)
            
            # Create unified instrument data entry
            instrument_data[config_name] = {
                'model': model,  # Compiled VoigtModel object
                'wave': config_data['wave'],
                'flux': config_data['flux'],
                'error': config_data['error']
            }
        
        return instrument_data

    # System management methods
    def add_system(self):
        """Add new ion system with full custom support"""
        if not self.current_config:
            QMessageBox.warning(self, "No Configuration", "Please select a configuration first")
            return
            
        dialog = SystemDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            system_data = dialog.get_system_data()
            
            # Validate system data
            if not system_data['ion']:
                QMessageBox.warning(self, "Invalid Ion", "Please specify an ion name")
                return
                
            if not system_data['transitions']:
                QMessageBox.warning(self, "Invalid Transitions", "Please specify at least one transition wavelength")
                return
            
            # Add to config systems
            self.config_systems[self.current_config].append(system_data)
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_status(f"Added system: {system_data['ion']} at z={system_data['z']}")
    
    def edit_system(self):
        """Edit selected system"""
        items = self.systems_tree.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Selection", "Please select a system to edit")
            return
            
        # Get selected system
        system_id = items[0].data(0, Qt.UserRole)
        systems = self.config_systems.get(self.current_config, [])
        
        # Find system by ID
        for i, system in enumerate(systems):
            if system['id'] == system_id:
                # Create dialog with existing values
                dialog = SystemDialog(system_data=system, parent=self)
                
                if dialog.exec_() == QDialog.Accepted:
                    # Update system
                    new_system_data = dialog.get_system_data()
                    systems[i] = new_system_data
                    
                    self.update_systems_display()
                    self.update_system_combo()
                    self.update_status(f"Updated system: {new_system_data['ion']} at z={new_system_data['z']}")
                break
    
    def delete_system(self):
        """Delete selected system"""
        items = self.systems_tree.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Selection", "Please select a system to delete")
            return
            
        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   "Are you sure you want to delete this system?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            system_id = items[0].data(0, Qt.UserRole)
            systems = self.config_systems.get(self.current_config, [])
            
            # Remove system
            for i, system in enumerate(systems):
                if system['id'] == system_id:
                    del systems[i]
                    break
            
            # Clear related parameters
            keys_to_remove = []
            for key in self.config_parameters:
                if key[0] == self.current_config and key[1] == system_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.config_parameters[key]
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_status("System deleted")

    # Parameter management methods
    def setup_manual_params(self):
        """Set up manual parameter estimation"""
        if not self.current_config or not self.current_system_id:
            QMessageBox.warning(self, "Selection Required", 
                              "Please select a configuration and system")
            return
            
        # Get system info
        systems = self.config_systems.get(self.current_config, [])
        current_system = None
        
        for system in systems:
            if system['id'] == self.current_system_id:
                current_system = system
                break
                
        if not current_system:
            QMessageBox.warning(self, "System Not Found", "Selected system not found")
            return
            
        # Create default parameters DataFrame
        import pandas as pd
        
        n_components = current_system['components']
        data = []
        
        for i in range(n_components):
            data.append({
                'Component': f'Comp_{i+1}',
                'N': 13.0,  # Default log column density
                'b': 20.0,  # Default b-parameter
                'v': 0.0    # Default velocity
            })
        
        params_df = pd.DataFrame(data)
        
        # Store parameters
        key = (self.current_config, self.current_system_id)
        self.config_parameters[key] = params_df
        
        # Update display
        self.load_parameter_table()
        self.update_status("Manual parameter setup completed")
    
    def clear_params(self):
        """Clear parameters for current system"""
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key in self.config_parameters:
            del self.config_parameters[key]
            
        self.params_table.setRowCount(0)
        self.update_status("Parameters cleared")
    
    def add_component(self):
        """Add component to current system"""
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            # Create new DataFrame
            import pandas as pd
            self.config_parameters[key] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            
        df = self.config_parameters[key]
        new_comp = len(df) + 1
        
        # Add new row
        import pandas as pd
        new_row = pd.DataFrame({
            'Component': [f'Comp_{new_comp}'],
            'N': [13.0],
            'b': [20.0],
            'v': [0.0]
        })
        
        self.config_parameters[key] = pd.concat([df, new_row], ignore_index=True)
        
        # Update display
        self.load_parameter_table()
        self.update_status(f"Added component {new_comp}")
    
    def delete_component(self):
        """Delete last component from current system"""
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        if len(df) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete the last component")
            return
            
        # Remove last row
        self.config_parameters[key] = df.iloc[:-1].reset_index(drop=True)
        
        # Update display
        self.load_parameter_table()
        self.update_status("Deleted last component")

    def show_master_theta_dialog(self):
        """Show master parameter array dialog"""
        if self.master_theta is None:
            QMessageBox.warning(self, "No Compilation", "No compiled model available")
            return
            
        # Create simple parameter dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Master Parameters")
        dialog.setModal(True)
        dialog.resize(400, 600)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        # Info
        n_params = len(self.master_theta)
        info_label = QLabel(f"Master Parameter Array: {n_params} parameters")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(info_label)
        
        # Table
        table = QTableWidget()
        table.setRowCount(n_params)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Index', 'Type', 'Value'])
        
        n_comp = n_params // 3
        for i in range(n_params):
            # Index
            table.setItem(i, 0, QTableWidgetItem(str(i)))
            
            # Type
            if i < n_comp:
                param_type = f'N_{i+1}'
            elif i < 2*n_comp:
                param_type = f'b_{i-n_comp+1}'
            else:
                param_type = f'v_{i-2*n_comp+1}'
            table.setItem(i, 1, QTableWidgetItem(param_type))
            
            # Value (editable)
            table.setItem(i, 2, QTableWidgetItem(f"{self.master_theta[i]:.6f}"))
        
        # Make index and type read-only
        for i in range(n_params):
            for col in [0, 1]:
                item = table.item(i, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # Get updated parameters
            try:
                updated_theta = np.zeros(n_params)
                for i in range(n_params):
                    item = table.item(i, 2)
                    updated_theta[i] = float(item.text())
                
                # Update master theta
                self.master_theta = updated_theta
                self.update_status("Master parameters updated")
                    
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Invalid parameter values: {e}")

    # Event handlers
    def on_config_tree_selection_changed(self):
        """Handle configuration tree selection"""
        items = self.config_tree.selectedItems()
        if items:
            config_name = items[0].data(0, Qt.UserRole)
            self.current_config_combo.setCurrentText(config_name)
    
    def on_current_config_changed(self, config_name):
        """Handle current configuration change"""
        self.current_config = config_name
        self.update_systems_display()
        self.update_system_combo()
    
    def on_system_selection_changed(self):
        """Handle system selection change"""
        items = self.systems_tree.selectedItems()
        self.edit_system_btn.setEnabled(len(items) > 0)
        self.delete_system_btn.setEnabled(len(items) > 0)
    
    def on_system_changed(self, system_text):
        """Handle system combo change"""
        if not system_text or not self.current_config:
            return
            
        # Extract system ID from combo text
        parts = system_text.split(' - ')
        if len(parts) >= 2:
            self.current_system_id = parts[0]
            self.load_parameter_table()
    
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
                    
                # Update status
                if self.master_theta is not None:
                    self.update_status("Parameter updated (compile models to propagate changes)")
                    
            except ValueError:
                # Invalid input - revert to original value
                self.load_parameter_table()
                self.update_status("Invalid parameter value - reverted to original")

    # Display update methods
    def update_systems_display(self):
        """Update systems tree display"""
        self.systems_tree.clear()
        
        if not self.current_config:
            return
            
        systems = self.config_systems.get(self.current_config, [])
        for i, system in enumerate(systems):
            item = QTreeWidgetItem()
            item.setText(0, f"{system['ion']} z={system['z']:.4f}")
            
            # Show parameter status
            key = (self.current_config, system['id'])
            n_params = len(self.config_parameters.get(key, []))
            n_transitions = len(system['transitions'])
            item.setText(1, f"{n_transitions} lines, {system['components']} comp, {n_params} params")
            item.setData(0, Qt.UserRole, system['id'])
            self.systems_tree.addTopLevelItem(item)
    
    def update_system_combo(self):
        """Update system selector combo"""
        self.system_combo.clear()
        
        if not self.current_config:
            return
            
        systems = self.config_systems.get(self.current_config, [])
        for system in systems:
            text = f"{system['id']} - {system['ion']} z={system['z']:.4f}"
            self.system_combo.addItem(text)
    
    def load_parameter_table(self):
        """Load parameters into table"""
        self.params_table.setRowCount(0)
        
        if not self.current_config or not self.current_system_id:
            return
            
        key = (self.current_config, self.current_system_id)
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        self.params_table.setRowCount(len(df))
        
        for i, row in df.iterrows():
            for j, (col_name, value) in enumerate(row.items()):
                if j < 4:  # Only show first 4 columns
                    item = QTableWidgetItem(str(value))
                    self.params_table.setItem(i, j, item)
    
    def update_status(self, message=""):
        """Update status text"""
        if message:
            self.status_text.append(message)
        else:
            # Default status
            n_configs = len([c for c in self.configurations.values() if c['wave'] is not None])
            total_systems = sum(len(systems) for systems in self.config_systems.values())
            self.status_text.clear()
            self.status_text.append(f"Configurations: {n_configs}, Total systems: {total_systems}")
            
            if self.master_theta is not None:
                n_params = len(self.master_theta)
                self.status_text.append(f"Compiled: {n_params} parameters ready")
    
    def clear_state(self):
        """Clear all state (for project loading)"""
        self.configurations = {}
        self.config_systems = {}
        self.config_parameters = {}
        self.current_config = None
        self.current_system_id = None
        self.master_config = None
        self.master_theta = None
        self.param_manager = None
        
        # Clear UI
        self.config_tree.clear()
        self.current_config_combo.clear()
        self.systems_tree.clear()
        self.system_combo.clear()
        self.params_table.setRowCount(0)
        self.status_text.clear()
        
        # Disable buttons
        self.show_master_theta_btn.setEnabled(False)
        self.edit_system_btn.setEnabled(False)
        self.delete_system_btn.setEnabled(False)