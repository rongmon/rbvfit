#!/usr/bin/env python
"""
Simplified Model Setup Tab for rbvfit 2.0 GUI

Clean, streamlined interface using FitConfiguration and direct VoigtModel creation.
Focus on simplicity and common use cases.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                            QGroupBox, QTreeWidget, QTreeWidgetItem, QPushButton,
                            QLabel, QComboBox, QTableWidget, QTableWidgetItem, 
                            QHeaderView, QMessageBox, QDialog, QFormLayout,
                            QDoubleSpinBox, QSpinBox, QLineEdit, QDialogButtonBox,
                            QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit import vfit_mcmc as mc


# Ion templates for quick setup
ION_TEMPLATES = {
    "CIV": {
        "ion": "CIV", 
        "transitions": [1548.2, 1550.8],
        "description": "C IV doublet (1548, 1551 Å)"
    },
    "SiIV": {
        "ion": "SiIV",
        "transitions": [1393.8, 1402.8],
        "description": "Si IV doublet (1394, 1403 Å)"
    },
    "MgII": {
        "ion": "MgII",
        "transitions": [2796.4, 2803.5],
        "description": "Mg II doublet (2796, 2804 Å)"
    },
    "OVI": {
        "ion": "OVI",
        "transitions": [1031.9, 1037.6],
        "description": "O VI doublet (1032, 1038 Å)"
    },
    "OI": {
        "ion": "OI",
        "transitions": [1302.2],
        "description": "O I 1302 Å"
    },
    "SiII": {
        "ion": "SiII", 
        "transitions": [1260.4, 1304.4, 1526.7],
        "description": "Si II (1260, 1304, 1527 Å)"
    },
    "FeII": {
        "ion": "FeII",
        "transitions": [1608.5, 2374.5, 2382.8, 2586.7, 2600.2],
        "description": "Fe II multiplet"
    },
    "AlIII": {
        "ion": "AlIII",
        "transitions": [1854.7, 1862.8],
        "description": "Al III doublet (1855, 1863 Å)"
    },
    "CIII": {
        "ion": "CIII",
        "transitions": [977.0],
        "description": "C III 977 Å"
    },
    "NV": {
        "ion": "NV", 
        "transitions": [1238.8, 1242.8],
        "description": "N V doublet (1239, 1243 Å)"
    },
    "HI_Lya": {
        "ion": "HI",
        "transitions": [1215.7],
        "description": "Lyman-α (1216 Å)"
    },
    "HI_Lyb": {
        "ion": "HI", 
        "transitions": [1025.7],
        "description": "Lyman-β (1026 Å)"
    },
    "Custom": {
        "ion": "",
        "transitions": [],
        "description": "Custom ion/transitions"
    }
}


class SystemDialog(QDialog):
    """Dialog for adding/editing absorption systems"""
    
    def __init__(self, system_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add/Edit Absorption System")
        self.setModal(True)
        self.resize(500, 400)
        
        # Store data
        self.system_data = system_data or {}
        
        # Create UI
        self.setup_ui()
        
        # Load existing data
        if system_data:
            self.load_system_data(system_data)
    
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Basic system properties
        form_layout = QFormLayout()
        
        # Redshift
        self.redshift_spin = QDoubleSpinBox()
        self.redshift_spin.setDecimals(6)
        self.redshift_spin.setRange(-0.1, 10.0)
        self.redshift_spin.setValue(0.0)
        form_layout.addRow("Redshift (z):", self.redshift_spin)
        
        # Ion template selection
        self.ion_combo = QComboBox()
        self.ion_combo.addItems(list(ION_TEMPLATES.keys()))
        self.ion_combo.currentTextChanged.connect(self.on_ion_template_changed)
        form_layout.addRow("Ion Template:", self.ion_combo)
        
        # Custom ion name (for custom template)
        self.custom_ion_edit = QLineEdit()
        self.custom_ion_edit.setPlaceholderText("e.g., CII, SiIII")
        form_layout.addRow("Custom Ion:", self.custom_ion_edit)
        
        # Transitions
        self.transitions_edit = QLineEdit()
        self.transitions_edit.setPlaceholderText("e.g., 1548.2, 1550.8")
        form_layout.addRow("Transitions (Å):", self.transitions_edit)
        
        # Number of components
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 10)
        self.components_spin.setValue(1)
        form_layout.addRow("Components:", self.components_spin)
        
        layout.addLayout(form_layout)
        
        # Description text
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        layout.addWidget(self.description_label)
        
        # Update description initially
        self.on_ion_template_changed(self.ion_combo.currentText())
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def on_ion_template_changed(self, template_name):
        """Handle ion template selection"""
        if template_name not in ION_TEMPLATES:
            return
            
        template = ION_TEMPLATES[template_name]
        
        # Update description
        self.description_label.setText(template["description"])
        
        # Update fields
        if template_name == "Custom":
            self.custom_ion_edit.setEnabled(True)
            self.transitions_edit.clear()
        else:
            self.custom_ion_edit.setEnabled(False)
            self.custom_ion_edit.clear()
            
            # Set transitions
            transitions_str = ", ".join(f"{t:.1f}" for t in template["transitions"])
            self.transitions_edit.setText(transitions_str)
    
    def load_system_data(self, data):
        """Load existing system data into dialog"""
        self.redshift_spin.setValue(data.get('redshift', 0.0))
        self.components_spin.setValue(data.get('components', 1))
        
        # Try to match ion template
        ion_name = data.get('ion', '')
        transitions = data.get('transitions', [])
        
        matched_template = None
        for template_name, template_data in ION_TEMPLATES.items():
            if (template_data['ion'] == ion_name and 
                set(template_data['transitions']) == set(transitions)):
                matched_template = template_name
                break
        
        if matched_template:
            self.ion_combo.setCurrentText(matched_template)
        else:
            # Use custom
            self.ion_combo.setCurrentText("Custom")
            self.custom_ion_edit.setText(ion_name)
            transitions_str = ", ".join(f"{t:.1f}" for t in transitions)
            self.transitions_edit.setText(transitions_str)
    
    def get_system_data(self):
        """Get system data from dialog"""
        template_name = self.ion_combo.currentText()
        
        if template_name == "Custom":
            ion_name = self.custom_ion_edit.text().strip()
        else:
            ion_name = ION_TEMPLATES[template_name]["ion"]
        
        # Parse transitions
        transitions_text = self.transitions_edit.text().strip()
        try:
            transitions = [float(x.strip()) for x in transitions_text.split(',') if x.strip()]
        except ValueError:
            transitions = []
        
        return {
            'redshift': self.redshift_spin.value(),
            'ion': ion_name,
            'transitions': transitions,
            'components': self.components_spin.value(),
            'template': template_name
        }


class ModelSetupTab(QWidget):
    """Simplified model setup tab using FitConfiguration"""
    
    # Signal emitted when model is ready
    model_updated = pyqtSignal(dict, dict, dict)  # instrument_data, theta_dict, bounds_dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.systems = []  # List of system dictionaries
        self.configurations = {}  # From data tab: {name: {wave, flux, error, fwhm}}
        self.current_config = None
        self.compiled_models = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the main UI"""
        layout = QVBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel: System management
        left_panel = self.create_system_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Configuration and compilation
        right_panel = self.create_config_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 300])
    
    def create_system_panel(self):
        """Create the absorption systems management panel"""
        group = QGroupBox("Absorption Systems")
        layout = QVBoxLayout(group)
        
        # Systems table
        self.systems_table = QTableWidget()
        self.systems_table.setColumnCount(5)
        self.systems_table.setHorizontalHeaderLabels([
            "Redshift", "Ion", "Transitions", "Components", "Template"
        ])
        
        # Make table read-only
        self.systems_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.systems_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Resize columns
        header = self.systems_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.systems_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_system_btn = QPushButton("Add System")
        self.add_system_btn.clicked.connect(self.add_system)
        button_layout.addWidget(self.add_system_btn)
        
        self.edit_system_btn = QPushButton("Edit System")
        self.edit_system_btn.clicked.connect(self.edit_system)
        button_layout.addWidget(self.edit_system_btn)
        
        self.remove_system_btn = QPushButton("Remove System")
        self.remove_system_btn.clicked.connect(self.remove_system)
        button_layout.addWidget(self.remove_system_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return group
    
    def create_config_panel(self):
        """Create the configuration and compilation panel"""
        group = QGroupBox("Model Configuration & Compilation")
        layout = QVBoxLayout(group)
        
        # FitConfiguration preview
        config_group = QGroupBox("Current Configuration")
        config_layout = QVBoxLayout(config_group)
        
        self.config_preview = QTextEdit()
        self.config_preview.setMaximumHeight(150)
        self.config_preview.setFont(QFont("monospace"))
        self.config_preview.setReadOnly(True)
        config_layout.addWidget(self.config_preview)
        
        layout.addWidget(config_group)
        
        # Instrument FWHM settings
        fwhm_group = QGroupBox("Instrument FWHM Settings")
        fwhm_layout = QVBoxLayout(fwhm_group)
        
        self.fwhm_table = QTableWidget()
        self.fwhm_table.setColumnCount(2)
        self.fwhm_table.setHorizontalHeaderLabels(["Instrument", "FWHM"])
        self.fwhm_table.horizontalHeader().setStretchLastSection(True)
        fwhm_layout.addWidget(self.fwhm_table)
        
        layout.addWidget(fwhm_group)
        
        # Compilation
        compile_group = QGroupBox("Model Compilation")
        compile_layout = QVBoxLayout(compile_group)
        
        self.compile_btn = QPushButton("Compile Models")
        self.compile_btn.clicked.connect(self.compile_models)
        self.compile_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        compile_layout.addWidget(self.compile_btn)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setFont(QFont("monospace", 9))
        self.status_text.setReadOnly(True)
        compile_layout.addWidget(self.status_text)
        
        layout.addWidget(compile_group)
        
        return group
    
    def add_system(self):
        """Add a new absorption system"""
        dialog = SystemDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            system_data = dialog.get_system_data()
            
            # Validate
            if not system_data['ion'] or not system_data['transitions']:
                QMessageBox.warning(self, "Invalid System", 
                                  "Please specify ion name and transitions.")
                return
            
            self.systems.append(system_data)
            self.update_systems_table()
            self.update_config_preview()
    
    def edit_system(self):
        """Edit selected absorption system"""
        row = self.systems_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Please select a system to edit.")
            return
        
        system_data = self.systems[row]
        dialog = SystemDialog(system_data, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            updated_data = dialog.get_system_data()
            
            # Validate
            if not updated_data['ion'] or not updated_data['transitions']:
                QMessageBox.warning(self, "Invalid System", 
                                  "Please specify ion name and transitions.")
                return
            
            self.systems[row] = updated_data
            self.update_systems_table()
            self.update_config_preview()
    
    def remove_system(self):
        """Remove selected absorption system"""
        row = self.systems_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Please select a system to remove.")
            return
        
        reply = QMessageBox.question(self, "Confirm Removal", 
                                   "Remove selected absorption system?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            del self.systems[row]
            self.update_systems_table()
            self.update_config_preview()
    
    def update_systems_table(self):
        """Update the systems table display"""
        self.systems_table.setRowCount(len(self.systems))
        
        for row, system in enumerate(self.systems):
            # Redshift
            self.systems_table.setItem(row, 0, QTableWidgetItem(f"{system['redshift']:.6f}"))
            
            # Ion
            self.systems_table.setItem(row, 1, QTableWidgetItem(system['ion']))
            
            # Transitions
            trans_str = ", ".join(f"{t:.1f}" for t in system['transitions'])
            self.systems_table.setItem(row, 2, QTableWidgetItem(trans_str))
            
            # Components
            self.systems_table.setItem(row, 3, QTableWidgetItem(str(system['components'])))
            
            # Template
            self.systems_table.setItem(row, 4, QTableWidgetItem(system.get('template', 'Custom')))
    
    def update_config_preview(self):
        """Update the FitConfiguration preview"""
        if not self.systems:
            self.config_preview.setText("No absorption systems defined.\nUse 'Add System' to get started.")
            return
        
        try:
            # Create temporary config to show structure
            config = FitConfiguration()
            
            for system in self.systems:
                config.add_system(
                    z=system['redshift'],
                    ion=system['ion'],
                    transitions=system['transitions'],
                    components=system['components']
                )
            
            # Show summary
            preview_text = config.summary()
            
            # Add parameter count info
            total_components = sum(s['components'] for s in self.systems)
            total_params = total_components * 3  # N, b, v for each component
            
            preview_text += f"\n\nTotal Components: {total_components}"
            preview_text += f"\nTotal Parameters: {total_params} (N, b, v)"
            
            self.config_preview.setText(preview_text)
            
        except Exception as e:
            self.config_preview.setText(f"Configuration Error:\n{str(e)}")
    
    def update_configurations(self, configurations):
        """Update available instrument configurations from data tab"""
        self.configurations = configurations
        self.update_fwhm_table()
        self.update_status("Configurations updated from data tab")
    
    def update_fwhm_table(self):
        """Update the FWHM settings table"""
        self.fwhm_table.setRowCount(len(self.configurations))
        
        for row, (name, config_data) in enumerate(self.configurations.items()):
            # Instrument name
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.fwhm_table.setItem(row, 0, name_item)
            
            # FWHM (editable)
            current_fwhm = config_data.get('fwhm', 6.5)
            fwhm_item = QTableWidgetItem(f"{current_fwhm:.1f}")
            self.fwhm_table.setItem(row, 1, fwhm_item)
    
    def compile_models(self):
        """Compile VoigtModel objects for all instruments"""
        if not self.systems:
            QMessageBox.warning(self, "No Systems", "Please add absorption systems first.")
            return
        
        if not self.configurations:
            QMessageBox.warning(self, "No Data", "Please load spectroscopic data first.")
            return
        
        try:
            self.update_status("Starting model compilation...")
            
            # Create FitConfiguration
            config = FitConfiguration()
            
            for system in self.systems:
                config.add_system(
                    z=system['redshift'],
                    ion=system['ion'],
                    transitions=system['transitions'],
                    components=system['components']
                )
            
            self.update_status("FitConfiguration created successfully")
            
            # Get updated FWHM values from table
            fwhm_values = {}
            for row in range(self.fwhm_table.rowCount()):
                name = self.fwhm_table.item(row, 0).text()
                fwhm_text = self.fwhm_table.item(row, 1).text()
                try:
                    fwhm_values[name] = float(fwhm_text)
                except ValueError:
                    fwhm_values[name] = 6.5  # Default
            
            # Create VoigtModel objects for each instrument
            voigt_models = {}
            for name, config_data in self.configurations.items():
                if config_data['wave'] is None:
                    continue
                
                fwhm = fwhm_values.get(name, 6.5)
                voigt_model = VoigtModel(config, FWHM=str(fwhm))
                voigt_models[name] = voigt_model
                
                self.update_status(f"Created VoigtModel for {name} (FWHM={fwhm})")
            
            # Build instrument_data dictionary
            instrument_data = {}
            for name, voigt_model in voigt_models.items():
                config_data = self.configurations[name]
                instrument_data[name] = {
                    'model': voigt_model,
                    'wave': config_data['wave'],
                    'flux': config_data['flux'],
                    'error': config_data['error']
                }
            
            self.update_status(f"Built instrument_data for {len(instrument_data)} instruments")
            
            # Create initial parameter guess and bounds
            total_components = sum(s['components'] for s in self.systems)
            
            # Simple initial guesses
            nguess = [14.0] * total_components  # log column density
            bguess = [20.0] * total_components  # Doppler parameter
            vguess = [0.0] * total_components   # velocity offset
            
            # Use rbvfit bounds utility
            bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
            
            # Prepare output dictionaries
            theta = np.concatenate([nguess, bguess, vguess])
            theta_dict = {'theta': theta, 'length': len(theta)}
            bounds_dict = {'lb': lb, 'ub': ub}
            
            # Store compiled models
            self.compiled_models = voigt_models
            
            # Emit signal to fitting tab
            self.model_updated.emit(instrument_data, theta_dict, bounds_dict)
            
            # Success message
            success_msg = (
                f"✓ Models compiled successfully!\n"
                f"  Instruments: {len(instrument_data)}\n"
                f"  Systems: {len(self.systems)}\n"
                f"  Components: {total_components}\n"
                f"  Parameters: {len(theta)}"
            )
            
            self.update_status(success_msg)
            QMessageBox.information(self, "Success", success_msg)
            
        except Exception as e:
            error_msg = f"Model compilation failed:\n{str(e)}"
            self.update_status(error_msg)
            QMessageBox.critical(self, "Compilation Error", error_msg)
    
    def update_status(self, message):
        """Update status display"""
        self.status_text.append(f"[{self.get_timestamp()}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)
    
    def get_timestamp(self):
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def clear_all(self):
        """Clear all systems and reset"""
        reply = QMessageBox.question(self, "Clear All", 
                                   "Clear all absorption systems?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.systems.clear()
            self.compiled_models.clear()
            self.update_systems_table()
            self.update_config_preview()
            self.status_text.clear()
            self.update_status("All systems cleared")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Test the widget
    widget = ModelSetupTab()
    widget.show()
    
    sys.exit(app.exec_())