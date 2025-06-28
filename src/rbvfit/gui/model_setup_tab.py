#!/usr/bin/env python
"""
rbvfit 2.0 Model Setup Tab - Redesigned
"""

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QTreeWidget, QTreeWidgetItem, QPushButton, QGroupBox,
                            QTableWidget, QTableWidgetItem, QHeaderView, QLabel,
                            QMessageBox, QSpinBox, QComboBox, QDoubleSpinBox,
                            QFormLayout, QDialog, QDialogButtonBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
import rbvfit.vfit_mcmc as mc
from rbvfit.gui.interactive_param_dialog import InteractiveParameterDialog, show_validation_warning


class AddSystemDialog(QDialog):
    """Dialog for adding new absorption system"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Absorption System")
        self.setModal(True)
        
        layout = QFormLayout()
        self.setLayout(layout)
        
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.0, 10.0)
        self.z_spin.setDecimals(6)
        self.z_spin.setValue(0.0)
        layout.addRow("Redshift:", self.z_spin)
        
        self.ion_combo = QComboBox()
        self.ion_combo.setEditable(True)
        self.ion_combo.addItem("")  # Empty option for auto-detect
        common_ions = ['HI', 'CII', 'CIV', 'SiII', 'SiIV', 'OI', 'OVI', 'MgII', 'FeII']
        self.ion_combo.addItems(common_ions)
        layout.addRow("Ion (empty = auto-detect):", self.ion_combo)
        
        self.transitions_edit = QLineEdit()
        self.transitions_edit.setPlaceholderText("e.g., 1548.2, 1550.3")
        layout.addRow("Transitions (Å):", self.transitions_edit)
        
        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(0.1, 20.0)
        self.fwhm_spin.setDecimals(2)
        self.fwhm_spin.setValue(2.5)
        layout.addRow("FWHM (pixels):", self.fwhm_spin)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_system_data(self):
        """Return system data as dict"""
        transitions_text = self.transitions_edit.text().strip()
        transitions = [float(x.strip()) for x in transitions_text.split(',') if x.strip()]
        
        ion_name = self.ion_combo.currentText().strip()
        
        return {
            'z': self.z_spin.value(),
            'ion': ion_name,  # Empty string if auto-detect
            'transitions': transitions,
            'fwhm': self.fwhm_spin.value()
        }


class ModelSetupTab(QWidget):
    """Tab for setting up absorption line models"""
    
    model_updated = pyqtSignal(object, dict)  # (compiled_model, mcmc_params)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.systems = []  # List of system dicts
        self.parameter_tables = {}  # Dict of system_id -> DataFrame
        self.current_system_id = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create the model setup interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Main splitter: systems | parameters
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        self.setup_systems_panel(main_splitter)
        self.setup_parameters_panel(main_splitter)
        
        main_splitter.setSizes([300, 500])
        
        # Bottom controls
        self.setup_bottom_controls(layout)
        
    def setup_systems_panel(self, parent):
        """Create systems management panel"""
        systems_widget = QWidget()
        systems_layout = QVBoxLayout()
        systems_widget.setLayout(systems_layout)
        parent.addWidget(systems_widget)
        
        # Systems group
        systems_group = QGroupBox("Absorption Systems")
        systems_group_layout = QVBoxLayout()
        systems_group.setLayout(systems_group_layout)
        systems_layout.addWidget(systems_group)
        
        # Systems list
        self.systems_tree = QTreeWidget()
        self.systems_tree.setHeaderLabels(['System', 'Details'])
        self.systems_tree.setColumnWidth(0, 150)
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
        self.progress_label = QLabel("No systems configured")
        params_layout.addWidget(self.progress_label)
        
    def setup_bottom_controls(self, parent_layout):
        """Create bottom control buttons"""
        controls_layout = QHBoxLayout()
        parent_layout.addLayout(controls_layout)
        
        controls_layout.addStretch()
        
        self.compile_btn = QPushButton("Compile Model")
        self.compile_btn.setEnabled(False)
        controls_layout.addWidget(self.compile_btn)
        
    def setup_connections(self):
        """Connect signals and slots"""
        self.add_system_btn.clicked.connect(self.add_system)
        self.edit_system_btn.clicked.connect(self.edit_system)
        self.delete_system_btn.clicked.connect(self.delete_system)
        
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        self.interactive_btn.clicked.connect(self.launch_interactive_params)
        self.manual_btn.clicked.connect(self.setup_manual_params)
        self.clear_params_btn.clicked.connect(self.clear_params)
        
        self.add_component_btn.clicked.connect(self.add_component)
        self.delete_component_btn.clicked.connect(self.delete_component)
        
        self.compile_btn.clicked.connect(self.compile_model)
        
        self.systems_tree.itemSelectionChanged.connect(self.on_system_selection_changed)
        
    def add_system(self):
        """Add new absorption system"""
        dialog = AddSystemDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            system_data = dialog.get_system_data()
            
            if not system_data['transitions']:
                QMessageBox.warning(self, "Invalid Input", "Please specify at least one transition")
                return
                
            # Auto-detect ion if empty
            if not system_data['ion']:
                try:
                    from rbvfit import rb_setline as rb
                    line_info = rb.rb_setline(system_data['transitions'][0], 'closest')
                    detected_ion = line_info['name'][0].split()[0]
                    system_data['ion'] = detected_ion
                    print(f"Auto-detected ion: {detected_ion}")
                except Exception as e:
                    QMessageBox.warning(self, "Auto-detect Failed", 
                                      f"Could not auto-detect ion from wavelength {system_data['transitions'][0]:.1f}Å\n"
                                      f"Error: {str(e)}\nPlease enter ion name manually.")
                    return
                
            system_id = f"sys_{len(self.systems)}"
            system_data['id'] = system_id
            
            self.systems.append(system_data)
            self.parameter_tables[system_id] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            
            self.update_systems_display()
            self.update_progress()
            
    def edit_system(self):
        """Edit selected system"""
        current_item = self.systems_tree.currentItem()
        if not current_item:
            return
            
        system_id = current_item.data(0, Qt.UserRole)
        system_data = next((s for s in self.systems if s['id'] == system_id), None)
        
        if system_data:
            dialog = AddSystemDialog(self)
            dialog.z_spin.setValue(system_data['z'])
            dialog.ion_combo.setCurrentText(system_data['ion'])
            dialog.transitions_edit.setText(', '.join(map(str, system_data['transitions'])))
            dialog.fwhm_spin.setValue(system_data['fwhm'])
            
            if dialog.exec_() == QDialog.Accepted:
                new_data = dialog.get_system_data()
                system_data.update(new_data)
                self.update_systems_display()
                
    def delete_system(self):
        """Delete selected system"""
        current_item = self.systems_tree.currentItem()
        if not current_item:
            return
            
        system_id = current_item.data(0, Qt.UserRole)
        
        reply = QMessageBox.question(self, "Delete System", 
                                   f"Delete system {system_id}?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.systems = [s for s in self.systems if s['id'] != system_id]
            if system_id in self.parameter_tables:
                del self.parameter_tables[system_id]
                
            self.update_systems_display()
            self.update_progress()
            
    def update_systems_display(self):
        """Update systems tree display"""
        self.systems_tree.clear()
        self.system_combo.clear()
        
        for system in self.systems:
            item = QTreeWidgetItem()
            item.setText(0, f"{system['ion']} z={system['z']:.3f}")
            item.setText(1, f"{len(system['transitions'])} transitions, FWHM={system['fwhm']}")
            item.setData(0, Qt.UserRole, system['id'])
            self.systems_tree.addTopLevelItem(item)
            
            self.system_combo.addItem(f"{system['ion']} z={system['z']:.3f}", system['id'])
            
    def on_system_selection_changed(self):
        """Handle system selection change"""
        current_item = self.systems_tree.currentItem()
        self.edit_system_btn.setEnabled(current_item is not None)
        self.delete_system_btn.setEnabled(current_item is not None)
        
    def on_system_changed(self, text):
        """Handle system combo change"""
        if text:
            system_id = self.system_combo.currentData()
            self.current_system_id = system_id
            self.load_parameter_table()
            
    def launch_interactive_params(self):
        """Launch interactive parameter estimation"""
        if not self.current_system_id:
            QMessageBox.warning(self, "No System", "Please select a system first")
            return
            
        # Get current system data
        system_data = next((s for s in self.systems if s['id'] == self.current_system_id), None)
        if not system_data:
            return
            
        # Get spectrum data from main window
        if not hasattr(self.main_window, 'get_spectrum_data'):
            QMessageBox.warning(self, "No Data", "No spectrum data available")
            return
            
        spec_data = self.main_window.get_spectrum_data()
        if not spec_data:
            QMessageBox.warning(self, "No Data", "No spectrum data loaded")
            return
            
        # Use first available spectrum
        first_key = list(spec_data.keys())[0]
        spectrum_data = spec_data[first_key]
        
        # Launch interactive dialog
        dialog = InteractiveParameterDialog(system_data, spectrum_data, self)
        dialog.parameters_ready.connect(self.on_interactive_parameters_ready)
        dialog.exec_()
        
    def on_interactive_parameters_ready(self, df):
        """Handle interactive parameters result"""
        if self.current_system_id:
            self.parameter_tables[self.current_system_id] = df.copy()
            self.load_parameter_table()
            self.update_progress()
        
    def setup_manual_params(self):
        """Set up manual parameter entry"""
        if not self.current_system_id:
            QMessageBox.warning(self, "No System", "Please select a system first")
            return
            
        # Add default component if table is empty
        df = self.parameter_tables[self.current_system_id]
        if df.empty:
            self.add_component()
            
    def clear_params(self):
        """Clear parameters for current system"""
        if self.current_system_id:
            self.parameter_tables[self.current_system_id] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])
            self.load_parameter_table()
            self.update_progress()
            
    def add_component(self):
        """Add new component to current system"""
        if not self.current_system_id:
            return
            
        df = self.parameter_tables[self.current_system_id]
        new_component = len(df) + 1
        
        new_row = pd.DataFrame({
            'Component': [new_component],
            'N': [13.5],  # Default log column density
            'b': [25.0],  # Default Doppler parameter
            'v': [0.0]    # Default velocity
        })
        
        self.parameter_tables[self.current_system_id] = pd.concat([df, new_row], ignore_index=True)
        self.load_parameter_table()
        self.update_progress()
        
    def delete_component(self):
        """Delete selected component"""
        if not self.current_system_id:
            return
            
        current_row = self.params_table.currentRow()
        if current_row >= 0:
            df = self.parameter_tables[self.current_system_id]
            self.parameter_tables[self.current_system_id] = df.drop(df.index[current_row]).reset_index(drop=True)
            
            # Renumber components
            df = self.parameter_tables[self.current_system_id]
            if not df.empty:
                df['Component'] = range(1, len(df) + 1)
                
            self.load_parameter_table()
            self.update_progress()
            
    def load_parameter_table(self):
        """Load parameter table for current system"""
        if not self.current_system_id:
            self.params_table.setRowCount(0)
            return
            
        df = self.parameter_tables[self.current_system_id]
        
        self.params_table.setRowCount(len(df))
        
        for i, row in df.iterrows():
            self.params_table.setItem(i, 0, QTableWidgetItem(str(row['Component'])))
            self.params_table.setItem(i, 1, QTableWidgetItem(f"{row['N']:.2f}"))
            self.params_table.setItem(i, 2, QTableWidgetItem(f"{row['b']:.1f}"))
            self.params_table.setItem(i, 3, QTableWidgetItem(f"{row['v']:.1f}"))
            
        # Connect item changed signal
        self.params_table.itemChanged.connect(self.on_parameter_changed)
        
    def on_parameter_changed(self, item):
        """Handle parameter table changes"""
        if not self.current_system_id:
            return
            
        row = item.row()
        col = item.column()
        
        try:
            value = float(item.text())
            df = self.parameter_tables[self.current_system_id]
            
            if col == 1:  # N column
                df.iloc[row, df.columns.get_loc('N')] = value
            elif col == 2:  # b column
                df.iloc[row, df.columns.get_loc('b')] = value
            elif col == 3:  # v column
                df.iloc[row, df.columns.get_loc('v')] = value
                
            self.validate_parameters()
            
        except ValueError:
            show_validation_warning(self, "Please enter a valid number")
            self.load_parameter_table()  # Reload to reset invalid value
            
    def validate_parameters(self):
        """Validate parameter ranges"""
        if not self.current_system_id:
            return
            
        df = self.parameter_tables[self.current_system_id]
        
        warnings = []
        for i, row in df.iterrows():
            if row['N'] < 10 or row['N'] > 22:
                warnings.append(f"Component {i+1}: N outside typical range [10-22]")
            if row['b'] < 5 or row['b'] > 200:
                warnings.append(f"Component {i+1}: b outside typical range [5-200] km/s")
                
        if warnings:
            show_validation_warning(self, "\n".join(warnings))
            
    def update_progress(self):
        """Update progress indicator"""
        total_systems = len(self.systems)
        if total_systems == 0:
            self.progress_label.setText("No systems configured")
            self.compile_btn.setEnabled(False)
            return
            
        completed_systems = sum(1 for sys_id in self.parameter_tables.keys() 
                              if not self.parameter_tables[sys_id].empty)
        
        self.progress_label.setText(f"Parameters set for {completed_systems}/{total_systems} systems")
        self.compile_btn.setEnabled(completed_systems == total_systems and total_systems > 0)
        
    def get_current_theta(self):
        """Get combined theta array for all systems"""
        if not self.systems:
            return None
            
        theta_parts = []
        
        for system in self.systems:
            system_id = system['id']
            if system_id not in self.parameter_tables or self.parameter_tables[system_id].empty:
                return None
                
            df = self.parameter_tables[system_id]
            n_vals = df['N'].values
            b_vals = df['b'].values
            v_vals = df['v'].values
            
            theta_parts.extend([n_vals, b_vals, v_vals])
            
        return np.concatenate(theta_parts)
        
    def get_current_model(self):
        """Get the current VoigtModel (before compilation)"""
        if not self.systems:
            return None
            
        # For now, create single instrument config
        # TODO: Support multi-instrument
        config = FitConfiguration()
        
        for system in self.systems:
            system_id = system['id']
            if system_id not in self.parameter_tables or self.parameter_tables[system_id].empty:
                continue
                
            df = self.parameter_tables[system_id]
            n_components = len(df)
            
            config.add_system(
                z=system['z'],
                ion=system['ion'],
                transitions=system['transitions'],
                components=n_components
            )
            
        # Use FWHM from first system for now
        fwhm = self.systems[0]['fwhm'] if self.systems else 2.5
        model = VoigtModel(config, FWHM=str(fwhm))
        
        return model
        
    def compile_model(self):
        """Compile model and prepare for fitting"""
        try:
            # Get current model
            model = self.get_current_model()
            if not model:
                QMessageBox.warning(self, "No Model", "No valid model to compile")
                return
                
            # Compile model
            compiled = model.compile(verbose=True)
            
            # Get theta
            theta = self.get_current_theta()
            if theta is None:
                QMessageBox.warning(self, "No Parameters", "No parameters available")
                return
                
            # Set bounds
            n_components_total = len(theta) // 3
            n_guess = theta[:n_components_total]
            b_guess = theta[n_components_total:2*n_components_total]
            v_guess = theta[2*n_components_total:]
            
            bounds, lb, ub = mc.set_bounds(n_guess, b_guess, v_guess)
            
            # Get spectrum data from main window
            if not hasattr(self.main_window, 'get_spectrum_data'):
                QMessageBox.warning(self, "No Data", "No spectrum data available")
                return
                
            spec_data = self.main_window.get_spectrum_data()
            if not spec_data:
                QMessageBox.warning(self, "No Data", "No spectrum data loaded")
                return
                
            # Use first available spectrum for now
            first_key = list(spec_data.keys())[0]
            wave = spec_data[first_key]['wave']
            flux = spec_data[first_key]['flux']
            error = spec_data[first_key].get('error', np.ones_like(flux) * 0.05)
            
            # Prepare MCMC parameters
            mcmc_params = {
                'wave': wave,
                'flux': flux,
                'error': error,
                'theta_guess': theta,
                'bounds': (lb, ub)
            }
            
            self.model_updated.emit(compiled, mcmc_params)
            QMessageBox.information(self, "Success", "Model compiled successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Compilation Error", f"Failed to compile model:\n{str(e)}")
            
    def set_spectra_data(self, spectra_data):
        """Set available spectra data"""
        self.spectra_data = spectra_data