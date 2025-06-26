#!/usr/bin/env python
"""
rbvfit 2.0 Model Setup Tab - PyQt5 Implementation

Interface for building absorption line models with component management.
"""

import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QTreeWidget, QTreeWidgetItem, QPushButton, QGroupBox,
                            QTableWidget, QTableWidgetItem, QHeaderView, QLabel,
                            QTextEdit, QMessageBox, QSpinBox, QComboBox, QMenu,
                            QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.gui.dialogs.add_system_dialog import AddSystemDialog
from rbvfit.gui.io import create_parameter_dataframe


class ModelSetupTab(QWidget):
    """Tab for setting up absorption line models"""
    
    model_updated = pyqtSignal(object)  # FitConfiguration object
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.config = FitConfiguration()
        self.parameter_df = pd.DataFrame()  # Track component parameters
        self.current_system = None
        self.current_ion = None
        self.fwhm_values = {}  # Store FWHM per instrument
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create the model setup interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Main splitter: systems tree | parameter management
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel: System tree
        self.setup_system_panel(main_splitter)
        
        # Right panel: Parameter management
        self.setup_parameter_panel(main_splitter)
        
        main_splitter.setSizes([400, 600])
        
    def setup_system_panel(self, parent):
        """Create system tree view"""
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        parent.addWidget(left_widget)
        
        # Systems group
        sys_group = QGroupBox("Absorption Systems")
        sys_layout = QVBoxLayout()
        sys_group.setLayout(sys_layout)
        left_layout.addWidget(sys_group)
        
        # System tree
        self.system_tree = QTreeWidget()
        self.system_tree.setHeaderLabels(['System/Ion', 'Components', 'Transitions'])
        self.system_tree.setToolTip("Tree view of all absorption systems and ions")
        sys_layout.addWidget(self.system_tree)
        
        # System control buttons
        sys_btn_layout = QHBoxLayout()
        
        self.add_system_btn = QPushButton("Add System")
        self.add_system_btn.setToolTip("Add new absorption system")
        sys_btn_layout.addWidget(self.add_system_btn)
        
        self.delete_system_btn = QPushButton("Delete System")
        self.delete_system_btn.setEnabled(False)
        self.delete_system_btn.setToolTip("Delete selected system")
        sys_btn_layout.addWidget(self.delete_system_btn)
        
        sys_layout.addLayout(sys_btn_layout)
        
        # FWHM settings group
        fwhm_group = QGroupBox("Instrumental FWHM Settings")
        fwhm_layout = QFormLayout()
        fwhm_group.setLayout(fwhm_layout)
        left_layout.addWidget(fwhm_group)
        
        # Default FWHM for new instruments
        self.default_fwhm_spin = QDoubleSpinBox()
        self.default_fwhm_spin.setRange(0.1, 10.0)
        self.default_fwhm_spin.setValue(2.5)
        self.default_fwhm_spin.setDecimals(2)
        self.default_fwhm_spin.setToolTip("Default FWHM for new instruments (pixels)")
        fwhm_layout.addRow("Default FWHM:", self.default_fwhm_spin)
        
        # FWHM table for loaded instruments
        self.fwhm_table = QTableWidget()
        self.fwhm_table.setColumnCount(2)
        self.fwhm_table.setHorizontalHeaderLabels(["Instrument", "FWHM"])
        self.fwhm_table.horizontalHeader().setStretchLastSection(True)
        self.fwhm_table.setMaximumHeight(150)
        self.fwhm_table.setToolTip("FWHM values for each loaded instrument")
        fwhm_layout.addRow("Instruments:", self.fwhm_table)
        
    def setup_parameter_panel(self, parent):
        """Create parameter management panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        parent.addWidget(right_widget)
        
        # Parameter controls group
        param_group = QGroupBox("Component Parameters")
        param_layout = QVBoxLayout()
        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)
        
        # Component count control
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Components:"))
        
        self.component_spin = QSpinBox()
        self.component_spin.setRange(1, 20)
        self.component_spin.setValue(1)
        self.component_spin.setToolTip("Number of velocity components")
        count_layout.addWidget(self.component_spin)
        
        self.add_component_btn = QPushButton("Add")
        self.add_component_btn.setToolTip("Add new component")
        count_layout.addWidget(self.add_component_btn)
        
        self.delete_component_btn = QPushButton("Delete")
        self.delete_component_btn.setEnabled(False)
        self.delete_component_btn.setToolTip("Delete selected component")
        count_layout.addWidget(self.delete_component_btn)
        
        self.duplicate_component_btn = QPushButton("Duplicate")
        self.duplicate_component_btn.setEnabled(False)
        self.duplicate_component_btn.setToolTip("Duplicate selected component")
        count_layout.addWidget(self.duplicate_component_btn)
        
        count_layout.addStretch()
        param_layout.addLayout(count_layout)
        
        # Parameter table
        self.param_table = QTableWidget()
        self.param_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.param_table.customContextMenuRequested.connect(self.show_context_menu)
        self.param_table.setToolTip("Component parameters: N (log column density), b (Doppler), v (velocity)")
        param_layout.addWidget(self.param_table)
        
        # Parameter action buttons
        param_btn_layout = QHBoxLayout()
        
        self.reset_params_btn = QPushButton("Reset Parameters")
        self.reset_params_btn.setToolTip("Reset all parameters to default values")
        param_btn_layout.addWidget(self.reset_params_btn)
        
        self.apply_params_btn = QPushButton("Apply to Model")
        self.apply_params_btn.setEnabled(False)
        self.apply_params_btn.setToolTip("Apply current parameters to model configuration")
        param_btn_layout.addWidget(self.apply_params_btn)
        
        param_layout.addLayout(param_btn_layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Fitting Mode:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Interactive", "Batch", "MCMC Only"])
        self.mode_combo.setToolTip("Parameter estimation mode")
        mode_layout.addWidget(self.mode_combo)
        
        mode_layout.addStretch()
        param_layout.addLayout(mode_layout)
        
        # Summary text
        summary_group = QGroupBox("Model Summary")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        right_layout.addWidget(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(100)
        self.summary_text.setReadOnly(True)
        self.summary_text.setToolTip("Summary of current model configuration")
        summary_layout.addWidget(self.summary_text)
        
        # Initialize with default component
        self.initialize_default_parameters()
        
    def initialize_default_parameters(self):
        """Initialize with default single component"""
        self.parameter_df = create_parameter_dataframe(1)
        self.setup_parameter_table()
        self.update_summary()
        
    def setup_parameter_table(self):
        """Setup parameter table with current data"""
        if self.parameter_df.empty:
            self.param_table.setRowCount(0)
            self.param_table.setColumnCount(0)
            return
            
        # Setup table structure
        self.param_table.setRowCount(len(self.parameter_df))
        self.param_table.setColumnCount(len(self.parameter_df.columns))
        self.param_table.setHorizontalHeaderLabels(self.parameter_df.columns)
        
        # Fill table with data
        for i, (idx, row) in enumerate(self.parameter_df.iterrows()):
            for j, col in enumerate(self.parameter_df.columns):
                item = QTableWidgetItem(str(row[col]))
                if col == 'Component':
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make component name read-only
                self.param_table.setItem(i, j, item)
                
        # Resize columns
        self.param_table.resizeColumnsToContents()
        header = self.param_table.horizontalHeader()
        header.setStretchLastSection(True)
        
        # Update button states
        self.update_button_states()
        
    def update_button_states(self):
        """Update button enabled states based on current selection"""
        has_selection = len(self.param_table.selectedItems()) > 0
        has_multiple = len(self.parameter_df) > 1
        
        self.delete_component_btn.setEnabled(has_selection and has_multiple)
        self.duplicate_component_btn.setEnabled(has_selection)
        
    def setup_connections(self):
        """Connect signals and slots"""
        # System controls
        self.add_system_btn.clicked.connect(self.add_system)
        self.delete_system_btn.clicked.connect(self.delete_system)
        self.system_tree.itemSelectionChanged.connect(self.on_system_selection)
        
        # Component controls
        self.component_spin.valueChanged.connect(self.on_component_count_changed)
        self.add_component_btn.clicked.connect(self.add_component)
        self.delete_component_btn.clicked.connect(self.delete_selected_component)
        self.duplicate_component_btn.clicked.connect(self.duplicate_component)
        
        # Parameter controls
        self.param_table.itemChanged.connect(self.on_parameter_changed)
        self.param_table.itemSelectionChanged.connect(self.update_button_states)
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        self.apply_params_btn.clicked.connect(self.apply_parameters)
        
        # FWHM controls
        self.fwhm_table.itemChanged.connect(self.on_fwhm_changed)
        self.default_fwhm_spin.valueChanged.connect(self.on_default_fwhm_changed)
        
        # Mode selection
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        
    def add_system(self):
        """Open dialog to add new system"""
        result = AddSystemDialog.get_system(self)
        if result:
            try:
                self.config.add_system(
                    z=result['redshift'],
                    ion=result['ion_name'],
                    transitions=result['transitions'],
                    components=result['components']
                )
                self.refresh_system_tree()
                self.update_summary()
                self.apply_params_btn.setEnabled(True)
                self.main_window.update_status(f"Added system: {result['ion_name']} at z={result['redshift']:.4f}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add system:\n{str(e)}")
                
    def delete_system(self):
        """Delete selected system"""
        current_item = self.system_tree.currentItem()
        if not current_item:
            return
            
        # Find system to delete (handle both system and ion items)
        if current_item.parent():  # Ion item
            system_item = current_item.parent()
            ion_name = current_item.text(0)
        else:  # System item
            system_item = current_item
            ion_name = None
            
        system_text = system_item.text(0)
        
        # Confirm deletion
        msg = f"Delete system '{system_text}'"
        if ion_name:
            msg += f" ion '{ion_name}'"
        msg += "?"
        
        reply = QMessageBox.question(self, "Confirm Deletion", msg,
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # TODO: Implement system/ion deletion in config
            # self.config.delete_system(...)
            self.refresh_system_tree()
            self.update_summary()
            self.main_window.update_status("System deleted")
            
    def on_system_selection(self):
        """Handle system tree selection"""
        current_item = self.system_tree.currentItem()
        self.delete_system_btn.setEnabled(current_item is not None)
        
        if current_item:
            if current_item.parent():  # Ion selected
                self.current_ion = current_item.text(0)
                self.current_system = current_item.parent().text(0)
            else:  # System selected
                self.current_system = current_item.text(0)
                self.current_ion = None
                
    def on_component_count_changed(self, count):
        """Handle component count change"""
        current_count = len(self.parameter_df)
        
        if count > current_count:
            # Add components
            for i in range(current_count, count):
                self.add_component(silent=True)
        elif count < current_count:
            # Remove components
            self.parameter_df = self.parameter_df.iloc[:count].reset_index(drop=True)
            self.setup_parameter_table()
            self.apply_params_btn.setEnabled(True)
            
    def add_component(self, silent=False):
        """Add new component"""
        if self.parameter_df.empty:
            self.parameter_df = create_parameter_dataframe(1)
        else:
            # Create new component with default values
            new_comp_num = len(self.parameter_df) + 1
            new_row = {
                'Component': f"C{new_comp_num}",
                'N': 13.5,      # log column density
                'N_err': 2.0,   # error/range
                'b': 15.0,      # Doppler parameter (km/s)
                'b_err': 10.0,  # error/range
                'v': 0.0,       # velocity offset (km/s)
                'v_err': 50.0   # error/range
            }
            
            new_row_df = pd.DataFrame([new_row])
            self.parameter_df = pd.concat([self.parameter_df, new_row_df], ignore_index=True)
            
        self.component_spin.setValue(len(self.parameter_df))
        self.setup_parameter_table()
        self.apply_params_btn.setEnabled(True)
        
        if not silent:
            self.main_window.update_status(f"Added component ({len(self.parameter_df)} total)")
            
    def delete_selected_component(self):
        """Delete selected component"""
        selected_rows = set()
        for item in self.param_table.selectedItems():
            selected_rows.add(item.row())
            
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No component selected")
            return
            
        if len(self.parameter_df) <= 1:
            QMessageBox.warning(self, "Warning", "Cannot delete the last component")
            return
            
        # Remove rows in reverse order
        for row in sorted(selected_rows, reverse=True):
            self.parameter_df = self.parameter_df.drop(self.parameter_df.index[row]).reset_index(drop=True)
            
        # Renumber components
        for i in range(len(self.parameter_df)):
            self.parameter_df.iloc[i, self.parameter_df.columns.get_loc('Component')] = f"C{i+1}"
            
        self.component_spin.setValue(len(self.parameter_df))
        self.setup_parameter_table()
        self.apply_params_btn.setEnabled(True)
        self.main_window.update_status(f"Deleted component ({len(self.parameter_df)} remaining)")
        
    def duplicate_component(self):
        """Duplicate selected component"""
        selected_rows = set()
        for item in self.param_table.selectedItems():
            selected_rows.add(item.row())
            
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "No component selected")
            return
            
        # Duplicate first selected row
        row_to_dup = min(selected_rows)
        if row_to_dup < len(self.parameter_df):
            dup_row = self.parameter_df.iloc[row_to_dup].copy()
            dup_row['Component'] = f"C{len(self.parameter_df) + 1}"
            
            new_row_df = pd.DataFrame([dup_row])
            self.parameter_df = pd.concat([self.parameter_df, new_row_df], ignore_index=True)
            
            self.component_spin.setValue(len(self.parameter_df))
            self.setup_parameter_table()
            self.apply_params_btn.setEnabled(True)
            self.main_window.update_status(f"Duplicated component ({len(self.parameter_df)} total)")
            
    def show_context_menu(self, position):
        """Show right-click context menu"""
        if self.param_table.itemAt(position) is None:
            return
            
        menu = QMenu(self)
        
        if len(self.parameter_df) > 1:
            delete_action = menu.addAction("Delete Component")
            delete_action.triggered.connect(self.delete_selected_component)
            
        duplicate_action = menu.addAction("Duplicate Component") 
        duplicate_action.triggered.connect(self.duplicate_component)
        
        if menu.actions():
            menu.exec_(self.param_table.mapToGlobal(position))
            
    def on_parameter_changed(self, item):
        """Handle parameter table changes"""
        # Update dataframe with new value
        row = item.row()
        col = item.column()
        try:
            value = float(item.text()) if col > 0 else item.text()
            self.parameter_df.iloc[row, col] = value
            self.apply_params_btn.setEnabled(True)
        except ValueError:
            # Revert invalid input
            old_value = self.parameter_df.iloc[row, col]
            item.setText(str(old_value))
            
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        n_components = len(self.parameter_df)
        self.parameter_df = create_parameter_dataframe(n_components)
        self.setup_parameter_table()
        self.apply_params_btn.setEnabled(True)
        self.main_window.update_status("Parameters reset to defaults")
        
    def apply_parameters(self):
        """Apply current parameters to model configuration"""
        try:
            # Update configuration with current parameters
            self.update_config_components()
            
            # Emit signal for other tabs
            self.model_updated.emit(self.config)
            
            self.apply_params_btn.setEnabled(False)
            self.update_summary()
            self.main_window.update_status("Parameters applied to model")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply parameters:\n{str(e)}")
            
    def update_config_components(self):
        """Update configuration with current component parameters"""
        if self.config is None or self.parameter_df.empty:
            return
            
        # TODO: Update config with parameter_df data
        # This requires implementing parameter update methods in FitConfiguration
        
    def on_fwhm_changed(self, item):
        """Handle FWHM table changes"""
        row = item.row()
        col = item.column()
        
        if col == 1:  # FWHM column
            try:
                fwhm = float(item.text())
                instrument = self.fwhm_table.item(row, 0).text()
                self.fwhm_values[instrument] = fwhm
                self.apply_params_btn.setEnabled(True)
                self.main_window.update_status(f"Updated FWHM for {instrument}: {fwhm}")
            except ValueError:
                QMessageBox.warning(self, "Warning", "Invalid FWHM value")
                item.setText("2.5")  # Reset to default
                
    def on_default_fwhm_changed(self, value):
        """Handle default FWHM change"""
        self.main_window.update_status(f"Default FWHM set to {value}")
        
    def on_mode_changed(self, mode):
        """Handle fitting mode change"""
        self.main_window.update_status(f"Fitting mode: {mode}")
        
    def update_data(self, spectra_data):
        """Update with new spectrum data"""
        # Update FWHM table with loaded instruments
        self.fwhm_table.setRowCount(len(spectra_data))
        
        for i, filename in enumerate(spectra_data.keys()):
            instrument = spectra_data[filename].get('basename', filename)
            
            # Instrument name
            item = QTableWidgetItem(instrument)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.fwhm_table.setItem(i, 0, item)
            
            # FWHM value (use existing or default)
            fwhm = self.fwhm_values.get(instrument, self.default_fwhm_spin.value())
            self.fwhm_values[instrument] = fwhm
            fwhm_item = QTableWidgetItem(str(fwhm))
            self.fwhm_table.setItem(i, 1, fwhm_item)
            
        self.fwhm_table.resizeColumnsToContents()
        
    def get_fwhm_for_instrument(self, instrument):
        """Get FWHM value for specified instrument"""
        return self.fwhm_values.get(instrument, self.default_fwhm_spin.value())
        
    def refresh_system_tree(self):
        """Refresh the system tree display"""
        self.system_tree.clear()
        
        if not self.config:
            return
            
        # TODO: Populate tree from config
        # for system in self.config.systems:
        #     system_item = QTreeWidgetItem([f"z={system.redshift:.4f}", "", ""])
        #     self.system_tree.addTopLevelItem(system_item)
        #     
        #     for ion in system.ions:
        #         ion_item = QTreeWidgetItem([ion.name, str(ion.components), str(len(ion.transitions))])
        #         system_item.addChild(ion_item)
        
        self.system_tree.expandAll()
        
    def update_summary(self):
        """Update model summary text"""
        if not self.config:
            self.summary_text.setPlainText("No model configured")
            return
            
        summary = f"Model Configuration:\n"
        summary += f"• Components: {len(self.parameter_df)}\n"
        summary += f"• Systems: {len(getattr(self.config, 'systems', []))}\n"
        summary += f"• Mode: {self.mode_combo.currentText()}\n"
        
        if self.fwhm_values:
            summary += f"• Instruments: {len(self.fwhm_values)}\n"
            
        self.summary_text.setPlainText(summary)
        
    def handle_spectrum_click(self, wavelength):
        """Handle spectrum click from fitting tab"""
        # Add component at clicked wavelength
        self.main_window.update_status(f"Spectrum clicked at {wavelength:.2f} Å")
        
    def refresh(self):
        """Refresh the tab"""
        self.refresh_system_tree()
        self.update_summary()
        
    def update_from_config(self, config_dict):
        """Update tab from loaded configuration"""
        # TODO: Implement configuration loading
        self.main_window.update_status("Configuration loaded")