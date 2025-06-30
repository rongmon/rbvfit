#!/usr/bin/env python
"""
Configuration & Data Tab for rbvfit 2.0 GUI

This tab handles instrument configuration setup and data loading/processing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                            QGroupBox, QListWidget, QListWidgetItem, QPushButton,
                            QLabel, QDoubleSpinBox, QLineEdit, QComboBox,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QFileDialog, QMessageBox, QTextEdit, QCheckBox,
                            QFormLayout, QSpinBox, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from rbvfit.gui.io import load_multiple_files, slice_spectrum, get_spectrum_info


class ConfigurationDialog(QDialog):
    """Dialog for creating/editing instrument configurations"""
    
    def __init__(self, config_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instrument Configuration")
        self.setModal(True)
        self.resize(400, 200)
        
        self.config_data = config_data or {}
        self.setup_ui()
        
    def setup_ui(self):
        """Create dialog interface"""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Configuration name
        self.name_edit = QLineEdit()
        self.name_edit.setText(self.config_data.get('name', ''))
        layout.addRow("Configuration Name:", self.name_edit)
        
        # FWHM setting
        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(0.1, 50.0)
        self.fwhm_spin.setDecimals(2)
        self.fwhm_spin.setValue(self.config_data.get('fwhm', 2.5))
        layout.addRow("FWHM (pixels):", self.fwhm_spin)
        
        # Description
        self.desc_edit = QLineEdit()
        self.desc_edit.setText(self.config_data.get('description', ''))
        layout.addRow("Description:", self.desc_edit)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_config_data(self):
        """Return configuration data"""
        return {
            'name': self.name_edit.text().strip(),
            'fwhm': self.fwhm_spin.value(),
            'description': self.desc_edit.text().strip(),
            'wave': None,
            'flux': None,
            'error': None,
            'filename': None,
            'wave_original': None,
            'flux_original': None,
            'error_original': None
        }


class WavelengthTrimDialog(QDialog):
    """Dialog for setting wavelength trimming ranges"""
    
    def __init__(self, wave_min, wave_max, current_min=None, current_max=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wavelength Trimming")
        self.setModal(True)
        self.resize(350, 150)
        
        self.wave_min = wave_min
        self.wave_max = wave_max
        
        self.setup_ui(current_min, current_max)
        
    def setup_ui(self, current_min, current_max):
        """Create dialog interface"""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Min wavelength
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(self.wave_min, self.wave_max)
        self.min_spin.setDecimals(2)
        self.min_spin.setValue(current_min if current_min is not None else self.wave_min)
        layout.addRow("Minimum λ (Å):", self.min_spin)
        
        # Max wavelength
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(self.wave_min, self.wave_max)
        self.max_spin.setDecimals(2)
        self.max_spin.setValue(current_max if current_max is not None else self.wave_max)
        layout.addRow("Maximum λ (Å):", self.max_spin)
        
        # Reset button
        reset_btn = QPushButton("Reset to Full Range")
        reset_btn.clicked.connect(self.reset_range)
        layout.addRow(reset_btn)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def reset_range(self):
        """Reset to full wavelength range"""
        self.min_spin.setValue(self.wave_min)
        self.max_spin.setValue(self.wave_max)
        
    def get_range(self):
        """Return selected wavelength range"""
        return self.min_spin.value(), self.max_spin.value()


class UnionBuildDialog(QDialog):
    """Dialog for building unions of overlapping spectral regions"""
    
    def __init__(self, configurations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Spectral Unions")
        self.setModal(True)
        self.resize(500, 400)
        
        self.configurations = configurations
        self.setup_ui()
        
    def setup_ui(self):
        """Create dialog interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Instructions
        instructions = QLabel(
            "Define overlapping wavelength regions to combine spectra.\n"
            "Enter comma-separated ranges like: 1200-1250, 1300-1350"
        )
        layout.addWidget(instructions)
        
        # Region input
        self.regions_edit = QTextEdit()
        self.regions_edit.setMaximumHeight(100)
        self.regions_edit.setPlaceholderText("1200-1250, 1300-1350")
        layout.addWidget(self.regions_edit)
        
        # Configuration selector for union target
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        self.target_combo = QComboBox()
        self.target_combo.addItems(list(self.configurations.keys()))
        form_layout.addRow("Union target configuration:", self.target_combo)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_union_data(self):
        """Return union configuration data"""
        regions_text = self.regions_edit.toPlainText().strip()
        target_config = self.target_combo.currentText()
        
        # Parse regions
        regions = []
        if regions_text:
            for region_str in regions_text.split(','):
                region_str = region_str.strip()
                if '-' in region_str:
                    try:
                        min_wave, max_wave = map(float, region_str.split('-'))
                        regions.append((min_wave, max_wave))
                    except ValueError:
                        continue
        
        return {
            'regions': regions,
            'target_config': target_config
        }


class ConfigurationDataTab(QWidget):
    """Tab for configuration management and data loading"""
    
    # Signals
    configurations_updated = pyqtSignal(dict)  # configurations dict
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.configurations = {}  # Dict of configuration_name -> config_data
        self.loaded_spectra = {}  # Dict of filename -> spectrum_data
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create the configuration and data interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Main splitter: configurations | data management | preview
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        self.setup_configurations_panel(main_splitter)
        self.setup_data_panel(main_splitter)
        self.setup_preview_panel(main_splitter)
        
        main_splitter.setSizes([300, 300, 400])
        
        # Bottom status
        self.status_label = QLabel("Ready - Create configurations and load data")
        layout.addWidget(self.status_label)
        
    def setup_configurations_panel(self, parent):
        """Create configurations management panel"""
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        parent.addWidget(config_widget)
        
        # Configurations group
        config_group = QGroupBox("Instrument Configurations")
        config_group_layout = QVBoxLayout()
        config_group.setLayout(config_group_layout)
        config_layout.addWidget(config_group)
        
        # Configurations list
        self.config_list = QListWidget()
        config_group_layout.addWidget(self.config_list)
        
        # Configuration controls
        config_controls_layout = QHBoxLayout()
        config_group_layout.addLayout(config_controls_layout)
        
        self.add_config_btn = QPushButton("Add Config")
        self.edit_config_btn = QPushButton("Edit")
        self.delete_config_btn = QPushButton("Delete")
        
        config_controls_layout.addWidget(self.add_config_btn)
        config_controls_layout.addWidget(self.edit_config_btn)
        config_controls_layout.addWidget(self.delete_config_btn)
        
        self.edit_config_btn.setEnabled(False)
        self.delete_config_btn.setEnabled(False)
        
        # Configuration info
        self.config_info = QTextEdit()
        self.config_info.setMaximumHeight(100)
        self.config_info.setReadOnly(True)
        config_group_layout.addWidget(self.config_info)
        
    def setup_data_panel(self, parent):
        """Create data management panel"""
        data_widget = QWidget()
        data_layout = QVBoxLayout()
        data_widget.setLayout(data_layout)
        parent.addWidget(data_widget)
        
        # Data loading group
        load_group = QGroupBox("Data Loading")
        load_group_layout = QVBoxLayout()
        load_group.setLayout(load_group_layout)
        data_layout.addWidget(load_group)
        
        self.load_files_btn = QPushButton("Load Spectrum Files")
        load_group_layout.addWidget(self.load_files_btn)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        load_group_layout.addWidget(self.file_list)
        
        # Data assignment group
        assign_group = QGroupBox("Data Assignment")
        assign_group_layout = QFormLayout()
        assign_group.setLayout(assign_group_layout)
        data_layout.addWidget(assign_group)
        
        self.config_selector = QComboBox()
        assign_group_layout.addRow("Configuration:", self.config_selector)
        
        self.file_selector = QComboBox()
        assign_group_layout.addRow("Spectrum File:", self.file_selector)
        
        self.assign_data_btn = QPushButton("Assign Data")
        assign_group_layout.addRow(self.assign_data_btn)
        
        # Processing group
        process_group = QGroupBox("Data Processing")
        process_group_layout = QVBoxLayout()
        process_group.setLayout(process_group_layout)
        data_layout.addWidget(process_group)
        
        self.trim_wavelength_btn = QPushButton("Trim Wavelengths")
        self.build_unions_btn = QPushButton("Build Unions")
        self.reset_data_btn = QPushButton("Reset to Original")
        
        process_group_layout.addWidget(self.trim_wavelength_btn)
        process_group_layout.addWidget(self.build_unions_btn)
        process_group_layout.addWidget(self.reset_data_btn)
        
        # Initially disable data controls
        self.assign_data_btn.setEnabled(False)
        self.trim_wavelength_btn.setEnabled(False)
        self.build_unions_btn.setEnabled(False)
        self.reset_data_btn.setEnabled(False)
        
    def setup_preview_panel(self, parent):
        """Create data preview panel"""
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_widget.setLayout(preview_layout)
        parent.addWidget(preview_widget)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        preview_layout.addLayout(preview_controls)
        
        preview_controls.addWidget(QLabel("Preview:"))
        self.preview_selector = QComboBox()
        preview_controls.addWidget(self.preview_selector)
        
        self.show_all_check = QCheckBox("Show All")
        preview_controls.addWidget(self.show_all_check)
        
        preview_controls.addStretch()
        
        # Plot canvas
        self.preview_figure = Figure(figsize=(8, 6))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas)
        
    def setup_connections(self):
        """Connect signals and slots"""
        # Configuration controls
        self.add_config_btn.clicked.connect(self.add_configuration)
        self.edit_config_btn.clicked.connect(self.edit_configuration)
        self.delete_config_btn.clicked.connect(self.delete_configuration)
        self.config_list.itemSelectionChanged.connect(self.on_config_selection_changed)
        
        # Data controls
        self.load_files_btn.clicked.connect(self.load_spectrum_files)
        self.assign_data_btn.clicked.connect(self.assign_data_to_config)
        self.trim_wavelength_btn.clicked.connect(self.trim_wavelengths)
        self.build_unions_btn.clicked.connect(self.build_unions)
        self.reset_data_btn.clicked.connect(self.reset_data)
        
        # Preview controls
        self.preview_selector.currentTextChanged.connect(self.update_preview)
        self.show_all_check.toggled.connect(self.update_preview)
        
    def add_configuration(self):
        """Add new instrument configuration"""
        dialog = ConfigurationDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            config_data = dialog.get_config_data()
            
            if not config_data['name']:
                QMessageBox.warning(self, "Invalid Name", "Configuration name cannot be empty")
                return
                
            if config_data['name'] in self.configurations:
                QMessageBox.warning(self, "Duplicate Name", 
                                  f"Configuration '{config_data['name']}' already exists")
                return
                
            self.configurations[config_data['name']] = config_data
            self.update_config_display()
            self.update_status()
            
    def edit_configuration(self):
        """Edit selected configuration"""
        current_item = self.config_list.currentItem()
        if not current_item:
            return
            
        config_name = current_item.text().split(' - ')[0]
        config_data = self.configurations[config_name]
        
        dialog = ConfigurationDialog(config_data, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            new_config_data = dialog.get_config_data()
            
            # Handle name change
            if new_config_data['name'] != config_name:
                if new_config_data['name'] in self.configurations:
                    QMessageBox.warning(self, "Duplicate Name", 
                                      f"Configuration '{new_config_data['name']}' already exists")
                    return
                    
                # Copy data to new name
                new_config_data.update({
                    'wave': config_data['wave'],
                    'flux': config_data['flux'],
                    'error': config_data['error'],
                    'filename': config_data['filename'],
                    'wave_original': config_data['wave_original'],
                    'flux_original': config_data['flux_original'],
                    'error_original': config_data['error_original']
                })
                
                del self.configurations[config_name]
                self.configurations[new_config_data['name']] = new_config_data
            else:
                # Update existing
                config_data.update(new_config_data)
                
            self.update_config_display()
            self.update_status()
            
    def delete_configuration(self):
        """Delete selected configuration"""
        current_item = self.config_list.currentItem()
        if not current_item:
            return
            
        config_name = current_item.text().split(' - ')[0]
        
        reply = QMessageBox.question(self, "Confirm Delete",
                                   f"Delete configuration '{config_name}'?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            del self.configurations[config_name]
            self.update_config_display()
            self.update_status()
            
    def on_config_selection_changed(self):
        """Handle configuration selection change"""
        current_item = self.config_list.currentItem()
        has_selection = current_item is not None
        
        self.edit_config_btn.setEnabled(has_selection)
        self.delete_config_btn.setEnabled(has_selection)
        
        if has_selection:
            config_name = current_item.text().split(' - ')[0]
            config_data = self.configurations[config_name]
            
            info_text = f"Name: {config_data['name']}\n"
            info_text += f"FWHM: {config_data['fwhm']} pixels\n"
            info_text += f"Description: {config_data.get('description', 'None')}\n"
            
            if config_data['filename']:
                info_text += f"Data: {config_data['filename']}\n"
                if config_data['wave'] is not None:
                    wave = config_data['wave']
                    info_text += f"λ range: {wave.min():.1f} - {wave.max():.1f} Å\n"
                    info_text += f"Points: {len(wave)}"
            else:
                info_text += "Data: Not assigned"
                
            self.config_info.setText(info_text)
        else:
            self.config_info.clear()
            
    def load_spectrum_files(self):
        """Load spectrum files"""
        file_dialog = QFileDialog()
        files, _ = file_dialog.getOpenFileNames(
            self,
            "Load Spectrum Files",
            "",
            "All supported (*.fits *.json *.txt *.dat);;FITS files (*.fits);;JSON files (*.json);;Text files (*.txt *.dat)"
        )
        
        if files:
            try:
                new_spectra = load_multiple_files(files)
                
                if new_spectra:
                    self.loaded_spectra.update(new_spectra)
                    self.update_file_display()
                    self.update_status()
                    
                    # Show summary
                    info_text = get_spectrum_info(new_spectra)
                    QMessageBox.information(self, "Data Loaded", f"Loaded spectra:\n{info_text}")
                else:
                    QMessageBox.warning(self, "Load Failed", "No valid spectrum data found in selected files")
                    
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load spectrum files:\n{str(e)}")
                
    def assign_data_to_config(self):
        """Assign selected spectrum data to selected configuration"""
        config_name = self.config_selector.currentText()
        filename = self.file_selector.currentText()
        
        if not config_name or not filename:
            QMessageBox.warning(self, "Selection Required", 
                              "Please select both configuration and spectrum file")
            return
            
        if filename not in self.loaded_spectra:
            QMessageBox.warning(self, "File Not Found", f"Spectrum file '{filename}' not found")
            return
            
        # Assign data to configuration
        spectrum_data = self.loaded_spectra[filename]
        config_data = self.configurations[config_name]
        
        config_data.update({
            'wave': spectrum_data['wave'].copy(),
            'flux': spectrum_data['flux'].copy(),
            'error': spectrum_data['error'].copy(),
            'filename': filename,
            'wave_original': spectrum_data['wave'].copy(),
            'flux_original': spectrum_data['flux'].copy(),
            'error_original': spectrum_data['error'].copy()
        })
        
        self.update_config_display()
        self.update_preview_display()
        self.update_controls_state()
        self.update_status()
        
        QMessageBox.information(self, "Data Assigned", 
                              f"Assigned '{filename}' to configuration '{config_name}'")
        
    def trim_wavelengths(self):
        """Trim wavelengths for selected configuration"""
        current_item = self.config_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a configuration")
            return
            
        config_name = current_item.text().split(' - ')[0]
        config_data = self.configurations[config_name]
        
        if config_data['wave'] is None:
            QMessageBox.warning(self, "No Data", "Configuration has no assigned data")
            return
            
        wave = config_data['wave']
        current_min, current_max = wave.min(), wave.max()
        orig_min, orig_max = config_data['wave_original'].min(), config_data['wave_original'].max()
        
        dialog = WavelengthTrimDialog(orig_min, orig_max, current_min, current_max, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            new_min, new_max = dialog.get_range()
            
            # Apply trimming
            self.trim_configuration_data(config_name, new_min, new_max)
            self.update_config_display()
            self.update_preview()
            self.update_status()
            
    def trim_configuration_data(self, config_name, wave_min, wave_max):
        """Apply wavelength trimming to configuration data"""
        config_data = self.configurations[config_name]
        
        # Use original data for trimming
        wave_orig = config_data['wave_original']
        flux_orig = config_data['flux_original']
        error_orig = config_data['error_original']
        
        # Apply mask
        mask = (wave_orig >= wave_min) & (wave_orig <= wave_max)
        
        config_data['wave'] = wave_orig[mask]
        config_data['flux'] = flux_orig[mask]
        config_data['error'] = error_orig[mask]
        
    def build_unions(self):
        """Build unions of overlapping spectral regions"""
        if len(self.configurations) < 2:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Need at least 2 configurations with data to build unions")
            return
            
        # Check that configurations have data
        configs_with_data = {name: config for name, config in self.configurations.items() 
                           if config['wave'] is not None}
        
        if len(configs_with_data) < 2:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Need at least 2 configurations with assigned data")
            return
            
        dialog = UnionBuildDialog(configs_with_data, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            union_data = dialog.get_union_data()
            
            if union_data['regions']:
                self.apply_union_building(union_data)
                self.update_preview()
                self.update_status()
            else:
                QMessageBox.warning(self, "No Regions", "No valid wavelength regions specified")
                
    def apply_union_building(self, union_data):
        """Apply union building to configurations"""
        # This is a placeholder - implement union building logic
        QMessageBox.information(self, "Union Building", 
                              f"Union building with {len(union_data['regions'])} regions\n"
                              f"Target: {union_data['target_config']}\n"
                              "Implementation needed based on specific requirements")
        
    def reset_data(self):
        """Reset all configuration data to original"""
        reply = QMessageBox.question(self, "Confirm Reset",
                                   "Reset all configuration data to original?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for config_data in self.configurations.values():
                if config_data['wave_original'] is not None:
                    config_data['wave'] = config_data['wave_original'].copy()
                    config_data['flux'] = config_data['flux_original'].copy()
                    config_data['error'] = config_data['error_original'].copy()
                    
            self.update_config_display()
            self.update_preview()
            self.update_status()
            
    def update_config_display(self):
        """Update configuration list display"""
        self.config_list.clear()
        self.config_selector.clear()
        self.preview_selector.clear()
        
        for config_name, config_data in self.configurations.items():
            # List widget item
            if config_data['filename']:
                item_text = f"{config_name} - {config_data['filename']}"
            else:
                item_text = f"{config_name} - No data"
            self.config_list.addItem(item_text)
            
            # Combo boxes
            self.config_selector.addItem(config_name)
            if config_data['wave'] is not None:
                self.preview_selector.addItem(config_name)
                
        self.update_controls_state()
        
    def update_file_display(self):
        """Update file list display"""
        self.file_list.clear()
        self.file_selector.clear()
        
        for filename, spectrum_data in self.loaded_spectra.items():
            # List widget
            wave = spectrum_data['wave']
            item_text = f"{spectrum_data['basename']} ({len(wave)} pts, {wave.min():.1f}-{wave.max():.1f} Å)"
            self.file_list.addItem(item_text)
            
            # Combo box
            self.file_selector.addItem(filename)
            
    def update_preview_display(self):
        """Update preview selector with configurations that have data"""
        self.preview_selector.clear()
        
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is not None:
                self.preview_selector.addItem(config_name)
                
    def update_preview(self):
        """Update spectrum preview plot"""
        self.preview_figure.clear()
        ax = self.preview_figure.add_subplot(111)
        
        if self.show_all_check.isChecked():
            # Show all configurations with data
            for config_name, config_data in self.configurations.items():
                if config_data['wave'] is not None:
                    ax.plot(config_data['wave'], config_data['flux'], 
                           label=f"{config_name} (FWHM={config_data['fwhm']})", alpha=0.7)
            ax.legend()
        else:
            # Show selected configuration
            config_name = self.preview_selector.currentText()
            if config_name and config_name in self.configurations:
                config_data = self.configurations[config_name]
                if config_data['wave'] is not None:
                    ax.plot(config_data['wave'], config_data['flux'], 'b-', alpha=0.7)
                    ax.fill_between(config_data['wave'], 
                                  config_data['flux'] - config_data['error'],
                                  config_data['flux'] + config_data['error'],
                                  alpha=0.3)
                    ax.set_title(f"{config_name} (FWHM={config_data['fwhm']})")
                    
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Normalized Flux')
        ax.grid(True, alpha=0.3)
        
        self.preview_canvas.draw()
        
    def update_controls_state(self):
        """Update control button states"""
        has_configs = bool(self.configurations)
        has_files = bool(self.loaded_spectra)
        has_config_selection = self.config_selector.currentText() != ""
        has_file_selection = self.file_selector.currentText() != ""
        
        #self.assign_data_btn.setEnabled(has_config_selection and has_file_selection)
        self.assign_data_btn.setEnabled(True)

        
        # Check if any configs have data
        has_data = any(config['wave'] is not None for config in self.configurations.values())
        
        self.trim_wavelength_btn.setEnabled(has_data)
        self.build_unions_btn.setEnabled(has_data)
        self.reset_data_btn.setEnabled(has_data)
        
    def update_status(self):
        """Update status label"""
        n_configs = len(self.configurations)
        n_files = len(self.loaded_spectra)
        n_assigned = sum(1 for config in self.configurations.values() if config['wave'] is not None)
        
        status = f"Configurations: {n_configs}, Files loaded: {n_files}, Data assigned: {n_assigned}"
        self.status_label.setText(status)
        
        # Emit signal if ready
        if n_assigned > 0:
            self.configurations_updated.emit(self.configurations)
            
    def get_configurations(self):
        """Return current configurations for other tabs"""
        return self.configurations.copy()
        
    def has_valid_configurations(self):
        """Check if there are valid configurations with data"""
        return any(config['wave'] is not None for config in self.configurations.values())