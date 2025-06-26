#!/usr/bin/env python
"""
rbvfit 2.0 Fitting Tab - PyQt5 Implementation

Interface for interactive fitting with spectrum display and wavelength slicing.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QComboBox, QCheckBox, QPushButton, QLabel,
                            QSlider, QDoubleSpinBox, QSpinBox, QProgressBar,
                            QTextEdit, QFormLayout, QMessageBox, QFrame,
                            QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont

# Always import matplotlib - no conditional checks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

# Full path imports - always available
from rbvfit.core.voigt_model import VoigtModel
from rbvfit.analysis.mcmc_fitting import run_mcmc_fit
from rbvfit.gui.io import slice_spectrum, reset_spectrum_slice


class MCMCThread(QThread):
    """Thread for running MCMC fits"""
    
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    fitting_completed = pyqtSignal(object)
    fitting_error = pyqtSignal(str)
    
    def __init__(self, model, mcmc_params):
        super().__init__()
        self.model = model
        self.mcmc_params = mcmc_params
        
    def run(self):
        """Run MCMC fitting in background"""
        try:
            self.status_update.emit("Starting MCMC fitting...")
            
            # Run the fit
            results = run_mcmc_fit(self.model, **self.mcmc_params)
            
            self.status_update.emit("MCMC fitting completed")
            self.fitting_completed.emit(results)
            
        except Exception as e:
            self.fitting_error.emit(str(e))


class FittingTab(QWidget):
    """Tab for interactive fitting and spectrum display"""
    
    spectrum_clicked = pyqtSignal(float)  # wavelength
    fitting_started = pyqtSignal(dict)    # mcmc_params
    fitting_completed = pyqtSignal(object)  # results
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.spectra_data = {}
        self.spectra_original = {}  # Keep original data
        self.current_instrument = None
        self.fit_config = None
        self.velocity_markers = []
        self.mcmc_thread = None
        
        # Matplotlib setup
        plt.style.use('default')
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create fitting interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Top controls
        self.setup_top_controls(layout)
        
        # Main content: spectrum display | controls
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left: Spectrum display with slicing
        self.setup_spectrum_panel(main_splitter)
        
        # Right: Parameter and MCMC controls
        self.setup_control_panel(main_splitter)
        
        main_splitter.setSizes([800, 400])
        
    def setup_top_controls(self, parent_layout):
        """Create top control bar"""
        control_frame = QFrame()
        control_layout = QHBoxLayout()
        control_frame.setLayout(control_layout)
        parent_layout.addWidget(control_frame)
        
        # Instrument selector
        control_layout.addWidget(QLabel("Instrument:"))
        self.instrument_combo = QComboBox()
        self.instrument_combo.setToolTip("Select spectrum to display")
        control_layout.addWidget(self.instrument_combo)
        
        # Wavelength range controls
        control_layout.addWidget(QLabel("λ Range:"))
        self.wave_min_spin = QDoubleSpinBox()
        self.wave_min_spin.setRange(0, 50000)
        self.wave_min_spin.setDecimals(2)
        self.wave_min_spin.setSuffix(" Å")
        self.wave_min_spin.setToolTip("Minimum wavelength")
        control_layout.addWidget(self.wave_min_spin)
        
        control_layout.addWidget(QLabel("to"))
        
        self.wave_max_spin = QDoubleSpinBox()
        self.wave_max_spin.setRange(0, 50000)
        self.wave_max_spin.setDecimals(2)
        self.wave_max_spin.setSuffix(" Å")
        self.wave_max_spin.setToolTip("Maximum wavelength")
        control_layout.addWidget(self.wave_max_spin)
        
        # Slice controls
        self.slice_btn = QPushButton("Slice")
        self.slice_btn.setToolTip("Apply wavelength slice")
        control_layout.addWidget(self.slice_btn)
        
        self.reset_slice_btn = QPushButton("Reset")
        self.reset_slice_btn.setToolTip("Reset to full spectrum")
        control_layout.addWidget(self.reset_slice_btn)
        
        control_layout.addStretch()
        
        # Display controls
        self.show_model_check = QCheckBox("Show Model")
        self.show_model_check.setToolTip("Display current model overlay")
        control_layout.addWidget(self.show_model_check)
        
        self.show_components_check = QCheckBox("Show Components")
        self.show_components_check.setToolTip("Display individual components")
        control_layout.addWidget(self.show_components_check)
        
    def setup_spectrum_panel(self, parent):
        """Create spectrum display panel"""
        spec_widget = QWidget()
        spec_layout = QVBoxLayout()
        spec_widget.setLayout(spec_layout)
        parent.addWidget(spec_widget)
        
        # Spectrum display group
        display_group = QGroupBox("Spectrum Display")
        display_layout = QVBoxLayout()
        display_group.setLayout(display_layout)
        spec_layout.addWidget(display_group)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setToolTip("Left click: add component, Right click: remove component")
        display_layout.addWidget(self.canvas)
        
        # Create main spectrum axis
        self.ax_spec = self.figure.add_subplot(111)
        self.ax_spec.set_xlabel('Wavelength (Å)')
        self.ax_spec.set_ylabel('Normalized Flux')
        self.ax_spec.grid(True, alpha=0.3)
        
        # Initialize empty plot elements
        self.spec_line = None
        self.error_fill = None
        self.model_line = None
        self.component_lines = []
        self.velocity_markers_plot = []
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_spectrum_click)
        
        # Zoom/pan controls
        zoom_layout = QHBoxLayout()
        
        self.zoom_check = QCheckBox("Zoom Mode")
        self.zoom_check.setToolTip("Enable zoom/pan mode")
        zoom_layout.addWidget(self.zoom_check)
        
        self.home_btn = QPushButton("Home")
        self.home_btn.setToolTip("Reset zoom to full range")
        zoom_layout.addWidget(self.home_btn)
        
        zoom_layout.addStretch()
        display_layout.addLayout(zoom_layout)
        
    def setup_control_panel(self, parent):
        """Create parameter and MCMC controls panel"""
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        parent.addWidget(control_widget)
        
        # Model preview group
        preview_group = QGroupBox("Model Preview")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        control_layout.addWidget(preview_group)
        
        self.preview_btn = QPushButton("Generate Preview")
        self.preview_btn.setToolTip("Generate model preview with current parameters")
        preview_layout.addWidget(self.preview_btn)
        
        # FWHM display table
        fwhm_group = QGroupBox("Instrumental FWHM")
        fwhm_layout = QVBoxLayout()
        fwhm_group.setLayout(fwhm_layout)
        control_layout.addWidget(fwhm_group)
        
        self.fwhm_table = QTableWidget()
        self.fwhm_table.setColumnCount(2)
        self.fwhm_table.setHorizontalHeaderLabels(["Instrument", "FWHM"])
        self.fwhm_table.horizontalHeader().setStretchLastSection(True)
        self.fwhm_table.setMaximumHeight(120)
        self.fwhm_table.setToolTip("FWHM values used for model convolution")
        fwhm_layout.addWidget(self.fwhm_table)
        
        # MCMC parameters group
        mcmc_group = QGroupBox("MCMC Parameters")
        mcmc_layout = QFormLayout()
        mcmc_group.setLayout(mcmc_layout)
        control_layout.addWidget(mcmc_group)
        
        # Number of walkers
        self.nwalkers_spin = QSpinBox()
        self.nwalkers_spin.setRange(10, 1000)
        self.nwalkers_spin.setValue(50)
        self.nwalkers_spin.setToolTip("Number of MCMC walkers")
        mcmc_layout.addRow("Walkers:", self.nwalkers_spin)
        
        # Number of steps
        self.nsteps_spin = QSpinBox()
        self.nsteps_spin.setRange(100, 50000)
        self.nsteps_spin.setValue(1000)
        self.nsteps_spin.setToolTip("Number of MCMC steps")
        mcmc_layout.addRow("Steps:", self.nsteps_spin)
        
        # Burn-in steps
        self.burnin_spin = QSpinBox()
        self.burnin_spin.setRange(0, 10000)
        self.burnin_spin.setValue(100)
        self.burnin_spin.setToolTip("Number of burn-in steps to discard")
        mcmc_layout.addRow("Burn-in:", self.burnin_spin)
        
        # Fitting controls
        fitting_group = QGroupBox("Fitting Control")
        fitting_layout = QVBoxLayout()
        fitting_group.setLayout(fitting_layout)
        control_layout.addWidget(fitting_group)
        
        # Fit button
        self.fit_btn = QPushButton("Start Fitting")
        self.fit_btn.setToolTip("Start MCMC fitting process")
        self.fit_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        fitting_layout.addWidget(self.fit_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fitting_layout.addWidget(self.progress_bar)
        
        # Stop button
        self.stop_btn = QPushButton("Stop Fitting")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop current fitting process")
        fitting_layout.addWidget(self.stop_btn)
        
        # Status text
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        self.status_text.setToolTip("Fitting status and messages")
        status_layout.addWidget(self.status_text)
        
        control_layout.addStretch()
        
    def setup_connections(self):
        """Connect signals and slots"""
        # Top controls
        self.instrument_combo.currentTextChanged.connect(self.on_instrument_changed)
        self.wave_min_spin.valueChanged.connect(self.on_wave_range_changed)
        self.wave_max_spin.valueChanged.connect(self.on_wave_range_changed)
        self.slice_btn.clicked.connect(self.apply_wavelength_slice)
        self.reset_slice_btn.clicked.connect(self.reset_wavelength_slice)
        self.show_model_check.toggled.connect(self.plot_spectrum)
        self.show_components_check.toggled.connect(self.plot_spectrum)
        
        # Spectrum controls
        self.zoom_check.toggled.connect(self.toggle_zoom_mode)
        self.home_btn.clicked.connect(self.reset_zoom)
        
        # Model controls
        self.preview_btn.clicked.connect(self.preview_model)
        
        # MCMC controls
        self.fit_btn.clicked.connect(self.start_fitting)
        self.stop_btn.clicked.connect(self.stop_fitting)
        
    def update_data(self, spectra_data):
        """Update with new spectrum data"""
        self.spectra_data = spectra_data.copy()
        self.spectra_original = spectra_data.copy()  # Keep originals
        
        # Update instrument combo
        self.instrument_combo.clear()
        for filename in spectra_data.keys():
            basename = spectra_data[filename].get('basename', filename)
            self.instrument_combo.addItem(basename, filename)
            
        # Update FWHM table
        self.update_fwhm_table()
        
        # Auto-select first instrument
        if self.instrument_combo.count() > 0:
            self.instrument_combo.setCurrentIndex(0)
            
        self.status_text.append(f"Loaded {len(spectra_data)} spectra")
        
    def update_fwhm_table(self):
        """Update FWHM display table"""
        self.fwhm_table.setRowCount(len(self.spectra_data))
        
        for i, filename in enumerate(self.spectra_data.keys()):
            basename = self.spectra_data[filename].get('basename', filename)
            
            # Instrument name
            item = QTableWidgetItem(basename)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.fwhm_table.setItem(i, 0, item)
            
            # FWHM value - get from model setup tab if available
            if hasattr(self.main_window, 'model_tab'):
                fwhm = self.main_window.model_tab.get_fwhm_for_instrument(basename)
            else:
                fwhm = 2.5  # Default
                
            fwhm_item = QTableWidgetItem(f"{fwhm:.2f}")
            fwhm_item.setFlags(fwhm_item.flags() & ~Qt.ItemIsEditable)
            self.fwhm_table.setItem(i, 1, fwhm_item)
            
        self.fwhm_table.resizeColumnsToContents()
        
    def update_model(self, fit_config):
        """Update with new model configuration"""
        self.fit_config = fit_config
        self.plot_spectrum()
        self.status_text.append("Model configuration updated")
        
    def on_instrument_changed(self, basename):
        """Handle instrument selection change"""
        if not basename:
            return
            
        # Find filename for this basename
        self.current_instrument = None
        for filename, data in self.spectra_data.items():
            if data.get('basename', filename) == basename:
                self.current_instrument = filename
                break
                
        if self.current_instrument:
            self.plot_spectrum()
            self.update_wavelength_range()
            self.status_text.append(f"Selected instrument: {basename}")
            
    def update_wavelength_range(self):
        """Update wavelength range controls"""
        if not self.current_instrument or self.current_instrument not in self.spectra_data:
            return
            
        spec_data = self.spectra_data[self.current_instrument]
        wavelength = spec_data['wavelength']
        
        # Update spin box ranges
        wave_min = np.min(wavelength)
        wave_max = np.max(wavelength)
        
        self.wave_min_spin.setRange(wave_min, wave_max)
        self.wave_max_spin.setRange(wave_min, wave_max)
        
        # Set current values to full range
        self.wave_min_spin.setValue(wave_min)
        self.wave_max_spin.setValue(wave_max)
        
    def on_wave_range_changed(self):
        """Handle wavelength range change"""
        if self.wave_min_spin.value() >= self.wave_max_spin.value():
            return  # Invalid range
            
        # Update plot limits if not in zoom mode
        if not self.zoom_check.isChecked():
            self.ax_spec.set_xlim(self.wave_min_spin.value(), self.wave_max_spin.value())
            self.canvas.draw()
            
    def apply_wavelength_slice(self):
        """Apply wavelength slice to current spectrum"""
        if not self.current_instrument:
            QMessageBox.warning(self, "Warning", "No spectrum selected")
            return
            
        wave_min = self.wave_min_spin.value()
        wave_max = self.wave_max_spin.value()
        
        if wave_min >= wave_max:
            QMessageBox.warning(self, "Warning", "Invalid wavelength range")
            return
            
        try:
            # Apply slice
            self.spectra_data[self.current_instrument] = slice_spectrum(
                self.spectra_original[self.current_instrument], 
                wave_min, wave_max
            )
            
            self.plot_spectrum()
            self.status_text.append(f"Applied slice: {wave_min:.2f} - {wave_max:.2f} Å")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to slice spectrum:\n{str(e)}")
            
    def reset_wavelength_slice(self):
        """Reset spectrum to original data"""
        if not self.current_instrument:
            return
            
        # Restore original data
        self.spectra_data[self.current_instrument] = self.spectra_original[self.current_instrument].copy()
        
        self.update_wavelength_range()
        self.plot_spectrum()
        self.status_text.append("Reset to full spectrum")
        
    def plot_spectrum(self):
        """Plot current spectrum with optional model overlay"""
        if not self.current_instrument or self.current_instrument not in self.spectra_data:
            return
            
        # Clear previous plot
        self.ax_spec.clear()
        self.ax_spec.set_xlabel('Wavelength (Å)')
        self.ax_spec.set_ylabel('Normalized Flux')
        self.ax_spec.grid(True, alpha=0.3)
        
        # Get spectrum data
        spec_data = self.spectra_data[self.current_instrument]
        wavelength = spec_data['wavelength']
        flux = spec_data['flux']
        error = spec_data.get('error', np.ones_like(flux) * 0.05)
        
        # Plot spectrum
        self.spec_line, = self.ax_spec.plot(wavelength, flux, 'k-', linewidth=0.8, label='Data')
        
        # Plot error region
        self.error_fill = self.ax_spec.fill_between(
            wavelength, flux - error, flux + error, 
            alpha=0.3, color='gray', label='Error'
        )
        
        # Plot velocity markers
        self.plot_velocity_markers()
        
        # Plot model if requested and available
        if self.show_model_check.isChecked() and self.fit_config:
            self.plot_model_overlay()
            
        # Set reasonable y-limits
        flux_med = np.median(flux)
        flux_std = np.std(flux)
        self.ax_spec.set_ylim(flux_med - 3*flux_std, flux_med + 3*flux_std)
        
        # Add legend
        self.ax_spec.legend(loc='upper right', fontsize=8)
        
        # Update canvas
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_velocity_markers(self):
        """Plot velocity component markers"""
        self.velocity_markers_plot.clear()
        
        for i, wave in enumerate(self.velocity_markers):
            line = self.ax_spec.axvline(wave, color='red', linestyle='--', alpha=0.7, linewidth=1)
            self.velocity_markers_plot.append(line)
            
            # Add component label
            ylim = self.ax_spec.get_ylim()
            self.ax_spec.text(wave, ylim[1]*0.95, f'C{i+1}', 
                            rotation=90, ha='right', va='top', 
                            color='red', fontsize=8)
                            
    def plot_model_overlay(self):
        """Plot model overlay on spectrum"""
        if not self.fit_config or not self.current_instrument:
            return
            
        try:
            # Get FWHM for current instrument
            basename = self.spectra_data[self.current_instrument].get('basename', self.current_instrument)
            if hasattr(self.main_window, 'model_tab'):
                fwhm = self.main_window.model_tab.get_fwhm_for_instrument(basename)
            else:
                fwhm = 2.5
                
            # Create VoigtModel
            model = VoigtModel(self.fit_config, FWHM=str(fwhm))
            
            # Generate model spectrum
            spec_data = self.spectra_data[self.current_instrument]
            wavelength = spec_data['wavelength']
            
            # Get current parameters from model setup tab
            if hasattr(self.main_window, 'model_tab') and not self.main_window.model_tab.parameter_df.empty:
                # Use parameters from GUI
                model_flux = model.evaluate_at_wavelengths(wavelength)
                
                # Plot total model
                self.model_line, = self.ax_spec.plot(wavelength, model_flux, 'r-', 
                                                   linewidth=1.5, label='Model', alpha=0.8)
                
                # Plot components if requested
                if self.show_components_check.isChecked():
                    self.plot_model_components(model, wavelength)
                    
            else:
                self.status_text.append("No parameters available for model preview")
                
        except Exception as e:
            self.status_text.append(f"Model plot error: {str(e)}")
            
    def plot_model_components(self, model, wavelength):
        """Plot individual model components"""
        try:
            # Get individual component models
            component_fluxes = model.evaluate_components_at_wavelengths(wavelength)
            
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            self.component_lines.clear()
            
            for i, comp_flux in enumerate(component_fluxes):
                color = colors[i % len(colors)]
                line, = self.ax_spec.plot(wavelength, comp_flux, '--', 
                                        color=color, linewidth=1, alpha=0.7,
                                        label=f'Comp {i+1}')
                self.component_lines.append(line)
                
        except Exception as e:
            self.status_text.append(f"Component plot error: {str(e)}")
            
    def on_spectrum_click(self, event):
        """Handle mouse clicks on spectrum"""
        if not event.inaxes == self.ax_spec or not event.xdata:
            return
            
        wavelength = event.xdata
        
        if event.button == 1:  # Left click - add component
            if wavelength not in self.velocity_markers:
                self.velocity_markers.append(wavelength)
                self.plot_spectrum()
                
                # Emit signal for model setup tab
                self.spectrum_clicked.emit(wavelength)
                
                self.main_window.update_status(
                    f"Added component at {wavelength:.2f} Å ({len(self.velocity_markers)} total)")
                    
        elif event.button == 3:  # Right click - remove nearest component
            if self.velocity_markers and event.xdata is not None:
                distances = [abs(m - event.xdata) for m in self.velocity_markers]
                nearest_idx = distances.index(min(distances))
                removed_wavelength = self.velocity_markers.pop(nearest_idx)
                self.plot_spectrum()
                
                self.main_window.update_status(
                    f"Removed component at {removed_wavelength:.2f} Å ({len(self.velocity_markers)} total)")
                    
    def toggle_zoom_mode(self, enabled):
        """Toggle zoom/pan mode"""
        if enabled:
            self.canvas.toolbar_visible = True
            self.status_text.append("Zoom mode enabled - use mouse to zoom/pan")
        else:
            self.canvas.toolbar_visible = False
            self.status_text.append("Zoom mode disabled")
            
    def reset_zoom(self):
        """Reset zoom to full range"""
        if self.current_instrument and self.current_instrument in self.spectra_data:
            spec_data = self.spectra_data[self.current_instrument]
            wavelength = spec_data['wavelength']
            self.ax_spec.set_xlim(np.min(wavelength), np.max(wavelength))
            self.canvas.draw()
            
    def preview_model(self):
        """Generate and display model preview"""
        if not self.fit_config:
            QMessageBox.warning(self, "Warning", "No model configuration available")
            return
            
        if not self.current_instrument:
            QMessageBox.warning(self, "Warning", "No spectrum selected")
            return
            
        self.status_text.clear()
        self.status_text.append("Generating model preview...")
        
        try:
            # Enable model display
            self.show_model_check.setChecked(True)
            self.plot_spectrum()
            
            self.status_text.append("Model preview generated successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate preview:\n{str(e)}")
            self.status_text.append(f"Preview error: {str(e)}")
            
    def start_fitting(self):
        """Start MCMC fitting process"""
        if not self.fit_config:
            QMessageBox.warning(self, "Warning", "No model configuration available")
            return
            
        if not self.current_instrument:
            QMessageBox.warning(self, "Warning", "No spectrum selected")
            return
            
        # Prepare MCMC parameters
        mcmc_params = {
            'nwalkers': self.nwalkers_spin.value(),
            'nsteps': self.nsteps_spin.value(),
            'burnin': self.burnin_spin.value()
        }
        
        # Create model with current FWHM
        basename = self.spectra_data[self.current_instrument].get('basename', self.current_instrument)
        if hasattr(self.main_window, 'model_tab'):
            fwhm = self.main_window.model_tab.get_fwhm_for_instrument(basename)
        else:
            fwhm = 2.5
            
        try:
            model = VoigtModel(self.fit_config, FWHM=str(fwhm))
            
            # Start fitting thread
            self.mcmc_thread = MCMCThread(model, mcmc_params)
            self.mcmc_thread.progress_update.connect(self.update_progress)
            self.mcmc_thread.status_update.connect(self.update_status)
            self.mcmc_thread.fitting_completed.connect(self.on_fitting_completed)
            self.mcmc_thread.fitting_error.connect(self.on_fitting_error)
            
            self.mcmc_thread.start()
            
            # Update UI
            self.fit_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            
            self.fitting_started.emit(mcmc_params)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start fitting:\n{str(e)}")
            
    def stop_fitting(self):
        """Stop current fitting process"""
        if self.mcmc_thread and self.mcmc_thread.isRunning():
            self.mcmc_thread.terminate()
            self.mcmc_thread.wait()
            
        self.reset_fitting_ui()
        self.status_text.append("Fitting stopped by user")
        
    def reset_fitting_ui(self):
        """Reset fitting UI elements"""
        self.fit_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.mcmc_thread = None
        
    @pyqtSlot(int)
    def update_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
        
    @pyqtSlot(str)
    def update_status(self, message):
        """Update status text"""
        self.status_text.append(message)
        
    @pyqtSlot(object)
    def on_fitting_completed(self, results):
        """Handle fitting completion"""
        self.reset_fitting_ui()
        self.status_text.append("Fitting completed successfully!")
        
        # Update model display with fitted parameters
        self.plot_spectrum()
        
        # Emit signal to main window
        self.fitting_completed.emit(results)
        
    @pyqtSlot(str)
    def on_fitting_error(self, error_msg):
        """Handle fitting error"""
        self.reset_fitting_ui()
        self.status_text.append(f"Fitting error: {error_msg}")
        QMessageBox.critical(self, "Fitting Error", f"MCMC fitting failed:\n{error_msg}")
        
    def refresh(self):
        """Refresh the tab"""
        self.plot_spectrum()
        self.update_fwhm_table()