#!/usr/bin/env python
"""
rbvfit 2.0 Fitting Tab - Enhanced with Parameter Bounds and Plot Range Controls

Enhanced fitting interface with parameter bounds editing, quick fit capability, and plot range controls.
Updated for unified vfit_mcmc interface.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QPushButton, QLabel, QComboBox, QSpinBox,
                            QDoubleSpinBox, QCheckBox, QProgressBar, QTextEdit,
                            QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                            QFormLayout, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import rbvfit.vfit_mcmc as mc
from rbvfit.core.unified_results import UnifiedResults

# Import shared dialog
from rbvfit.gui.shared_plot_range_dialog import PlotRangeDialog

class MCMCThread(QThread):
    """Thread for running MCMC to avoid blocking GUI"""
    
    status_update = pyqtSignal(str)
    fitting_completed = pyqtSignal(object)  # fitter object
    fitting_error = pyqtSignal(str)
    
    def __init__(self, instrument_data, theta, lb, ub, mcmc_params):
        super().__init__()
        self.instrument_data = instrument_data
        self.theta = theta
        self.lb = lb
        self.ub = ub
        self.mcmc_params = mcmc_params
        
    def run(self):
        """Run MCMC fitting"""
        try:
            self.status_update.emit("Starting MCMC fitting...")
            
            # Create fitter with unified interface
            fitter = mc.vfit(
                self.instrument_data,    # Unified instrument data dictionary
                self.theta, self.lb, self.ub,
                no_of_Chain=self.mcmc_params['n_walkers'],
                no_of_steps=self.mcmc_params['n_steps'],
                sampler=self.mcmc_params['sampler'],
                perturbation=self.mcmc_params['perturbation']
            )
            
            self.status_update.emit("Running MCMC chains...")
            
            # Run MCMC
            fitter.runmcmc(
                optimize=self.mcmc_params['optimize'],
                verbose=False
            )
            
            self.status_update.emit("MCMC completed successfully")
            self.fitting_completed.emit(fitter)
            
        except Exception as e:
            self.fitting_error.emit(str(e))


class ParameterBoundsTable(QTableWidget):
    """Table widget for editing parameter bounds and initial values"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_table()
        
    def setup_table(self):
        """Set up the parameter table"""
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(['Parameter', 'Initial', 'Lower', 'Upper', 'Units'])
        
        # Set column properties
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        
        # Enable editing for Initial, Lower, Upper columns
        self.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        
    def populate_parameters(self, param_names, theta, lb, ub):
        """Populate table with parameters and bounds"""
        self.setRowCount(len(param_names))
        
        for i, (name, initial, lower, upper) in enumerate(zip(param_names, theta, lb, ub)):
            # Parameter name (read-only)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(i, 0, name_item)
            
            # Initial value (editable)
            initial_item = QTableWidgetItem(f"{initial:.3f}")
            self.setItem(i, 1, initial_item)
            
            # Lower bound (editable)
            lb_item = QTableWidgetItem(f"{lower:.3f}")
            self.setItem(i, 2, lb_item)
            
            # Upper bound (editable)
            ub_item = QTableWidgetItem(f"{upper:.3f}")
            self.setItem(i, 3, ub_item)
            
            # Units (read-only)
            if name.startswith('N_'):
                units = "log cm⁻²"
            elif name.startswith('b_'):
                units = "km/s"
            elif name.startswith('v_'):
                units = "km/s"
            else:
                units = ""
                
            units_item = QTableWidgetItem(units)
            units_item.setFlags(units_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(i, 4, units_item)
            
    def get_parameters(self):
        """Get current parameter values and bounds from table"""
        n_params = self.rowCount()
        theta = np.zeros(n_params)
        lb = np.zeros(n_params)
        ub = np.zeros(n_params)
        
        for i in range(n_params):
            try:
                theta[i] = float(self.item(i, 1).text())
                lb[i] = float(self.item(i, 2).text())
                ub[i] = float(self.item(i, 3).text())
            except (ValueError, AttributeError):
                # Handle invalid entries
                theta[i] = 0.0
                lb[i] = -np.inf
                ub[i] = np.inf
                
        return theta, lb, ub
        
    def update_initial_values(self, new_theta):
        """Update only the initial values column (for quick fit results)"""
        for i in range(min(len(new_theta), self.rowCount())):
            self.setItem(i, 1, QTableWidgetItem(f"{new_theta[i]:.3f}"))


class FittingTab(QWidget):
    """Enhanced fitting tab with parameter bounds control"""
    
    # Signals
    fitting_started = pyqtSignal(dict)
    fitting_completed = pyqtSignal(object)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # Data storage - NEW: unified interface
        self.instrument_data = {}
        self.theta = None
        self.lb = None
        self.ub = None
        self.param_names = []
        self.current_instrument = None
        
        # Legacy support
        self.spectra_data = {}
        self.compiled_model = None
        self.current_theta = None
        
        # Plot range storage
        self.plot_ranges = {
            'xlim': None,  # None = auto, tuple = manual
            'ylim': None,
            'original_xlim': None,  # Store original auto ranges
            'original_ylim': None
        }
        
        # MCMC parameters
        self.mcmc_params = {
            'n_walkers': 20,
            'n_steps': 1000,
            'sampler': 'emcee',
            'perturbation': 1e-4,
            'optimize': True,
            'use_pool': True
        }
        
        # Fitting thread
        self.mcmc_thread = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create fitting interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: Controls
        self.setup_controls(splitter)
        
        # Right side: Spectrum plot
        self.setup_spectrum_plot(splitter)
        
        splitter.setSizes([400, 600])
        
    def setup_controls(self, parent):
        """Set up control panel"""
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        parent.addWidget(control_widget)
        
        # Instrument selection
        self.setup_instrument_selection(control_layout)
        
        # Parameter bounds table
        self.setup_parameter_bounds(control_layout)
        
        # MCMC controls
        self.setup_mcmc_controls(control_layout)
        
        # Fitting controls and status
        self.setup_fitting_controls(control_layout)
        
    def setup_instrument_selection(self, parent_layout):
        """Create instrument selection"""
        inst_group = QGroupBox("Data Selection")
        inst_layout = QFormLayout()
        inst_group.setLayout(inst_layout)
        parent_layout.addWidget(inst_group)
        
        self.instrument_combo = QComboBox()
        self.instrument_combo.setToolTip("Select instrument/dataset for viewing")
        inst_layout.addRow("Instrument:", self.instrument_combo)
        
        # Show model checkbox
        self.show_model_check = QCheckBox("Show Model Overlay")
        self.show_model_check.setToolTip("Overlay current model on spectrum")
        inst_layout.addRow(self.show_model_check)
        
    def setup_parameter_bounds(self, parent_layout):
        """Create parameter bounds table"""
        bounds_group = QGroupBox("Parameter Values & Bounds")
        bounds_layout = QVBoxLayout()
        bounds_group.setLayout(bounds_layout)
        parent_layout.addWidget(bounds_group)
        
        # Quick fit button (disabled for multi-instrument for now)
        quick_fit_layout = QHBoxLayout()
        bounds_layout.addLayout(quick_fit_layout)
        
        self.quick_fit_btn = QPushButton("Quick Fit")
        self.quick_fit_btn.setEnabled(False)
        self.quick_fit_btn.setToolTip("Quick optimization (single instrument only)")
        quick_fit_layout.addWidget(self.quick_fit_btn)
        
        quick_fit_layout.addStretch()
        
        # Parameter table
        self.param_bounds_table = ParameterBoundsTable()
        self.param_bounds_table.setMaximumHeight(200)
        bounds_layout.addWidget(self.param_bounds_table)
        
    def setup_mcmc_controls(self, parent_layout):
        """Create MCMC parameter controls"""
        mcmc_group = QGroupBox("MCMC Settings")
        mcmc_layout = QFormLayout()
        mcmc_group.setLayout(mcmc_layout)
        parent_layout.addWidget(mcmc_group)
        
        # Sampler selection
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(['emcee', 'zeus'])
        self.sampler_combo.setCurrentText(self.mcmc_params['sampler'])
        self.sampler_combo.setToolTip("MCMC sampler algorithm")
        mcmc_layout.addRow("Sampler:", self.sampler_combo)
        
        # Number of walkers
        self.walkers_spin = QSpinBox()
        self.walkers_spin.setRange(10, 200)
        self.walkers_spin.setValue(self.mcmc_params['n_walkers'])
        self.walkers_spin.setToolTip("Number of MCMC walkers")
        mcmc_layout.addRow("Walkers:", self.walkers_spin)
        
        # Number of steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 10000)
        self.steps_spin.setValue(self.mcmc_params['n_steps'])
        self.steps_spin.setToolTip("Number of MCMC steps per walker")
        mcmc_layout.addRow("Steps:", self.steps_spin)
        
        # Perturbation
        self.perturbation_spin = QDoubleSpinBox()
        self.perturbation_spin.setRange(1e-6, 1e-1)
        self.perturbation_spin.setDecimals(6)
        self.perturbation_spin.setValue(self.mcmc_params['perturbation'])
        self.perturbation_spin.setToolTip("Walker initialization perturbation")
        mcmc_layout.addRow("Perturbation:", self.perturbation_spin)
        
        # Options
        self.optimize_check = QCheckBox("Pre-optimize")
        self.optimize_check.setChecked(self.mcmc_params['optimize'])
        self.optimize_check.setToolTip("Optimize walker positions before MCMC")
        mcmc_layout.addRow(self.optimize_check)
        
        self.pool_check = QCheckBox("Use multiprocessing")
        self.pool_check.setChecked(self.mcmc_params['use_pool'])
        self.pool_check.setToolTip("Use multiple CPU cores")
        mcmc_layout.addRow(self.pool_check)
        
    def setup_fitting_controls(self, parent_layout):
        """Create fitting buttons and status"""
        fit_layout = QVBoxLayout()
        parent_layout.addLayout(fit_layout)
        
        # Fit button
        self.fit_btn = QPushButton("Start MCMC Fitting")
        self.fit_btn.setEnabled(False)
        self.fit_btn.setToolTip("Start MCMC fitting with current parameters")
        fit_layout.addWidget(self.fit_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fit_layout.addWidget(self.progress_bar)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        fit_layout.addWidget(self.status_text)
        
        fit_layout.addStretch()
        
    def setup_spectrum_plot(self, parent):
        """Create spectrum plotting area with range controls"""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        parent.addWidget(plot_widget)
        
        # Plot controls toolbar
        plot_controls = QHBoxLayout()
        plot_layout.addLayout(plot_controls)
        
        # Range controls
        self.set_range_btn = QPushButton("Set Range")
        self.set_range_btn.setToolTip("Set custom plot range")
        plot_controls.addWidget(self.set_range_btn)
        
        self.reset_range_btn = QPushButton("Reset Range")
        self.reset_range_btn.setToolTip("Reset to auto range")
        plot_controls.addWidget(self.reset_range_btn)
        
        plot_controls.addStretch()
        
        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax_spec = self.figure.add_subplot(111)
        plot_layout.addWidget(self.canvas)
        
        # Connect range control signals
        self.set_range_btn.clicked.connect(self.set_custom_range)
        self.reset_range_btn.clicked.connect(self.reset_range)
        
    def setup_connections(self):
        """Connect signals and slots"""
        self.instrument_combo.currentTextChanged.connect(self.on_instrument_changed)
        self.show_model_check.toggled.connect(self.plot_spectrum)
        self.quick_fit_btn.clicked.connect(self.run_quick_fit)
        self.fit_btn.clicked.connect(self.start_fitting)

    def set_model_data(self, instrument_data, theta, bounds):
        """
        Set model data from the updated Model Setup tab.
        
        Parameters
        ----------
        instrument_data : dict
            Dictionary with instrument data for vfit_mcmc unified interface
            Format: {'instrument_name': {'model': func, 'wave': array, 'flux': array, 'error': array}}
        theta : np.ndarray
            Global parameter array [N1,N2,...,b1,b2,...,v1,v2,...]
        bounds : dict
            Dictionary with 'lb' and 'ub' arrays for lower and upper bounds
        """
        print(f"Setting model data: {len(instrument_data)} instruments, {len(theta)} parameters")
        
        # Store the unified interface data
        self.instrument_data = instrument_data
        self.theta = theta.copy()
        self.lb = bounds['lb'].copy()
        self.ub = bounds['ub'].copy()
        
        # Generate parameter names from theta structure
        n_params = len(theta)
        n_comp = n_params // 3
        param_names = []
        
        # Add all N parameters first, then b, then v (matches theta structure)
        for i in range(n_comp):
            param_names.append(f'N_c{i+1}')
        for i in range(n_comp):
            param_names.append(f'b_c{i+1}')
        for i in range(n_comp):
            param_names.append(f'v_c{i+1}')
        self.param_names = param_names
        
        # Update instrument selector
        self.instrument_combo.clear()
        for name in instrument_data.keys():
            self.instrument_combo.addItem(name)
        
        # Set first instrument as current
        if instrument_data:
            self.current_instrument = list(instrument_data.keys())[0]
            
        # Convert instrument_data to spectra_data format for plotting compatibility
        self.spectra_data = {}
        for name, data in instrument_data.items():
            self.spectra_data[name] = {
                'wave': data['wave'],
                'flux': data['flux'],
                'error': data['error']
            }
        
        # Populate parameter table
        self.param_bounds_table.populate_parameters(param_names, theta, self.lb, self.ub)
        
        # Enable controls
        self.fit_btn.setEnabled(True)
        # Quick fit disabled for multi-instrument
        #self.quick_fit_btn.setEnabled(len(instrument_data) == 1)
        self.quick_fit_btn.setEnabled(True)
        
        # Update status
        self.status_text.clear()
        self.status_text.append("Model data loaded:")
        self.status_text.append(f"  Instruments: {len(instrument_data)}")
        self.status_text.append(f"  Parameters: {n_params} ({n_comp} components)")
        for name, data in instrument_data.items():
            wave = data['wave']
            self.status_text.append(f"  {name}: {len(wave)} points, {wave.min():.1f}-{wave.max():.1f} Å")
        
        # Reset plot ranges and update plot
        self.reset_plot_ranges()
        self.plot_spectrum()
        
        print(f"✓ Fitting tab ready: {len(instrument_data)} instruments, {n_params} parameters")

    # LEGACY SUPPORT: Keep the old interface for backward compatibility
    def set_compiled_model(self, compiled_model, mcmc_params):
        """Set the compiled model for fitting - backward compatible"""
        print("Warning: Using legacy set_compiled_model interface")
        self.compiled_model = compiled_model
        
        # Extract theta from mcmc_params (existing interface)
        if 'theta' in mcmc_params:
            theta = mcmc_params['theta']
        else:
            # Fallback: try to get from main window
            theta = getattr(self.main_window, 'theta', None)
            if theta is None:
                QMessageBox.warning(self, "No Parameters", "No parameter values available")
                return
                
        self.current_theta = theta.copy()
        
        # This is a legacy single-instrument setup
        # Convert to new interface if possible
        if hasattr(self.main_window, 'spectra_data'):
            spectra_data = self.main_window.spectra_data
            if spectra_data:
                # Create fake instrument_data from legacy data
                instrument_data = {}
                for name, data in spectra_data.items():
                    instrument_data[name] = {
                        'model': compiled_model.model_flux,  # Same model for all
                        'wave': data['wave'],
                        'flux': data['flux'],
                        'error': data['error']
                    }
                
                # Create bounds
                n_params = len(theta)
                n_comp = n_params // 3
                nguess = theta[:n_comp]
                bguess = theta[n_comp:2*n_comp]
                vguess = theta[2*n_comp:]
                bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
                bounds_dict = {'lb': lb, 'ub': ub}
                
                # Use new interface
                self.set_model_data(instrument_data, theta, bounds_dict)
                return
        
        # Fallback to old behavior if conversion fails
        self.status_text.append("Legacy model interface - limited functionality")
        
    def set_spectra_data(self, spectra_data):
        """Update spectra data - legacy support"""
        print("Warning: Using legacy set_spectra_data interface")
        self.spectra_data = spectra_data
        
        # Reset plot ranges for new data
        self.reset_plot_ranges()
        
        # Update instrument combo
        self.instrument_combo.clear()
        self.instrument_combo.addItems(list(spectra_data.keys()))
        
        if spectra_data:
            self.current_instrument = list(spectra_data.keys())[0]
            self.plot_spectrum()
        
    def reset_plot_ranges(self):
        """Reset plot ranges when new data is loaded"""
        self.plot_ranges = {
            'xlim': None,
            'ylim': [-0.02,1.5], 
            'original_xlim': None,
            'original_ylim': None
        }
            
    def on_instrument_changed(self, instrument_name):
        """Handle instrument selection change"""
        if instrument_name:
            self.current_instrument = instrument_name
            self.reset_plot_ranges()  # Reset ranges for different instrument
            self.plot_spectrum()
            
    def plot_spectrum(self):
        """Plot the current spectrum with custom range support"""
        if not self.current_instrument or not self.spectra_data:
            return
            
        self.ax_spec.clear()
        
        spec_data = self.spectra_data[self.current_instrument]
        wave = spec_data['wave']
        flux = spec_data['flux']
        error = spec_data.get('error', np.ones_like(flux) * 0.05)
        
        # Plot spectrum
        self.ax_spec.step(wave, flux, 'k-', linewidth=0.8, label='Data')
        self.ax_spec.fill_between(wave, flux - error, flux + error, 
                                alpha=0.3, color='gray', label='Error')
        
        # Plot model if available and requested
        if self.show_model_check.isChecked() and self.theta is not None:
            try:
                # Get current parameters from table
                current_theta, _, _ = self.param_bounds_table.get_parameters()
                
                # Use the model function from instrument_data
                if self.current_instrument in self.instrument_data:
                    model_func = self.instrument_data[self.current_instrument]['model']
                    model_flux = model_func(current_theta, wave)
                    self.ax_spec.plot(wave, model_flux, 'r-', alpha=0.8, linewidth=2, label='Model')
                elif self.compiled_model:  # Legacy fallback
                    model_flux = self.compiled_model.model_flux(current_theta, wave)
                    self.ax_spec.plot(wave, model_flux, 'r-', alpha=0.8, linewidth=2, label='Model')
                    
            except Exception as e:
                print(f"Could not plot model: {e}")
                
        # Store original ranges if not set
        if self.plot_ranges['original_xlim'] is None:
            data_xlim = (wave.min(), wave.max())
            flux_med = np.median(flux)
            flux_std = np.std(flux)
            data_ylim = (flux_med - 3*flux_std, flux_med + 3*flux_std)
            
            self.plot_ranges['original_xlim'] = data_xlim
            self.plot_ranges['original_ylim'] = data_ylim
            
        # Apply custom ranges or auto ranges
        if self.plot_ranges['xlim'] is not None:
            self.ax_spec.set_xlim(self.plot_ranges['xlim'])
        else:
            self.ax_spec.set_xlim(self.plot_ranges['original_xlim'])
            
        if self.plot_ranges['ylim'] is not None:
            self.ax_spec.set_ylim(self.plot_ranges['ylim'])
        else:
            self.ax_spec.set_ylim(self.plot_ranges['original_ylim'])
                
        # Formatting
        self.ax_spec.legend(loc='upper right', fontsize=8)
        self.ax_spec.set_xlabel('Wavelength (Å)')
        self.ax_spec.set_ylabel('Normalized Flux')
        self.ax_spec.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def set_custom_range(self):
        """Open range setting dialog - non-modal to prevent GUI freeze"""
        current_xlim = self.plot_ranges['xlim']
        current_ylim = self.plot_ranges['ylim']
        original_xlim = self.plot_ranges['original_xlim']
        original_ylim = self.plot_ranges['original_ylim']
        
        dialog = PlotRangeDialog(current_xlim, current_ylim, original_xlim, original_ylim, self)
        
        # Make it non-modal to prevent GUI freezing
        dialog.setModal(False)
        
        # Connect accepted signal to handle result
        def on_dialog_accepted():
            if hasattr(dialog, 'get_ranges'):
                xlim, ylim = dialog.get_ranges()
                self.plot_ranges['xlim'] = xlim
                self.plot_ranges['ylim'] = ylim
                self.plot_spectrum()
            dialog.deleteLater()
        
        dialog.accepted.connect(on_dialog_accepted)
        dialog.show()  # Use show() instead of exec_() to prevent blocking
            
    def reset_range(self):
        """Reset to original auto ranges"""
        self.plot_ranges['xlim'] = None
        self.plot_ranges['ylim'] = None
        self.plot_spectrum()
        
    def run_quick_fit(self):
        """Run quick scipy optimization - supports single and multi-instrument"""
        if not self.current_instrument or not self.instrument_data:
            QMessageBox.warning(self, "No Model", "No model data available")
            return
            
        try:
            # Get current parameters and bounds
            theta, lb, ub = self.param_bounds_table.get_parameters()
            
            self.status_text.append("Running quick fit...")
            
            # Use scipy optimization directly
            from scipy.optimize import minimize
            
            def chi2_multi_instrument(params):
                """Chi-squared for all instruments combined"""
                try:
                    total_chi2 = 0.0
                    
                    # Sum chi-squared across all instruments
                    for name, inst_data in self.instrument_data.items():
                        wave = inst_data['wave']
                        flux = inst_data['flux']
                        error = inst_data['error']
                        model_func = inst_data['model']
                        
                        model = model_func(params, wave)
                        chi2_contrib = np.sum(((flux - model) / error) ** 2)
                        total_chi2 += chi2_contrib
                    
                    return total_chi2
                except:
                    return 1e10  # Return large value if model evaluation fails
            
            # Show info about optimization
            n_instruments = len(self.instrument_data)
            if n_instruments == 1:
                self.status_text.append(f"Optimizing single instrument: {self.current_instrument}")
            else:
                self.status_text.append(f"Optimizing {n_instruments} instruments jointly")
            
            # Run scipy optimization
            result = minimize(chi2_multi_instrument, theta, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            best_theta = result.x
            
            # Update table with optimized parameters
            self.param_bounds_table.update_initial_values(best_theta)
            
            # Update plot
            self.plot_spectrum()
            
            # Calculate and display chi-squared breakdown
            try:
                total_chi2 = 0.0
                total_points = 0
                
                self.status_text.append("Chi-squared breakdown:")
                
                for name, inst_data in self.instrument_data.items():
                    wave = inst_data['wave']
                    flux = inst_data['flux']
                    error = inst_data['error']
                    model_func = inst_data['model']
                    
                    model_flux = model_func(best_theta, wave)
                    chi2_inst = np.sum(((flux - model_flux) / error) ** 2)
                    ndof_inst = len(flux)
                    chi2_reduced_inst = chi2_inst / ndof_inst
                    
                    total_chi2 += chi2_inst
                    total_points += ndof_inst
                    
                    self.status_text.append(f"  {name}: χ²/N = {chi2_reduced_inst:.3f}")
                
                # Overall chi-squared
                total_ndof = total_points - len(best_theta)
                total_chi2_reduced = total_chi2 / total_ndof if total_ndof > 0 else total_chi2
                
                self.status_text.append(f"✅ Quick fit completed: Total χ²/ν = {total_chi2_reduced:.3f}")
                
            except Exception as e:
                self.status_text.append(f"✅ Quick fit completed (chi2 calculation failed: {str(e)})")
                
        except Exception as e:
            QMessageBox.critical(self, "Quick Fit Error", f"Quick fit failed:\n{str(e)}")
            self.status_text.append(f"❌ Quick fit error: {str(e)}")
                
    def start_fitting(self):
        """Start MCMC fitting using unified interface"""
        if not self.instrument_data:
            QMessageBox.warning(self, "No Model", "No model data available for fitting")
            return
            
        try:
            # Get current parameters and bounds
            theta, lb, ub = self.param_bounds_table.get_parameters()
            
            # Validate bounds
            if not np.all(lb <= theta) or not np.all(theta <= ub):
                QMessageBox.warning(self, "Invalid Bounds", 
                                  "Initial values must be within bounds")
                return
                
            # Update MCMC parameters from GUI
            self.mcmc_params.update({
                'sampler': self.sampler_combo.currentText(),
                'n_walkers': self.walkers_spin.value(),
                'n_steps': self.steps_spin.value(),
                'perturbation': self.perturbation_spin.value(),
                'optimize': self.optimize_check.isChecked(),
                'use_pool': self.pool_check.isChecked()
            })
            
            # Disable UI during fitting
            self.fit_btn.setEnabled(False)
            self.quick_fit_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Start MCMC thread with unified interface
            self.mcmc_thread = MCMCThread(
                self.instrument_data, theta, lb, ub, self.mcmc_params
            )
            self.mcmc_thread.status_update.connect(self.on_status_update)
            self.mcmc_thread.fitting_completed.connect(self.on_fitting_completed)
            self.mcmc_thread.fitting_error.connect(self.on_fitting_error)
            self.mcmc_thread.start()
            
            self.fitting_started.emit(self.mcmc_params)
            
            self.status_text.append(f"Started MCMC fitting:")
            self.status_text.append(f"  Instruments: {', '.join(self.instrument_data.keys())}")
            self.status_text.append(f"  Sampler: {self.mcmc_params['sampler']}")
            self.status_text.append(f"  Walkers: {self.mcmc_params['n_walkers']}")
            self.status_text.append(f"  Steps: {self.mcmc_params['n_steps']}")
            
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Failed to start fitting:\n{str(e)}")
            self.reset_fitting_ui()
            
    @pyqtSlot(str)
    def on_status_update(self, message):
        """Handle status updates"""
        self.status_text.append(message)
        
    @pyqtSlot(object)
    def on_fitting_completed(self, fitter):
        """Handle fitting completion"""
        self.reset_fitting_ui()
        
        # Update parameter table with best-fit values
        if hasattr(fitter, 'best_theta'):
            best_theta = fitter.best_theta
        elif hasattr(fitter, 'theta_best'):
            best_theta = fitter.theta_best
        else:
            # Extract from samples
            try:
                samples = fitter.get_samples(flat=True, burn_in=0.5)
                best_theta = np.median(samples, axis=0)
            except:
                best_theta = self.theta  # Fallback to original
                
        self.param_bounds_table.update_initial_values(best_theta)
        
        # Create results object
        try:
            if hasattr(self.main_window, 'get_current_model'):
                model = self.main_window.get_current_model()
                if model:
                    #results = fr.FitResults(fitter, model)
                    results = UnifiedResults(fitter, model)  # ← NEW LINE

                else:
                    results = fitter  # Fallback
            else:
                results = fitter
        except:
            results = fitter  # Fallback if FitResults creation fails
            
        self.status_text.append("Fitting completed successfully")
        
        # Calculate basic statistics
        try:
            if hasattr(fitter, 'get_samples'):
                samples = fitter.get_samples(flat=True, burn_in=0.5)
                n_samples = len(samples)
                acceptance_rate = getattr(fitter, 'acceptance_fraction', 'N/A')
                self.status_text.append(f"Final samples: {n_samples}")
                if acceptance_rate != 'N/A':
                    if hasattr(acceptance_rate, '__len__'):
                        acc_mean = np.mean(acceptance_rate)
                        self.status_text.append(f"Acceptance rate: {acc_mean:.3f}")
                    else:
                        self.status_text.append(f"Acceptance rate: {acceptance_rate:.3f}")
        except:
            pass  # Skip stats if not available
            
        self.fitting_completed.emit(results)
        
        # Update model display
        self.plot_spectrum()
        
    @pyqtSlot(str)
    def on_fitting_error(self, error_msg):
        """Handle fitting error"""
        self.reset_fitting_ui()
        self.status_text.append(f"Fitting error: {error_msg}")
        QMessageBox.critical(self, "Fitting Error", f"MCMC fitting failed:\n{error_msg}")
        
    def reset_fitting_ui(self):
        """Reset UI after fitting"""
        self.fit_btn.setEnabled(True)
        # Re-enable quick fit only for single instrument
        self.quick_fit_btn.setEnabled(True)
        #self.quick_fit_btn.setEnabled(len(self.instrument_data) == 1)        
        self.progress_bar.setVisible(False)
        
    def get_current_theta(self):
        """Return current theta parameters from table"""
        if self.param_bounds_table.rowCount() > 0:
            theta, _, _ = self.param_bounds_table.get_parameters()
            return theta
        return self.theta
        
    def refresh(self):
        """Refresh the tab"""
        self.plot_spectrum()