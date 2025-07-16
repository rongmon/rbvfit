#!/usr/bin/env python
"""
Updated fitting_tab.py - Enhanced for Unified vfit Interface

Key Changes:
- Cleaned up to use only unified vfit interface
- Removed legacy support code  
- Streamlined MCMC parameter handling
- Better integration with UnifiedResults
- Improved error handling and status reporting
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
            self.status_update.emit("Creating vfit fitter...")
            
            # Create fitter with unified interface
            fitter = mc.vfit(
                self.instrument_data,    # Unified instrument data dictionary
                self.theta, self.lb, self.ub,
                no_of_Chain=self.mcmc_params['n_walkers'],
                no_of_steps=self.mcmc_params['n_steps'],
                sampler=self.mcmc_params['sampler'],
                perturbation=self.mcmc_params['perturbation']
            )
            
            self.status_update.emit("Starting MCMC sampling...")
            
            # Run MCMC
            fitter.runmcmc(
                optimize=self.mcmc_params['optimize'],
                verbose=False,  # Allow vfit to print its own status
                use_pool=self.mcmc_params['use_pool'],
                progress=True  
            )
            
            self.status_update.emit("MCMC completed successfully")
            self.fitting_completed.emit(fitter)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fitting_error.emit(str(e))


class ParameterBoundsTable(QTableWidget):
    """Table widget for editing parameter bounds and initial values"""
    
    def __init__(self):
        super().__init__()
        self.setup_table()
        
    def setup_table(self):
        """Set up table structure"""
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Parameter', 'Initial', 'Lower', 'Upper'])
        
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultSectionSize(100)
        
    def set_parameters(self, param_names, theta, lb, ub):
        """Set parameter data"""
        n_params = len(theta)
        self.setRowCount(n_params)
        
        for i in range(n_params):
            # Parameter name (read-only)
            name_item = QTableWidgetItem(param_names[i] if i < len(param_names) else f"Param_{i}")
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.setItem(i, 0, name_item)
            
            # Initial value (editable)
            self.setItem(i, 1, QTableWidgetItem(f"{theta[i]:.6f}"))
            
            # Lower bound (editable)
            self.setItem(i, 2, QTableWidgetItem(f"{lb[i]:.6f}"))
            
            # Upper bound (editable)
            self.setItem(i, 3, QTableWidgetItem(f"{ub[i]:.6f}"))
        
        self.resizeColumnsToContents()
        
    def get_parameters(self):
        """Get parameter values from table"""
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
            self.setItem(i, 1, QTableWidgetItem(f"{new_theta[i]:.6f}"))


class FittingTab(QWidget):
    """Enhanced fitting tab with unified vfit interface"""
    
    # Signals
    fitting_started = pyqtSignal(dict)
    fitting_completed = pyqtSignal(object)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # Data storage - UNIFIED INTERFACE ONLY
        self.instrument_data = {}
        self.theta = None
        self.lb = None
        self.ub = None
        self.param_names = []
        self.current_instrument = None
        
        # Plot range storage
        self.plot_ranges = {
            'xlim': None,  # None = auto, tuple = manual
            'ylim': None,
            'original_xlim': None,  # Store original auto ranges
            'original_ylim': None
        }
        
        # MCMC parameters
        self.mcmc_params = {
            'n_walkers': 50,
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
        
        # Left panel: Controls and parameters
        self.setup_controls_panel(splitter)
        
        # Right panel: Plot and status
        self.setup_plot_panel(splitter)
        
        splitter.setSizes([400, 600])
        
    def setup_controls_panel(self, parent):
        """Create controls panel"""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        parent.addWidget(controls_widget)
        
        # MCMC settings group
        mcmc_group = QGroupBox("MCMC Settings")
        mcmc_layout = QFormLayout()
        mcmc_group.setLayout(mcmc_layout)
        controls_layout.addWidget(mcmc_group)
        
        # Sampler selection
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(['emcee', 'zeus'])
        self.sampler_combo.setCurrentText(self.mcmc_params['sampler'])
        mcmc_layout.addRow("Sampler:", self.sampler_combo)
        
        # Number of walkers
        self.walkers_spin = QSpinBox()
        self.walkers_spin.setRange(10, 500)
        self.walkers_spin.setValue(self.mcmc_params['n_walkers'])
        mcmc_layout.addRow("Walkers:", self.walkers_spin)
        
        # Number of steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 10000)
        self.steps_spin.setValue(self.mcmc_params['n_steps'])
        mcmc_layout.addRow("Steps:", self.steps_spin)
        
        # Perturbation
        self.perturbation_spin = QDoubleSpinBox()
        self.perturbation_spin.setRange(1e-6, 1e-2)
        self.perturbation_spin.setDecimals(6)
        self.perturbation_spin.setValue(self.mcmc_params['perturbation'])
        mcmc_layout.addRow("Perturbation:", self.perturbation_spin)
        
        # Options
        self.optimize_check = QCheckBox("Optimize starting guess")
        self.optimize_check.setChecked(self.mcmc_params['optimize'])
        mcmc_layout.addRow(self.optimize_check)
        
        self.pool_check = QCheckBox("Use multiprocessing")
        self.pool_check.setChecked(self.mcmc_params['use_pool'])
        mcmc_layout.addRow(self.pool_check)
        
        # Fitting controls
        fitting_group = QGroupBox("Fitting Controls")
        fitting_layout = QVBoxLayout()
        fitting_group.setLayout(fitting_layout)
        controls_layout.addWidget(fitting_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.quick_fit_btn = QPushButton("Quick Fit")
        self.fit_btn = QPushButton("Run MCMC")
        
        button_layout.addWidget(self.quick_fit_btn)
        button_layout.addWidget(self.fit_btn)
        fitting_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fitting_layout.addWidget(self.progress_bar)
        
        # Parameter bounds table
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_layout = QVBoxLayout()
        bounds_group.setLayout(bounds_layout)
        controls_layout.addWidget(bounds_group)
        
        self.param_bounds_table = ParameterBoundsTable()
        bounds_layout.addWidget(self.param_bounds_table)
        
    def setup_plot_panel(self, parent):
        """Create plot panel"""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        parent.addWidget(plot_widget)
        
        # Plot controls
        plot_controls_layout = QHBoxLayout()
        plot_layout.addLayout(plot_controls_layout)
        
        # Instrument selector
        plot_controls_layout.addWidget(QLabel("Instrument:"))
        self.instrument_combo = QComboBox()
        plot_controls_layout.addWidget(self.instrument_combo)
        
        # Plot range button
        self.plot_range_btn = QPushButton("Set Plot Range")
        plot_controls_layout.addWidget(self.plot_range_btn)
        
        plot_controls_layout.addStretch()
        
        # Plot area
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        plot_layout.addWidget(self.status_text)
        
    def setup_connections(self):
        """Connect signals and slots"""
        # MCMC parameter updates
        self.sampler_combo.currentTextChanged.connect(self.update_mcmc_params)
        self.walkers_spin.valueChanged.connect(self.update_mcmc_params)
        self.steps_spin.valueChanged.connect(self.update_mcmc_params)
        self.perturbation_spin.valueChanged.connect(self.update_mcmc_params)
        self.optimize_check.toggled.connect(self.update_mcmc_params)
        self.pool_check.toggled.connect(self.update_mcmc_params)
        
        # Fitting controls
        self.quick_fit_btn.clicked.connect(self.run_quick_fit)
        self.fit_btn.clicked.connect(self.run_mcmc_fit)
        
        # Plot controls
        self.instrument_combo.currentTextChanged.connect(self.plot_spectrum)
        self.plot_range_btn.clicked.connect(self.set_plot_range)
        
    def update_mcmc_params(self):
        """Update MCMC parameters from UI"""
        self.mcmc_params.update({
            'sampler': self.sampler_combo.currentText(),
            'n_walkers': self.walkers_spin.value(),
            'n_steps': self.steps_spin.value(),
            'perturbation': self.perturbation_spin.value(),
            'optimize': self.optimize_check.isChecked(),
            'use_pool': self.pool_check.isChecked()
        })
    
    def set_model_data(self, instrument_data, theta, bounds):
        """Set model data from Tab 2 - UNIFIED INTERFACE"""
        self.instrument_data = instrument_data
        self.theta = theta.copy()
        self.lb = bounds['lb'].copy()
        self.ub = bounds['ub'].copy()
        
        # Generate parameter names
        n_params = len(theta)
        n_comp = n_params // 3
        self.param_names = []
        # N parameters first
        for i in range(n_comp):
            self.param_names.append(f'N_{i+1}')
        # Then b parameters  
        for i in range(n_comp):
            self.param_names.append(f'b_{i+1}')
        # Then v parameters
        for i in range(n_comp):
            self.param_names.append(f'v_{i+1}')
        # Update UI
        self.update_instrument_selector()
        self.update_parameter_table()
        self.update_fitting_controls()
        
        # Status update
        self.status_text.clear()
        self.status_text.append(f"Model data loaded:")
        self.status_text.append(f"  Instruments: {len(instrument_data)}")
        self.status_text.append(f"  Parameters: {n_params} ({n_comp} components)")
        for name, data in instrument_data.items():
            wave = data['wave']
            self.status_text.append(f"  {name}: {len(wave)} points, {wave.min():.1f}-{wave.max():.1f} Å")
        
        # Reset plot ranges and update plot
        self.reset_plot_ranges()
        self.plot_spectrum()
        
        print(f"✓ Fitting tab ready: {len(instrument_data)} instruments, {n_params} parameters")

    def update_instrument_selector(self):
        """Update instrument selector combo"""
        self.instrument_combo.clear()
        self.instrument_combo.addItems(list(self.instrument_data.keys()))
        if self.instrument_data:
            self.current_instrument = list(self.instrument_data.keys())[0]
    
    def update_parameter_table(self):
        """Update parameter bounds table"""
        if self.theta is not None:
            self.param_bounds_table.set_parameters(self.param_names, self.theta, self.lb, self.ub)
    
    def update_fitting_controls(self):
        """Update fitting control states"""
        has_data = len(self.instrument_data) > 0
        self.quick_fit_btn.setEnabled(has_data)
        self.fit_btn.setEnabled(has_data)
        
        # Update walker count recommendation
        if self.theta is not None:
            n_params = len(self.theta)
            recommended_walkers = max(50, 2 * n_params)
            self.walkers_spin.setValue(recommended_walkers)
    
    def reset_plot_ranges(self):
        """Reset plot ranges to auto"""
        self.plot_ranges = {
            'xlim': None,
            'ylim': None,
            'original_xlim': None,
            'original_ylim': None
        }
    
    def plot_spectrum(self):
        """Plot spectrum for current instrument"""
        self.figure.clear()
        
        if not self.instrument_data or not self.current_instrument:
            self.canvas.draw()
            return
            
        self.current_instrument = self.instrument_combo.currentText()
        if not self.current_instrument or self.current_instrument not in self.instrument_data:
            self.canvas.draw()
            return
        
        #grab values from the param_table
        current_theta, _, _ = self.param_bounds_table.get_parameters()   
        data = self.instrument_data[self.current_instrument]
        wave = data['wave']
        flux = data['flux']
        error = data['error']
        
        ax = self.figure.add_subplot(111)
        
        # Plot data
        ax.step(wave, flux, 'k-', where='mid', label='Data')
        ax.step(wave, error, 'r-', where='mid', alpha=0.7, label='Error')
        
        # Plot model if available
        if current_theta is not None:
            try:
                #data['model'].print_info()
                model_flux = data['model'].evaluate(current_theta, wave)
                ax.plot(wave, model_flux, 'b-', linewidth=2, label='Model')
            except Exception as e:
                self.status_text.append(f"Model evaluation error: {str(e)}")
        
        # Set ranges
        if self.plot_ranges['xlim'] is not None:
            ax.set_xlim(self.plot_ranges['xlim'])
        else:
            xlim = ax.get_xlim()
            self.plot_ranges['original_xlim'] = xlim
            
        if self.plot_ranges['ylim'] is not None:
            ax.set_ylim(self.plot_ranges['ylim'])
        else:
            ylim = ax.get_ylim()
            self.plot_ranges['original_ylim'] = ylim
        
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'{self.current_instrument} Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def set_plot_range(self):
        """Set custom plot range"""
        if not self.plot_ranges['original_xlim']:
            return
        


        current_xlim = self.plot_ranges['xlim'] or self.plot_ranges['original_xlim']
        current_ylim = self.plot_ranges['ylim'] or self.plot_ranges['original_ylim']


        # Default ranges for velocity plots
        original_xlim = (current_xlim)  # Default velocity range
        original_ylim = (-0.02, 1.5)  # Default flux range
        
        dialog = PlotRangeDialog(
            current_xlim=current_xlim,
            current_ylim=current_ylim, 
            original_xlim=original_xlim,
            original_ylim=original_ylim,
            parent=self
        )

        if dialog.exec_() == QDialog.Accepted:
            xlim, ylim = dialog.get_ranges()
            self.plot_ranges['xlim'] = xlim
            self.plot_ranges['ylim'] = ylim
            self.plot_spectrum()
    
    def run_quick_fit(self):
        """Run quick scipy optimization fit"""
        if not self.instrument_data or self.theta is None:
            QMessageBox.warning(self, "No Data", "No model data available for fitting")
            return
            
        try:
            # Get current parameters from table
            theta, lb, ub = self.param_bounds_table.get_parameters()
            
            self.status_text.append("Starting quick fit (scipy optimization)...")
            
            # Create temporary fitter for quick fit
            fitter = mc.vfit(self.instrument_data, theta, lb, ub)
            
            # Run quick fit
            best_theta, best_errors = fitter.fit_quick(verbose=True)
            
            # Update parameter table
            self.param_bounds_table.update_initial_values(best_theta)
            self.theta = best_theta.copy()
            
            # Update plot
            self.plot_spectrum()
            
            self.status_text.append("Quick fit completed successfully")
            self.status_text.append(f"Best-fit parameters updated in table")
            
        except Exception as e:
            QMessageBox.critical(self, "Quick Fit Error", f"Quick fit failed:\n{str(e)}")
            self.status_text.append(f"Quick fit error: {str(e)}")
    
    def run_mcmc_fit(self):
        """Run MCMC fitting"""
        if not self.instrument_data or self.theta is None:
            QMessageBox.warning(self, "No Data", "No model data available for fitting")
            return
            
        try:
            # Get current parameters from table
            theta, lb, ub = self.param_bounds_table.get_parameters()
            
            # Update MCMC parameters
            self.update_mcmc_params()
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
        if hasattr(fitter, 'theta_best') and fitter.theta_best is not None:
            best_theta = fitter.theta_best
        elif hasattr(fitter, 'best_theta') and fitter.best_theta is not None:
            best_theta = fitter.best_theta
        else:
            # Extract from samples
            try:
                if hasattr(fitter, 'get_samples'):
                    samples = fitter.get_samples(flat=True, burn_in=0.2)
                    best_theta = np.median(samples, axis=0)
                else:
                    # Fallback to sampler
                    samples = fitter.sampler.get_chain(flat=True, discard=int(0.2 * fitter.no_of_steps))
                    best_theta = np.median(samples, axis=0)
            except:
                best_theta = self.theta  # Fallback to original
                
        self.param_bounds_table.update_initial_values(best_theta)
        self.theta = best_theta.copy()
        
        # Create results object for Tab 4
        try:
            results = UnifiedResults(fitter)
            self.status_text.append("Created UnifiedResults object")
        except Exception as e:
            # Fallback to fitter object
            results = fitter
            self.status_text.append(f"Warning: Could not create UnifiedResults: {e}")
            
        self.status_text.append("Fitting completed successfully")
        
        # Calculate basic statistics
        try:
            if hasattr(fitter, 'sampler'):
                # Get acceptance fraction
                acc_frac = fitter.sampler.acceptance_fraction
                if hasattr(acc_frac, '__len__'):
                    acc_mean = np.mean(acc_frac)
                    self.status_text.append(f"Acceptance rate: {acc_mean:.3f}")
                else:
                    self.status_text.append(f"Acceptance rate: {acc_frac:.3f}")
                    
                # Get autocorrelation time (if available)
                try:
                    tau = fitter.sampler.get_autocorr_time(quiet=True)
                    tau_mean = np.mean(tau)
                    self.status_text.append(f"Autocorrelation time: {tau_mean:.1f}")
                except:
                    pass  # Skip if not available
                    
        except Exception as e:
            self.status_text.append(f"Could not compute diagnostics: {e}")
            
        # Emit completion signal
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
        self.quick_fit_btn.setEnabled(True)
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
        
    def clear_state(self):
        """Clear all state (for project loading)"""
        self.instrument_data = {}
        self.theta = None
        self.lb = None
        self.ub = None
        self.param_names = []
        self.current_instrument = None
        
        # Clear UI
        self.instrument_combo.clear()
        self.param_bounds_table.setRowCount(0)
        self.status_text.clear()
        self.figure.clear()
        self.canvas.draw()
        
        # Reset plot ranges
        self.reset_plot_ranges()
        
        # Reset controls
        self.fit_btn.setEnabled(False)
        self.quick_fit_btn.setEnabled(False)
        self.progress_bar.setVisible(False)