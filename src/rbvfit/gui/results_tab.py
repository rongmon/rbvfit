#!/usr/bin/env python
"""
rbvfit 2.0 Results Tab - Updated with Real Data Integration

Interface for viewing and exporting fit results with actual MCMC data.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QPushButton, QTableWidget, QTableWidgetItem,
                            QTextEdit, QTabWidget, QFileDialog, QMessageBox,
                            QHeaderView, QLabel, QComboBox, QCheckBox, QApplication,QDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from rbvfit.gui.shared_plot_range_dialog import PlotRangeDialog  # or wherever it's located

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

from rbvfit.gui.io import export_results_csv, export_results_latex


class ResultsTab(QWidget):
    """Tab for displaying and exporting fit results"""
    
    export_requested = pyqtSignal(str, str)  # format, filename
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.results = None
        self.parameter_data = []
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create results interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Top controls
        self.setup_top_controls(layout)
        
        # Main content: plots | statistics and tables
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left: Plot area
        self.setup_plot_area(main_splitter)
        
        # Right: Statistics and parameter table
        self.setup_results_panel(main_splitter)
        
        main_splitter.setSizes([700, 500])
        
    def setup_top_controls(self, parent_layout):
        """Create top control bar"""
        control_layout = QHBoxLayout()
        parent_layout.addLayout(control_layout)
        
        # Export buttons
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.setToolTip("Export parameter table to CSV file")
        control_layout.addWidget(self.export_csv_btn)
        
        self.export_latex_btn = QPushButton("Export LaTeX")
        self.export_latex_btn.setEnabled(False)
        self.export_latex_btn.setToolTip("Export results as LaTeX table")
        control_layout.addWidget(self.export_latex_btn)
        
        self.save_plots_btn = QPushButton("Save Plots")
        self.save_plots_btn.setEnabled(False)
        self.save_plots_btn.setToolTip("Save all plots to files")
        control_layout.addWidget(self.save_plots_btn)

        # NEW: Add save results button
        self.save_results_btn = QPushButton("Save Results")
        self.save_results_btn.setEnabled(False)
        self.save_results_btn.setToolTip("Save complete fit results to HDF5 file for later analysis")
        control_layout.addWidget(self.save_results_btn)        

        # NEW: Add trace plots button
        self.trace_plots_btn = QPushButton("Show Trace Plots")
        self.trace_plots_btn.setEnabled(False)
        self.trace_plots_btn.setToolTip("Display MCMC chain trace plots for convergence assessment")
        control_layout.addWidget(self.trace_plots_btn)
        
        control_layout.addStretch()
        
        # Plot controls
        self.show_components_check = QCheckBox("Show Components")
        self.show_components_check.setChecked(True)
        self.show_components_check.setToolTip("Show individual Voigt components")
        control_layout.addWidget(self.show_components_check)
        
        self.show_residuals_check = QCheckBox("Show Residuals")
        self.show_residuals_check.setChecked(True)
        self.show_residuals_check.setToolTip("Show fit residuals")
        control_layout.addWidget(self.show_residuals_check)
        
    def setup_plot_area(self, parent):
        """Create plot area with tabs"""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        parent.addWidget(plot_widget)
        
        if not HAS_MATPLOTLIB:
            plot_layout.addWidget(QLabel("Matplotlib not available - plots disabled"))
            return

        # NEW: Add instrument selection controls BEFORE plot tabs
        instrument_controls = QHBoxLayout()
        plot_layout.addLayout(instrument_controls)
        
        instrument_controls.addWidget(QLabel("Show results for:"))
        self.instrument_combo = QComboBox()
        self.instrument_combo.setToolTip("Select instrument/configuration to display")
        self.instrument_combo.setMinimumWidth(150)
        instrument_controls.addWidget(self.instrument_combo)
        
        instrument_controls.addStretch()

            
        # Plot tabs
        self.plot_tabs = QTabWidget()
        plot_layout.addLayout(QHBoxLayout())  # Controls will go here
        plot_layout.addWidget(self.plot_tabs)
        
        # Model comparison tab
        self.comparison_widget = QWidget()
        comparison_layout = QVBoxLayout()
        self.comparison_widget.setLayout(comparison_layout)
        
        # Comparison plot controls
        comparison_controls = QHBoxLayout()
        comparison_layout.addLayout(comparison_controls)
        
        self.set_ranges_btn = QPushButton("Set Plot Ranges")
        comparison_controls.addWidget(self.set_ranges_btn)
        comparison_controls.addStretch()
        
        # Set default ranges
        self.plot_ranges = {
            'x_min': None, 'x_max': None,
            'y_min': -0.02, 'y_max': 1.5  # Default y-ranges as requested
        }
        
        # Comparison plot canvas
        self.comparison_figure = Figure(figsize=(12, 8))
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        comparison_layout.addWidget(self.comparison_canvas)
        
        self.plot_tabs.addTab(self.comparison_widget, "Model vs Data")




        
        # Corner plot tab
        self.corner_widget = QWidget()
        corner_layout = QVBoxLayout()
        self.corner_widget.setLayout(corner_layout)
        
        # Corner plot controls
        corner_controls = QHBoxLayout()
        corner_layout.addLayout(corner_controls)
        
        corner_controls.addWidget(QLabel("Parameters:"))
        self.param_selector = QComboBox()
        self.param_selector.addItems(["All parameters", "N parameters", "b parameters", "v parameters"])
        self.param_selector.setToolTip("Select which parameters to show in corner plot")
        corner_controls.addWidget(self.param_selector)
        
        self.update_corner_btn = QPushButton("Generate Corner Plot")
        self.update_corner_btn.setEnabled(False)
        self.update_corner_btn.setToolTip("Update corner plot")
        corner_controls.addWidget(self.update_corner_btn)
        
        self.export_corner_btn = QPushButton("Export")
        self.export_corner_btn.setEnabled(False)
        self.export_corner_btn.setToolTip("Export corner plot")
        corner_controls.addWidget(self.export_corner_btn)
        
        corner_controls.addStretch()
        
        # Corner plot canvas
        self.corner_figure = Figure(figsize=(8, 8))
        self.corner_canvas = FigureCanvas(self.corner_figure)
        corner_layout.addWidget(self.corner_canvas)
        #placeholder corner plot. will generate on demand.
        self.corner_figure.clear()
        ax = self.corner_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Click "Generate Corner Plot" to create\n(May take time for complex models)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        self.corner_canvas.draw()
        
        self.plot_tabs.addTab(self.corner_widget, "Corner Plot")        
        
        # Velocity plot tab
        self.velocity_widget = QWidget()
        velocity_layout = QVBoxLayout()
        self.velocity_widget.setLayout(velocity_layout)
        
        # Velocity plot controls
        vel_controls = QHBoxLayout()
        velocity_layout.addLayout(vel_controls)
        
        vel_controls.addWidget(QLabel("Velocity Range:"))
        self.vel_range_combo = QComboBox()
        self.vel_range_combo.addItems(["Auto", "±200 km/s", "±500 km/s", "±1000 km/s"])
        vel_controls.addWidget(self.vel_range_combo)
        vel_controls.addStretch()
        
        # Velocity plot canvas
        self.velocity_figure = Figure(figsize=(12, 8))
        self.velocity_canvas = FigureCanvas(self.velocity_figure)
        velocity_layout.addWidget(self.velocity_canvas)
        
        self.plot_tabs.addTab(self.velocity_widget, "Velocity Space")
        
    def setup_results_panel(self, parent):
        """Create results statistics and parameter table"""
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        parent.addWidget(right_widget)
        
        # Fit statistics
        stats_group = QGroupBox("Fit Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        self.stats_text.setToolTip("Statistical summary of fit quality")
        stats_layout.addWidget(self.stats_text)
        
        # Parameter table
        table_group = QGroupBox("Best-fit Parameters")
        table_layout = QVBoxLayout()
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)
        
        # Parameter table widget
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(4)
        self.param_table.setHorizontalHeaderLabels(['Parameter', 'Best Fit', '±Error', 'Units'])
        self.param_table.setToolTip("Best-fit parameter values with uncertainties")
        
        # Set table properties
        header = self.param_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        table_layout.addWidget(self.param_table)
        
        # Table control buttons
        table_btn_layout = QHBoxLayout()
        table_layout.addLayout(table_btn_layout)
        
        self.copy_table_btn = QPushButton("Copy to Clipboard")
        self.copy_table_btn.setEnabled(False)
        self.copy_table_btn.setToolTip("Copy parameter table to clipboard")
        table_btn_layout.addWidget(self.copy_table_btn)
        
        self.export_table_btn = QPushButton("Export Table")
        self.export_table_btn.setEnabled(False)
        self.export_table_btn.setToolTip("Export parameter table to file")
        table_btn_layout.addWidget(self.export_table_btn)
        
        table_btn_layout.addStretch()
        
    def setup_connections(self):
        """Connect signals and slots"""
        # Export buttons
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_latex_btn.clicked.connect(self.export_latex)
        self.save_plots_btn.clicked.connect(self.save_plots)
        self.trace_plots_btn.clicked.connect(self.show_trace_plots)
        self.save_results_btn.clicked.connect(self.save_results)

        
        # Plot controls
        self.show_components_check.toggled.connect(self.update_plots)
        self.show_residuals_check.toggled.connect(self.update_plots)
        self.param_selector.currentTextChanged.connect(self.update_corner_plot)
        self.vel_range_combo.currentTextChanged.connect(self.update_velocity_plot)
        self.set_ranges_btn.clicked.connect(self.set_plot_ranges)
        self.instrument_combo.currentTextChanged.connect(self.on_instrument_changed)

        
        # Corner plot controls
        self.update_corner_btn.clicked.connect(self.update_corner_plot)
        self.export_corner_btn.clicked.connect(self.export_corner_plot)
        
        # Table controls
        self.copy_table_btn.clicked.connect(self.copy_table_to_clipboard)
        self.export_table_btn.clicked.connect(self.export_parameter_table)
        
    def set_results(self, results):
        """Set results - compatibility method for main window"""
        self.update_results(results)
        
    def update_results(self, results):
        """Update display with new fit results"""
        self.results = results
        
        if results is None:
            self.clear_results()
            return


        self.populate_instrument_dropdown()
            
        # Enable controls
        self.export_csv_btn.setEnabled(True)
        self.export_latex_btn.setEnabled(True)
        self.save_plots_btn.setEnabled(True)
        self.trace_plots_btn.setEnabled(True)  # NEW: Enable trace plots button
        self.update_corner_btn.setEnabled(True)
        self.export_corner_btn.setEnabled(True)
        self.copy_table_btn.setEnabled(True)
        self.export_table_btn.setEnabled(True)
        self.save_results_btn.setEnabled(True)

        
        # Update all displays
        self.update_statistics()
        self.update_parameter_table()
        
        if HAS_MATPLOTLIB:
            self.update_plots()
            
    def clear_results(self):
        """Clear all result displays"""
        # Disable controls
        self.export_csv_btn.setEnabled(False)
        self.export_latex_btn.setEnabled(False)
        self.save_plots_btn.setEnabled(False)
        self.trace_plots_btn.setEnabled(False)  # NEW: Disable trace plots button
        self.update_corner_btn.setEnabled(False)
        self.export_corner_btn.setEnabled(False)
        self.copy_table_btn.setEnabled(False)
        self.export_table_btn.setEnabled(False)
        self.save_results_btn.setEnabled(False)

        if hasattr(self, 'instrument_combo'):
            self.instrument_combo.clear()
    
        
        # Clear displays
        self.stats_text.clear()
        self.stats_text.append("No results available")
        
        self.param_table.setRowCount(0)
        
        if HAS_MATPLOTLIB:
            self.corner_figure.clear()
            self.comparison_figure.clear()
            self.velocity_figure.clear()
            self.corner_canvas.draw()
            self.comparison_canvas.draw()
            self.velocity_canvas.draw()
            
    def save_results(self):
        """Save fit results with file dialog."""
        if not hasattr(self, 'results') or self.results is None:
            QMessageBox.warning(self, "No Results", "No fit results available to save.")
            return
        
        
        # Open file save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Fit Results",
            "fit_results.h5",  # Default filename
            "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        
        if filename:
            try:
                #
                self.results.save(filename)
                
                QMessageBox.information(
                    self, 
                    "Save Successful", 
                    f"Results saved successfully to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Save Failed", 
                    f"Failed to save results:\n{str(e)}"
                )
    
    def populate_instrument_dropdown(self):
        """Populate instrument dropdown with available instruments/configurations"""
        self.instrument_combo.clear()
        
        try:
            # Check if results has multi-instrument data
            if hasattr(self.results, 'instrument_data') and self.results.instrument_data:
                # Multi-instrument case
                for instrument_name in self.results.instrument_data.keys():
                    self.instrument_combo.addItem(instrument_name)
            elif hasattr(self.results, 'fitter') and hasattr(self.results.fitter, 'instrument_data'):
                # Alternative access pattern
                for instrument_name in self.results.fitter.instrument_data.keys():
                    self.instrument_combo.addItem(instrument_name)
            else:
                # Single instrument case - check main window for configurations
                if hasattr(self.main_window, 'configurations') and self.main_window.configurations:
                    for config_name in self.main_window.configurations.keys():
                        self.instrument_combo.addItem(config_name)
                else:
                    # Fallback - add generic entry
                    self.instrument_combo.addItem("Primary")
                    
            # Select first item by default
            if self.instrument_combo.count() > 0:
                self.instrument_combo.setCurrentIndex(0)
                
        except Exception as e:
            print(f"Warning: Could not populate instrument dropdown: {e}")
            # Fallback
            self.instrument_combo.addItem("Primary")

    def show_trace_plots(self):
        """Display MCMC trace plots in a new window"""
        if self.results is None:
            QMessageBox.warning(self, "No Results", "No fit results available for trace plots")
            return
            
        try:
            # Use the existing FitResults.chain_trace_plot() method
            trace_fig = self.results.chain_trace_plot()
            
            if trace_fig is not None:
                # Show the plot
                trace_fig.show()
                self.main_window.update_status("Trace plots displayed - check for convergence")
            else:
                QMessageBox.warning(self, "Plot Error", "Could not generate trace plots")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating trace plots:\n{str(e)}")


    def on_instrument_changed(self):
        """Handle instrument selection change"""
        if self.results is None:
            return
            
        # Update all plots with selected instrument
        self.update_plots()
        
        selected_instrument = self.instrument_combo.currentText()
        if selected_instrument:
            self.main_window.update_status(f"Displaying results for: {selected_instrument}")


    def set_plot_ranges(self):
        """Open plot range dialog"""
        if self.results is None:
            QMessageBox.warning(self, "No Data", "No results available to set ranges for")
            return
        
        # Extract current ranges, providing defaults for None values
        x_min = self.plot_ranges.get('x_min')
        x_max = self.plot_ranges.get('x_max') 
        y_min = self.plot_ranges.get('y_min', -0.02)
        y_max = self.plot_ranges.get('y_max', 1.5)
        
        # Provide sensible defaults if None
        if x_min is None or x_max is None:
            current_xlim = None  # This signals auto-range
        else:
            current_xlim = (x_min, x_max)
            
        current_ylim = (y_min, y_max)        
        # Set original ranges (can be None if auto)
        original_xlim = None  # or extract from somewhere if available
        original_ylim = (-0.02, 1.5)  # Reasonable defaults  # or extract from somewhere if available
        
        dialog = PlotRangeDialog(
            current_xlim=current_xlim,
            current_ylim=current_ylim, 
            original_xlim=original_xlim,
            original_ylim=original_ylim,
            parent=self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            xlim, ylim = dialog.get_ranges()
            
            # Update plot_ranges dict to match results_tab format
            if xlim:
                self.plot_ranges['x_min'] = xlim[0]
                self.plot_ranges['x_max'] = xlim[1]
            else:
                self.plot_ranges['x_min'] = None
                self.plot_ranges['x_max'] = None
                
            if ylim:
                self.plot_ranges['y_min'] = ylim[0]
                self.plot_ranges['y_max'] = ylim[1]
            else:
                self.plot_ranges['y_min'] = -0.02  # Default
                self.plot_ranges['y_max'] = 1.5    # Default
                
            self.update_model_comparison()  # Refresh plot with new ranges    
    def update_statistics(self):
        """Update fit statistics display with real data"""
        self.stats_text.clear()
        
        if self.results is None:
            self.stats_text.append("No results available")
            return
            
        try:
            # Extract real statistics from FitResults object
            if hasattr(self.results, 'fitter') and self.results.fitter:
                fitter = self.results.fitter
                
                # Get basic info
                n_samples = len(getattr(fitter, 'samples', []))
                n_params = len(getattr(fitter, 'best_theta', []))
                n_data_points = len(getattr(fitter, 'fnorm', []))
                
                # Calculate chi-squared
                try:
                    chi2 = self.results.chi_squared()
                    chi2_reduced = chi2['reduced_chi2']
                except Exception:
                    chi2_reduced = None
                
                # Get convergence diagnostics
                if hasattr(self.results, 'convergence_diagnostics'):
                    c = self.results.convergence_diagnostics()
                    conv_status = c['overall_status']
                    acceptance_rate = c['acceptance_fraction']['mean']
                    
                    # Safe extraction for Zeus-specific metrics
                    if 'gelman_rubin' in c and c['gelman_rubin'] is not None:
                        rhat_max = c['gelman_rubin']['max_r_hat']
                    else:
                        rhat_max = None
                    
                    # Safe extraction for effective sample size
                    if 'effective_sample_size' in c and c['effective_sample_size'] is not None:
                        min_eff_size = c['effective_sample_size']['min_n_eff']
                    else:
                        min_eff_size = None
                    
                    # Safe extraction for Emcee-specific metrics
                    if 'autocorr_time' in c and c['autocorr_time'] is not None:
                        mean_autocorr = c['autocorr_time']['mean_tau']
                    else:
                        mean_autocorr = None
                    
                else:
                    # Fallback to fitter object
                    conv_status = 'Unknown'
                    rhat_max = None
                    acceptance_rate = getattr(fitter, 'acceptance_rate', None)
                    min_eff_size = None
                    mean_autocorr = None
                
                # Safe formatting function
                def safe_format(value, format_spec):
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        return "N/A"
                    try:
                        return f"{value:{format_spec}}"
                    except:
                        return "N/A"
                
                # Create HTML statistics display
                stats_html = f"""
                <h4>Convergence</h4>
                <p style="color: {'green' if conv_status == 'GOOD' else 'orange' if conv_status == 'MARGINAL' else 'red'};"><b>{conv_status} convergence</b></p>
                <h4>Fit Quality</h4>
                <table>
                <tr><td><b>χ²/ν:</b></td><td>{safe_format(chi2_reduced, '.3f')}</td></tr>
                <tr><td><b>Parameters:</b></td><td>{n_params}</td></tr>
                <tr><td><b>Data points:</b></td><td>{n_data_points}</td></tr>
                </table>
                
                <h4>MCMC Diagnostics</h4>
                <table>
                <tr><td><b>Samples:</b></td><td>{n_samples}</td></tr>
                <tr><td><b>Chains:</b></td><td>{getattr(fitter, 'no_of_Chain', 'N/A')}</td></tr>
                <tr><td><b>Acceptance:</b></td><td>{safe_format(acceptance_rate, '.3f')}</td></tr>
                <tr><td><b>R̂ (max):</b></td><td>{safe_format(rhat_max, '.3f')}</td></tr>
                <tr><td><b>Min N_eff:</b></td><td>{safe_format(min_eff_size, '.0f')}</td></tr>
                <tr><td><b>Mean τ_autocorr:</b></td><td>{safe_format(mean_autocorr, '.1f')}</td></tr>
                </table>                
                """
                
                self.stats_text.setHtml(stats_html)
                
            else:
                self.stats_text.append("Results object missing fitter data")
                
        except Exception as e:
            self.stats_text.append(f"Error extracting statistics: {str(e)}")

    def update_parameter_table(self):
        """Update parameter table with real results"""
        # Clear existing data
        self.param_table.setRowCount(0)
        self.parameter_data = []
        
        if self.results is None:
            return
            
        try:
            # Get parameter summary from results
            if hasattr(self.results, 'parameter_summary'):
                param_summary = self.results.parameter_summary(verbose=False)
                
                # Populate table with real parameters
                for i, name in enumerate(param_summary.names):
                    best_fit = param_summary.best_fit[i]
                    error = param_summary.errors[i]
                    
                    # Determine units based on parameter type
                    if name.startswith('N_'):
                        units = "log cm⁻²"
                        best_fit_str = f"{best_fit:.3f}"
                        error_str = f"{error:.3f}"
                    elif name.startswith('b_'):
                        units = "km/s"
                        best_fit_str = f"{best_fit:.1f}"
                        error_str = f"{error:.1f}"
                    elif name.startswith('v_'):
                        units = "km/s"
                        best_fit_str = f"{best_fit:.1f}"
                        error_str = f"{error:.1f}"
                    else:
                        units = ""
                        best_fit_str = f"{best_fit:.3f}"
                        error_str = f"{error:.3f}"
                    
                    self.parameter_data.append((name, best_fit_str, error_str, units))
                    
            elif hasattr(self.results, 'fitter') and hasattr(self.results.fitter, 'best_theta'):
                # Fallback: extract basic info from fitter
                fitter = self.results.fitter
                best_theta = fitter.best_theta
                
                # Try to get parameter names from model
                param_names = []
                if hasattr(self.results, 'model') and hasattr(self.results.model, 'param_manager'):
                    param_names = self.results.model.param_manager.get_parameter_names()
                else:
                    # Generic parameter names
                    n_params = len(best_theta)
                    n_comp = n_params // 3
                    param_names = []
                    for i in range(n_comp):
                        param_names.extend([f"N_c{i+1}", f"b_c{i+1}", f"v_c{i+1}"])
                
                # Get uncertainties if available
                if hasattr(fitter, 'samples') and len(fitter.samples) > 0:
                    errors = np.std(fitter.samples, axis=0)
                else:
                    errors = np.zeros_like(best_theta)
                
                # Populate table
                for i, (name, value, error) in enumerate(zip(param_names, best_theta, errors)):
                    if name.startswith('N_'):
                        units = "log cm⁻²"
                        value_str = f"{value:.3f}"
                        error_str = f"{error:.3f}"
                    elif name.startswith('b_'):
                        units = "km/s"
                        value_str = f"{value:.1f}"
                        error_str = f"{error:.1f}"
                    elif name.startswith('v_'):
                        units = "km/s"
                        value_str = f"{value:.1f}"
                        error_str = f"{error:.1f}"
                    else:
                        units = ""
                        value_str = f"{value:.3f}"
                        error_str = f"{error:.3f}"
                    
                    self.parameter_data.append((name, value_str, error_str, units))
                    
            # Update table widget
            self.param_table.setRowCount(len(self.parameter_data))
            
            for row, (param_name, best_fit, error, units) in enumerate(self.parameter_data):
                self.param_table.setItem(row, 0, QTableWidgetItem(param_name))
                self.param_table.setItem(row, 1, QTableWidgetItem(best_fit))
                self.param_table.setItem(row, 2, QTableWidgetItem(error))
                self.param_table.setItem(row, 3, QTableWidgetItem(units))
                
                # Make table read-only
                for col in range(4):
                    item = self.param_table.item(row, col)
                    if item:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        
        except Exception as e:
            # Fallback to error message
            self.param_table.setRowCount(1)
            self.param_table.setItem(0, 0, QTableWidgetItem(f"Error: {str(e)}"))
            for col in range(1, 4):
                self.param_table.setItem(0, col, QTableWidgetItem(""))
                
    def update_plots(self):
        """Update all plots"""
        if HAS_MATPLOTLIB and self.results:
            selected_instrument = self.instrument_combo.currentText() if hasattr(self, 'instrument_combo') else None
            
            # Update corner plot (instrument-independent)
            #self.update_corner_plot()  # Uncommented per original
            
            # Update model comparison plot with selected instrument
            self.update_model_comparison_with_instrument(selected_instrument)
            
            # Update velocity plot with selected instrument  
            self.update_velocity_plot()



            
    def update_corner_plot(self):
        """Update corner plot with real MCMC samples"""
        if not HAS_MATPLOTLIB:
            return
            
        self.corner_figure.clear()
        
        if self.results is None:
            ax = self.corner_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No results to display', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            try:
                # Get MCMC samples
                if hasattr(self.results, 'get_samples'):
                    samples = self.results.get_samples()
                elif hasattr(self.results, 'fitter') and hasattr(self.results.fitter, 'samples'):
                    samples = self.results.fitter.samples
                else:
                    samples = None
                
                if samples is not None and len(samples) > 0:
                    # Get parameter names
                    if hasattr(self.results, 'parameter_summary'):
                        param_summary = self.results.parameter_summary(verbose=False)
                        param_names = param_summary.names
                    else:
                        # Generic names
                        n_params = samples.shape[1]
                        param_names = [f"θ_{i+1}" for i in range(n_params)]
                    
                    # Filter parameters based on selection
                    selected_params = self.param_selector.currentText()
                    if "N parameters" in selected_params:
                        indices = [i for i, name in enumerate(param_names) if name.startswith('N_')]
                    elif "b parameters" in selected_params:
                        indices = [i for i, name in enumerate(param_names) if name.startswith('b_')]
                    elif "v parameters" in selected_params:
                        indices = [i for i, name in enumerate(param_names) if name.startswith('v_')]
                    else:  # All parameters
                        indices = list(range(len(param_names)))
                    
                    if indices and HAS_CORNER:
                        # Create corner plot with real data
                        filtered_samples = samples[:, indices]
                        filtered_names = [param_names[i] for i in indices]
                        
                        corner.corner(filtered_samples, labels=filtered_names, 
                                    fig=self.corner_figure, 
                                    show_titles=True, title_kwargs={"fontsize": 10})
                    else:
                        # Fallback if corner not available
                        ax = self.corner_figure.add_subplot(111)
                        ax.text(0.5, 0.5, 'Corner plot package not available\nInstall with: pip install corner', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        ax.set_xticks([])
                        ax.set_yticks([])
                else:
                    ax = self.corner_figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'No MCMC samples available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
            except Exception as e:
                ax = self.corner_figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Error creating corner plot:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                
        self.corner_canvas.draw()
        
    def update_model_comparison(self):
        """Update model vs data comparison plot with real data"""
        if not HAS_MATPLOTLIB:
            return
            
        self.comparison_figure.clear()
        
        if self.results is None:
            ax = self.comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No results to display',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            try:
                # Get data and model from results
                fitter = self.results.fitter
                wave = fitter.wave_obs
                flux = fitter.fnorm
                error = fitter.enorm
                
                # Calculate best-fit model
                if hasattr(self.results, 'model'):
                    model_flux = self.results.model.evaluate(fitter.best_theta, wave)
                else:
                    model_flux = np.ones_like(flux)  # Fallback
                
                # Create subplots
                show_residuals = self.show_residuals_check.isChecked()
                if show_residuals:
                    ax1 = self.comparison_figure.add_subplot(211)
                    ax2 = self.comparison_figure.add_subplot(212, sharex=ax1)
                else:
                    ax1 = self.comparison_figure.add_subplot(111)
                    ax2 = None
                
                # Main plot: data vs model
                ax1.step(wave, flux, 'k-', where='mid', linewidth=1, alpha=0.8, label='Data')
                ax1.fill_between(wave, flux - error, flux + error, 
                               alpha=0.3, color='gray', label='Error')
                ax1.plot(wave, model_flux, 'r-', linewidth=2, label='Best-fit Model')
                
                # Add components if requested
                if self.show_components_check.isChecked():
                    # TODO: Add individual component plotting if model supports it
                    pass
                
                ax1.set_ylabel('Normalized Flux')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                ax1.set_title('Model vs Data Comparison')
                
                if not show_residuals:
                    ax1.set_xlabel('Wavelength (Å)')
                
                # Residuals subplot
                if show_residuals and ax2:
                    residuals = (flux - model_flux) / error
                    ax2.step(wave, residuals, 'k-', where='mid', linewidth=1, alpha=0.8)
                    ax2.axhline(0, color='r', linestyle='-', alpha=0.7)
                    ax2.fill_between(wave, -1, 1, alpha=0.3, color='gray', label='±1σ')
                    ax2.set_xlabel('Wavelength (Å)')
                    ax2.set_ylabel('Residuals (σ)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
            except Exception as e:
                ax = self.comparison_figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Error creating model plot:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw()
        
    def update_velocity_plot(self):
        """Update velocity space plot with real data"""
        if not HAS_MATPLOTLIB:
            return
            
        self.velocity_figure.clear()
        
        if self.results is None:
            ax = self.velocity_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No results to display',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            try:
                # Use FitResults velocity plotting method if available
                if hasattr(self.results, 'plot_velocity_fits'):
                    # Clear figure and use FitResults plotting
                    self.velocity_figure.clear()
                    
                    # Get velocity range setting
                    vel_range_text = self.vel_range_combo.currentText()
                    if vel_range_text == "±200 km/s":
                        velocity_range = (-200, 200)
                    elif vel_range_text == "±500 km/s":
                        velocity_range = (-500, 500)
                    elif vel_range_text == "±1000 km/s":
                        velocity_range = (-1000, 1000)
                    else:
                        velocity_range = None  # Auto
                    
                    # Create velocity plots
                    show_components = self.show_components_check.isChecked()
                    
                    # Call the FitResults plotting method
                    fig = self.results.plot_velocity_fits(
                        show_components=show_components,
                        velocity_range=velocity_range,
                        show_rail_system=True,
                        figure=self.velocity_figure
                    )
                else:
                    # Fallback: basic velocity plot
                    ax = self.velocity_figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'Velocity plotting not available in results object',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    
            except Exception as e:
                ax = self.velocity_figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Error creating velocity plot:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                
        self.velocity_canvas.draw()


    def update_model_comparison_with_instrument(self, instrument_name):
        """Update model comparison plot for specific instrument"""
        if not instrument_name:
            # Fallback to original method if no instrument selected
            self.update_model_comparison()
            return
            
        try:
            self.comparison_figure.clear()
            
            # For now, call the original method but add instrument info to title
            self.update_model_comparison()
            
            # Add instrument name to the plot title
            axes = self.comparison_figure.get_axes()
            if axes:
                current_title = axes[0].get_title()
                if instrument_name and instrument_name != "Primary":
                    axes[0].set_title(f"{current_title} ({instrument_name})")
                    
        except Exception as e:
            print(f"Error updating model plot for {instrument_name}: {e}")
            # Fallback to original method
            self.update_model_comparison()
    

    def update_velocity_plot_with_instrument(self, instrument_name):
        """Update velocity plot for specific instrument"""
        if not instrument_name:
            # Fallback to original method if no instrument selected
            self.update_velocity_plot()
            return
            
        try:
            # Try to use instrument-specific plotting if available
            if hasattr(self.results, 'plot_velocity_fits'):
                self.velocity_figure.clear()
                
                # Get velocity range setting
                vel_range_text = self.vel_range_combo.currentText()
                if vel_range_text == "±200 km/s":
                    velocity_range = (-200, 200)
                elif vel_range_text == "±500 km/s":
                    velocity_range = (-500, 500)
                elif vel_range_text == "±1000 km/s":
                    velocity_range = (-1000, 1000)
                else:
                    velocity_range = None
                    
                show_components = self.show_components_check.isChecked()
                
                # Try to pass instrument parameter if supported
                try:
                    fig = self.results.plot_velocity_fits(
                        show_components=show_components,
                        velocity_range=velocity_range,
                        show_rail_system=True,
                        instrument=instrument_name,  # Try to pass instrument
                        figure=self.velocity_figure
                    )
                except TypeError:
                    # If instrument parameter not supported, fall back
                    fig = self.results.plot_velocity_fits(
                        show_components=show_components,
                        velocity_range=velocity_range,
                        show_rail_system=True,
                        figure=self.velocity_figure
                    )
                    
                    # Add instrument info to title
                    axes = self.velocity_figure.get_axes()
                    if axes and instrument_name != "Primary":
                        current_title = axes[0].get_title()
                        axes[0].set_title(f"{current_title} ({instrument_name})")
                        
                self.velocity_canvas.draw()
            else:
                # Fallback to original method
                self.update_velocity_plot()
                
        except Exception as e:
            print(f"Error updating velocity plot for {instrument_name}: {e}")
            # Fallback to original method
            self.update_velocity_plot()
    
    
            
    def export_csv(self):
        """Export parameter table to CSV"""
        if not self.parameter_data:
            QMessageBox.warning(self, "No Data", "No parameter data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Parameters as CSV", "",
            "CSV files (*.csv);;All files (*.*)")
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Parameter', 'Best_Fit', 'Error', 'Units'])
                    writer.writerows(self.parameter_data)
                    
                self.main_window.update_status(f"Parameters exported: {filename}")
                QMessageBox.information(self, "Export Complete", 
                                      f"Parameters exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")
                
    def export_latex(self):
        """Export parameter table to LaTeX"""
        if not self.parameter_data:
            QMessageBox.warning(self, "No Data", "No parameter data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Parameters as LaTeX", "",
            "LaTeX files (*.tex);;All files (*.*)")
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("\\begin{table}[ht]\n")
                    f.write("\\centering\n")
                    f.write("\\begin{tabular}{llll}\n")
                    f.write("\\hline\n")
                    f.write("Parameter & Best Fit & Error & Units \\\\\n")
                    f.write("\\hline\n")
                    
                    for param, value, error, units in self.parameter_data:
                        # Escape underscores for LaTeX
                        param_latex = param.replace('_', '\\_')
                        f.write(f"{param_latex} & {value} & {error} & {units} \\\\\n")
                    
                    f.write("\\hline\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\caption{rbvfit Parameter Results}\n")
                    f.write("\\label{tab:rbvfit_results}\n")
                    f.write("\\end{table}\n")
                    
                self.main_window.update_status(f"LaTeX table exported: {filename}")
                QMessageBox.information(self, "Export Complete", 
                                      f"LaTeX table exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export LaTeX:\n{str(e)}")
                
    def save_plots(self):
        """Save all plots to files"""
        if self.results is None:
            QMessageBox.warning(self, "No Results", "No results to save")
            return
            
        # Get directory to save in
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Plots")
        if not directory:
            return
            
        try:
            saved_files = []
            
            # Save corner plot
            if HAS_MATPLOTLIB:
                corner_file = f"{directory}/corner_plot.png"
                self.corner_figure.savefig(corner_file, dpi=300, bbox_inches='tight')
                saved_files.append("corner_plot.png")
                
                # Save model comparison
                comparison_file = f"{directory}/model_comparison.png"
                self.comparison_figure.savefig(comparison_file, dpi=300, bbox_inches='tight')
                saved_files.append("model_comparison.png")
                
                # Save velocity plot
                velocity_file = f"{directory}/velocity_plot.png"
                self.velocity_figure.savefig(velocity_file, dpi=300, bbox_inches='tight')
                saved_files.append("velocity_plot.png")
                
            self.main_window.update_status(f"Plots saved to {directory}")
            QMessageBox.information(self, "Save Complete", 
                                  f"Saved plots:\n" + "\n".join(saved_files))
                                  
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save plots:\n{str(e)}")
            
    def export_corner_plot(self):
        """Export corner plot to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export corner plot", "",
            "PNG files (*.png);;PDF files (*.pdf);;All files (*.*)")
        
        if filename:
            try:
                self.corner_figure.savefig(filename, dpi=300, bbox_inches='tight')
                self.main_window.update_status(f"Corner plot saved: {filename}")
                QMessageBox.information(self, "Export Complete", 
                                      f"Corner plot saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
                
    def copy_table_to_clipboard(self):
        """Copy parameter table to clipboard"""
        if not self.parameter_data:
            QMessageBox.warning(self, "No Data", "No parameter data to copy")
            return
            
        try:
            # Create tab-separated text
            text = "Parameter\tBest Fit\t±Error\tUnits\n"
            for param_data in self.parameter_data:
                text += "\t".join(param_data) + "\n"
                
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            
            self.main_window.update_status("Parameter table copied to clipboard")
            QMessageBox.information(self, "Copy Complete", 
                                  "Parameter table copied to clipboard")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy table:\n{str(e)}")
            
    def export_parameter_table(self):
        """Export parameter table to file"""
        if not self.parameter_data:
            QMessageBox.warning(self, "No Data", "No parameter data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export parameter table", "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)")
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Parameter', 'Best_Fit', 'Error', 'Units'])
                        writer.writerows(self.parameter_data)
                else:
                    with open(filename, 'w') as f:
                        f.write("Parameter\tBest_Fit\tError\tUnits\n")
                        for param_data in self.parameter_data:
                            f.write("\t".join(param_data) + "\n")
                        
                self.main_window.update_status(f"Parameter table exported: {filename}")
                QMessageBox.information(self, "Export Complete", 
                                      f"Parameter table exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export table:\n{str(e)}")
                
    def refresh(self):
        """Refresh the results display"""
        if self.results:
            self.update_plots()