#!/usr/bin/env python
"""
rbvfit 2.0 Results Tab - Updated with Real Data Integration

Interface for viewing and exporting fit results with actual MCMC data.
"""

import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QPushButton, QTableWidget, QTableWidgetItem,
                            QTextEdit, QTabWidget, QFileDialog, QMessageBox,
                            QHeaderView, QLabel, QComboBox, QCheckBox, QApplication,
                            QDialog, QLineEdit)
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

from rbvfit.gui import plotting_gui


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

        # Load previously saved results
        self.load_results_btn = QPushButton("Load Results")
        self.load_results_btn.setToolTip("Load fit results from a previously saved HDF5 file")
        control_layout.addWidget(self.load_results_btn)

        # NEW: Add trace plots button
        self.trace_plots_btn = QPushButton("Show Trace Plots")
        self.trace_plots_btn.setEnabled(False)
        self.trace_plots_btn.setToolTip("Display MCMC chain trace plots for convergence assessment")
        control_layout.addWidget(self.trace_plots_btn)
        
        # Burn-in controls
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Burn-in fraction:"))
        self.burnin_edit = QLineEdit()
        self.burnin_edit.setFixedWidth(60)
        self.burnin_edit.setPlaceholderText("0.20")
        self.burnin_edit.setToolTip("Fraction of MCMC chain to discard as burn-in (0.0 – 0.9)")
        control_layout.addWidget(self.burnin_edit)

        self.apply_burnin_btn = QPushButton("Apply")
        self.apply_burnin_btn.setEnabled(False)
        self.apply_burnin_btn.setToolTip("Re-compute statistics and plots with the new burn-in fraction")
        control_layout.addWidget(self.apply_burnin_btn)

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

        # Velocity plot ranges (persisted across redraws)
        self.velocity_plot_ranges = {
            'vel_min': -600, 'vel_max': 600,
            'y_min': -0.02, 'y_max': 1.5
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
        
        self.set_velocity_ranges_btn = QPushButton("Set Plot Range")
        self.set_velocity_ranges_btn.setToolTip("Set custom velocity and flux plot ranges")
        vel_controls.addWidget(self.set_velocity_ranges_btn)
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
        # Burn-in
        self.apply_burnin_btn.clicked.connect(self.apply_burnin)

        # Export buttons
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_latex_btn.clicked.connect(self.export_latex)
        self.save_plots_btn.clicked.connect(self.save_plots)
        self.trace_plots_btn.clicked.connect(self.show_trace_plots)
        self.save_results_btn.clicked.connect(self.save_results)
        self.load_results_btn.clicked.connect(self.load_results)

        # Plot controls
        self.show_components_check.toggled.connect(self.update_plots)
        self.show_residuals_check.toggled.connect(self.update_plots)
        self.param_selector.currentTextChanged.connect(self.update_corner_plot)
        self.set_velocity_ranges_btn.clicked.connect(self.set_velocity_plot_ranges)
        self.set_ranges_btn.clicked.connect(self.set_plot_ranges)
        self.instrument_combo.currentTextChanged.connect(self.on_instrument_changed)

        
        # Corner plot controls
        self.update_corner_btn.clicked.connect(self.update_corner_plot)
        self.export_corner_btn.clicked.connect(self.export_corner_plot)
        
        # Table controls
        self.copy_table_btn.clicked.connect(self.copy_table_to_clipboard)
        self.export_table_btn.clicked.connect(self.export_parameter_table)



    def set_velocity_plot_ranges(self):
        """Set plot ranges specifically for velocity plots"""

        # Always initialize from stored ranges (not axes, which are tick-extended)
        current_xlim = (self.velocity_plot_ranges['vel_min'], self.velocity_plot_ranges['vel_max'])
        current_ylim = (self.velocity_plot_ranges['y_min'], self.velocity_plot_ranges['y_max'])
        original_xlim = (-600, 600)
        original_ylim = (-0.02, 1.5)

        dialog = PlotRangeDialog(
            current_xlim=current_xlim,
            current_ylim=current_ylim,
            original_xlim=original_xlim,
            original_ylim=original_ylim,
            parent=self
        )
        
        # Use blocking dialog
        if dialog.exec_() == QDialog.Accepted:
            xlim, ylim = dialog.get_ranges()

            if xlim:
                self.velocity_plot_ranges['vel_min'] = xlim[0]
                self.velocity_plot_ranges['vel_max'] = xlim[1]
            if ylim:
                self.velocity_plot_ranges['y_min'] = ylim[0]
                self.velocity_plot_ranges['y_max'] = ylim[1]

            self.update_velocity_plot()  # Refresh plot with persisted ranges
    
        

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

        # Show current burn-in fraction in the textbox
        try:
            burnin_frac = results.burnin_steps / results.n_steps if results.n_steps > 0 else 0.2
            self.burnin_edit.setText(f"{burnin_frac:.2f}")
        except Exception:
            self.burnin_edit.setText("0.20")
        self.apply_burnin_btn.setEnabled(True)

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
        self.trace_plots_btn.setEnabled(False)
        self.update_corner_btn.setEnabled(False)
        self.export_corner_btn.setEnabled(False)
        self.copy_table_btn.setEnabled(False)
        self.export_table_btn.setEnabled(False)
        self.save_results_btn.setEnabled(False)
        self.apply_burnin_btn.setEnabled(False)
        self.burnin_edit.clear()

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
            
    def apply_burnin(self):
        """Re-slice MCMC chain with user-specified burn-in fraction and refresh all displays."""
        if self.results is None:
            return

        text = self.burnin_edit.text().strip()
        try:
            frac = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                f"'{text}' is not a valid number. Enter a fraction between 0.0 and 0.9.")
            return

        if not (0.0 <= frac < 0.9):
            QMessageBox.warning(self, "Out of Range",
                                "Burn-in fraction must be between 0.0 and 0.9.")
            return

        try:
            chain = self.results.chain  # (n_steps, n_walkers, n_params)
            n_steps = chain.shape[0]
            burnin_steps = int(frac * n_steps)

            # Reslice the chain
            post_chain = chain[burnin_steps:]
            self.results.samples = post_chain.reshape(-1, chain.shape[-1])
            self.results.burnin_steps = burnin_steps

            # Refresh all displays
            self.update_statistics()
            self.update_parameter_table()
            if HAS_MATPLOTLIB:
                self.update_plots()

            self.main_window.update_status(
                f"Burn-in updated: {burnin_steps} steps ({frac:.0%}) discarded, "
                f"{len(self.results.samples):,} samples remaining"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply burn-in:\n{str(e)}")

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
    
    def load_results(self):
        """Load fit results from a previously saved HDF5 file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Fit Results",
            "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )

        if not filename:
            return

        try:
            from rbvfit.core.unified_results import UnifiedResults
            results = UnifiedResults.load(filename)

            # Pass into the results tab and enable the Results tab in main window
            self.update_results(results)
            self.main_window.tab_widget.setTabEnabled(3, True)
            self.main_window.update_status(f"Results loaded from: {Path(filename).name}")

            QMessageBox.information(
                self,
                "Load Successful",
                f"Results loaded from:\n{filename}\n\n"
                f"Instruments: {list(results.instrument_data.keys())}\n"
                f"Samples: {len(results.samples):,}\n"
                f"Parameters: {len(results.best_fit)}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to load results:\n{str(e)}")

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
        """Update fit statistics display with UnifiedResults data"""
        self.stats_text.clear()
        
        if self.results is None:
            self.stats_text.append("No results available")
            return
            
        try:
            # Get basic info from UnifiedResults
            n_samples = len(self.results.samples)
            n_params = len(self.results.best_fit)
            n_data_points = sum(len(data['wave']) for data in self.results.instrument_data.values())
            
            # Calculate chi-squared
            try:
                chi2 = self.results.chi_squared()
                chi2_reduced = chi2['reduced_chi2']
            except Exception:
                chi2_reduced = None
            
            # Get convergence diagnostics
            c = self.results.convergence_diagnostics(verbose=False)
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
                
        except Exception as e:
            # Fallback if convergence diagnostics fail
            self.stats_text.append(f"Error extracting statistics: {str(e)}")
            return
        
        # Color code convergence status
        if conv_status == "GOOD":
            conv_color = "green"
        elif conv_status == "MARGINAL":
            conv_color = "orange"
        elif conv_status == "POOR":
            conv_color = "red"
        else:
            conv_color = "gray"
        
        # Generate HTML stats display
        stats_html = f"""
        <h3>Fit Statistics</h3>
        <table border="1" cellpadding="4" style="border-collapse: collapse;">
        <tr><td><b>Convergence</b></td><td style="color: {conv_color}; font-weight:     bold;">{conv_status}</td></tr>
        <tr><td><b>Samples</b></td><td>{n_samples:,}</td></tr>
        <tr><td><b>Parameters</b></td><td>{n_params}</td></tr>
        <tr><td><b>Data Points</b></td><td>{n_data_points:,}</td></tr>
        <tr><td><b>Acceptance Rate</b></td><td>{acceptance_rate:.3f}</td></tr>
        """
        
        if chi2_reduced is not None:
            stats_html += f"<tr><td><b>χ²/ν</b></td><td>{chi2_reduced:.3f}</td></tr>"
        else:
            stats_html += f"<tr><td><b>χ²/ν</b></td><td>N/A</td></tr>"
        
        if rhat_max is not None:
            stats_html += f"<tr><td><b>Max R-hat</b></td><td>{rhat_max:.3f}</td></tr>"
        
        if min_eff_size is not None:
            stats_html += f"<tr><td><b>Min N_eff</b></td><td>{min_eff_size:.0f}</td></tr>"
        
        if mean_autocorr is not None:
            stats_html += f"<tr><td><b>Mean τ</b></td><td>{mean_autocorr:.1f}</td></tr>"
        
        stats_html += "</table>"
        
        self.stats_text.setHtml(stats_html)    

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
                param_summary = self.results.parameter_summary()
                
                # Populate table with real parameters
                for i, name in enumerate(param_summary.names):
                    best_fit = param_summary.best_fit[i]
                    error = param_summary.errors[i]
                    
                    # Determine units based on parameter type
                    if name.startswith('logN_'):
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
        if not HAS_MATPLOTLIB or self.results is None:
            return
        
        # Get parameter filter
        param_filter = self.param_selector.currentText()
        if "N parameters" in param_filter:
            filter_type = "N"
        elif "b parameters" in param_filter:
            filter_type = "b"
        elif "v parameters" in param_filter:
            filter_type = "v"
        else:
            filter_type = "all"
        
        # Use plotting_gui function
        plotting_gui.plot_corner_custom(self.corner_figure, self.results, filter_type)
        self.corner_canvas.draw()

    def update_model_comparison(self):
        """Update model vs data comparison plot"""
        if not HAS_MATPLOTLIB or self.results is None:
            return
        
        show_components = self.show_components_check.isChecked()
        show_residuals = self.show_residuals_check.isChecked()
        selected_instrument = self.instrument_combo.currentText()

        
        # Use plotting_gui function
        plotting_gui.plot_model_comparison_custom(
            self.comparison_figure, 
            self.results, 
            show_components, 
            show_residuals,
            self.plot_ranges,
            selected_instrument
        )
        self.comparison_canvas.draw()


    def update_velocity_plot(self):
        """Update velocity space plot using persisted ranges"""
        if not HAS_MATPLOTLIB or self.results is None:
            return

        velocity_range = (self.velocity_plot_ranges['vel_min'], self.velocity_plot_ranges['vel_max'])
        yrange = (self.velocity_plot_ranges['y_min'], self.velocity_plot_ranges['y_max'])
        
        show_components = self.show_components_check.isChecked()
        selected_instrument = self.instrument_combo.currentText()
        
        # Use simplified plotting function
        plotting_gui.plot_velocity_space_custom(
            self.velocity_figure,
            self.results,
            velocity_range,
            show_components,
            True,  # show_rail
            selected_instrument,
            yrange=yrange
        )
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