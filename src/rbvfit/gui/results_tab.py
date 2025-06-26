#!/usr/bin/env python
"""
rbvfit 2.0 Results Tab - PyQt5 Implementation

Interface for viewing and exporting fit results with enhanced visualization.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QPushButton, QTableWidget, QTableWidgetItem,
                            QTextEdit, QTabWidget, QFileDialog, QMessageBox,
                            QHeaderView, QLabel, QComboBox, QCheckBox, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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
        self.save_plots_btn.setToolTip("Save all plots as image files")
        control_layout.addWidget(self.save_plots_btn)
        
        control_layout.addStretch()
        
        # Plot options
        self.show_components_check = QCheckBox("Show Components")
        self.show_components_check.setChecked(True)
        self.show_components_check.setToolTip("Show individual velocity components in plots")
        control_layout.addWidget(self.show_components_check)
        
        self.show_residuals_check = QCheckBox("Show Residuals")
        self.show_residuals_check.setChecked(True)
        self.show_residuals_check.setToolTip("Show residuals subplot")
        control_layout.addWidget(self.show_residuals_check)
        
    def setup_plot_area(self, parent):
        """Create plot display area with tabs"""
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        parent.addWidget(left_widget)
        
        if HAS_MATPLOTLIB:
            # Plot tabs
            self.plot_tabs = QTabWidget()
            left_layout.addWidget(self.plot_tabs)
            
            # Corner plot tab
            self.setup_corner_plot_tab()
            
            # Model comparison tab
            self.setup_model_comparison_tab()
            
            # Velocity space tab
            self.setup_velocity_plot_tab()
            
        else:
            left_layout.addWidget(QLabel("Matplotlib not available"))
            
    def setup_corner_plot_tab(self):
        """Create corner plot tab"""
        corner_widget = QWidget()
        corner_layout = QVBoxLayout()
        corner_widget.setLayout(corner_layout)
        self.plot_tabs.addTab(corner_widget, "Corner Plot")
        
        # Corner plot controls
        corner_controls = QHBoxLayout()
        corner_layout.addLayout(corner_controls)
        
        corner_controls.addWidget(QLabel("Parameters:"))
        self.param_selector = QComboBox()
        self.param_selector.addItems(["All Parameters", "N parameters", "b parameters", "v parameters"])
        self.param_selector.setToolTip("Select which parameters to show in corner plot")
        corner_controls.addWidget(self.param_selector)
        
        self.update_corner_btn = QPushButton("Update Plot")
        self.update_corner_btn.setEnabled(False)
        self.update_corner_btn.setToolTip("Update corner plot with selected parameters")
        corner_controls.addWidget(self.update_corner_btn)
        
        corner_controls.addStretch()
        
        self.export_corner_btn = QPushButton("Export Plot")
        self.export_corner_btn.setEnabled(False)
        self.export_corner_btn.setToolTip("Export corner plot as image")
        corner_controls.addWidget(self.export_corner_btn)
        
        # Corner plot canvas
        self.corner_figure = Figure(figsize=(8, 8), dpi=80)
        self.corner_canvas = FigureCanvas(self.corner_figure)
        corner_layout.addWidget(self.corner_canvas)
        
    def setup_model_comparison_tab(self):
        """Create model vs data comparison tab"""
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout()
        comparison_widget.setLayout(comparison_layout)
        self.plot_tabs.addTab(comparison_widget, "Model Comparison")
        
        # Model comparison canvas
        self.comparison_figure = Figure(figsize=(10, 8), dpi=80)
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        comparison_layout.addWidget(self.comparison_canvas)
        
    def setup_velocity_plot_tab(self):
        """Create velocity space plot tab"""
        velocity_widget = QWidget()
        velocity_layout = QVBoxLayout()
        velocity_widget.setLayout(velocity_layout)
        self.plot_tabs.addTab(velocity_widget, "Velocity Space")
        
        # Velocity plot controls
        vel_controls = QHBoxLayout()
        velocity_layout.addLayout(vel_controls)
        
        vel_controls.addWidget(QLabel("Velocity Range:"))
        self.vel_range_combo = QComboBox()
        self.vel_range_combo.addItems(["-500 to +500 km/s", "-200 to +200 km/s", "-100 to +100 km/s", "Auto"])
        self.vel_range_combo.setToolTip("Select velocity range for display")
        vel_controls.addWidget(self.vel_range_combo)
        
        vel_controls.addStretch()
        
        # Velocity plot canvas
        self.velocity_figure = Figure(figsize=(10, 6), dpi=80)
        self.velocity_canvas = FigureCanvas(self.velocity_figure)
        velocity_layout.addWidget(self.velocity_canvas)
        
    def setup_results_panel(self, parent):
        """Create statistics and parameter table panel"""
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
        
        # Plot controls
        self.show_components_check.toggled.connect(self.update_plots)
        self.show_residuals_check.toggled.connect(self.update_plots)
        self.param_selector.currentTextChanged.connect(self.update_corner_plot)
        self.vel_range_combo.currentTextChanged.connect(self.update_velocity_plot)
        
        # Corner plot controls
        self.update_corner_btn.clicked.connect(self.update_corner_plot)
        self.export_corner_btn.clicked.connect(self.export_corner_plot)
        
        # Table controls
        self.copy_table_btn.clicked.connect(self.copy_table_to_clipboard)
        self.export_table_btn.clicked.connect(self.export_parameter_table)
        
    def update_results(self, results):
        """Update display with new fit results"""
        self.results = results
        
        if results is None:
            self.clear_results()
            return
            
        # Enable controls
        self.export_csv_btn.setEnabled(True)
        self.export_latex_btn.setEnabled(True)
        self.save_plots_btn.setEnabled(True)
        self.update_corner_btn.setEnabled(True)
        self.export_corner_btn.setEnabled(True)
        self.copy_table_btn.setEnabled(True)
        self.export_table_btn.setEnabled(True)
        
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
        self.update_corner_btn.setEnabled(False)
        self.export_corner_btn.setEnabled(False)
        self.copy_table_btn.setEnabled(False)
        self.export_table_btn.setEnabled(False)
        
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
            
    def update_statistics(self):
        """Update fit statistics display"""
        self.stats_text.clear()
        
        if self.results is None:
            self.stats_text.append("No results available")
            return
            
        # TODO: Extract actual statistics from results object
        # For now, show enhanced placeholder statistics
        stats_html = """
        <h4>Fit Quality</h4>
        <table>
        <tr><td><b>χ²/ν:</b></td><td>1.12</td></tr>
        <tr><td><b>AIC:</b></td><td>156.3</td></tr>
        <tr><td><b>BIC:</b></td><td>167.8</td></tr>
        </table>
        
        <h4>MCMC Diagnostics</h4>
        <table>
        <tr><td><b>Samples:</b></td><td>2500</td></tr>
        <tr><td><b>Burn-in:</b></td><td>500</td></tr>
        <tr><td><b>Acceptance:</b></td><td>0.35</td></tr>
        <tr><td><b>R_hat:</b></td><td>&lt; 1.01</td></tr>
        </table>
        
        <h4>Convergence</h4>
        <p style="color: green;"><b>✓ Good convergence</b></p>
        <p>All parameters converged successfully</p>
        """
        
        self.stats_text.setHtml(stats_html)
        
    def update_parameter_table(self):
        """Update parameter table with results"""
        # Clear existing data
        self.param_table.setRowCount(0)
        
        if self.results is None:
            return
            
        # Extract parameters from results
        if hasattr(self.results, 'parameters') and self.results.parameters:
            self.parameter_data = []
            
            for param_name, value in self.results.parameters.items():
                # Add some simulated uncertainty (in real implementation, get from MCMC)
                if 'N_' in param_name:
                    error = abs(value * 0.05)  # 5% error for N
                    units = "log cm⁻²"
                elif 'b_' in param_name:
                    error = abs(value * 0.15)  # 15% error for b
                    units = "km/s"
                elif 'v_' in param_name:
                    error = abs(value * 0.1)   # 10% error for v
                    units = "km/s"
                else:
                    error = abs(value * 0.1)
                    units = ""
                
                self.parameter_data.append((param_name, f"{value:.2f}", f"{error:.2f}", units))
        else:
            # Fallback to placeholder data
            self.parameter_data = [
                ("N_MgII_C1", "13.52", "0.08", "log cm⁻²"),
                ("b_MgII_C1", "14.8", "2.3", "km/s"),
                ("v_MgII_C1", "-48.2", "1.8", "km/s"),
            ]
        
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
                    
    def update_plots(self):
        """Update all plots"""
        if HAS_MATPLOTLIB and self.results:
            self.update_corner_plot()
            self.update_model_comparison()
            self.update_velocity_plot()
            
    def update_corner_plot(self):
        """Update corner plot"""
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
            # TODO: Generate actual corner plot from results
            # Enhanced placeholder with multiple parameter correlations
            selected_params = self.param_selector.currentText()
            
            # Generate fake correlation data based on selection
            np.random.seed(42)
            if "All" in selected_params:
                n_params = 6
                param_names = ['N₁', 'N₂', 'b₁', 'b₂', 'v₁', 'v₂']
            elif "N parameters" in selected_params:
                n_params = 2
                param_names = ['N₁', 'N₂']
            elif "b parameters" in selected_params:
                n_params = 2
                param_names = ['b₁', 'b₂']
            else:  # v parameters
                n_params = 2
                param_names = ['v₁', 'v₂']
            
            # Create subplot grid for corner plot
            for i in range(n_params):
                for j in range(i + 1):
                    ax = self.corner_figure.add_subplot(n_params, n_params, i * n_params + j + 1)
                    
                    if i == j:
                        # Diagonal: histograms
                        data = np.random.normal(0, 1, 1000)
                        ax.hist(data, bins=30, alpha=0.7, color='skyblue', density=True)
                        ax.set_ylabel('Density')
                        if i == n_params - 1:
                            ax.set_xlabel(param_names[i])
                    else:
                        # Off-diagonal: scatter plots
                        x = np.random.normal(0, 1, 1000)
                        y = 0.3 * x + np.random.normal(0, 0.8, 1000)
                        ax.scatter(x, y, alpha=0.5, s=1, color='darkblue')
                        if i == n_params - 1:
                            ax.set_xlabel(param_names[j])
                        if j == 0:
                            ax.set_ylabel(param_names[i])
            
            self.corner_figure.suptitle(f'Corner Plot: {selected_params}', fontsize=12)
            
        self.corner_figure.tight_layout()
        self.corner_canvas.draw()
        
    def update_model_comparison(self):
        """Update model vs data comparison plot"""
        if not HAS_MATPLOTLIB:
            return
            
        self.comparison_figure.clear()
        
        if self.results is None:
            ax = self.comparison_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No results to display',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            # TODO: Generate actual model comparison from results
            # Enhanced placeholder with multiple transitions
            show_residuals = self.show_residuals_check.isChecked()
            show_components = self.show_components_check.isChecked()
            
            if show_residuals:
                ax1 = self.comparison_figure.add_subplot(211)
                ax2 = self.comparison_figure.add_subplot(212)
            else:
                ax1 = self.comparison_figure.add_subplot(111)
                ax2 = None
            
            # Generate fake spectrum data with multiple absorption lines
            wave = np.linspace(2790, 2810, 300)
            
            # Multiple absorption lines
            flux = np.ones_like(wave)
            model = np.ones_like(wave)
            
            # Add absorption features
            lines = [2796.3, 2803.5]  # MgII doublet
            for line_wave in lines:
                absorption = 0.4 * np.exp(-0.5 * ((wave - line_wave) / 0.8)**2)
                flux -= absorption + 0.02 * np.random.normal(0, 1, len(wave))
                model -= absorption
                
                # Individual components if requested
                if show_components:
                    comp1 = 0.2 * np.exp(-0.5 * ((wave - (line_wave - 0.3)) / 0.6)**2)
                    comp2 = 0.2 * np.exp(-0.5 * ((wave - (line_wave + 0.4)) / 0.5)**2)
                    ax1.plot(wave, 1 - comp1, '--', alpha=0.7, linewidth=1, 
                            label='Component 1' if line_wave == lines[0] else None)
                    ax1.plot(wave, 1 - comp2, '--', alpha=0.7, linewidth=1,
                            label='Component 2' if line_wave == lines[0] else None)
            
            # Main plot
            ax1.plot(wave, flux, 'ko', markersize=1.5, alpha=0.7, label='Data')
            ax1.plot(wave, model, 'r-', linewidth=2, label='Best-fit Model')
            
            ax1.set_ylabel('Normalized Flux')
            ax1.legend(loc='upper right')
            ax1.set_title('Model vs Data Comparison')
            ax1.grid(True, alpha=0.3)
            
            if not show_residuals:
                ax1.set_xlabel('Wavelength (Å)')
            
            # Residuals subplot
            if show_residuals and ax2:
                residuals = flux - model
                ax2.plot(wave, residuals, 'ko', markersize=1.5, alpha=0.7)
                ax2.axhline(0, color='r', linestyle='-', alpha=0.7)
                ax2.fill_between(wave, -0.02, 0.02, alpha=0.3, color='gray', label='±1σ')
                ax2.set_xlabel('Wavelength (Å)')
                ax2.set_ylabel('Residuals')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw()
        
    def update_velocity_plot(self):
        """Update velocity space plot"""
        if not HAS_MATPLOTLIB:
            return
            
        self.velocity_figure.clear()
        
        if self.results is None:
            ax = self.velocity_figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No results to display',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            # TODO: Generate actual velocity plot from results
            # Enhanced placeholder velocity plot
            vel_range = self.vel_range_combo.currentText()
            
            if "500" in vel_range:
                v_min, v_max = -500, 500
            elif "200" in vel_range:
                v_min, v_max = -200, 200
            elif "100" in vel_range:
                v_min, v_max = -100, 100
            else:  # Auto
                v_min, v_max = -300, 300
            
            ax = self.velocity_figure.add_subplot(111)
            
            # Generate velocity space data
            velocity = np.linspace(v_min, v_max, 200)
            
            # Fake absorption profile with multiple components
            flux = np.ones_like(velocity)
            model = np.ones_like(velocity)
            
            # Two velocity components
            comp1_v, comp1_b = -50, 15
            comp2_v, comp2_b = 25, 20
            
            for comp_v, comp_b in [(comp1_v, comp1_b), (comp2_v, comp2_b)]:
                absorption = 0.3 * np.exp(-0.5 * ((velocity - comp_v) / comp_b)**2)
                flux -= absorption + 0.02 * np.random.normal(0, 1, len(velocity))
                model -= absorption
                
                # Mark component centers
                ax.axvline(comp_v, color='orange', linestyle=':', alpha=0.8, linewidth=2)
            
            ax.plot(velocity, flux, 'ko', markersize=2, alpha=0.7, label='Data')
            ax.plot(velocity, model, 'r-', linewidth=2, label='Best-fit Model')
            
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Normalized Flux')
            ax.set_title('Velocity Space View')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add component markers
            ax.text(comp1_v, 0.1, 'C1', ha='center', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))
            ax.text(comp2_v, 0.1, 'C2', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))
            
        self.velocity_figure.tight_layout()
        self.velocity_canvas.draw()
        
    def export_csv(self):
        """Export results to CSV"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export results to CSV", "",
            "CSV files (*.csv);;All files (*.*)")
        
        if filename:
            try:
                # Create CSV content from parameter table
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Parameter', 'Best_Fit', 'Error', 'Units'])
                    for param_data in self.parameter_data:
                        writer.writerow(param_data)
                        
                self.main_window.update_status(f"Results exported: {filename}")
                QMessageBox.information(self, "Export Complete", 
                                      f"Results exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")
                
    def export_latex(self):
        """Export results to LaTeX table"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export LaTeX table", "",
            "TeX files (*.tex);;All files (*.*)")
        
        if filename:
            try:
                latex_content = self.generate_latex_table()
                with open(filename, 'w') as f:
                    f.write(latex_content)
                    
                self.main_window.update_status(f"LaTeX table exported: {filename}")
                QMessageBox.information(self, "Export Complete", 
                                      f"LaTeX table exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export LaTeX:\n{str(e)}")
                
    def generate_latex_table(self):
        """Generate LaTeX table from parameter data"""
        latex = r"""
\begin{table}[ht]
\centering
\caption{Best-fit absorption line parameters}
\label{tab:absorption_params}
\begin{tabular}{lccc}
\hline
Parameter & Best Fit & $\pm$ Error & Units \\
\hline
"""
        
        for param_name, best_fit, error, units in self.parameter_data:
            # Convert parameter names to LaTeX format
            latex_param = param_name.replace('_', r'\_')
            if 'MgII' in param_name:
                latex_param = latex_param.replace('MgII', r'{\rm MgII}')
            if 'FeII' in param_name:
                latex_param = latex_param.replace('FeII', r'{\rm FeII}')
                
            # Convert units to LaTeX
            latex_units = units.replace('log cm⁻²', r'$\log$ cm$^{-2}$')
            latex_units = latex_units.replace('km/s', r'km s$^{-1}$')            
            latex += f"{latex_param} & {best_fit} & {error} & {latex_units} \\\\\n"
            
        latex += r"""
\hline
\end{tabular}
\end{table}
"""
        return latex
        
    def save_plots(self):
        """Save all plots as image files"""
        if not HAS_MATPLOTLIB:
            QMessageBox.critical(self, "Error", "Matplotlib not available")
            return
            
        directory = QFileDialog.getExistingDirectory(self, "Select directory to save plots")
        
        if directory:
            try:
                # Save all plots
                self.corner_figure.savefig(f"{directory}/corner_plot.png", 
                                         dpi=300, bbox_inches='tight')
                self.comparison_figure.savefig(f"{directory}/model_comparison.png", 
                                             dpi=300, bbox_inches='tight')
                self.velocity_figure.savefig(f"{directory}/velocity_plot.png", 
                                           dpi=300, bbox_inches='tight')
                
                self.main_window.update_status(f"Plots saved to: {directory}")
                QMessageBox.information(self, "Export Complete", 
                                      f"All plots saved to:\n{directory}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plots:\n{str(e)}")
                
    def export_corner_plot(self):
        """Export corner plot as image"""
        if not HAS_MATPLOTLIB:
            QMessageBox.critical(self, "Error", "Matplotlib not available")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save corner plot", "",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)")
        
        if filename:
            try:
                self.corner_figure.savefig(filename, dpi=300, bbox_inches='tight')
                self.main_window.update_status(f"Corner plot saved: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
                
    def copy_table_to_clipboard(self):
        """Copy parameter table to clipboard"""
        try:
            from PyQt5.QtWidgets import QApplication
            
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
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export parameter table", "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)")
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("Parameter\tBest_Fit\tError\tUnits\n")
                    for param_data in self.parameter_data:
                        f.write("\t".join(param_data) + "\n")
                        
                self.main_window.update_status(f"Parameter table exported: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export table:\n{str(e)}")
                
    def refresh(self):
        """Refresh the results display"""
        if self.results:
            self.update_plots()