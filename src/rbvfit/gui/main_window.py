#!/usr/bin/env python
"""
Updated rbvfit 2.0 Main GUI Window - Enhanced for Unified vfit Interface

Key Changes:
- Streamlined signal flow for unified vfit interface
- Better error handling and status reporting
- Improved project save/load functionality
- Clean separation of concerns between tabs
- Command-line argument support for loading config files
"""
import json
import datetime
import rbvfit as v
import sys
from pathlib import Path
import pandas as pd
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QWidget, QMenuBar, QStatusBar, QFileDialog,
                            QMessageBox, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QIcon

sys.path.append(str(Path(__file__).parent.parent))

from rbvfit.gui.config_data_tab import ConfigurationDataTab
from rbvfit.gui.model_setup_tab import ModelSetupTab
from rbvfit.gui.fitting_tab import FittingTab  
from rbvfit.gui.results_tab import ResultsTab
from rbvfit.gui import io


class GuiSignals(QObject):
    """Central signal hub for inter-tab communication"""
    configurations_updated = pyqtSignal(dict)  # From Tab 1 to Tab 2
    model_updated = pyqtSignal(dict, dict, dict)  # From Tab 2 to Tab 3: (instrument_data, theta_dict, bounds)
    fitting_completed = pyqtSignal(object)  # From Tab 3 to Tab 4
    results_updated = pyqtSignal(object)  # For any results updates


class UpdatedRbvfitGUI(QMainWindow):
    """Updated main GUI window with unified vfit interface support"""
    
    def __init__(self, config_file=None):
        super().__init__()
        self.setWindowTitle("rbvfit - Voigt Profile Fitting")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data storage
        self.configurations = {}
        self.instrument_data = None
        self.theta = None
        self.bounds = None
        self.fit_results = None
        
        # Signal hub
        self.signals = GuiSignals()
        
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
        
        # Load config file if provided
        if config_file:
            self.load_config_file_on_startup(config_file)
        
    def load_config_file_on_startup(self, config_file):
        """Load config file provided via command line"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                # Use a QTimer to load after the GUI is fully initialized
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(100, lambda: self.load_project_file(str(config_path)))
            else:
                self.status_bar.showMessage(f"Warning: Config file not found: {config_file}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading config file: {e}")
    
    def load_project_file(self, filename):
        """Load project file (used by startup loader)"""
        try:
            # Load and validate using io
            project_data = io.load_project_data(filename)
            
            # Clear current state
            self._clear_all_tabs()
            
            # Restore configurations with auto-loading
            configs_data = project_data.get('configurations', {})
            restored_configs, missing_files = io.deserialize_configurations(configs_data)
            
            # Set configurations
            self.configurations = restored_configs
            self.config_data_tab.configurations = restored_configs
            self.config_data_tab.update_config_display()
            if hasattr(self.config_data_tab, 'update_status'):
                self.config_data_tab.update_status()
            
            # Restore model setup
            config_systems = project_data.get('config_systems', {})
            config_parameters = io.deserialize_parameters(project_data.get('config_parameters', {}))
            
            self.model_setup_tab.config_systems = config_systems
            self.model_setup_tab.config_parameters = config_parameters
            self.model_setup_tab.current_config = project_data.get('current_config')
            self.model_setup_tab.current_system_id = project_data.get('current_system_id')
            
            # Restore master theta if available
            master_theta_data = project_data.get('master_theta')
            if master_theta_data:
                self.model_setup_tab.master_theta = io.deserialize_master_theta(master_theta_data)
            
            # Update model setup displays
            self.model_setup_tab.set_configurations(restored_configs)
            
            # Restore GUI state
            current_tab = project_data.get('current_tab', 0)
            tab_enabled = project_data.get('tab_enabled', [True, False, False, False])
            
            for i, enabled in enumerate(tab_enabled):
                if i < 4:
                    self.tab_widget.setTabEnabled(i, enabled)
            
            if current_tab < 4:
                self.tab_widget.setCurrentIndex(current_tab)
            
            # Show results
            loaded_configs = sum(1 for config in restored_configs.values() if config['wave'] is not None)
            total_configs = len(restored_configs)
            
            if missing_files:
                missing_text = "\n".join(missing_files)
                QMessageBox.warning(self, "Some Files Missing", 
                                  f"Project loaded with {loaded_configs}/{total_configs} data files.\n\n"
                                  f"Missing files:\n{missing_text}\n\n"
                                  f"Reload these files in Configuration & Data tab.")
            else:
                QMessageBox.information(self, "Project Loaded",
                                      f"Project loaded successfully!\n\n"
                                      f"Configurations: {total_configs}\n"
                                      f"Data loaded: {loaded_configs}/{total_configs}")
            
            self.status_bar.showMessage(f"Project loaded: {Path(filename).name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{str(e)}")
        
    def setup_ui(self):
        """Create main interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        # Set custom stylesheet for better tab visibility
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-bottom-color: #c0c0c0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                color: #000000;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)


        layout.addWidget(self.tab_widget)



        
        # Create tabs in workflow order
        self.config_data_tab = ConfigurationDataTab(self)
        self.model_setup_tab = ModelSetupTab(self)
        self.fitting_tab = FittingTab(self)
        self.results_tab = ResultsTab(self)
        
        # Add tabs
        self.tab_widget.addTab(self.config_data_tab, "1. Configuration & Data")
        self.tab_widget.addTab(self.model_setup_tab, "2. Model Setup")
        self.tab_widget.addTab(self.fitting_tab, "3. Fitting")
        self.tab_widget.addTab(self.results_tab, "4. Results")
        
        # Initially disable tabs 2-4
        self.tab_widget.setTabEnabled(1, False)  # Model Setup
        self.tab_widget.setTabEnabled(2, False)  # Fitting
        self.tab_widget.setTabEnabled(3, False)  # Results
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Configure instruments and load data to begin")
        
    def setup_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Project management
        save_project_action = QAction('Save Project...', self)
        save_project_action.setShortcut('Ctrl+S')
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        load_project_action = QAction('Load Project...', self)
        load_project_action.setShortcut('Ctrl+O')
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)
        
        file_menu.addSeparator()
        
        # Export options
        export_menu = file_menu.addMenu('Export')
        
        export_config_action = QAction('Export Configuration...', self)
        export_config_action.triggered.connect(self.export_configuration)
        export_menu.addAction(export_config_action)
        
        export_results_action = QAction('Export Results...', self)
        export_results_action.triggered.connect(self.export_results)
        export_menu.addAction(export_results_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        workflow_action = QAction('Workflow Guide', self)
        workflow_action.triggered.connect(self.show_workflow_guide)
        help_menu.addAction(workflow_action)
        
        about_action = QAction('About rbvfit', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Connect inter-tab signals"""
        # Tab 1 -> Tab 2: Configurations ready
        self.config_data_tab.configurations_updated.connect(self.on_configurations_updated)
        
        # Tab 2 -> Tab 3: Models compiled
        self.model_setup_tab.model_updated.connect(self.on_model_updated)
        
        # Tab 3 -> Tab 4: Fitting completed
        self.fitting_tab.fitting_completed.connect(self.on_fitting_completed)
        
        # Tab switching validation
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    def on_tab_changed(self, index):
        """Handle tab change - update status"""
        tab_names = [
            "Configuration & Data",
            "Model Setup", 
            "Fitting",
            "Results"
        ]
        
        if index < len(tab_names):
            self.status_bar.showMessage(f"Current tab: {tab_names[index]}")
        
    def on_configurations_updated(self, configurations):
        """Handle configuration update from Tab 1"""
        self.configurations = configurations
        
        # Pass to Model Setup tab
        self.model_setup_tab.set_configurations(configurations)
        
        # Enable Model Setup tab
        has_valid_configs = any(config['wave'] is not None for config in configurations.values())
        self.tab_widget.setTabEnabled(1, has_valid_configs)
        
        if has_valid_configs:
            n_configs = len([c for c in configurations.values() if c['wave'] is not None])
            self.status_bar.showMessage(f"Configurations ready: {n_configs} - Set up ion systems in Model Setup tab")
        else:
            self.status_bar.showMessage("No valid configurations - Assign data to configurations")
            
    def on_model_updated(self, instrument_data, theta_dict, bounds):
        """Handle model compilation from Tab 2"""
        self.instrument_data = instrument_data
        self.theta = theta_dict['theta']  # Extract numpy array from dict
        self.bounds = bounds
        
        # Pass to Fitting tab
        self.fitting_tab.set_model_data(instrument_data, self.theta, bounds)
        
        # Enable Fitting tab
        self.tab_widget.setTabEnabled(2, True)
        
        n_instruments = len(instrument_data)
        n_params = len(self.theta)
        self.status_bar.showMessage(f"Models compiled: {n_instruments} instruments, {n_params} parameters - Ready for fitting")
        
    def on_fitting_completed(self, fit_results):
        """Handle fitting completion from Tab 3"""
        self.fit_results = fit_results

        # Pass to Results tab
        self.results_tab.set_results(fit_results)
        
        # Enable Results tab
        self.tab_widget.setTabEnabled(3, True)
        
        # Update status
        try:
            if hasattr(fit_results, 'best_fit'):
                n_params = len(fit_results.best_fit)
                self.status_bar.showMessage(f"Fitting completed: {n_params} parameters - View results in Results tab")
            else:
                self.status_bar.showMessage("Fitting completed - View results in Results tab")
        except:
            self.status_bar.showMessage("Fitting completed - View results in Results tab")
    

    def save_project(self):
        """Save complete project state - simplified using io.py"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            "rbvfit Projects (*.rbv);;JSON files (*.json)")
        
        if filename:
            try:
                # Collect project data using io functions
                project_data = {
                    # Metadata
                    'version': v.__version__,
                    'created': datetime.datetime.now().isoformat(),
                    'rbvfit_version': v.__version__,
                    
                    # Tab 1: Configurations - use io serialization
                    'configurations': io.serialize_configurations(self.configurations),
                    
                    # Tab 2: Model Setup
                    'config_systems': getattr(self.model_setup_tab, 'config_systems', {}),
                    'config_parameters': io.serialize_parameters(
                        getattr(self.model_setup_tab, 'config_parameters', {})
                    ),
                    'current_config': getattr(self.model_setup_tab, 'current_config', None),
                    'current_system_id': getattr(self.model_setup_tab, 'current_system_id', None),
                    
                    # Master theta if available
                    'master_theta': io.serialize_master_theta(
                        getattr(self.model_setup_tab, 'master_theta', None)
                    ),
                    
                    # GUI state
                    'current_tab': self.tab_widget.currentIndex(),
                    'tab_enabled': [self.tab_widget.isTabEnabled(i) for i in range(4)]
                }
                
                # Save using io
                io.save_project_data(project_data, filename)
                
                # Success message with summary
                summary = io.create_project_summary(project_data)
                n_configs = len(project_data['configurations'])
                n_systems = sum(len(systems) for systems in project_data['config_systems'].values())
                
                self.status_bar.showMessage(f"Project saved: {Path(filename).name}")
                QMessageBox.information(self, "Project Saved", 
                                      f"Project saved successfully!\n\n"
                                      f"Configurations: {n_configs}\n"
                                      f"Systems: {n_systems}\n"
                                      f"File: {Path(filename).name}")
                    
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}")
    
    def load_project(self):
        """Load complete project state - simplified using io.py"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            "rbvfit Projects (*.rbv);;JSON files (*.json)")
        
        if filename:
            self.load_project_file(filename)
    
    def export_configuration(self):
        """Export configuration metadata only - clean version"""
        if not self.configurations:
            QMessageBox.warning(self, "No Configurations", "No configurations to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "",
            "JSON files (*.json);;All files (*)")
        
        if filename:
            try:
                # Export only configuration metadata (no data arrays)
                export_data = {
                    'version': v.__version__,
                    'exported': datetime.datetime.now().isoformat(),
                    'type': 'rbvfit_configuration_export',
                    'configurations': {}
                }
                
                for name, config in self.configurations.items():
                    export_data['configurations'][name] = {
                        'name': config['name'],
                        'fwhm': config['fwhm'],
                        'description': config.get('description', ''),
                        'filename': config.get('filename', ''),
                        'filepath': config.get('filepath', ''),
                        'basename': config.get('basename', ''),
                        'file_directory': config.get('file_directory', ''),
                        'has_data': config['wave'] is not None,
                        'current_data_points': len(config['wave']) if config['wave'] is not None else 0,
                        'current_wavelength_range': [float(config['wave'].min()), float(config['wave'].max())] if config['wave'] is not None else None,
                        'original_wavelength_range': [float(config['wave_original'].min()), float(config['wave_original'].max())] if config.get('wave_original') is not None else None,
                        'trim_range': config.get('trim_range'),
                        'processing_steps': config.get('processing_steps', []),
                        'is_trimmed': config.get('wave_original') is not None and config['wave'] is not None and len(config['wave']) != len(config['wave_original'])
                    }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                QMessageBox.information(self, "Export Complete", 
                                      f"Configuration exported to {Path(filename).name}\n\n"
                                      f"Exported {len(export_data['configurations'])} configurations")
                    
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export configuration:\n{str(e)}")
    
    def _clear_all_tabs(self):
        """Clear state in all tabs - keep this function"""
        # Clear main window state
        self.configurations = {}
        self.instrument_data = None
        self.theta = None
        self.bounds = None
        self.fit_results = None
        
        # Clear tab states
        for tab in [self.config_data_tab, self.model_setup_tab, self.fitting_tab, self.results_tab]:
            if hasattr(tab, 'clear_state'):
                tab.clear_state()
        
        # Reset tab enablement
        self.tab_widget.setTabEnabled(1, False)  # Model Setup
        self.tab_widget.setTabEnabled(2, False)  # Fitting  
        self.tab_widget.setTabEnabled(3, False)  # Results
        self.tab_widget.setCurrentIndex(0)       # Go to first tab    

    def export_results(self):
        """Export fitting results"""
        if self.fit_results is None:
            QMessageBox.warning(self, "No Results", "No fitting results to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", 
            "", "CSV Files (*.csv);;JSON Files (*.json)")
        
        if filename:
            try:
                # Use results tab export functionality if available
                if hasattr(self.results_tab, 'export_results'):
                    self.results_tab.export_results(filename)
                else:
                    # Basic export
                    if filename.endswith('.json'):
                        # Export as JSON
                        export_data = {
                            'results_type': str(type(self.fit_results)),
                            'exported': str(pd.Timestamp.now()),
                            'version': '2.0_unified'
                        }
                        
                        # Add basic results info if available
                        if hasattr(self.fit_results, 'best_fit'):
                            export_data['best_fit'] = self.fit_results.best_fit.tolist()
                        
                        with open(filename, 'w') as f:
                            json.dump(export_data, f, indent=2)
                    else:
                        # Export as CSV (basic)
                        QMessageBox.information(self, "Export Info", 
                                              "CSV export requires results analysis. Use Results tab for detailed export.")
                        return
                
                QMessageBox.information(self, "Export Complete", 
                                      f"Results exported to:\n{filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
        
    def show_workflow_guide(self):
        """Show workflow guide dialog"""
        guide_text = """
rbvfit 2.0 Workflow Guide (Unified Interface)

1. CONFIGURATION & DATA TAB
   • Create instrument configurations with FWHM settings
   • Load spectrum files
   • Assign data to configurations
   • Trim wavelengths and build unions as needed

2. MODEL SETUP TAB
   • Select configuration
   • Add ion systems (redshift, ion, transitions, components)
   • Estimate parameters using interactive tools
   • Compile models for all configurations

3. FITTING TAB
   • Run MCMC fitting with compiled models
   • Monitor progress and convergence
   • Adjust fitting parameters if needed

4. RESULTS TAB
   • Analyze fit results and parameter uncertainties
   • Generate plots and export data
   • Save results for publication

KEY FEATURES:
• Multi-instrument support with shared parameters
• Interactive parameter estimation
• Unified vfit_mcmc interface
• Configuration-aware workflow

PROJECT SAVE/LOAD:
• Save Project (Ctrl+S): Save configuration, systems, and parameters
• Load Project (Ctrl+O): Restore your complete setup
• Lightweight: Only saves setup, not raw data or fit results
• Smart file handling: Checks for missing data files on load

UNIFIED INTERFACE BENEFITS:
• Consistent API across single and multi-instrument fitting
• Automatic model compilation and optimization
• Improved performance and reliability
• Better error handling and diagnostics
        """
        
        QMessageBox.information(self, "Workflow Guide", guide_text)
        
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
rbvfit 2.0 - Voigt Profile Fitting (Unified Interface)

A Python package for fitting Voigt profiles to absorption line spectra
with support for multi-instrument datasets and MCMC parameter estimation.

Version: {getattr(v, '__version__', '2.0')} (Unified Interface)
Author: rbvfit Development Team

Features:
• Multi-instrument configuration management
• Interactive parameter estimation
• Unified vfit_mcmc interface
• MCMC fitting with emcee and zeus samplers
• Comprehensive results analysis
• Modern PyQt5 interface
• Project save/load functionality

Unified Interface Benefits:
• Consistent API for single and multi-instrument fitting
• Automatic model compilation and optimization
• Improved performance and error handling
• Better integration with modern MCMC samplers

For documentation and updates, visit:
https://github.com/rbvfit/rbvfit
        """
        
        QMessageBox.about(self, "About rbvfit", about_text)
        
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.showMessage(message)
        
    def closeEvent(self, event):
        """Handle application close"""
        reply = QMessageBox.question(self, "Exit rbvfit",
                                   "Are you sure you want to exit?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main application entry point with command-line argument support"""
    app = QApplication(sys.argv)
    app.setApplicationName("rbvfit 2.0")
    app.setApplicationVersion(getattr(v, '__version__', '2.0'))
    
    # Set application icon if available
    icon_path = Path(__file__).parent / "icons" / "rbvfit.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Check for config file argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Create and show main window
    window = UpdatedRbvfitGUI(config_file=config_file)
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()