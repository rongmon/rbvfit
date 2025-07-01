#!/usr/bin/env python
"""
Updated rbvfit 2.0 Main GUI Window - 4-Tab Redesign
"""

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
    """Updated main GUI window with 4-tab redesigned interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rbvfit 2.0 - Voigt Profile Fitting (4-Tab Redesign)")
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
        
    def setup_ui(self):
        """Create main interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs in new order
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
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        next_tab_action = QAction('Next Tab', self)
        next_tab_action.setShortcut('Ctrl+Tab')
        next_tab_action.triggered.connect(self.next_tab)
        view_menu.addAction(next_tab_action)
        
        prev_tab_action = QAction('Previous Tab', self)
        prev_tab_action.setShortcut('Ctrl+Shift+Tab')
        prev_tab_action.triggered.connect(self.prev_tab)
        view_menu.addAction(prev_tab_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        workflow_action = QAction('Workflow Guide', self)
        workflow_action.triggered.connect(self.show_workflow_guide)
        help_menu.addAction(workflow_action)
        
        about_action = QAction('About rbvfit', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Connect signals between tabs"""
        # Tab 1 (Config & Data) -> Tab 2 (Model Setup)
        self.config_data_tab.configurations_updated.connect(self.on_configurations_updated)
        
        # Tab 2 (Model Setup) -> Tab 3 (Fitting)
        self.model_setup_tab.model_updated.connect(self.on_model_updated)
        
        # Tab 3 (Fitting) -> Tab 4 (Results)
        self.fitting_tab.fitting_completed.connect(self.on_fitting_completed)
        
        # Tab navigation
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    def on_configurations_updated(self, configurations):
        """Handle configurations update from Tab 1"""
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
        
        # Auto-switch to Results tab
        self.tab_widget.setCurrentIndex(3)
        
        self.status_bar.showMessage("Fitting completed - View results in Results tab")
        
    def on_tab_changed(self, index):
        """Handle tab change"""
        tab_names = ["Configuration & Data", "Model Setup", "Fitting", "Results"]
        if 0 <= index < len(tab_names):
            self.status_bar.showMessage(f"Current tab: {tab_names[index]}")
            
    def next_tab(self):
        """Switch to next tab"""
        current = self.tab_widget.currentIndex()
        next_index = (current + 1) % self.tab_widget.count()
        
        # Only switch if tab is enabled
        if self.tab_widget.isTabEnabled(next_index):
            self.tab_widget.setCurrentIndex(next_index)
            
    def prev_tab(self):
        """Switch to previous tab"""
        current = self.tab_widget.currentIndex()
        prev_index = (current - 1) % self.tab_widget.count()
        
        # Only switch if tab is enabled
        if self.tab_widget.isTabEnabled(prev_index):
            self.tab_widget.setCurrentIndex(prev_index)

    def get_current_model(self):
        """Return current uncompiled model for results creation"""
        if hasattr(self.model_setup_tab, 'voigt_model'):
            return self.model_setup_tab.voigt_model  # Return the original VoigtModel
        return None

    # ==================== PROJECT SAVE/LOAD IMPLEMENTATION ====================
                
    def save_project(self):
        """Save complete project state"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "",
            "rbvfit Projects (*.rbv);;JSON files (*.json);;All files (*.*)")
        
        if filename:
            try:
                # Collect all project state
                project_data = self._collect_project_data()
                
                # Use io module to save
                io.save_project_data(project_data, filename)
                
                # Show success message
                summary = io.create_project_summary(project_data)
                self.status_bar.showMessage(f"Project saved: {Path(filename).name}")
                QMessageBox.information(self, "Project Saved", 
                                      f"Project saved successfully!\n\n{summary}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}")
                
    def load_project(self):
        """Load complete project state"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "",
            "rbvfit Projects (*.rbv);;JSON files (*.json);;All files (*.*)")
        
        if filename:
            try:
                # Validate file first
                file_info = io.validate_project_file(filename)
                
                if not file_info['valid']:
                    QMessageBox.critical(self, "Invalid File", 
                                       f"Cannot load project file:\n{file_info['error']}")
                    return
                
                # Check version compatibility
                if file_info['version'] != '2.0':
                    reply = QMessageBox.question(self, "Version Mismatch", 
                                               f"Project file version: {file_info['version']}\n"
                                               f"Current rbvfit version: 2.0\n\n"
                                               f"Continue loading? (may cause issues)",
                                               QMessageBox.Yes | QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
                
                # Load project data using io module
                project_data = io.load_project_data(filename)
                
                # Clear current state and restore
                self._clear_all_tabs()
                self._restore_project_data(project_data)
                
                # Show success message with summary
                summary = io.create_project_summary(project_data)
                missing_files = io.check_missing_files(project_data)
                
                message = f"Project loaded successfully!\n\n{summary}"
                if missing_files:
                    message += f"\n⚠️ Note: {len(missing_files)} data files need to be reassigned in the Configuration & Data tab."
                
                self.status_bar.showMessage(f"Project loaded: {Path(filename).name}")
                QMessageBox.information(self, "Project Loaded", message)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{str(e)}")

    def _collect_project_data(self) -> dict:
        """Collect complete project state from all tabs"""
        project_data = {
            'version': '2.0',
            'created': getattr(self, '_project_created', None) or str(pd.Timestamp.now()),
            
            # Tab 1: Configuration & Data
            'configurations': io.serialize_configurations(self.configurations),
            
            # Tab 2: Model Setup
            'config_systems': getattr(self.model_setup_tab, 'config_systems', {}),
            'config_parameters': io.serialize_parameters(
                getattr(self.model_setup_tab, 'config_parameters', {})
            ),
            'current_config': getattr(self.model_setup_tab, 'current_config', None),
            'current_system_id': getattr(self.model_setup_tab, 'current_system_id', None),
            
            # Master theta (if compiled)
            'master_theta': io.serialize_master_theta(
                getattr(self.model_setup_tab.current_collection_result, 'master_theta', None)
                if hasattr(self.model_setup_tab, 'current_collection_result') and 
                   self.model_setup_tab.current_collection_result else None
            ),
            'collection_result_info': io.serialize_collection_info(
                getattr(self.model_setup_tab, 'current_collection_result', None)
            ),
            
            # GUI state
            'gui_state': {
                'current_tab': self.tab_widget.currentIndex(),
                'tab_enabled': [self.tab_widget.isTabEnabled(i) for i in range(4)]
            }
        }
        
        return project_data

    def _restore_project_data(self, project_data: dict) -> None:
        """Restore complete project state to all tabs"""
        import pandas as pd
        
        # Restore Tab 1: Configurations
        configurations = io.deserialize_configurations(project_data.get('configurations', {}))
        if hasattr(self.config_data_tab, '_restore_configurations'):
            self.config_data_tab._restore_configurations(configurations)
        else:
            # Fallback: set configurations directly
            self.configurations = configurations
            self.config_data_tab.configurations = configurations
            if hasattr(self.config_data_tab, 'update_config_display'):
                self.config_data_tab.update_config_display()
                self.config_data_tab.update_status()
        
        # Restore Tab 2: Model Setup
        config_systems = project_data.get('config_systems', {})
        config_parameters = io.deserialize_parameters(project_data.get('config_parameters', {}))
        current_config = project_data.get('current_config')
        current_system_id = project_data.get('current_system_id')
        
        if hasattr(self.model_setup_tab, '_restore_model_setup'):
            self.model_setup_tab._restore_model_setup(
                config_systems, config_parameters, current_config, current_system_id
            )
        else:
            # Fallback: set attributes directly
            self.model_setup_tab.config_systems = config_systems
            self.model_setup_tab.config_parameters = config_parameters
            self.model_setup_tab.current_config = current_config
            self.model_setup_tab.current_system_id = current_system_id
        
        # Restore master theta if available
        master_theta_data = project_data.get('master_theta')
        collection_info = project_data.get('collection_result_info')
        
        if master_theta_data and collection_info:
            master_theta = io.deserialize_master_theta(master_theta_data)
            if hasattr(self.model_setup_tab, '_restore_master_theta'):
                self.model_setup_tab._restore_master_theta(master_theta, collection_info)
            else:
                # Create minimal collection result for viewing
                self._create_minimal_collection_result(master_theta, collection_info)
        
        # Restore GUI state
        gui_state = project_data.get('gui_state', {})
        tab_enabled = gui_state.get('tab_enabled', [True, False, False, False])
        for i, enabled in enumerate(tab_enabled):
            if i < self.tab_widget.count():
                self.tab_widget.setTabEnabled(i, enabled)
                
        current_tab = gui_state.get('current_tab', 0)
        if current_tab < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(current_tab)
        
        # Update main window state
        self.configurations = configurations
        
        # Emit signals to update dependent tabs
        if configurations:
            self.config_data_tab.configurations_updated.emit(configurations)

    def _clear_all_tabs(self) -> None:
        """Clear state in all tabs"""
        # Clear main window state
        self.configurations = {}
        self.instrument_data = None
        self.theta = None
        self.bounds = None
        self.fit_results = None
        
        # Clear tab states if they have clear methods
        for tab in [self.config_data_tab, self.model_setup_tab, self.fitting_tab, self.results_tab]:
            if hasattr(tab, 'clear_state'):
                tab.clear_state()
        
        # Reset tab enablement
        self.tab_widget.setTabEnabled(1, False)  # Model Setup
        self.tab_widget.setTabEnabled(2, False)  # Fitting  
        self.tab_widget.setTabEnabled(3, False)  # Results
        self.tab_widget.setCurrentIndex(0)       # Go to first tab
        
        # Clear status
        self.status_bar.showMessage("Project cleared - Ready to load new project or start fresh")

    def _create_minimal_collection_result(self, master_theta, collection_info) -> None:
        """Create minimal collection result for restored master theta"""
        from types import SimpleNamespace
        import numpy as np
        
        # Create minimal master systems from saved info
        master_systems = []
        for sys_info in collection_info.get('system_info', []):
            system = SimpleNamespace(
                z=sys_info['z'],
                ion=sys_info['ion'], 
                transitions=sys_info['transitions'],
                components=sys_info['components'],
                source_instrument=sys_info['source_instrument']
            )
            master_systems.append(system)
        
        # Create minimal collection result
        minimal_result = SimpleNamespace(
            master_theta=master_theta,
            master_systems=master_systems,
            instrument_mappings=[],  # Empty for now
            master_config=None,
            collection_log=[f"Restored from project file ({len(master_theta)} parameters)"]
        )
        
        self.model_setup_tab.current_collection_result = minimal_result

    # ==================== EXISTING EXPORT METHODS ====================
                
    def export_configuration(self):
        """Export configuration to file"""
        if not self.configurations:
            QMessageBox.warning(self, "No Configuration", "No configurations to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "",
            "JSON files (*.json);;All files (*.*)")
        
        if filename:
            try:
                # Export configurations (without the actual data arrays)
                export_data = {}
                for name, config in self.configurations.items():
                    export_data[name] = {
                        'name': config['name'],
                        'fwhm': config['fwhm'],
                        'description': config.get('description', ''),
                        'filename': config.get('filename', '')
                    }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                QMessageBox.information(self, "Export Complete", 
                                      f"Configuration exported to {Path(filename).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export configuration:\n{str(e)}")
                
    def export_results(self):
        """Export fit results"""
        if not self.fit_results:
            QMessageBox.warning(self, "No Results", "No fit results to export")
            return
            
        # For now, delegate to Results tab
        if hasattr(self.results_tab, 'export_csv'):
            self.results_tab.export_csv()
        else:
            QMessageBox.information(self, "Export Results", 
                                  "Results export functionality is handled in the Results tab")
            
    def show_workflow_guide(self):
        """Show workflow guide dialog"""
        guide_text = """
rbvfit 2.0 Workflow Guide

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
        """
        
        QMessageBox.information(self, "Workflow Guide", guide_text)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
rbvfit 2.0 - Voigt Profile Fitting

A Python package for fitting Voigt profiles to absorption line spectra
with support for multi-instrument datasets and MCMC parameter estimation.

Version: 2.0 (4-Tab Redesign)
Author: rbvfit Development Team

Features:
• Multi-instrument configuration management
• Interactive parameter estimation
• MCMC fitting with emcee and zeus samplers
• Comprehensive results analysis
• Modern PyQt5 interface
• Project save/load functionality

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
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("rbvfit 2.0")
    app.setApplicationVersion("2.0")
    
    # Set application icon if available
    icon_path = Path(__file__).parent / "icons" / "rbvfit.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Create and show main window
    window = UpdatedRbvfitGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()