#!/usr/bin/env python
"""
rbvfit 2.0 Main GUI Window - PyQt5 Implementation

Modern 3-tab interface for Voigt profile fitting.
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QHBoxLayout, QWidget, QMenuBar, QStatusBar, QFileDialog,
                            QMessageBox, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QIcon

# Add the rbvfit path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rbvfit.gui.io import load_multiple_files, get_spectrum_info
from rbvfit.gui.model_setup_tab import ModelSetupTab
from rbvfit.gui.fitting_tab import FittingTab  
from rbvfit.gui.results_tab import ResultsTab


class GuiSignals(QObject):
    """Central signal hub for inter-tab communication"""
    data_loaded = pyqtSignal(dict)  # spectra_dict
    model_updated = pyqtSignal(object)  # fit_config
    spectrum_clicked = pyqtSignal(float)  # wavelength
    component_added = pyqtSignal(dict)  # component_data
    results_updated = pyqtSignal(object)  # fit_results


class RbvfitGUI(QMainWindow):
    """Main GUI window with 3-tab interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rbvfit 2.0 - Voigt Profile Fitting")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.spectra_data = {}
        self.fit_config = None
        self.fit_results = None
        
        # Signal hub
        self.signals = GuiSignals()
        
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
        
    def setup_ui(self):
        """Create main interface"""
        # Central widget with tab layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.model_tab = ModelSetupTab(self)
        self.fitting_tab = FittingTab(self)
        self.results_tab = ResultsTab(self)
        
        self.tab_widget.addTab(self.model_tab, "Model Setup")
        self.tab_widget.addTab(self.fitting_tab, "Fitting")
        self.tab_widget.addTab(self.results_tab, "Results")
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load data to begin")
        
    def setup_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Data...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.setStatusTip("Load spectrum data files")
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        save_config_action = QAction("Save Configuration...", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.setStatusTip("Save current model configuration")
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)
        
        load_config_action = QAction("Load Configuration...", self)
        load_config_action.setStatusTip("Load model configuration from file")
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit rbvfit")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.setStatusTip("Refresh all tabs")
        refresh_action.triggered.connect(self.refresh_tabs)
        view_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.setStatusTip("About rbvfit 2.0")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Connect signals between tabs"""
        # Data loading updates fitting tab
        self.signals.data_loaded.connect(self.fitting_tab.update_data)
        
        # Model updates propagate to fitting tab
        self.signals.model_updated.connect(self.fitting_tab.update_model)
        self.model_tab.model_updated.connect(self.fitting_tab.update_model)
        
        # Spectrum clicks from fitting tab to model tab
        self.fitting_tab.spectrum_clicked.connect(self.model_tab.handle_spectrum_click)
        
        # Results updates propagate to results tab
        self.signals.results_updated.connect(self.results_tab.update_results)
        
        # Fitting completion updates results
        self.fitting_tab.fitting_completed.connect(self.on_fitting_completed)
        
    def on_fitting_completed(self, results):
        """Handle fitting completion"""
        self.fit_results = results
        self.signals.results_updated.emit(results)
        self.tab_widget.setCurrentIndex(2)  # Switch to results tab
        self.update_status("Fitting completed - results available")
        
    def load_data(self):
        """Load spectrum data files"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("All files (*.*)")
        file_dialog.setWindowTitle("Select spectrum files")
        
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            
            try:
                self.spectra_data = load_multiple_files(filenames)
                info = get_spectrum_info(self.spectra_data)
                self.status_bar.showMessage(f"Loaded {len(filenames)} files")
                
                # Emit signal to update other tabs
                self.signals.data_loaded.emit(self.spectra_data)
                
                QMessageBox.information(self, "Data Loaded", 
                                      f"Successfully loaded:\n{info}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to load data:\n{str(e)}")
                
    def save_config(self):
        """Save current configuration"""
        if self.fit_config is None:
            QMessageBox.warning(self, "Warning", "No configuration to save")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "", 
            "JSON files (*.json);;All files (*.*)")
        
        if filename:
            try:
                from gui.io import save_configuration
                save_configuration(self.fit_config, filename)
                self.status_bar.showMessage(f"Configuration saved: {Path(filename).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to save configuration:\n{str(e)}")
                
    def load_config(self):
        """Load configuration from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", "",
            "JSON files (*.json);;All files (*.*)")
        
        if filename:
            try:
                from gui.io import load_configuration
                config_dict = load_configuration(filename)
                # TODO: Convert dict back to FitConfiguration object
                self.status_bar.showMessage(f"Configuration loaded: {Path(filename).name}")
                self.model_tab.update_from_config(config_dict)
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to load configuration:\n{str(e)}")
                
    def refresh_tabs(self):
        """Refresh all tabs"""
        self.model_tab.refresh()
        self.fitting_tab.refresh() 
        self.results_tab.refresh()
        
    def show_about(self):
        """Show about dialog"""
        about_text = """<h3>rbvfit 2.0 GUI</h3>
        <p>A modern interface for Voigt profile fitting of 
        absorption lines in astronomical spectra.</p>
        
        <p><b>Features:</b></p>
        <ul>
        <li>Multi-system absorption line fitting</li>
        <li>Interactive component selection</li>
        <li>MCMC parameter estimation</li>
        <li>Multi-instrument support</li>
        <li>Results visualization and export</li>
        </ul>
        
        <p><b>Built with PyQt5 and matplotlib</b></p>
        """
        QMessageBox.about(self, "About rbvfit 2.0", about_text)
        
    def update_status(self, message: str):
        """Update status bar"""
        self.status_bar.showMessage(message)
        
    def closeEvent(self, event):
        """Handle application close"""
        reply = QMessageBox.question(self, 'Exit', 
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("rbvfit 2.0")
    app.setOrganizationName("rbvfit")
    
    # Set application style
    app.setStyle('Fusion')
    
    gui = RbvfitGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()