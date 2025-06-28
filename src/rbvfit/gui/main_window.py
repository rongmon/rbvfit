#!/usr/bin/env python
"""
rbvfit 2.0 Main GUI Window - Updated
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QWidget, QMenuBar, QStatusBar, QFileDialog,
                            QMessageBox, QAction)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QIcon

sys.path.append(str(Path(__file__).parent.parent))

from rbvfit.gui.io import load_multiple_files, get_spectrum_info
from rbvfit.gui.model_setup_tab import ModelSetupTab
from rbvfit.gui.fitting_tab import FittingTab  
from rbvfit.gui.results_tab import ResultsTab


class GuiSignals(QObject):
    """Central signal hub for inter-tab communication"""
    data_loaded = pyqtSignal(dict)
    model_updated = pyqtSignal(object, dict)  # compiled_model, mcmc_params
    spectrum_clicked = pyqtSignal(float)
    component_added = pyqtSignal(dict)
    results_updated = pyqtSignal(object)


class RbvfitGUI(QMainWindow):
    """Main GUI window with 4-tab interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rbvfit 2.0 - Voigt Profile Fitting")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.spectra_data = {}
        self.compiled_model = None
        self.current_model = None
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
        
        # Create tabs
        self.model_tab = ModelSetupTab(self)
        self.fitting_tab = FittingTab(self)
        self.results_tab = ResultsTab(self)
        
        # Add tabs
        self.tab_widget.addTab(self.model_tab, "Model Setup")
        self.tab_widget.addTab(self.fitting_tab, "Fitting")
        self.tab_widget.addTab(self.results_tab, "Results")
        
        # Initially disable fitting and results tabs
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load spectrum data to begin")
        
    def setup_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load Spectra...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_spectra)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        save_project_action = QAction('Save Project...', self)
        save_project_action.setShortcut('Ctrl+S')
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        load_project_action = QAction('Load Project...', self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About rbvfit', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Connect signals between tabs"""
        # Model setup to fitting
        self.model_tab.model_updated.connect(self.on_model_updated)
        
        # Fitting to results
        self.fitting_tab.fitting_completed.connect(self.on_fitting_completed)
        
        # Central signal connections
        self.signals.data_loaded.connect(self.model_tab.set_spectra_data)
        self.signals.data_loaded.connect(self.fitting_tab.set_spectra_data)
        
    def load_spectra(self):
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
                # Load files using rbvfit IO
                spectra_data = load_multiple_files(files)
                
                if spectra_data:
                    self.spectra_data = spectra_data
                    self.signals.data_loaded.emit(spectra_data)
                    
                    # Update status
                    n_files = len(spectra_data)
                    self.status_bar.showMessage(f"Loaded {n_files} spectrum file(s)")
                    
                    # Show summary
                    info_text = get_spectrum_info(spectra_data)  # Pass the full dict, not individual data
                    QMessageBox.information(self, "Data Loaded", f"Loaded spectra:\n{info_text}")
                    
                else:
                    QMessageBox.warning(self, "Load Failed", "No valid spectrum data found in selected files")
                    
            except Exception as e:
                # Print full traceback to terminal for debugging
                import traceback
                print("FULL ERROR TRACEBACK:")
                traceback.print_exc()
                
                QMessageBox.critical(self, "Load Error", f"Failed to load spectrum files:\n{str(e)}")
                
    def on_model_updated(self, compiled_model, mcmc_params):
        """Handle model compilation"""
        self.compiled_model = compiled_model
        self.current_model = self.model_tab.get_current_model()
        
        # Enable fitting tab
        self.tab_widget.setTabEnabled(1, True)
        
        # Set model in fitting tab
        self.fitting_tab.set_compiled_model(compiled_model, mcmc_params)
        
        self.status_bar.showMessage("Model compiled - ready for fitting")
        
    def on_fitting_completed(self, results):
        """Handle fitting completion"""
        self.fit_results = results
        
        # Enable results tab
        self.tab_widget.setTabEnabled(2, True)
        
        # Set results in results tab
        self.results_tab.set_results(results)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)
        
        self.status_bar.showMessage("Fitting completed - results available")
        
    def get_spectrum_data(self):
        """Return current spectrum data"""
        return self.spectra_data
        
    def get_current_model(self):
        """Return current VoigtModel"""
        return self.current_model
        
    def get_current_theta(self):
        """Return current theta parameters"""
        if self.model_tab:
            return self.model_tab.get_current_theta()
        return None
        
    def save_project(self):
        """Save current project"""
        # TODO: Implement project saving
        QMessageBox.information(self, "Coming Soon", "Project saving will be implemented")
        
    def load_project(self):
        """Load saved project"""
        # TODO: Implement project loading
        QMessageBox.information(self, "Coming Soon", "Project loading will be implemented")
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        rbvfit 2.0 - Voigt Profile Fitting
        
        Bayesian absorption line fitting with:
        • Interactive parameter estimation
        • Multi-system and multi-instrument support
        • MCMC and quick fitting options
        • Enhanced results analysis
        
        Version 2.0 - Enhanced by Rongmon Bordoloi, 2025
        """
        QMessageBox.about(self, "About rbvfit", about_text)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("rbvfit 2.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = RbvfitGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()