#!/usr/bin/env python
"""
Restored Model Setup Tab for rbvfit 2.0 - Full Functionality

Restored with proper:
- Custom ion support with templates
- Interactive parameter estimation with spectrum display
- Working parameter management
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                            QGroupBox, QPushButton, QLabel, QComboBox, QTreeWidget,
                            QTreeWidgetItem, QHeaderView, QTableWidget, QTableWidgetItem,
                            QTextEdit, QFormLayout, QSpinBox, QDoubleSpinBox,
                            QMessageBox, QDialog, QDialogButtonBox, QLineEdit,
                            QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QBrush, QColor

# Import rbvfit components
from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.parameter_manager import ParameterManager
import rbvfit.vfit_mcmc as mc
from rbvfit import rb_setline 
# Try to import interactive parameter dialog
try:
    from rbvfit.gui.interactive_param_dialog import InteractiveParameterDialog
    HAS_INTERACTIVE_DIALOG = True
except ImportError:
    HAS_INTERACTIVE_DIALOG = False


ION_TEMPLATES = {
    "Custom (manual entry)": {"ion": "", "transitions": []},

    # Low-ionization species
    "CII":     {"ion": "CII",     "transitions": [1334.532]},
    "CIII":    {"ion": "CIII",    "transitions": [977.020]},
    "CIV":     {"ion": "CIV",     "transitions": [1548.195, 1550.770]},
    "OI":      {"ion": "OI",      "transitions": [1302.168]},
    "SiII":    {"ion": "SiII",    "transitions": [1190.416, 1193.290, 1260.422, 1304.370, 1526.707, 1808.012]},
    "SiIV":    {"ion": "SiIV",    "transitions": [1393.755, 1402.770]},
    "AlIII":   {"ion": "AlIII",   "transitions": [1854.716, 1862.790]},
    "FeII":    {"ion": "FeII",    "transitions": [1608.451, 2344.214, 2374.461, 2382.765, 2586.650, 2600.173]},
    "FeIII":   {"ion": "FeIII",   "transitions": [1122.526]},
    "MgII":    {"ion": "MgII",    "transitions": [2796.352, 2803.531]},

    # Intermediate/high-ionization species
    "NV":      {"ion": "NV",      "transitions": [1238.821, 1242.804]},
    "OVI":     {"ion": "OVI",     "transitions": [1031.926, 1037.617]},
    "OV":      {"ion": "OV",      "transitions": [629.730]},
    "OIV":     {"ion": "OIV",     "transitions": [608.397, 787.711]},
    "OIII":    {"ion": "OIII",    "transitions": [832.927]},
    "SIV":     {"ion": "SIV",     "transitions": [1062.664]},
    "SV":      {"ion": "SV",      "transitions": [786.468, 933.378]},
    "SVI":     {"ion": "SVI",     "transitions": [933.378, 944.523]},
    "NeVIII":  {"ion": "NeVIII",  "transitions": [770.409, 780.324]},
    "MgX":     {"ion": "MgX",     "transitions": [609.793, 624.941]},
    "ArVIII":  {"ion": "ArVIII",  "transitions": [700.242]},
    "NIII":    {"ion": "NIII",    "transitions": [989.799]},

    # Hydrogen series
    "Lyman-α": {"ion": "HI",      "transitions": [1215.670]},
    "Lyman-β": {"ion": "HI",      "transitions": [1025.722]},
    "HI-series": {
        "ion": "HI",
        "transitions": [
            917.180, 918.129, 919.351, 920.963, 923.150, 926.226,
            930.748, 937.803, 949.743, 972.537, 1025.722, 1215.670
        ]
    }
}


class SystemDialog(QDialog):
    """Dialog for adding/editing ion systems with templates and custom support"""
    
    def __init__(self, system_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ion System Configuration")
        self.setModal(True)
        self.resize(450, 350)
        
        self.system_data = system_data or {}
        self.setup_ui()
        
    def setup_ui(self):
        """Create dialog interface with ion templates"""
        layout = QFormLayout()
        self.setLayout(layout)
        
        # Ion Template Dropdown
        self.template_combo = QComboBox()
        self.template_combo.setToolTip("Select predefined ion template or custom entry")
        for template_name in ION_TEMPLATES.keys():
            self.template_combo.addItem(template_name)
        self.template_combo.setCurrentText("Custom (manual entry)")
        layout.addRow("Ion Template:", self.template_combo)
        
        # Add a separator line
        layout.addRow("", QLabel("─" * 40))
        
        # Redshift
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-0.1, 20.0)
        self.z_spin.setDecimals(6)
        self.z_spin.setValue(self.system_data.get('z', 0.0))
        layout.addRow("Redshift (z):", self.z_spin)
        
        # Ion name (can be custom)
        self.ion_edit = QLineEdit()
        self.ion_edit.setText(self.system_data.get('ion', ''))
        self.ion_edit.setPlaceholderText("e.g., CIV, OI, SiII, or custom ion")
        layout.addRow("Ion:", self.ion_edit)
        
        # Transitions (can be custom wavelengths)
        self.transitions_edit = QLineEdit()
        transitions = self.system_data.get('transitions', [])
        if transitions:
            self.transitions_edit.setText(', '.join(map(str, transitions)))
        self.transitions_edit.setPlaceholderText("e.g., 1548.2, 1550.3 (rest wavelengths in Å)")
        layout.addRow("Transitions (Å):", self.transitions_edit)
        
        # Components
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 99)
        self.components_spin.setValue(self.system_data.get('components', 1))
        layout.addRow("Components:", self.components_spin)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(buttons)
        
        # Connect signals
        self.template_combo.currentTextChanged.connect(self.on_template_changed)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # If editing existing system, try to match template
        if self.system_data:
            self.match_existing_template()
    
    def on_template_changed(self, template_name):
        """Handle template selection change"""
        if template_name not in ION_TEMPLATES:
            return
            
        template = ION_TEMPLATES[template_name]
        
        if template_name == "Custom (manual entry)":
            # Clear fields for manual entry
            if not self.system_data:  # Only clear if not editing existing
                self.ion_edit.clear()
                self.transitions_edit.clear()
        else:
            # Populate fields from template
            self.ion_edit.setText(template["ion"])
            
            if template["transitions"]:
                transitions_str = ', '.join(map(str, template["transitions"]))
                self.transitions_edit.setText(transitions_str)
            else:
                self.transitions_edit.clear()
    
    def match_existing_template(self):
        """Try to match existing system data to a template"""
        if not self.system_data:
            return
            
        existing_ion = self.system_data.get('ion', '').strip()
        existing_transitions = set(self.system_data.get('transitions', []))
        
        # Look for matching template
        for template_name, template in ION_TEMPLATES.items():
            if template_name == "Custom (manual entry)":
                continue
                
            template_ion = template["ion"]
            template_transitions = set(template["transitions"])
            
            # Check for exact match (ion name and transitions)
            if (existing_ion.upper() == template_ion.upper() and 
                existing_transitions == template_transitions):
                self.template_combo.setCurrentText(template_name)
                return
                
            # Check for partial match (ion name and subset of transitions)
            if (existing_ion.upper() == template_ion.upper() and 
                existing_transitions.issubset(template_transitions)):
                self.template_combo.setCurrentText(template_name)
                return
        
        # No match found, keep "Custom"
        self.template_combo.setCurrentText("Custom (manual entry)")
            
    def get_system_data(self):
        """Get system data from dialog"""
        transitions_text = self.transitions_edit.text().strip()
        transitions = []
        detected_ion = None
        
        if transitions_text:
            for trans in transitions_text.split(','):
                try:
                    wavelength = float(trans.strip())
                    line = rb_setline.rb_setline(wavelength, 'closest')  # Fix import
                    name = line['name'][0].split(' ')
                    transitions.append(line['wave'][0])
                    
                    # Only set ion from first valid transition
                    if detected_ion is None:
                        detected_ion = name[0]
                        
                except ValueError:
                    continue
        
        # Use detected ion or fallback to manual entry
        ion_name = detected_ion if detected_ion else self.ion_edit.text().strip()
        
        return {
            'z': self.z_spin.value(),
            'ion': ion_name,
            'components': self.components_spin.value(),
            'transitions': transitions,
            'id': f"{self.z_spin.value():.6f}_{ion_name}"  # Use same ion for ID
        }

class TieDialog(QDialog):
    """Dialog for tying v and/or b across ions at the same redshift"""

    def __init__(self, systems_by_z, parent=None):
        """
        systems_by_z : dict
            Redshift -> list of untied ion names available at that redshift
        """
        super().__init__(parent)
        self.setWindowTitle("Tie Ions")
        self.setModal(True)
        self.systems_by_z = systems_by_z

        layout = QFormLayout()
        self.setLayout(layout)

        self.z_combo = QComboBox()
        for z in systems_by_z:
            self.z_combo.addItem(f"{z:.6f}", z)
        layout.addRow("Redshift:", self.z_combo)

        self.ref_combo = QComboBox()
        self.ref_combo.setToolTip("The reference ion sets the component count and the tied values")
        layout.addRow("Reference ion:", self.ref_combo)

        self.ion_checks_widget = QWidget()
        self.ion_checks_layout = QVBoxLayout()
        self.ion_checks_layout.setContentsMargins(0, 0, 0, 0)
        self.ion_checks_widget.setLayout(self.ion_checks_layout)
        layout.addRow("Tie to reference:", self.ion_checks_widget)
        self.ion_checks = []

        self.v_check = QCheckBox("Tie velocities (v)")
        self.v_check.setChecked(True)
        self.b_check = QCheckBox("Tie Doppler parameters (b)")
        layout.addRow(self.v_check)
        layout.addRow(self.b_check)
        layout.addRow(QLabel("Column densities (N) are never tied."))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.z_combo.currentIndexChanged.connect(self.on_z_changed)
        self.ref_combo.currentTextChanged.connect(self.on_ref_changed)
        self.on_z_changed()

    def on_z_changed(self):
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        self.ref_combo.addItems(self.systems_by_z.get(self.z_combo.currentData(), []))
        self.ref_combo.blockSignals(False)
        self.on_ref_changed()

    def on_ref_changed(self):
        for check in self.ion_checks:
            self.ion_checks_layout.removeWidget(check)
            check.deleteLater()
        self.ion_checks = []
        ref = self.ref_combo.currentText()
        for ion in self.systems_by_z.get(self.z_combo.currentData(), []):
            if ion != ref:
                check = QCheckBox(ion)
                check.setChecked(True)
                self.ion_checks_layout.addWidget(check)
                self.ion_checks.append(check)

    def validate_and_accept(self):
        if not any(c.isChecked() for c in self.ion_checks):
            QMessageBox.warning(self, "No Ions", "Select at least one ion to tie to the reference")
            return
        if not (self.v_check.isChecked() or self.b_check.isChecked()):
            QMessageBox.warning(self, "Nothing to Tie", "Select v and/or b")
            return
        if self.b_check.isChecked() and not self.v_check.isChecked():
            reply = QMessageBox.question(
                self, "Tie b without v",
                "Tying b without v assumes component i of each ion is the same cloud, "
                "even though their velocities can drift apart.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def get_tie(self):
        ions = [self.ref_combo.currentText()] + [c.text() for c in self.ion_checks if c.isChecked()]
        return {'z': self.z_combo.currentData(), 'ions': ions,
                'v': self.v_check.isChecked(), 'b': self.b_check.isChecked()}


class ModelSetupTab(QWidget):
    """Clean tab for setting up absorption line models with full functionality"""
    
    model_updated = pyqtSignal(dict, dict, dict)  # (instrument_data, theta_dict, bounds)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.configurations = {}  # From Tab 1
        self.global_systems = []  # Flat list of all ion systems (shared across configs)
        self.config_parameters = {}  # Dict: system_id -> DataFrame
        self.global_ties = []  # List of {'z', 'ions', 'v', 'b'}; first ion is the reference
        self.current_config = None
        self.current_system_id = None
        
        # Simple unified approach
        self.master_config = None
        self.master_theta = None
        self.param_manager = None

        # Voigt method: 'wofz' (default, exact) or 'fast' (Tepper-García)
        self.voigt_method = 'wofz'

        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create the model setup interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Main splitter: configurations | systems | parameters
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        self.setup_config_panel(main_splitter)
        self.setup_systems_panel(main_splitter)
        self.setup_parameters_panel(main_splitter)
        
        main_splitter.setSizes([250, 300, 450])
        
        # Bottom controls
        self.setup_bottom_controls(layout)
        
    def setup_config_panel(self, parent):
        """Create configuration selection panel"""
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        parent.addWidget(config_widget)
        
        # Configuration group
        config_group = QGroupBox("Available Configurations")
        config_group_layout = QVBoxLayout()
        config_group.setLayout(config_group_layout)
        config_layout.addWidget(config_group)
        
        # Configuration list
        self.config_tree = QTreeWidget()
        self.config_tree.setHeaderLabels(['Configuration', 'Info'])
        config_group_layout.addWidget(self.config_tree)
        
        # Current config selector
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Current:"))
        self.current_config_combo = QComboBox()
        current_layout.addWidget(self.current_config_combo)
        config_group_layout.addLayout(current_layout)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setReadOnly(True)
        config_layout.addWidget(self.status_text)
        
    def setup_systems_panel(self, parent):
        """Create systems management panel"""
        systems_widget = QWidget()
        systems_layout = QVBoxLayout()
        systems_widget.setLayout(systems_layout)
        parent.addWidget(systems_widget)
        
        # Systems group
        systems_group = QGroupBox("Ion Systems")
        systems_group_layout = QVBoxLayout()
        systems_group.setLayout(systems_group_layout)
        systems_layout.addWidget(systems_group)
        
        # Systems tree
        self.systems_tree = QTreeWidget()
        self.systems_tree.setHeaderLabels(['System', 'Details'])
        systems_group_layout.addWidget(self.systems_tree)
        
        # System controls
        system_controls = QHBoxLayout()
        self.add_system_btn = QPushButton("Add System")
        self.edit_system_btn = QPushButton("Edit")
        self.delete_system_btn = QPushButton("Delete")

        system_controls.addWidget(self.add_system_btn)
        system_controls.addWidget(self.edit_system_btn)
        system_controls.addWidget(self.delete_system_btn)
        systems_group_layout.addLayout(system_controls)

        tie_controls = QHBoxLayout()
        self.tie_btn = QPushButton("Tie Ions...")
        self.tie_btn.setToolTip("Share velocities and/or b parameters across ions at the same redshift")
        self.untie_btn = QPushButton("Untie")
        self.untie_btn.setToolTip("Remove the tie group containing the selected system")
        self.untie_btn.setEnabled(False)
        tie_controls.addWidget(self.tie_btn)
        tie_controls.addWidget(self.untie_btn)
        systems_group_layout.addLayout(tie_controls)

        # Initially disable edit/delete
        self.edit_system_btn.setEnabled(False)
        self.delete_system_btn.setEnabled(False)
        
    def setup_parameters_panel(self, parent):
        """Create parameters management panel"""
        params_widget = QWidget()
        params_layout = QVBoxLayout()
        params_widget.setLayout(params_layout)
        parent.addWidget(params_widget)
        
        # Parameters group
        params_group = QGroupBox("Parameter Estimation")
        params_group_layout = QVBoxLayout()
        params_group.setLayout(params_group_layout)
        params_layout.addWidget(params_group)
        
        # System selector
        system_layout = QHBoxLayout()
        system_layout.addWidget(QLabel("System:"))
        self.system_combo = QComboBox()
        system_layout.addWidget(self.system_combo)
        params_group_layout.addLayout(system_layout)
        
        # Parameter estimation controls
        param_controls = QHBoxLayout()
        
        # Interactive button (only if available)
        if HAS_INTERACTIVE_DIALOG:
            self.interactive_btn = QPushButton("Interactive Parameters")
            self.interactive_btn.setToolTip("Click on spectrum to estimate parameters")
            param_controls.addWidget(self.interactive_btn)
        else:
            self.interactive_btn = None
            
        self.manual_btn = QPushButton("Manual Setup")
        self.clear_params_btn = QPushButton("Clear")
        
        param_controls.addWidget(self.manual_btn)
        param_controls.addWidget(self.clear_params_btn)
        params_group_layout.addLayout(param_controls)
        
        # Component controls
        comp_controls = QHBoxLayout()
        self.add_component_btn = QPushButton("Add Component")
        self.delete_component_btn = QPushButton("Delete Component")
        
        comp_controls.addWidget(self.add_component_btn)
        comp_controls.addWidget(self.delete_component_btn)
        params_group_layout.addLayout(comp_controls)
        
        # Parameters table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(4)
        self.params_table.setHorizontalHeaderLabels(['Component', 'N (log)', 'b (km/s)', 'v (km/s)'])
        header = self.params_table.horizontalHeader()
        header.setStretchLastSection(True)
        params_group_layout.addWidget(self.params_table)
        
    def setup_bottom_controls(self, layout):
        """Create bottom control buttons"""
        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)
        
        # Master theta button
        self.show_master_theta_btn = QPushButton("Show Master Parameters")
        self.show_master_theta_btn.setEnabled(False)
        controls_layout.addWidget(self.show_master_theta_btn)
        
        controls_layout.addStretch()

        # Voigt method selector
        self.fast_voigt_check = QCheckBox("Fast Voigt approximation")
        self.fast_voigt_check.setChecked(False)
        self.fast_voigt_check.setToolTip(
            "Use the Tepper-García (2006) approximation instead of the exact\n"
            "scipy wofz Faddeeva function.\n\n"
            "Try using without multiprocessing for faster results.\n\n"
            "~3.7x faster Voigt computation; safe for weak/intermediate\n"
            "absorbers (damping parameter a < 0.01).\n\n"
            "NOT validated for strong DLA damping wings (a > 0.1).\n"
            "Leave unchecked for publication-quality fits."
        )
        controls_layout.addWidget(self.fast_voigt_check)

        # Compile button
        self.compile_btn = QPushButton("Compile Models")
        controls_layout.addWidget(self.compile_btn)
        
    def setup_connections(self):
        """Connect signals and slots"""
        # Configuration selection
        self.config_tree.itemSelectionChanged.connect(self.on_config_tree_selection_changed)
        self.current_config_combo.currentTextChanged.connect(self.on_current_config_changed)
        
        # System management
        self.add_system_btn.clicked.connect(self.add_system)
        self.edit_system_btn.clicked.connect(self.edit_system)
        self.delete_system_btn.clicked.connect(self.delete_system)
        self.tie_btn.clicked.connect(self.tie_ions)
        self.untie_btn.clicked.connect(self.untie_ions)
        self.systems_tree.itemSelectionChanged.connect(self.on_system_selection_changed)
        
        # Parameter management
        self.system_combo.currentTextChanged.connect(self.on_system_changed)
        if self.interactive_btn:
            self.interactive_btn.clicked.connect(self.launch_interactive_params)
        self.manual_btn.clicked.connect(self.setup_manual_params)
        self.clear_params_btn.clicked.connect(self.clear_params)
        self.add_component_btn.clicked.connect(self.add_component)
        self.delete_component_btn.clicked.connect(self.delete_component)
        self.fast_voigt_check.toggled.connect(self._on_voigt_method_changed)
        self.params_table.itemChanged.connect(self.on_parameter_table_edited)

        # Compilation
        self.compile_btn.clicked.connect(self.compile_models)
        self.show_master_theta_btn.clicked.connect(self.show_master_theta_dialog)

    def _on_voigt_method_changed(self, checked):
        """Update stored voigt_method when the checkbox is toggled."""
        self.voigt_method = 'fast' if checked else 'wofz'

    def set_configurations(self, configurations):
        """Set configurations from Tab 1"""
        self.configurations = configurations
        self.update_config_display()
        self.update_status()
        
    def update_config_display(self):
        """Update configuration tree display"""
        self.config_tree.clear()
        self.current_config_combo.clear()
        
        n_systems = len(self.global_systems)
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is not None:  # Only show configs with data
                item = QTreeWidgetItem()
                item.setText(0, config_name)
                item.setText(1, f"FWHM={config_data['fwhm']}, {n_systems} systems")
                item.setData(0, Qt.UserRole, config_name)
                self.config_tree.addTopLevelItem(item)
                self.current_config_combo.addItem(config_name)

    def launch_interactive_params(self):
        """Launch interactive parameter estimation with spectrum display"""
        if not self.current_system_id:
            QMessageBox.warning(self, "Selection Required",
                              "Please select a system")
            return

        # Find system in global list
        current_system = next(
            (s for s in self.global_systems if s['id'] == self.current_system_id), None
        )
        if not current_system:
            QMessageBox.warning(self, "System Not Found", "Selected system not found")
            return

        # Build all_spectra dict from every config that has data
        all_spectra = {
            name: {'wave': cfg['wave'], 'flux': cfg['flux'], 'error': cfg['error']}
            for name, cfg in self.configurations.items()
            if cfg['wave'] is not None
        }

        if not all_spectra:
            QMessageBox.warning(self, "No Data", "No configurations have data loaded")
            return

        try:
            tie = self._tie_for(current_system)
            if tie:
                dialog = InteractiveParameterDialog(current_system, all_spectra, parent=self,
                                                    tied_systems=self._tie_members(tie))
                dialog.group_parameters_ready.connect(self.on_group_parameters_ready)
            else:
                dialog = InteractiveParameterDialog(current_system, all_spectra, parent=self)
                dialog.parameters_ready.connect(self.on_interactive_parameters_ready)
            dialog.exec_()
            if getattr(dialog, 'n_fallbacks', None):
                self.update_status("No absorption measured for " + ", ".join(dialog.n_fallbacks)
                                   + ": starting log N set to a default; check these before fitting")

        except Exception as e:
            QMessageBox.critical(self, "Estimation Error",
                               f"Interactive parameter estimation failed:\n{str(e)}")
            self.setup_manual_params()

    def on_interactive_parameters_ready(self, df):
        """Handle interactive parameters result"""
        if self.current_system_id:
            self.config_parameters[self.current_system_id] = df.copy()
            self._sync_ties()
            self.load_parameter_table()
            self.update_systems_display()
            self.update_status("Interactive parameter estimation completed")

    def on_group_parameters_ready(self, tables):
        """Handle interactive result for a tie group (one table per ion)"""
        for system_id, df in tables.items():
            self.config_parameters[system_id] = df.copy()
        self._sync_ties()
        self.load_parameter_table()
        self.update_systems_display()
        self.update_status(f"Interactive parameters set for {len(tables)} tied ions")

    # ---------------- Tie management ----------------
    def _find_system(self, z, ion):
        """Global system dict for (z, ion), or None"""
        return next((s for s in self.global_systems
                     if abs(s['z'] - z) < 1e-6 and s['ion'] == ion), None)

    def _tie_for(self, system):
        """Tie group containing the system dict, or None"""
        for tie in self.global_ties:
            if abs(tie['z'] - system['z']) < 1e-6 and system['ion'] in tie['ions']:
                return tie
        return None

    def _tie_members(self, tie):
        """System dicts of a tie group, reference first"""
        return [self._find_system(tie['z'], ion) for ion in tie['ions']]

    def _tied_columns(self, system):
        """Parameter columns of this system that mirror its reference ion"""
        tie = self._tie_for(system)
        if not tie or tie['ions'][0] == system['ion']:
            return []
        return [col for col in ('v', 'b') if tie[col]]

    def _sync_ties(self):
        """
        Make each tied ion match its reference: same number of components,
        and tied columns (v and/or b) copied. N is kept for existing rows.
        If the reference has no parameters yet, the first member that does
        is used instead.
        """
        for tie in self.global_ties:
            members = self._tie_members(tie)
            source = next((s for s in members if s['id'] in self.config_parameters), None)
            if source is None:
                continue
            src = self.config_parameters[source['id']]
            cols = [col for col in ('v', 'b') if tie[col]]
            for member in members:
                if member is source:
                    continue
                df = self.config_parameters.get(member['id'])
                if df is None or len(df) != len(src):
                    rows = []
                    for i in range(len(src)):
                        if df is not None and i < len(df):
                            rows.append(df.iloc[i].to_dict())
                        else:
                            rows.append({'Component': src.iloc[i]['Component'], 'N': 13.0,
                                         'b': src.iloc[i]['b'], 'v': src.iloc[i]['v']})
                    df = pd.DataFrame(rows, columns=['Component', 'N', 'b', 'v'])
                else:
                    df = df.copy()
                for col in cols:
                    df[col] = src[col].values
                self.config_parameters[member['id']] = df
                member['components'] = len(df)
            source['components'] = len(src)

    def tie_ions(self):
        """Create a tie group from the dialog"""
        systems_by_z = {}
        for system in self.global_systems:
            if self._tie_for(system):
                continue
            z = next((zk for zk in systems_by_z if abs(zk - system['z']) < 1e-6), system['z'])
            systems_by_z.setdefault(z, []).append(system['ion'])
        systems_by_z = {z: ions for z, ions in systems_by_z.items() if len(ions) >= 2}
        if not systems_by_z:
            QMessageBox.information(self, "Nothing to Tie",
                                    "Tying needs at least two untied ions at the same redshift.")
            return

        dialog = TieDialog(systems_by_z, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        tie = dialog.get_tie()
        self.global_ties.append(tie)
        self._sync_ties()
        self.update_systems_display()
        self.load_parameter_table()
        tied = '+'.join(p for p in ('v', 'b') if tie[p])
        self.update_status(f"Tied {tied}: {', '.join(tie['ions'])} at z={tie['z']:.4f} "
                           f"(reference {tie['ions'][0]})")

    def untie_ions(self):
        """Remove the tie group containing the selected system"""
        system = self._selected_system()
        tie = self._tie_for(system) if system else None
        if not tie:
            return
        self.global_ties.remove(tie)
        self.update_systems_display()
        self.load_parameter_table()
        self.update_status(f"Removed tie: {', '.join(tie['ions'])}")

    def _selected_system(self):
        items = self.systems_tree.selectedItems()
        if not items:
            return None
        system_id = items[0].data(0, Qt.UserRole)
        return next((s for s in self.global_systems if s['id'] == system_id), None)

    def _drop_from_ties(self, system):
        """Remove a system from its tie group (whole group if <2 ions remain)"""
        tie = self._tie_for(system)
        if not tie:
            return
        tie['ions'] = [ion for ion in tie['ions'] if ion != system['ion']]
        if len(tie['ions']) < 2:
            self.global_ties.remove(tie)

    def compile_models(self):
        """Compile models for unified vfit interface"""
        if not self.configurations:
            QMessageBox.warning(self, "No Configurations", 
                              "No configurations available. Please set up configurations and data first.")
            return
            
        total_systems = len(self.global_systems)
        if total_systems == 0:
            QMessageBox.warning(self, "No Systems", 
                              "No ion systems defined. Please add systems before compiling.")
            return
            
        # Check if any parameters are estimated
        if not self.config_parameters:
            QMessageBox.warning(self, "No Parameters", 
                              "No parameters estimated. Please estimate parameters for your systems first.")
            return
            
        try:
            self.update_status("Creating master configuration...")


            missing = [s['ion'] for s in self.global_systems
                       if s['id'] not in self.config_parameters]
            if missing:
                QMessageBox.warning(self, "Missing Parameters",
                                    f"No parameters estimated for: {', '.join(missing)}")
                return

            # Tied ions follow their reference (component count and tied values)
            self._sync_ties()

            # Update component counts to match actual parameter rows
            for system in self.global_systems:
                if system['id'] in self.config_parameters:
                    system['components'] = len(self.config_parameters[system['id']])

            # Create master FitConfiguration from global systems
            self.master_config = FitConfiguration()
            for system in self.global_systems:
                self.master_config.add_system(
                    z=system['z'],
                    ion=system['ion'],
                    transitions=system['transitions'],
                    components=system['components']
                )
            for tie in self.global_ties:
                self.master_config.tie_ions(z=self._find_system(tie['z'], tie['ions'][0])['z'],
                                            ions=tie['ions'], v=tie['v'], b=tie['b'])
            
            self.update_status("Collecting parameters...")
            
            # Collect parameters in the model's theta order: systems (grouped
            # by redshift), then ions within each system
            all_N = []
            all_b = []
            all_v = []
            
            for abs_system in self.master_config.systems:
                for ion_group in abs_system.ion_groups:
                    system = self._find_system(abs_system.redshift, ion_group.ion_name)
                    df = self.config_parameters[system['id']]
                    for _, row in df.iterrows():
                        all_N.append(row['N'])
                        all_b.append(row['b'])
                        all_v.append(row['v'])
            
            # Create master theta array
            self.master_theta = np.array(all_N + all_b + all_v)
            
            self.update_status("Building instrument data...")
            
            # Build unified instrument data
            instrument_data = self.build_unified_instrument_data()
            
            self.update_status("Creating parameter bounds...")
            
            # Create bounds using vfit_mcmc
            n_comp = len(self.master_theta) // 3
            nguess = self.master_theta[:n_comp]
            bguess = self.master_theta[n_comp:2*n_comp]
            vguess = self.master_theta[2*n_comp:]
            
            bounds, lb, ub = mc.set_bounds(nguess, bguess, vguess)
            
            # Emit to fitting tab
            theta_dict = {'theta': self.master_theta, 'length': len(self.master_theta)}
            bounds_dict = {'lb': lb, 'ub': ub}
            self.model_updated.emit(instrument_data, theta_dict, bounds_dict)
            
            # Enable master theta button
            self.show_master_theta_btn.setEnabled(True)
            
            self.update_status("Compilation completed successfully!")
            
            n_free = int(np.sum(ParameterManager(self.master_config).get_free_mask()))
            tie_info = ""
            if n_free < len(self.master_theta):
                tie_info = f"\nTied parameters: {len(self.master_theta) - n_free} ({n_free} free)"
            QMessageBox.information(self, "Success", 
                                  f"Models compiled successfully!\n"
                                  f"Configurations: {len(instrument_data)}\n"
                                  f"Total parameters: {len(self.master_theta)}\n"
                                  f"Systems: {total_systems}" + tie_info)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Compilation Error", f"Failed to compile models:\n{str(e)}")
            self.update_status(f"Compilation failed: {str(e)}")
    
    def build_unified_instrument_data(self):
        """Build unified instrument data dictionary for vfit interface"""
        instrument_data = {}
        
        for config_name, config_data in self.configurations.items():
            if config_data['wave'] is None:
                continue
                
            # Create VoigtModel with instrument-specific FWHM
            fwhm_unit = config_data.get('fwhm_unit', 'pixels')
            if fwhm_unit == 'km/s':
                fwhm_pix = mean_fwhm_pixels(config_data['fwhm'], config_data['wave'])
            else:
                fwhm_pix = config_data['fwhm']
            model = VoigtModel(self.master_config, FWHM=str(fwhm_pix),
                               voigt_method=self.voigt_method)
            
            # Compile model
            #compiled_model = model.compile(verbose=False)
            
            # Create unified instrument data entry
            instrument_data[config_name] = {
                'model': model,  # Compiled VoigtModel object
                'wave': config_data['wave'],
                'flux': config_data['flux'],
                'error': config_data['error']
            }
        
        return instrument_data

    # System management methods
    def add_system(self):
        """Add new ion system with full custom support"""
        if not self.current_config:
            QMessageBox.warning(self, "No Configuration", "Please select a configuration first")
            return
            
        dialog = SystemDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            system_data = dialog.get_system_data()
            
            # Validate system data
            if not system_data['ion']:
                QMessageBox.warning(self, "Invalid Ion", "Please specify an ion name")
                return
                
            if not system_data['transitions']:
                QMessageBox.warning(self, "Invalid Transitions", "Please specify at least one transition wavelength")
                return
            
            # Add to global systems
            self.global_systems.append(system_data)
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_status(f"Added system: {system_data['ion']} at z={system_data['z']}")
    
    def edit_system(self):
        """Edit selected system"""
        items = self.systems_tree.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Selection", "Please select a system to edit")
            return
            
        system_id = items[0].data(0, Qt.UserRole)
        for i, system in enumerate(self.global_systems):
            if system['id'] == system_id:
                dialog = SystemDialog(system_data=system, parent=self)
                if dialog.exec_() == QDialog.Accepted:
                    new_system_data = dialog.get_system_data()
                    if (new_system_data['ion'] != system['ion'] or
                            abs(new_system_data['z'] - system['z']) >= 1e-6) and self._tie_for(system):
                        self._drop_from_ties(system)
                        self.update_status(f"{system['ion']} removed from its tie group (ion/redshift changed)")
                    self.global_systems[i] = new_system_data
                    self.update_systems_display()
                    self.update_system_combo()
                    self.update_status(f"Updated system: {new_system_data['ion']} at z={new_system_data['z']}")
                break
    
    def delete_system(self):
        """Delete selected system"""
        items = self.systems_tree.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Selection", "Please select a system to delete")
            return
            
        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   "Are you sure you want to delete this system?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            system_id = items[0].data(0, Qt.UserRole)
            for s in self.global_systems:
                if s['id'] == system_id:
                    self._drop_from_ties(s)
            self.global_systems = [s for s in self.global_systems if s['id'] != system_id]
            self.config_parameters.pop(system_id, None)
            
            self.update_systems_display()
            self.update_system_combo()
            self.update_status("System deleted")

    # Parameter management methods
    def setup_manual_params(self):
        """Set up manual parameter estimation"""
        if not self.current_config or not self.current_system_id:
            QMessageBox.warning(self, "Selection Required", 
                              "Please select a configuration and system")
            return
            
        current_system = next(
            (s for s in self.global_systems if s['id'] == self.current_system_id), None
        )
        if not current_system:
            QMessageBox.warning(self, "System Not Found", "Selected system not found")
            return

        import pandas as pd
        n_components = current_system['components']
        data = []
        
        for i in range(n_components):
            data.append({
                'Component': f'Comp_{i+1}',
                'N': 13.0,  # Default log column density
                'b': 20.0,  # Default b-parameter
                'v': 0.0    # Default velocity
            })
        
        params_df = pd.DataFrame(data)
        
        self.config_parameters[self.current_system_id] = params_df
        self._sync_ties()
        
        # Update display
        self.load_parameter_table()
        self.update_status("Manual parameter setup completed")
    
    def clear_params(self):
        """Clear parameters for current system"""
        if not self.current_system_id:
            return
        self.config_parameters.pop(self.current_system_id, None)
            
        self.params_table.setRowCount(0)
        self.update_status("Parameters cleared")
    
    def _component_target_id(self):
        """System whose components are edited: the reference ion if tied"""
        system = next((s for s in self.global_systems if s['id'] == self.current_system_id), None)
        tie = self._tie_for(system) if system else None
        if tie:
            return self._find_system(tie['z'], tie['ions'][0])['id']
        return self.current_system_id

    def add_component(self):
        """Add component to current system"""
        if not self.current_system_id:
            return

        key = self._component_target_id()
        if key not in self.config_parameters:
            import pandas as pd
            self.config_parameters[key] = pd.DataFrame(columns=['Component', 'N', 'b', 'v'])

        df = self.config_parameters[key]
        new_comp = len(df) + 1
        
        # Add new row
        import pandas as pd
        new_row = pd.DataFrame({
            'Component': [f'Comp_{new_comp}'],
            'N': [13.0],
            'b': [20.0],
            'v': [0.0]
        })
        self.config_parameters[key] = pd.concat([df, new_row], ignore_index=True)
        self._sync_ties()
        
        # Update display
        self.load_parameter_table()
        self.update_status(f"Added component {new_comp}")
    
    def delete_component(self):
        """Delete last component from current system"""
        if not self.current_system_id:
            return

        key = self._component_target_id()
        if key not in self.config_parameters:
            return

        df = self.config_parameters[key]
        if len(df) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete the last component")
            return

        self.config_parameters[key] = df.iloc[:-1].reset_index(drop=True)
        self._sync_ties()
        
        # Update display
        self.load_parameter_table()
        self.update_status("Deleted last component")

    def show_master_theta_dialog(self):
        """Show master parameter array dialog"""
        if self.master_theta is None:
            QMessageBox.warning(self, "No Compilation", "No compiled model available")
            return
            
        # Create simple parameter dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Master Parameters")
        dialog.setModal(True)
        dialog.resize(400, 600)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        # Info
        n_params = len(self.master_theta)
        info_label = QLabel(f"Master Parameter Array: {n_params} parameters")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(info_label)
        
        # Table
        table = QTableWidget()
        table.setRowCount(n_params)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Index', 'Type', 'Value'])
        
        n_comp = n_params // 3
        for i in range(n_params):
            # Index
            table.setItem(i, 0, QTableWidgetItem(str(i)))
            
            # Type
            if i < n_comp:
                param_type = f'N_{i+1}'
            elif i < 2*n_comp:
                param_type = f'b_{i-n_comp+1}'
            else:
                param_type = f'v_{i-2*n_comp+1}'
            table.setItem(i, 1, QTableWidgetItem(param_type))
            
            # Value (editable)
            table.setItem(i, 2, QTableWidgetItem(f"{self.master_theta[i]:.6f}"))
        
        # Make index and type read-only
        for i in range(n_params):
            for col in [0, 1]:
                item = table.item(i, col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # Get updated parameters
            try:
                updated_theta = np.zeros(n_params)
                for i in range(n_params):
                    item = table.item(i, 2)
                    updated_theta[i] = float(item.text())
                
                # Update master theta
                self.master_theta = updated_theta
                self.update_status("Master parameters updated")
                    
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Invalid parameter values: {e}")

    # Event handlers
    def on_config_tree_selection_changed(self):
        """Handle configuration tree selection"""
        items = self.config_tree.selectedItems()
        if items:
            config_name = items[0].data(0, Qt.UserRole)
            self.current_config_combo.setCurrentText(config_name)
    
    def on_current_config_changed(self, config_name):
        """Handle current configuration change"""
        self.current_config = config_name
        self.update_systems_display()
        self.update_system_combo()
    
    def on_system_selection_changed(self):
        """Handle system selection change"""
        items = self.systems_tree.selectedItems()
        has_selection = len(items) > 0
        self.edit_system_btn.setEnabled(has_selection)
        self.delete_system_btn.setEnabled(has_selection)
        system = self._selected_system()
        self.untie_btn.setEnabled(bool(system and self._tie_for(system)))
    
    def on_system_changed(self, system_text):
        """Handle system combo change"""
        if not system_text or not self.current_config:
            return
            
        # Extract system ID from combo text
        parts = system_text.split(' - ')
        if len(parts) >= 2:
            self.current_system_id = parts[0]
            self.load_parameter_table()
    
    def on_parameter_table_edited(self, item):
        """Handle parameter table cell edits"""
        if not self.current_system_id:
            return

        row = item.row()
        col = item.column()

        key = self.current_system_id
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        if row >= len(df):
            return
            
        # Map column to DataFrame column
        col_names = ['Component', 'N', 'b', 'v']
        if col < len(col_names):
            try:
                import pandas as pd
                col_name = col_names[col]
                col_dtype = df[col_name].dtype
                if pd.api.types.is_integer_dtype(col_dtype):
                    df.iloc[row, col] = int(float(item.text()))
                elif pd.api.types.is_float_dtype(col_dtype):
                    df.iloc[row, col] = float(item.text())
                else:
                    df.iloc[row, col] = str(item.text())
                    
                # Tied ions mirror their reference
                if col_name in ('v', 'b') and self.global_ties:
                    self._sync_ties()

                # Update status
                if self.master_theta is not None:
                    self.update_status("Parameter updated (compile models to propagate changes)")
                    
            except ValueError:
                # Invalid input - revert to original value
                self.load_parameter_table()
                self.update_status("Invalid parameter value - reverted to original")

    # Display update methods
    def update_systems_display(self):
        """Update systems tree display — shows all global systems"""
        self.systems_tree.clear()
        for system in self.global_systems:
            item = QTreeWidgetItem()
            item.setText(0, f"{system['ion']} z={system['z']:.4f}")
            n_params = len(self.config_parameters.get(system['id'], []))
            n_transitions = len(system['transitions'])
            details = f"{n_transitions} lines, {system['components']} comp, {n_params} params"
            tie = self._tie_for(system)
            if tie:
                tied = '+'.join(p for p in ('v', 'b') if tie[p])
                if tie['ions'][0] == system['ion']:
                    details += f", tie reference ({tied})"
                else:
                    details += f", {tied} tied to {tie['ions'][0]}"
            item.setText(1, details)
            item.setData(0, Qt.UserRole, system['id'])
            self.systems_tree.addTopLevelItem(item)
    
    def update_system_combo(self):
        """Update system selector combo — always shows all global systems"""
        self.system_combo.clear()
        for system in self.global_systems:
            text = f"{system['id']} - {system['ion']} z={system['z']:.4f}"
            self.system_combo.addItem(text)
    
    def load_parameter_table(self):
        """Load parameters into table"""
        self.params_table.setRowCount(0)

        if not self.current_system_id:
            return

        key = self.current_system_id
        if key not in self.config_parameters:
            return
            
        df = self.config_parameters[key]
        self.params_table.setRowCount(len(df))

        system = next((s for s in self.global_systems if s['id'] == key), None)
        tie = self._tie_for(system) if system else None
        tied_cols = self._tied_columns(system) if system else []
        
        self.params_table.blockSignals(True)
        for i, row in df.iterrows():
            for j, (col_name, value) in enumerate(row.items()):
                if j < 4:  # Only show first 4 columns
                    item = QTableWidgetItem(str(value))
                    if col_name in tied_cols:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setForeground(QBrush(QColor('gray')))
                        item.setToolTip(f"Tied to {tie['ions'][0]} — edit it there")
                    self.params_table.setItem(i, j, item)
        self.params_table.blockSignals(False)
    
    def update_status(self, message=""):
        """Update status text"""
        if message:
            self.status_text.append(message)
        else:
            # Default status
            n_configs = len([c for c in self.configurations.values() if c['wave'] is not None])
            total_systems = len(self.global_systems)
            self.status_text.clear()
            self.status_text.append(f"Configurations: {n_configs}, Total systems: {total_systems}")
            
            if self.master_theta is not None:
                n_params = len(self.master_theta)
                self.status_text.append(f"Compiled: {n_params} parameters ready")
    
    def clear_state(self):
        """Clear all state (for project loading)"""
        self.configurations = {}
        self.global_systems = []
        self.config_parameters = {}
        self.global_ties = []
        self.current_config = None
        self.current_system_id = None
        self.master_config = None
        self.master_theta = None
        self.param_manager = None
        
        # Clear UI
        self.config_tree.clear()
        self.current_config_combo.clear()
        self.systems_tree.clear()
        self.system_combo.clear()
        self.params_table.setRowCount(0)
        self.status_text.clear()
        
        # Disable buttons
        self.show_master_theta_btn.setEnabled(False)
        self.untie_btn.setEnabled(False)
        self.edit_system_btn.setEnabled(False)
        self.delete_system_btn.setEnabled(False)