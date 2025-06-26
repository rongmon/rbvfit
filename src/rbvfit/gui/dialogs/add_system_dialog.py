#!/usr/bin/env python
"""
rbvfit 2.0 Add System Dialog - PyQt5 Implementation

Dialog for adding new absorption systems with auto-detection.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                            QLineEdit, QDoubleSpinBox, QListWidget, QPushButton,
                            QLabel, QMessageBox, QCheckBox, QListWidgetItem,
                            QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator

from rbvfit.gui.io import detect_ion_from_wavelength

try:
    from rbvfit import rb_setline as rb
    HAS_RB_SETLINE = True
except ImportError:
    HAS_RB_SETLINE = False


class AddSystemDialog(QDialog):
    """Dialog for adding new absorption systems"""
    
    system_added = pyqtSignal(dict)  # Signal emitted when system is added
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Absorption System")
        self.setModal(True)
        self.resize(500, 600)
        
        self.transitions = []  # List of wavelengths
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Create dialog interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # System parameters group
        sys_group = QGroupBox("System Parameters")
        sys_layout = QFormLayout()
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        # Redshift input
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.0, 10.0)
        self.z_spin.setDecimals(4)
        self.z_spin.setValue(0.0)
        self.z_spin.setSingleStep(0.001)
        self.z_spin.setToolTip("Redshift of the absorption system")
        sys_layout.addRow("Redshift:", self.z_spin)
        
        # Ion name input
        self.ion_edit = QLineEdit()
        self.ion_edit.setPlaceholderText("e.g., MgII, FeII, OVI (leave empty for auto-detect)")
        self.ion_edit.setToolTip("Ion name - leave empty to auto-detect from first transition")
        sys_layout.addRow("Ion Name:", self.ion_edit)
        
        # Transitions group
        trans_group = QGroupBox("Transitions")
        trans_layout = QVBoxLayout()
        trans_group.setLayout(trans_layout)
        layout.addWidget(trans_group)
        
        # Add transition input
        add_trans_layout = QHBoxLayout()
        trans_layout.addLayout(add_trans_layout)
        
        self.wave_edit = QLineEdit()
        self.wave_edit.setPlaceholderText("Enter wavelength in Å")
        self.wave_edit.setValidator(QDoubleValidator(0.0, 10000.0, 3))
        self.wave_edit.setToolTip("Rest wavelength of transition in Angstroms")
        add_trans_layout.addWidget(QLabel("Wavelength (Å):"))
        add_trans_layout.addWidget(self.wave_edit)
        
        self.add_trans_btn = QPushButton("Add Transition")
        self.add_trans_btn.setToolTip("Add this transition to the system")
        add_trans_layout.addWidget(self.add_trans_btn)
        
        # Auto-detect button
        self.auto_detect_btn = QPushButton("Auto-detect Ion")
        self.auto_detect_btn.setEnabled(False)
        self.auto_detect_btn.setToolTip("Auto-detect ion name from first transition")
        add_trans_layout.addWidget(self.auto_detect_btn)
        
        # Transition list
        self.trans_list = QListWidget()
        self.trans_list.setToolTip("List of transitions for this system")
        trans_layout.addWidget(QLabel("Selected Transitions:"))
        trans_layout.addWidget(self.trans_list)
        
        # Remove transition button
        self.remove_trans_btn = QPushButton("Remove Selected")
        self.remove_trans_btn.setEnabled(False)
        self.remove_trans_btn.setToolTip("Remove selected transition from list")
        trans_layout.addWidget(self.remove_trans_btn)
        
        # Preview area
        preview_group = QGroupBox("System Preview")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(100)
        self.preview_text.setReadOnly(True)
        self.preview_text.setToolTip("Preview of the system to be added")
        preview_layout.addWidget(self.preview_text)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setEnabled(False)
        self.ok_btn.setDefault(True)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.ok_btn)
        
        self.update_preview()
        
    def setup_connections(self):
        """Connect signals and slots"""
        self.add_trans_btn.clicked.connect(self.add_transition)
        self.remove_trans_btn.clicked.connect(self.remove_transition)
        self.auto_detect_btn.clicked.connect(self.auto_detect_ion)
        
        self.wave_edit.returnPressed.connect(self.add_transition)
        self.trans_list.itemSelectionChanged.connect(self.on_selection_changed)
        
        self.z_spin.valueChanged.connect(self.update_preview)
        self.ion_edit.textChanged.connect(self.update_preview)
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
    def add_transition(self):
        """Add transition to the list"""
        try:
            wavelength = float(self.wave_edit.text())
            if wavelength <= 0:
                raise ValueError("Wavelength must be positive")
                
            if wavelength in self.transitions:
                QMessageBox.warning(self, "Warning", 
                                  f"Transition {wavelength:.3f} Å already added")
                return
                
            self.transitions.append(wavelength)
            
            # Get transition info if available
            trans_info = f"{wavelength:.3f} Å"
            if HAS_RB_SETLINE:
                try:
                    line_info = rb.rb_setline(wavelength, 'closest')
                    trans_info += f" ({line_info['name'][0]})"
                except Exception:
                    pass
            
            item = QListWidgetItem(trans_info)
            item.setData(Qt.UserRole, wavelength)
            self.trans_list.addItem(item)
            
            self.wave_edit.clear()
            self.auto_detect_btn.setEnabled(len(self.transitions) > 0)
            self.update_preview()
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid wavelength")
            
    def remove_transition(self):
        """Remove selected transition"""
        current_item = self.trans_list.currentItem()
        if current_item:
            wavelength = current_item.data(Qt.UserRole)
            self.transitions.remove(wavelength)
            
            row = self.trans_list.row(current_item)
            self.trans_list.takeItem(row)
            
            self.auto_detect_btn.setEnabled(len(self.transitions) > 0)
            self.update_preview()
            
    def on_selection_changed(self):
        """Handle transition list selection changes"""
        has_selection = self.trans_list.currentItem() is not None
        self.remove_trans_btn.setEnabled(has_selection)
        
    def auto_detect_ion(self):
        """Auto-detect ion name from first transition"""
        if not self.transitions:
            return
            
        first_wavelength = self.transitions[0]
        detected_ion = detect_ion_from_wavelength(first_wavelength)
        
        if detected_ion:
            reply = QMessageBox.question(
                self, "Auto-detect Ion",
                f"Detected ion: {detected_ion}\n\nUse this ion name?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.ion_edit.setText(detected_ion)
        else:
            QMessageBox.information(
                self, "Auto-detect", 
                "Could not auto-detect ion from transition wavelength"
            )
            
    def update_preview(self):
        """Update system preview"""
        z = self.z_spin.value()
        ion = self.ion_edit.text().strip()
        n_trans = len(self.transitions)
        
        if not ion:
            ion = "Auto-detect"
            
        preview = f"System: {ion} at z = {z:.4f}\n"
        preview += f"Transitions: {n_trans} selected"
        
        if self.transitions:
            trans_str = ", ".join([f"{w:.1f}" for w in sorted(self.transitions)])
            preview += f"\nWavelengths: {trans_str} Å"
            
        self.preview_text.setPlainText(preview)
        
        # Enable OK button if we have at least one transition
        self.ok_btn.setEnabled(n_trans > 0)
        
    def get_result(self):
        """Get the dialog result"""
        ion_name = self.ion_edit.text().strip()
        
        # Auto-detect ion if empty
        if not ion_name and self.transitions:
            ion_name = detect_ion_from_wavelength(self.transitions[0])
            if not ion_name:
                ion_name = "Unknown"
                
        return {
            'redshift': self.z_spin.value(),
            'ion_name': ion_name,
            'transitions': sorted(self.transitions),
            'components': 1  # Default - will be updated via interactive mode
        }
        
    @classmethod
    def get_system(cls, parent=None):
        """Static method to get system data"""
        dialog = cls(parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_result()
        return None