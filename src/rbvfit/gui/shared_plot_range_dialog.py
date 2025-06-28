#!/usr/bin/env python
"""
Shared Plot Range Dialog for rbvfit 2.0 GUI

This dialog can be used by both Model Tab and Fitting Tab for setting custom plot ranges.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QLabel, QDoubleSpinBox, QCheckBox, QPushButton)
from PyQt5.QtCore import Qt


class PlotRangeDialog(QDialog):
    """Dialog for setting custom plot ranges"""
    
    def __init__(self, current_xlim, current_ylim, original_xlim, original_ylim, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Plot Range")
        self.setModal(True)
        self.resize(350, 200)
        
        self.original_xlim = original_xlim
        self.original_ylim = original_ylim
        
        self.setup_ui(current_xlim, current_ylim)
        
    def setup_ui(self, current_xlim, current_ylim):
        """Create dialog interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # X Range controls
        x_group = QGroupBox("X Range (Wavelength)")
        x_layout = QHBoxLayout()
        x_group.setLayout(x_layout)
        layout.addWidget(x_group)
        
        x_layout.addWidget(QLabel("Min:"))
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-10000, 10000)
        self.xmin_spin.setDecimals(2)
        self.xmin_spin.setValue(current_xlim[0] if current_xlim else 0)
        x_layout.addWidget(self.xmin_spin)
        
        x_layout.addWidget(QLabel("Max:"))
        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setRange(-10000, 10000)
        self.xmax_spin.setDecimals(2)
        self.xmax_spin.setValue(current_xlim[1] if current_xlim else 1)
        x_layout.addWidget(self.xmax_spin)
        
        self.auto_x_check = QCheckBox("Auto X")
        self.auto_x_check.setChecked(current_xlim is None)
        self.auto_x_check.toggled.connect(self.on_auto_x_toggled)
        x_layout.addWidget(self.auto_x_check)
        
        # Y Range controls
        y_group = QGroupBox("Y Range (Flux)")
        y_layout = QHBoxLayout()
        y_group.setLayout(y_layout)
        layout.addWidget(y_group)
        
        y_layout.addWidget(QLabel("Min:"))
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-10, 10)
        self.ymin_spin.setDecimals(3)
        self.ymin_spin.setValue(current_ylim[0] if current_ylim else 0)
        y_layout.addWidget(self.ymin_spin)
        
        y_layout.addWidget(QLabel("Max:"))
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-10, 10)
        self.ymax_spin.setDecimals(3)
        self.ymax_spin.setValue(current_ylim[1] if current_ylim else 1)
        y_layout.addWidget(self.ymax_spin)
        
        self.auto_y_check = QCheckBox("Auto Y")
        self.auto_y_check.setChecked(current_ylim is None)
        self.auto_y_check.toggled.connect(self.on_auto_y_toggled)
        y_layout.addWidget(self.auto_y_check)
        
        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setToolTip("Reset to original auto-calculated ranges")
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # Connect signals
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.apply_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Update initial state
        self.on_auto_x_toggled(self.auto_x_check.isChecked())
        self.on_auto_y_toggled(self.auto_y_check.isChecked())
        
    def on_auto_x_toggled(self, checked):
        """Handle auto X checkbox"""
        self.xmin_spin.setEnabled(not checked)
        self.xmax_spin.setEnabled(not checked)
        
    def on_auto_y_toggled(self, checked):
        """Handle auto Y checkbox"""
        self.ymin_spin.setEnabled(not checked)
        self.ymax_spin.setEnabled(not checked)
        
    def reset_to_original(self):
        """Reset to original ranges"""
        if self.original_xlim:
            self.xmin_spin.setValue(self.original_xlim[0])
            self.xmax_spin.setValue(self.original_xlim[1])
            self.auto_x_check.setChecked(False)
            
        if self.original_ylim:
            self.ymin_spin.setValue(self.original_ylim[0])
            self.ymax_spin.setValue(self.original_ylim[1])
            self.auto_y_check.setChecked(False)
            
    def get_ranges(self):
        """Get selected ranges"""
        if self.auto_x_check.isChecked():
            xlim = None
        else:
            xlim = (self.xmin_spin.value(), self.xmax_spin.value())
            
        if self.auto_y_check.isChecked():
            ylim = None
        else:
            ylim = (self.ymin_spin.value(), self.ymax_spin.value())
            
        return xlim, ylim