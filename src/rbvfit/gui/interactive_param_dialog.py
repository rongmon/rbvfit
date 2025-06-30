#!/usr/bin/env python
"""
Clean Interactive Parameter Dialog for rbvfit2 GUI
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QMessageBox, QComboBox, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QDoubleSpinBox,
                            QFormLayout,QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from rbvfit import guess_profile_parameters_interactive as g
from rbvfit import rb_setline as line

def calculate_velocity(wave, wrest, zabs):
    """Calculate velocity array from wavelength"""
    c = 2.9979e5  # km/s
    wave_rest = wave / (1.0 + zabs)
    vel = (wave_rest - wrest) * c / wrest
    return vel



def _get_default_b_from_transition(transition_name: str) -> float:
    """
    Get default b-parameter based on transition ionization state.
    
    Parameters
    ----------
    transition_name : str
        Transition name from rb_setline, e.g., 'HI 1215', 'OVI 1031'
        
    Returns
    -------
    float
        Default b-parameter in km/s
    """
    # Default fallback value
    default_b = 25.0
    
    # Extract ionization state (roman numerals)
    if 'I ' in transition_name or transition_name.endswith('I'):
        # Neutral species (I)
        default_b = 22.0
    elif 'II ' in transition_name or transition_name.endswith('II'):
        # Singly ionized (II) 
        default_b = 18.0
    elif 'III ' in transition_name or transition_name.endswith('III'):
        # Doubly ionized (III)
        default_b = 25.0
    elif 'IV ' in transition_name or transition_name.endswith('IV'):
        # Triply ionized (IV)
        default_b = 30.0
    elif any(ion in transition_name for ion in ['V ', 'VI ', 'VII ', 'VIII ']):
        # Higher ionization states
        default_b = 40.0
        
    return default_b

class RangeDialog(QDialog):
    """Dialog for setting plot ranges"""
    
    def __init__(self, current_xlim, current_ylim, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Plot Range")
        self.setModal(True)
        
        layout = QFormLayout()
        self.setLayout(layout)
        
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-10000, 10000)
        self.xmin_spin.setValue(current_xlim[0])
        layout.addRow("X min:", self.xmin_spin)
        
        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setRange(-10000, 10000)
        self.xmax_spin.setValue(current_xlim[1])
        layout.addRow("X max:", self.xmax_spin)
        
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-20000, 20000)
        self.ymin_spin.setDecimals(3)
        self.ymin_spin.setValue(current_ylim[0])
        layout.addRow("Y min:", self.ymin_spin)
        
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-2, 2)
        self.ymax_spin.setDecimals(3)
        self.ymax_spin.setValue(current_ylim[1])
        layout.addRow("Y max:", self.ymax_spin)
        
        # Buttons
        button_layout = QHBoxLayout()
        layout.addRow(button_layout)
        
        reset_btn = QPushButton("Reset")
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(cancel_btn)
        
        reset_btn.clicked.connect(self.reset_range)
        apply_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        self.original_xlim = current_xlim
        self.original_ylim = current_ylim
        
    def reset_range(self):
        """Reset to original range"""
        self.xmin_spin.setValue(self.original_xlim[0])
        self.xmax_spin.setValue(self.original_xlim[1])
        self.ymin_spin.setValue(self.original_ylim[0])
        self.ymax_spin.setValue(self.original_ylim[1])
        
    def get_ranges(self):
        """Get the selected ranges"""
        xlim = (self.xmin_spin.value(), self.xmax_spin.value())
        ylim = (self.ymin_spin.value(), self.ymax_spin.value())
        return xlim, ylim


class BParameterDialog(QDialog):
    """Dialog for setting b parameters"""
    
    def __init__(self, n_estimates, v_selected, transitions, transition_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Doppler Parameters (b)")
        self.setModal(True)
        self.resize(450, 300)
        self.transition_name = transition_name
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        info_label = QLabel("Set Doppler parameter (b) for each component:")
        layout.addWidget(info_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Component', 'N (est.)', 'v (km/s)', 'b (km/s)'])
        self.table.setRowCount(len(n_estimates))
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        # Get smart default b based on transition
        default_b = _get_default_b_from_transition(transition_name)
        
        # Populate table
        for i, (n_est, v_sel) in enumerate(zip(n_estimates, v_selected)):
            # Component number
            self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # N estimate (read-only)
            n_item = QTableWidgetItem(f"{n_est:.2f}")
            n_item.setFlags(n_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 1, n_item)
            
            # v selected (read-only)
            v_item = QTableWidgetItem(f"{v_sel:.1f}")
            v_item.setFlags(v_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 2, v_item)
            
            # b parameter (editable) - smart default based on transition
            self.table.setItem(i, 3, QTableWidgetItem(f"{default_b:.1f}"))
            
        layout.addWidget(self.table)
        
        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
    def get_b_parameters(self):
        """Get the b parameters from table"""
        b_params = []
        for i in range(self.table.rowCount()):
            try:
                b_val = float(self.table.item(i, 3).text())
                b_params.append(b_val)
            except (ValueError, AttributeError):
                b_params.append(_get_default_b_from_transition(self.transition_name))
        return b_params


class VelocitySelector:
    """Clean velocity selection class"""
    
    def __init__(self, wave, flux, error, zabs, wrest):
        self.wave = wave
        self.flux = flux
        self.error = error
        self.zabs = zabs
        

        # Quickly look up the transition we're using
        line_info = line.rb_setline(wrest, 'closest', 'atom')
        self.transition_name = line_info['name'][0]
        self.f0 = line_info['fval'][0]
        self.wrest = line_info['wave'][0] 

        # Calculate velocity
        self.vel = calculate_velocity(wave, self.wrest, zabs)
        
        # Storage for selections
        self.vel_guess = []
        self.markers = []
        self.ax = None
        
    def setup_plot(self, ax):
        """Set up the velocity plot"""
        self.ax = ax
        ax.clear()
        
        # Plot spectrum
        ax.step(self.vel, self.flux, 'k-', linewidth=0.8, label='Flux')
        ax.fill_between(self.vel, self.flux - self.error, self.flux + self.error, 
                       alpha=0.3, color='gray', label='Error')
        
        ax.set_xlim([-600, 600])
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Select Components - {self.wrest:.2f} Å')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    def add_component(self, vel_pos):
        """Add velocity component"""
        self.vel_guess.append(vel_pos)
        
        # Add visual marker
        line = self.ax.axvline(vel_pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add label
        ylim = self.ax.get_ylim()
        text = self.ax.text(vel_pos, ylim[1]*0.9, f'C{len(self.vel_guess)}', 
                           ha='center', va='top', color='red', fontweight='bold')
        
        self.markers.extend([line, text])
        
    def remove_nearest(self, vel_pos):
        """Remove nearest component"""
        if not self.vel_guess:
            return
            
        # Find nearest
        distances = [abs(v - vel_pos) for v in self.vel_guess]
        nearest_idx = distances.index(min(distances))
        
        # Remove from list
        removed_vel = self.vel_guess.pop(nearest_idx)
        print(f"Removed component at {removed_vel:.1f} km/s")  # DEBUG
        
        # Just clear markers - let the parent redraw everything
        self.clear_markers()
    
    # DON'T redraw here - let the calling code handle it            
    def clear_markers(self):
        """Clear all visual markers"""
        for marker in self.markers:
            marker.remove()
        self.markers = []
        
    def clear_all(self):
        """Clear everything"""
        self.clear_markers()
        self.vel_guess = []


class InteractiveParameterDialog(QDialog):
    """Clean interactive parameter dialog"""
    
    parameters_ready = pyqtSignal(object)  # DataFrame
    
    def __init__(self, system_data, spectrum_data, parent=None):
        super().__init__(parent)
        self.system_data = system_data
        self.spectrum_data = spectrum_data
        self.velocity_selector = None
        self.original_xlim = [-600, 600]
        self.original_ylim = None
        
        self.setWindowTitle(f"Interactive Parameters - {system_data['ion']} z={system_data['z']:.3f}")
        self.setModal(True)
        self.resize(1000, 700)
        
        self.setup_ui()
        self.start_selection()
        
    def setup_ui(self):
        """Create UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header info
        info_text = f"System: {self.system_data['ion']} at z={self.system_data['z']:.6f}"
        layout.addWidget(QLabel(info_text))
        
        # Transition selector for multi-transition ions
        if len(self.system_data['transitions']) > 1:
            trans_layout = QHBoxLayout()
            layout.addLayout(trans_layout)
            
            trans_layout.addWidget(QLabel("Transition:"))
            self.transition_combo = QComboBox()
            
            for trans in self.system_data['transitions']:
                label = f"{self.system_data['ion']} {trans:.1f} Å"
                self.transition_combo.addItem(label, trans)
                
            trans_layout.addWidget(self.transition_combo)
            trans_layout.addStretch()
            
            self.transition_combo.currentIndexChanged.connect(self.change_transition)
        else:
            self.transition_combo = None
            
        # Instructions
        inst_text = ("Left click: add | Right click: remove nearest | 'r': reset | 'c': clear all\n"
             "'z': zoom in | 'o': zoom out | 'x'/'X': set xmin/xmax | 'y'/'Y': set ymin/ymax | 'm': manual range")
        layout.addWidget(QLabel(inst_text))        
        # Plot
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        # ENABLE KEYBOARD FOCUS
        self.canvas.setFocusPolicy(Qt.ClickFocus)  # Allow canvas to receive focus
        self.canvas.setFocus()  # Give it focus initially

        layout.addWidget(self.canvas)
        
        # Status
        self.status_label = QLabel("Click on absorption features...")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        
        self.done_btn = QPushButton("Done - Set Parameters")
        cancel_btn = QPushButton("Cancel")
        
        button_layout.addStretch()
        button_layout.addWidget(self.done_btn)
        button_layout.addWidget(cancel_btn)
        
        
        # Connect
        self.done_btn.clicked.connect(self.finish_selection)
        cancel_btn.clicked.connect(self.reject)
        
        # Matplotlib events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)
        
    def start_selection(self):
        """Start velocity selection"""
        try:
            wrest = self.get_current_transition()
            wave = self.spectrum_data['wave']
            flux = self.spectrum_data['flux']
            error = self.spectrum_data.get('error', np.ones_like(flux) * 0.05)
            zabs = self.system_data['z']
            
            self.velocity_selector = VelocitySelector(wave, flux, error, zabs, wrest)
            self.velocity_selector.setup_plot(self.ax)
            
            if self.original_ylim is None:
                self.original_ylim = self.ax.get_ylim()
                
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start selection:\n{str(e)}")
            self.reject()
            
    def get_current_transition(self):
        """Get currently selected transition"""
        if self.transition_combo:
            return self.transition_combo.currentData()
        else:
            return self.system_data['transitions'][0]
            
    def change_transition(self):
        """Handle transition change"""
        if self.velocity_selector:
            wrest = self.get_current_transition()
            wave = self.spectrum_data['wave']
            flux = self.spectrum_data['flux']
            error = self.spectrum_data.get('error', np.ones_like(flux) * 0.05)
            zabs = self.system_data['z']
            
            # Save current selections
            old_selections = self.velocity_selector.vel_guess.copy()
            
            # Create new selector
            self.velocity_selector = VelocitySelector(wave, flux, error, zabs, wrest)
            self.velocity_selector.setup_plot(self.ax)
            
            # Restore selections
            for vel in old_selections:
                self.velocity_selector.add_component(vel)
                
            self.canvas.draw()
            
    def on_click(self, event):
        """Handle mouse clicks"""
        if not self.velocity_selector or not event.inaxes:
            return
            
        # ENSURE CANVAS HAS FOCUS for keyboard events
        self.canvas.setFocus()
        
        if event.button == 1:  # Left click - add component
            self.velocity_selector.add_component(event.xdata)
            self.canvas.draw()
            self.update_status()
            
        elif event.button == 3:  # Right click - remove nearest component
            if self.velocity_selector.vel_guess:  # Only if components exist
                self.velocity_selector.remove_nearest(event.xdata)
    
            # Redraw all remaining components
            for vel in self.velocity_selector.vel_guess:
                self.velocity_selector.add_component(vel)
            self.canvas.draw()
            self.update_status()
                

    def on_key(self, event):
        """Handle key presses"""
        print(f"Key pressed: '{event.key}'")  # DEBUG
        
        if not self.velocity_selector:
            return
        
        # Get current mouse position for boundary setting
        if hasattr(event, 'xdata') and hasattr(event, 'ydata') and event.xdata is not None:
            x = event.xdata
            y = event.ydata
        else:
            # Fallback to center if no mouse position
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x = (xlim[0] + xlim[1]) / 2
            y = (ylim[0] + ylim[1]) / 2
            
        if event.key == 'r':  # Reset range
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw()
            
        elif event.key == 'c':  # Clear all
            self.velocity_selector.clear_all()
            self.canvas.draw()
            self.update_status()
            
        elif event.key == 'z':  # Zoom in
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * 0.8
            y_range = (ylim[1] - ylim[0]) * 0.8
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            self.canvas.draw()
            
        elif event.key == 'o':  # Zoom out
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * 1.2
            y_range = (ylim[1] - ylim[0]) * 1.2
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            self.canvas.draw()
            
        # SIMPLE BOUNDARY SETTING
        elif event.key == 'x':  # Set x-min (left boundary) at cursor
            current_xlim = self.ax.get_xlim()
            self.ax.set_xlim(x, current_xlim[1])
            self.canvas.draw()
            
        elif event.key == 'X':  # Set x-max (right boundary) at cursor
            current_xlim = self.ax.get_xlim()
            self.ax.set_xlim(current_xlim[0], x)
            self.canvas.draw()
            
        elif event.key == 'y':  # Set y-min (bottom boundary) at cursor
            current_ylim = self.ax.get_ylim()
            self.ax.set_ylim(y, current_ylim[1])
            self.canvas.draw()
            
        elif event.key == 'Y':  # Set y-max (top boundary) at cursor
            current_ylim = self.ax.get_ylim()
            self.ax.set_ylim(current_ylim[0], y)
            self.canvas.draw()
            
        # MANUAL RANGE INPUT
        elif event.key == 'm':  # Manual range input popup
            from PyQt5.QtWidgets import QInputDialog
            
            # X-range input
            current_xlim = self.ax.get_xlim()
            xlim_str, ok = QInputDialog.getText(self, 'Manual X-Limits', 
                                               f'Input x-range (current: {current_xlim[0]:.1f},{current_xlim[1]:.1f}):',
                                               text=f'{current_xlim[0]:.1f},{current_xlim[1]:.1f}')
            if ok and xlim_str:
                try:
                    xlimit = [float(val.strip()) for val in xlim_str.split(',')]
                    if len(xlimit) == 2:
                        self.ax.set_xlim(xlimit)
                        
                        # Y-range input
                        current_ylim = self.ax.get_ylim()
                        ylim_str, ok2 = QInputDialog.getText(self, 'Manual Y-Limits', 
                                                           f'Input y-range (current: {current_ylim[0]:.2f},{current_ylim[1]:.2f}):',
                                                           text=f'{current_ylim[0]:.2f},{current_ylim[1]:.2f}')
                        if ok2 and ylim_str:
                            ylimit = [float(val.strip()) for val in ylim_str.split(',')]
                            if len(ylimit) == 2:
                                self.ax.set_ylim(ylimit)
                        
                        self.canvas.draw()
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Please enter two numbers separated by comma")    
            
    def update_status(self):
        """Update status"""
        n = len(self.velocity_selector.vel_guess) if self.velocity_selector else 0
        self.status_label.setText(f"Selected {n} components")
        
    def finish_selection(self):
        """Finish and get parameters"""
        if not self.velocity_selector or not self.velocity_selector.vel_guess:
            return
            
        # Get velocity and column density estimates using quick_nv_estimate
        wave_rest = self.velocity_selector.wave / (1.0 + self.velocity_selector.zabs)
        vel, nv = g.quick_nv_estimate(wave_rest, self.velocity_selector.flux, 
                                     self.velocity_selector.wrest, self.velocity_selector.f0)
        
        # Estimate column densities for each selected velocity
        n_estimates = []
        for vel_guess in self.velocity_selector.vel_guess:
            # Find velocity range around the guess (±10 km/s)
            vel_mask = (vel >= vel_guess - 10.0) & (vel <= vel_guess + 10.0)
            if np.any(vel_mask):
                # Sum column density in this velocity range
                n_total = np.sum(nv[vel_mask])
                n_estimate = np.log10(np.maximum(n_total, 1e12))  # Avoid log(0)
                n_estimates.append(np.clip(n_estimate, 12.0, 16.0))
            else:
                # Fallback if no data in range
                n_estimates.append(13.5)
                
        # Get b parameters
        b_dialog = BParameterDialog(n_estimates, self.velocity_selector.vel_guess, 
                                   self.system_data['transitions'], 
                                   self.velocity_selector.transition_name, self)
        
        if b_dialog.exec_() == QDialog.Accepted:
            b_params = b_dialog.get_b_parameters()
            
            # Create result
            result_df = pd.DataFrame({
                'Component': range(1, len(n_estimates) + 1),
                'N': n_estimates,
                'b': b_params,
                'v': self.velocity_selector.vel_guess
            })
            
            self.parameters_ready.emit(result_df)
            self.accept()


def show_validation_warning(parent, message):
    """Standard validation warning"""
    QMessageBox.warning(parent, "Parameter Validation", message)