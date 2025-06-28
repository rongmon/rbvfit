from __future__ import print_function
import numpy as np
from typing import Tuple, List, Optional, Any, Dict, Union
import matplotlib.figure
import matplotlib.axes
from rbvfit.rb_vfit import rb_veldiff 
# Converting To velocities
from rbvfit import rb_setline as line

# Check if we're specifically in a Jupyter notebook (not just IPython)
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        IN_JUPYTER = True  # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
        IN_JUPYTER = False  # Terminal running IPython
    else:
        IN_JUPYTER = False  # Other type of shell
except (NameError, AttributeError):
    IN_JUPYTER = False  # Probably standard Python interpreter

# Set up matplotlib backend appropriately
import matplotlib
if not IN_JUPYTER:
    # For non-Jupyter environments, use an interactive backend
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            print("Warning: No interactive matplotlib backend available. Plots may not display properly.")
            matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Try to import ipywidgets, fall back to None if not available
try:
    import ipywidgets as widgets
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    widgets = None
    IPYWIDGETS_AVAILABLE = False

# Try to import display function, create fallback if not available
try:
    from IPython.display import display
except ImportError:
    def display(obj: Any) -> None:
        """Fallback display function for non-IPython environments"""
        if hasattr(obj, 'value'):
            print(obj.value)
        else:
            print(obj)


def quick_nv_estimate(wave: np.ndarray, norm_flx: np.ndarray, wrest: float, f0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate column density from absorption line profile using apparent optical depth method.
    
    This function calculates the column density per velocity bin from normalized flux
    using the apparent optical depth approximation. Useful for quick estimates of
    absorption line column densities.
    
    Parameters
    ----------
    wave : np.ndarray
        Wavelength array in Angstroms
    norm_flx : np.ndarray
        Normalized flux array (continuum normalized to 1.0)
    wrest : float
        Rest wavelength of the transition in Angstroms
    f0 : float
        Oscillator strength of the transition
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        vel : velocity array in km/s relative to rest wavelength
        n : column density per velocity bin in cm^-2
        
    Notes
    -----
    Uses the apparent optical depth method from Savage & Sembach 1991.
    Formula: N_v = tau_a / (2.654e-15 * f * lambda_r)
    where tau_a = ln(1/F_norm) is the apparent optical depth.
    """
    # All done in rest frame
    spl = 2.9979e5  # speed of light in km/s
    vel = (wave - wrest * (1.0 + 0.)) * spl / (wrest * (1.0 + 0.))
    lambda_r = wave / (1 + 0.)    
    
    # check for infinite optical depth
    q = np.where((norm_flx <= 0.))
    norm_flx[q] = 0.01
    
    # compute apparent optical depth
    Tau_a = np.log(1. / norm_flx)
    
    # REMEMBER WE ARE SWITCHING TO VELOCITY HERE
    del_vel_j = np.diff(vel)
    del_vel_j = np.append([del_vel_j[0]], del_vel_j)
    
    # Column density per pixel as a function of velocity
    nv = Tau_a / ((2.654e-15) * f0 * lambda_r)  # in units cm^-2 / (km s^-1), SS91 
    n = nv * del_vel_j  # column density per bin obtained by multiplying differential Nv by bin width
    return vel, n 

class gui_set_clump(object):
    """
    Interactive GUI for setting initial velocity guesses for absorption line fitting.
    
    This class creates an interactive matplotlib plot where users can click to set
    initial velocity guesses for absorption line cloud components. Works in both
    Jupyter notebooks (with widget support) and command-line environments.
    
    Attributes
    ----------
    vel : np.ndarray
        Velocity array in km/s
    wrest : float
        Rest wavelength in Angstroms
    zabs : float
        Absorption redshift
    flux : np.ndarray
        Flux array (normalized)
    error : np.ndarray
        Error array
    wave : np.ndarray
        Wavelength array in Angstroms
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object
    vel_guess : List[float]
        List of velocity guesses from user clicks
    w : Union[widgets.HTML, SimpleWidget]
        Widget for displaying click information
    bguess : np.ndarray
        Doppler parameter guesses (set by input_b_guess)
    vguess : np.ndarray
        Velocity guesses array (set by input_b_guess)
    nguess : np.ndarray
        Column density guesses (set by input_b_guess)
    """
    
    def __init__(self, wave: np.ndarray, flux: np.ndarray, error: np.ndarray, 
                 zabs: float, wrest: float, xlim: List[float] = [-600., 600.], 
                 **kwargs) -> None:
        """
        Initialize the interactive velocity guess GUI.
        
        Parameters
        ----------
        wave : np.ndarray
            Observed wavelength array in Angstroms
        flux : np.ndarray
            Normalized flux array
        error : np.ndarray
            Error array for the flux
        zabs : float
            Absorption redshift
        wrest : float
            Rest wavelength of the line in Angstroms
        xlim : List[float], optional
            Velocity limits for the plot in km/s, by default [-600., 600.]
        **kwargs
            Additional keyword arguments (unused)
            
        Notes
        -----
        Creates an interactive plot where:
        - Left mouse clicks or 'a' key add velocity guesses
        - Right mouse clicks or 'r' key remove nearest velocity guess
        - 'q' or ESC key finish interaction and close plot
        - Works in Jupyter notebooks with widgets or command-line with console output
        """
        self.vel: np.ndarray = rb_veldiff(wrest, wave / (1. + zabs))
        self.wrest: float = wrest
        self.zabs: float = zabs
        self.flux: np.ndarray = flux
        self.error: np.ndarray = error
        self.wave: np.ndarray = wave
    
        self.fig: matplotlib.figure.Figure
        self.ax: matplotlib.axes.Axes
        self.fig, self.ax = plt.subplots()
        
        # This is where you feed in your velocity and flux to be fit
        self.ax.step(self.vel, self.flux)
        self.ax.set_xlim(xlim)
        self.ax.set_xlabel('Velocity (km/s)')
        self.ax.set_ylabel('Normalized Flux')
        self.ax.set_title(f'Interactive Velocity Guess Tool - λ_rest = {wrest:.2f} Å')
        
        # Create widget or fallback depending on availability
        if IPYWIDGETS_AVAILABLE and IN_JUPYTER:
            self.w = widgets.HTML()
            display(self.w)
        else:
            # Simple fallback object that mimics HTML widget behavior
            class SimpleWidget:
                def __init__(self) -> None:
                    self.value: str = ""
            self.w: Union[widgets.HTML, SimpleWidget] = SimpleWidget()
            print("Interactive mode:")
            print("  - Left click or 'a' key: add velocity guess")
            print("  - Right click or 'r' key: remove nearest velocity guess")
            print("  - 'q' or ESC key: finish and close plot")
            
        self.vel_guess: List[float] = []
        self.vel_markers: List[matplotlib.lines.Line2D] = []  # Keep track of plotted markers
        self.interactive_mode: bool = True  # Flag to control interactive mode
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid1 = self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        
        # Show the plot properly based on environment
        if not IN_JUPYTER:
            plt.ion()  # Turn on interactive mode
            self.fig.show()  # Show this specific figure
            # Force the plot to display and stay interactive
            plt.pause(0.001)  # Small pause to ensure display
            
        # Initialize attributes that will be set by input_b_guess()
        self.bguess: Optional[np.ndarray] = None
        self.vguess: Optional[np.ndarray] = None
        self.nguess: Optional[np.ndarray] = None
        
        # Set default b-values based on transition type
        self._set_default_b_values()
        
    def _set_default_b_values(self) -> None:
        """
        Set default b-parameter values based on the transition ionization state.
        """
        if not self.vel_guess:
            return
            
        # Get transition information
        line_info: Dict[str, Any] = line.rb_setline(self.wrest, 'closest', 'atom')
        transition_name = line_info['name'][0] if len(line_info['name']) > 0 else ""
        
        # Parse ionization state and set appropriate default b-value
        default_b = self._get_default_b_from_transition(transition_name)
        
        # Set default values
        n_clouds = len(self.vel_guess)
        self.bguess = np.full(n_clouds, default_b)
        self.vguess = np.array(self.vel_guess)
        
        # Estimate column densities using AOD method
        vel, nv = quick_nv_estimate(self.wave / (1. + self.zabs), self.flux, 
                                   line_info['wave'], line_info['fval'])
        self.nguess = np.zeros(n_clouds)
        for i in range(n_clouds):
            qq = np.where((vel < self.vguess[i] + 10.) & (vel > self.vguess[i] - 10.))
            self.nguess[i] = np.log10(np.sum(nv[qq]) + 1e-20)  # Add small value to avoid log(0)
            
        print(f"Set default b-values based on {transition_name}: {default_b:.1f} km/s for all {n_clouds} components")
        
    def _get_default_b_from_transition(self, transition_name: str) -> float:
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
        
    def input_b_guess(self) -> None:
        """
        Interactively set Doppler parameter (b-value) guesses for each velocity component.
        
        This method updates the default b-values that were automatically set based on
        the transition type. It estimates column densities for each velocity guess using 
        the apparent optical depth method, then prompts the user to input Doppler parameter guesses.
        
        Updates
        -------
        bguess : np.ndarray
            Array of Doppler parameter guesses in km/s
        vguess : np.ndarray
            Array of velocity guesses in km/s (copy of vel_guess)
        nguess : np.ndarray
            Array of log10 column density guesses in cm^-2
            
        Notes
        -----
        Column density estimates are made by integrating the apparent optical depth
        within ±10 km/s of each velocity guess. The user is then prompted to enter
        Doppler parameter values for each component, with current defaults shown.
        """
        if not self.vel_guess:
            print("No velocity guesses available. Add some velocity guesses first.")
            return
            
        # Ensure we have current defaults
        if self.bguess is None:
            self._set_default_b_values()
        
        # Now set up the model fitting parameters.
        n_clouds: int = len(self.vel_guess)
        
        for i in range(n_clouds):
            # Show current default and ask for new value
            current_b = self.bguess[i]
            prompt = (f'Guess b for line {i+1}/{n_clouds}, '
                     f'vel = {self.vguess[i]:.1f} km/s, '
                     f'col = {self.nguess[i]:.1f}, '
                     f'current b = {current_b:.1f} km/s\n'
                     f'Enter new b-value (or press Enter to keep {current_b:.1f}): ')
            
            tmp_b = input(prompt).strip()
            if tmp_b:  # User entered a value
                self.bguess[i] = np.double(tmp_b)
            # Otherwise keep the current default value
            
    def onclick(self, event) -> None:
        """
        Handle mouse click events on the plot.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse click event containing position information
            
        Notes
        -----
        Left click (button 1): Adds the clicked x-coordinate (velocity) to the vel_guess list
        Right click (button 3): Removes the nearest velocity guess to the clicked position
        """
        if event.inaxes != self.ax or not self.interactive_mode:
            return
            
        message = ('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % 
                  (event.button, event.x, event.y, event.xdata, event.ydata))
        self.w.value = message
        
        if event.button == 1:  # Left click - add point
            self._add_velocity_guess(event.xdata, event.ydata)
        elif event.button == 3:  # Right click - remove nearest point
            self._remove_nearest_velocity_guess(event.xdata)
            
    def _add_velocity_guess(self, xdata: float, ydata: float) -> None:
        """Add a velocity guess at the specified position."""
        # In non-Jupyter environments, print to console
        if not (IPYWIDGETS_AVAILABLE and IN_JUPYTER):
            print(f"Added velocity guess: {xdata:.2f} km/s")
            
        self.vel_guess.append(xdata)
        marker = self.ax.plot(xdata, ydata, 'r+', markersize=10)[0]
        self.vel_markers.append(marker)
        
        # Update the plot
        if IN_JUPYTER:
            plt.draw()
        else:
            self.fig.canvas.draw()
            
    def _finish_velocity_selection(self) -> None:
        """
        Finish the interactive velocity selection.
        """
        self.interactive_mode = False
        
        # Set default b-values now that we have velocity guesses
        self._set_default_b_values()
        
        # Close the figure in non-Jupyter environments
        if not IN_JUPYTER:
            plt.close(self.fig)
        
        # Print summary
        n_guesses = len(self.vel_guess)
        if n_guesses == 0:
            print("No velocity guesses added.")
            return
            
        print(f"\nFinished velocity selection!")
        print(f"Added {n_guesses} velocity guess{'es' if n_guesses != 1 else ''}: {[f'{v:.1f}' for v in sorted(self.vel_guess)]} km/s")
        print("Run .input_b_guess() to customize b-parameter values.")
            
    def _remove_nearest_velocity_guess(self, xdata: float) -> None:
        """Remove the velocity guess nearest to the specified x position."""
        if not self.vel_guess:
            return
            
        # Find the nearest velocity guess
        distances = [abs(v - xdata) for v in self.vel_guess]
        nearest_idx = distances.index(min(distances))
        
        # Remove from lists
        removed_vel = self.vel_guess.pop(nearest_idx)
        removed_marker = self.vel_markers.pop(nearest_idx)
        
        # Remove marker from plot
        removed_marker.remove()
        
        # In non-Jupyter environments, print to console
        if not (IPYWIDGETS_AVAILABLE and IN_JUPYTER):
            print(f"Removed velocity guess: {removed_vel:.2f} km/s")
        
        # Update the plot
        if IN_JUPYTER:
            plt.draw()
        else:
            self.fig.canvas.draw()
        
    def onpress(self, event) -> None:
        """
        Handle key press events on the plot.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key press event
            
        Notes
        -----
        'a' key: adds a velocity guess at the current mouse position
        'r' key: removes the nearest velocity guess to the current mouse position
        'q' or 'escape' key: finish interactive mode and close plot
        """
        if event.inaxes != self.ax or not self.interactive_mode:
            return
            
        if event.key == 'a':
            self._add_velocity_guess(event.xdata, event.ydata)
        elif event.key == 'r':
            self._remove_nearest_velocity_guess(event.xdata)
        elif event.key in ['q', 'escape']:
            self._finish_velocity_selection()
                
    def get_guesses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current velocity, column density, and Doppler parameter guesses.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            vguess : velocity guesses in km/s
            nguess : log10 column density guesses in cm^-2  
            bguess : Doppler parameter guesses in km/s
            
        Raises
        ------
        ValueError
            If input_b_guess() has not been called yet
        """
        if self.vguess is None or self.nguess is None or self.bguess is None:
            raise ValueError("Must call input_b_guess() before getting guesses")
        return self.vguess, self.nguess, self.bguess