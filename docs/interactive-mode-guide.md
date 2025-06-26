# Interactive Parameter Guessing Guide
[← Back to Main Documentation](../README.md)

The interactive parameter guessing tool in rbvfit provides an intuitive way to identify absorption line components and set initial parameter estimates for MCMC fitting. This guide covers both the workflow and technical details.

## 🎯 Overview

Interactive mode allows you to:
- **Visually identify** absorption components by clicking on velocity features
- **Set initial guesses** for column density (N), Doppler parameter (b), and velocity (v)
- **Customize parameters** interactively before starting the fit
- **Work seamlessly** in both Jupyter notebooks and command-line environments

## 🚀 Quick Start

```python
from rbvfit import guess_profile_parameters_interactive as g

# Set up interactive GUI
tab = g.gui_set_clump(wave_obs, flux, error, zabs, wrest=1548.5, xlim=[-600, 600])

# Interactive parameter input (GUI will appear)
tab.input_b_guess()

# Extract parameters for fitting
nguess = tab.nguess  # Column density guesses
bguess = tab.bguess  # Doppler parameter guesses  
vguess = tab.vguess  # Velocity guesses
```

## 🖱️ Interactive Controls

### Velocity Selection Mode

When the GUI opens, you'll see a plot of your spectrum in velocity space. Use these controls:

| Action | Method | Description |
|--------|--------|-------------|
| **Add velocity guess** | Left click or `a` key | Click on absorption features to mark component centers |
| **Remove velocity guess** | Right click or `r` key | Remove the nearest velocity marker |
| **Finish selection** | `q` key or `ESC` | Complete velocity selection and move to parameter input |

### Parameter Input Mode

After velocity selection, you'll be prompted to enter:

- **Column density (N)**: log₁₀ values in cm⁻² (typical range: 12-20)
- **Doppler parameter (b)**: Values in km/s (typical range: 10-100)
- **Velocity offsets**: Fine-tune the velocity guesses if needed

## 🔧 Technical Details

### Class Structure

The interactive tool is built around the `gui_set_clump` class:

```python
class gui_set_clump:
    """
    Interactive GUI for setting initial velocity guesses for absorption line fitting.
    
    Attributes
    ----------
    vel : np.ndarray
        Velocity array in km/s
    flux : np.ndarray
        Normalized flux array
    vel_guess : List[float]
        List of velocity guesses from user clicks
    nguess, bguess, vguess : np.ndarray
        Final parameter arrays after input_b_guess()
    """
```

### Environment Compatibility

The tool automatically adapts to your environment:

**Jupyter Notebooks:**
- Uses `ipywidgets` for enhanced interactivity
- Inline plotting with widget displays
- Rich HTML output for click feedback

**Command Line:**
- Uses `matplotlib` interactive backend
- Console output for feedback
- Cross-platform compatibility

### Default Parameter Values

The tool sets intelligent defaults based on transition type:

```python
def _set_default_b_values(self):
    """Set default b-parameter values based on ionization state."""
    if self.wrest > 1200:  # Low ionization (e.g., MgII, FeII)
        self.default_b = 20.0
    else:  # High ionization (e.g., OVI, NIII)  
        self.default_b = 30.0
```

## 📊 Workflow Integration

### Basic Workflow

```python
# 1. Load your data
wave, flux, error = load_spectrum('your_data.json')

# 2. Set up interactive mode
tab = g.gui_set_clump(wave, flux, error, zabs=0.348, wrest=2796.3)

# 3. Interactive parameter estimation
tab.input_b_guess()

# 4. Use parameters in fitting with FWHM configuration
config = FitConfiguration(FWHM='2.5')  # FWHM defined at configuration stage
config.add_system(z=zabs, ion='MgII', transitions=[2796.3, 2803.5], 
                  components=len(tab.nguess))

# 5. Create model (FWHM automatically extracted from configuration)
model = VoigtModel(config)
theta = np.concatenate([tab.nguess, tab.bguess, tab.vguess])

# 6. Run MCMC
fitter = mc.vfit(model.compile(), theta, bounds, wave, flux, error)
fitter.runmcmc()
```

### Multi-System Setup

For complex multi-system fits:

```python
# System 1: MgII at z=0.348
tab1 = g.gui_set_clump(wave, flux, error, zabs=0.348, wrest=2796.3)
tab1.input_b_guess()

# System 2: OVI at z=0.524  
tab2 = g.gui_set_clump(wave, flux, error, zabs=0.524, wrest=1031.9)
tab2.input_b_guess()

# Configure multi-system model with FWHM
config = FitConfiguration(FWHM='2.2')  # Define resolution at setup
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], 
                  components=len(tab1.nguess))
config.add_system(z=0.524, ion='OVI', transitions=[1031.9, 1037.6], 
                  components=len(tab2.nguess))
```

## 🛠️ Advanced Features

### Custom Velocity Ranges

```python
# Focus on specific velocity range
tab = g.gui_set_clump(wave, flux, error, zabs, wrest=1548.5, 
                      xlim=[-200, 200])  # Narrow range
```

### Parameter Validation

The tool includes built-in validation:

```python
def validate_parameters(self):
    """Validate that parameters are physically reasonable."""
    if any(n < 10 or n > 22 for n in self.nguess):
        warnings.warn("Column densities outside typical range [10, 22]")
    if any(b < 5 or b > 200 for b in self.bguess):
        warnings.warn("Doppler parameters outside typical range [5, 200] km/s")
```

### Error Handling

Common issues and solutions:

**Plot not appearing:**
```python
# Ensure interactive backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
plt.ion()
```

**Widget issues in Jupyter:**
```python
# Install widget extensions
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension
```

## 🎨 Customization

### Plot Styling

```python
# Customize the interactive plot
tab = g.gui_set_clump(wave, flux, error, zabs, wrest=1548.5)
tab.ax.set_ylim([0, 1.2])  # Adjust y-limits
tab.ax.grid(True, alpha=0.3)  # Add grid
tab.fig.set_size_inches(12, 6)  # Resize plot
```

### Parameter Bounds

```python
# Set custom parameter bounds after interactive input
tab.input_b_guess()

# Custom bounds for specific use case
bounds, lb, ub = mc.set_bounds(tab.nguess, tab.bguess, tab.vguess,
                               nlow=[12.0]*len(tab.nguess),    # Custom N lower bounds
                               nhigh=[18.0]*len(tab.nguess),   # Custom N upper bounds
                               blow=[10.0]*len(tab.bguess))    # Custom b lower bounds
```

### FWHM Handling with Interactive Mode

```python
# Interactive mode with FWHM configuration
tab = g.gui_set_clump(wave, flux, error, zabs=0.348, wrest=2796.3)
tab.input_b_guess()

# FWHM configuration approaches:

# 1. FWHM in pixels (direct)
config = FitConfiguration(FWHM='2.5')  # Pixels

# 2. Convert from km/s to pixels if needed
from rbvfit.core.voigt_model import mean_fwhm_pixels
FWHM_vel = 15.0  # km/s
FWHM_pixels = mean_fwhm_pixels(FWHM_vel, wave)
config = FitConfiguration(FWHM=str(FWHM_pixels))

# 3. Configure with parameters from interactive mode
config.add_system(z=zabs, ion='MgII', transitions=[2796.3, 2803.5], 
                  components=len(tab.nguess))
```

## 📋 Best Practices

### 1. Start Conservative
- Begin with fewer components than you think you need
- Add complexity gradually based on fit quality

### 2. Physical Reasoning
- Column densities: 12-16 (typical), 16-20 (strong), >20 (saturated)
- Doppler parameters: 10-30 km/s (thermal), 30-100 km/s (turbulent)
- Velocities: Based on kinematic structure expectations

### 3. Iterative Refinement
```python
# Initial fit
fitter.runmcmc(no_of_steps=100)  # Quick test

# Check results, then refine
if convergence_poor:
    # Adjust initial guesses and rerun
    tab.input_b_guess()  # Re-enter parameters
    fitter = mc.vfit(model, new_theta, bounds, wave, flux, error)
    fitter.runmcmc(no_of_steps=1000)  # Full fit
```

### 4. Validation
- Compare results with literature values
- Check for unphysical parameter values
- Verify component significance with model comparison

## 🔍 Troubleshooting

### Common Issues

**No GUI appears:**
- Check matplotlib backend: `matplotlib.get_backend()`
- Try different backend: `matplotlib.use('TkAgg')`
- Ensure X11 forwarding if using SSH

**Clicks not registering:**
- Ensure plot window has focus
- Try keyboard shortcuts (`a` to add, `r` to remove)
- Check for widget conflicts in Jupyter

**Parameters seem unreasonable:**
- Review physical expectations for your system
- Check units (velocities in km/s, N in log cm⁻²)
- Validate against literature measurements

### Debug Mode

```python
# Enable verbose output
tab = g.gui_set_clump(wave, flux, error, zabs, wrest=1548.5, verbose=True)
```

## 🌟 Tips for Success

1. **Prepare your data**: Ensure flux is normalized and error arrays are realistic
2. **Know your system**: Have approximate expectations for redshift and line strengths  
3. **Start simple**: Begin with single-component fits before adding complexity
4. **Iterate**: Use interactive mode to refine guesses based on initial fit results
5. **Validate**: Always compare final results with physical expectations

The interactive parameter guessing tool is designed to make the transition from data visualization to quantitative fitting as smooth as possible. Combined with rbvfit's robust MCMC engine, it provides a complete workflow for professional absorption line analysis.