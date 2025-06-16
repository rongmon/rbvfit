from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
import numpy as np
import matplotlib.pyplot as plt

# Multi-redshift, multi-ion system
config = FitConfiguration()

# System 1: z=0.348 - MgII doublet (2 components) + FeII (1 component)
config.add_system(z=0.348, ion='MgII', transitions=[2796.3, 2803.5], components=2)
config.add_system(z=0.348, ion='FeII', transitions=[2600.2], components=1)

# System 2: z=0.524 - OVI doublet (2 components)
config.add_system(z=0.524, ion='OVI', transitions=[1031.9, 1037.6], components=2)

# System 3: z=1.234 - CIV doublet (1 component) + SiII multiplet (3 components)
config.add_system(z=1.234, ion='CIV', transitions=[1548.2, 1550.8], components=1)
config.add_system(z=1.234, ion='SiII', transitions=[1190.42, 1193.29, 1260.42, 1526.71], components=3)

# Create model
model = VoigtModel(config, FWHM='6.5')


# Example theta array for this complex system
# System 1: MgII (2 comp) + FeII (1 comp) = 3 components
# System 2: OVI (2 comp) = 2 components  
# System 3: CIV (1 comp) + SiII (3 comp) = 4 components
# Total: 9 components = 27 parameters

example_theta = np.array([
    # N values (9 components total)
    13.8, 13.5,           # MgII z=0.348 (components 1,2)
    14.2,                 # FeII z=0.348 (component 1)
    14.5, 14.1,           # OVI z=0.524 (components 1,2)
    13.9,                 # CIV z=1.234 (component 1)
    13.2, 13.0, 12.8,     # SiII z=1.234 (components 1,2,3)
    
    # b values (9 components total)
    25.0, 35.0,           # MgII z=0.348
    20.0,                 # FeII z=0.348
    45.0, 50.0,           # OVI z=0.524
    30.0,                 # CIV z=1.234
    15.0, 20.0, 25.0,     # SiII z=1.234
    
    # v values (9 components total)
    -150.0, -50.0,        # MgII z=0.348
    -100.0,               # FeII z=0.348
    -200.0, -100.0,       # OVI z=0.524
    0.0,                  # CIV z=1.234
    -80.0, 0.0, 80.0      # SiII z=1.234
])


model_compile=model.compile()

# Create wavelength grids for different spectral regions
wave = np.linspace(1000, 4500, 10000)  # Covers MgII at z=0.348



# UV spectrum (OVI + CIV + SiII)  
flux = model_compile.model_flux(example_theta, wave)

transitions_info = [
    ("MgII 2796", 2796.3, 0.348),
    ("MgII 2803", 2803.5, 0.348),
    ("FeII 2600", 2600.2, 0.348),
    ("OVI 1031", 1031.9, 0.524),
    ("OVI 1037", 1037.6, 0.524),
    ("CIV 1548", 1548.2, 1.234),
    ("CIV 1550", 1550.8, 1.234),
    ("SiII 1190", 1190.42, 1.234),
    ("SiII 1193", 1193.29, 1.234),
    ("SiII 1260", 1260.42, 1.234),
    ("SiII 1526", 1526.71, 1.234)
]

for name, wave_rest, z in transitions_info:
    wave_obs = wave_rest * (1 + z)
    print(f"{name:12s}: {wave_rest:7.1f} Å → {wave_obs:7.1f} Å (z={z:.3f})")

    plt.axvline(wave_obs, color='red', linestyle='--', alpha=0.7)
    plt.text(wave_obs, 0.05, name, rotation=90, ha='center', fontsize=10)
    plt.plot()

plt.plot(wave,flux)
plt.ylim([0.05,1.2])
plt.show()

model.show_structure()
