

import numpy as np
import time
import matplotlib.pyplot as plt

# V2 imports only
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit import vfit_mcmc as mc

from rbcodes.utils.rb_spectrum import rb_spectrum
from pkg_resources import resource_filename

#read file
filename=resource_filename('rbvfit','example-data/test.fits')
sp=rb_spectrum.from_file(filename)


#Redshift of the absorber you are trying to fit
zabs=0.0

#Normalizing the spectrum
wave_full=sp.wavelength.value
flux=sp.flux.value/sp.co.value
error=sp.sig.value/sp.co.value


qt=np.isnan(flux)
flux[qt]=0.
error[qt]=0.

q=((wave_full/(1.+zabs) >1189.5) * (wave_full/(1.+zabs) < 1195.))#+((wave_full/(1.+zabs) >1524.5) * (wave_full/(1.+zabs) < 1527.5))

wave=wave_full[q]
flux=flux[q]
error=error[q]


#Which transitions to fit
lambda_rest = [1190.5,1193.5]#,1526.5]
lambda_rest1=[1025.7]
#Initial guess of clouds
nguess=[14.2,14.5]
bguess=[40.,30.]
vguess=[0.,0.]


#Setting the upper and lower limits for the fit. You can also do it by hand if you prefer
#bounds,lb,ub=mc.set_bounds(nguess,bguess,vguess)
bounds,lb,ub = mc.set_bounds(
    nguess, bguess, vguess,
    # Custom lower bounds
    Nlow=[12.0, 12.0],     # Broader N range
    blow=[5.0, 5.0],       # Broader b range  
    vlow=[-300.0, -300.0], # Broader v range
    # Custom upper bounds
    Nhi=[17.0, 17.0],      # Broader N range
    bhi=[100.0, 100.0],    # Broader b range
    vhi=[300.0, 300.0]     # Broader v range
)

#------------------------------------------------------

# Doing some book keeping to organize the guess
theta=np.concatenate((nguess,bguess,vguess))


print("\n=== Setting up V2 Model ===")
    
# Create V2 model
config = FitConfiguration()
config.add_system(z=zabs, ion='SiII', transitions=lambda_rest, components=1)
config.add_system(z=0.162005,ion='HI', transitions=lambda_rest1, components=1)
v2_model = VoigtModel(config, FWHM='6.5')
v2_compiled = v2_model.compile()
print("✓ V2 model created and compiled")


# MCMC settings
n_steps = 500
n_walkers = 20

print(f"\n{'='*50}")
print(f"Running mcmc fitting multi core")
print(f"{'='*50}")
    
start_time = time.time()
success = False
    
# Create fitter
fitter = mc.vfit(
    v2_compiled.model_flux, theta, lb, ub, wave, flux, error,
    no_of_Chain=n_walkers, 
    no_of_steps=n_steps,
    sampler='emcee',
    perturbation=1e-4  # Smaller perturbation for walker initialization
)


# Add this before runmcmc to debug:
print(f"Model expects: {v2_model.param_manager.structure['total_parameters']} params")
print(f"Theta provides: {len(theta)} params")
print(f"Initial theta: {theta}")
print(f"Bounds check: {np.all(theta >= lb)} and {np.all(theta <= ub)}")
print(f"Initial likelihood: {fitter.lnprob(theta)}")

# Try manual model evaluation
try:
    test_flux = v2_compiled.model_flux(theta, wave)
    print(f"Model evaluation successful: {np.isfinite(test_flux).all()}")
except Exception as e:
    print(f"Model evaluation failed: {e}")
        
# Run MCMC
fitter.runmcmc(optimize=True, verbose=True, use_pool=True)


#plot corner     
#fitter.plot_corner()
#fitter.fit_quick() 

elapsed_time=time.time()-start_time        
print(f"✓ mcmc completed in {elapsed_time:.1f} seconds")




#plot models
#mc.plot_model(v2_model,fitter,show_residuals=True)


# New modules for better analysis and model plotting

from rbvfit.core import fit_results as f
# Save results
results = f.FitResults(fitter, v2_model)
#results.save('my_fit.h5')

# Load and analyze
#results = f.FitResults.load('my_fit.h5')
results.print_fit_summary()
#results.corner_plot()#save_path='corner.pdf')
results.convergence_diagnostics()

# Visual chain inspection
#results.chain_trace_plot()#save_path='trace_plots.pdf')

# This is the main new feature - velocity space plots by ion!
#velocity_plots = results.plot_velocity_fits(
#    show_components=True,      # Show individual components
#    show_rail_system=True     # Show component position markers
#)

# For single ion systems, also try velocity range control:
#results.plot_velocity_fits(velocity_range=(-600, 600))