

import numpy as np
import time
import matplotlib.pyplot as plt

# V2 imports only
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel
from rbvfit import vfit_mcmc as mc

from linetools.spectra.xspectrum1d import XSpectrum1D  
from pkg_resources import resource_filename

#read file
filename=resource_filename('rbvfit','example-data/test.fits')
sp=XSpectrum1D.from_file(filename)


#Redshift of the absorber you are trying to fit
zabs=0.0

#Normalizing the spectrum
wave_full=sp.wavelength.value
flux=sp.flux.value/sp.co.value
error=sp.sig.value/sp.co.value


qt=np.isnan(flux)
flux[qt]=0.
error[qt]=0.

q=((wave_full/(1.+zabs) >1189.5) & (wave_full/(1.+zabs) < 1195.))

wave=wave_full[q]
flux=flux[q]
error=error[q]


#Which transitions to fit
lambda_rest = [1190.5,1193.5]
lambda_rest1=[1025.7]
#Initial guess of clouds
nguess=[13.8,13.8,13.5]
bguess=[70.,30.,30.]
vguess=[-180.,0.,0.]


#Setting the upper and lower limits for the fit. You can also do it by hand if you prefer
bounds,lb,ub=mc.set_bounds(nguess,bguess,vguess)


#------------------------------------------------------

# Doing some book keeping to organize the guess
theta=np.concatenate((nguess,bguess,vguess))


print("\n=== Setting up V2 Model ===")
    
# Create V2 model
config = FitConfiguration()
config.add_system(z=zabs, ion='SiII', transitions=lambda_rest, components=2)
config.add_system(z=0.162005,ion='HI', transitions=lambda_rest1, components=1)
v2_model = VoigtModel(config, FWHM='6.5')
v2_compiled = v2_model.compile()
print("✓ V2 model created and compiled")


# MCMC settings
n_steps = 500
n_walkers = 50

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
    sampler='emcee'
    )
        
# Run MCMC
fitter.runmcmc(optimize=False, verbose=False, use_pool=True)
        
# Extract results
samples = fitter._extract_samples(fitter.sampler, burnin=300)
best_fit = np.median(samples, axis=0)
best_model = v2_compiled.model_flux(best_fit, wave)
        
# Get sampler info
sampler_info = fitter.get_sampler_info()
        
elapsed_time = time.time() - start_time
success = True
        
print(f"✓ mcmc completed in {elapsed_time:.1f} seconds")
        
# Print diagnostics
if 'acceptance_fraction' in sampler_info:
    print(f"  Acceptance fraction: {sampler_info['acceptance_fraction']:.3f}")
if 'r_hat_max' in sampler_info:
    print(f"  R-hat max: {sampler_info['r_hat_max']:.3f}")

    
fitter.plot_corner()

#mc.plot_model(wave,flux,error,fitter,v2_compiled)
fig,ax = plt.subplots(1, 1, figsize=(10, 4))

# Top panel: Model fits
ax.step(wave, flux, 'k-', where='mid', linewidth=1.5, 
            label='Observed Data', alpha=0.8)
ax.fill_between(wave, flux - error, flux + error,
                     color='gray', alpha=0.3, step='mid', label='1σ Error')
ax.plot(wave, best_model, 'g--', linewidth=2, 
            label='Best Model', alpha=0.7)
plt.show()