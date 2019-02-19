from __future__ import print_function
from rbvfit import rb_vfit as r
import numpy as np
from astropy import units as u
from linetools.spectra.lsf import LSF
from astropy.convolution.kernels import CustomKernel
from astropy.convolution import convolve, Gaussian1DKernel



class create_voigt(object):

    def __init__(self, filename, FWHM = '6.5', grating='G130M',life_position='1'):
        self.filename = filename
        self.FWHM = FWHM
        self.grating = grating
        self.life_position=life_position
        self.read_model_par(filename)
        self.compile_model()
        self.use_custom_lsf(FWHM=FWHM,grating=grating,life_position=life_position)


    def read_model_par(self, filename):
        s = ascii.read(filename)

        self.NGuess = np.ndarray.tolist(s['NGuess'])
        self.bGuess = np.ndarray.tolist(s['bGuess'])
        self.vGuess = np.ndarray.tolist(s['vGuess'])

        Nlow = np.ndarray.tolist(s['Nlow'])
        blow = np.ndarray.tolist(s['blow'])
        vlow = np.ndarray.tolist(s['vlow'])

        NHI = np.ndarray.tolist(s['NHI'])
        bHI = np.ndarray.tolist(s['bHI'])
        vHI = np.ndarray.tolist(s['vHI'])

        self.zabs = s['z_abs']
        self.lambda_rest = s['lambda_rest']
        self.Nuissance = s['Nuissance']
        self.theta = np.array(self.NGuess + self.bGuess + self.vGuess)
        self.lb = np.array(Nlow + blow + vlow)
        self.ub = np.array(NHI + bHI + vHI)
        self.bounds = [self.lb, self.ub]

    def compile_model(self):
        nclump = int(len(self.zabs))
        line = r.model()

        for i in range(0, nclump):
            line.addline(self.lambda_rest[i], z=self.zabs[i])

        self.line = line

    def model_mcmc(self, theta, wave):
        line = self.line
        ss3, flx = r.create_model_simple(theta, wave, line)

        # Convolve data
        fmodel = convolve(flx, self.kernel ,boundary='extend')  
        return fmodel

    def model_fit(self, theta, wave):
        line = self.line
        ss3, flx = r.create_model_simple(theta, wave, line)
        # Convolve data
        fmodel = convolve(flx, self.kernel ,boundary='extend')  
        return fmodel, ss3

    def use_custom_lsf(self,FWHM='6.5',grating='G130M',life_position='1'):
        if FWHM=='COS':
            instr_config=dict(name='COS',grating=grating,life_position=life_position)
            coslsf=LSF(instr_config)
            s,data=coslsf.load_COS_data()
            #if 1150A   1200A   1250A   1300A   1350A   1400A   1450A
            kernel = CustomKernel(data['1300A'])
        else:
            COS_kernel=np.double(FWHM)/2.355 #6.5 pixels
            #window_size_number_of_points=np.round(FWHM /(2.355*np.abs(velgrid[2]-velgrid[1])))
            # Create kernel
            kernel = Gaussian1DKernel(stddev=COS_kernel)

        self.kernel=kernel

