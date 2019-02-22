from __future__ import print_function
from rbvfit import rb_vfit as r
import numpy as np
from astropy import units as u
from linetools.spectra.lsf import LSF
from astropy.convolution.kernels import CustomKernel
from astropy.convolution import convolve, Gaussian1DKernel



class create_voigt(object):
    """
    This object will create a model Voigt profile spectrum given an input set of paramters.

    Input:    logN :  Array with log column density of individual clumps
                 b :  Array with Doppler b parameter for individual clumps
                 v :  Array with line of sight velocity of individual clumps
              zabs :  Redshift of absorbers [For each clump].

                [Optional]
                FWHM  =    String giving Instrument FWHM in pixels (default = 6.5 pixel)
                           [ If given  FWHM= 'COS'= will take HST COS LSF at given grating and 
                           lifetime position. But will require linetools dependency.]
                grating = HST grating [only required if FWHM= 'COS']
                life_position = HST Lifetime position [only required if FWHM= 'COS']

    Currently not bothering about Nuissance parameters. Will be added later [RB]


    Working example:
        N=np.array([14.,13.])
        b=np.array([20.,21.])
        v=np.array([10.,-100.])
        zabs=np.array([0.,0.])
        theta=np.concatenate((N,b,v))
        lambda_rest = 1215.67 * np.ones((len(N),))


    """

    def __init__(self, zabs,lambda_rest, FWHM = '6.5', grating='G130M',life_position='1'):
        #self.filename = filename
        # Setting up model paramters
        self.FWHM = FWHM
        self.grating = grating
        self.life_position=life_position
        #self.NGuess = np.array(N)
        #self.bGuess = np.array(b)
        #self.vGuess = np.array(v)
        self.zabs = zabs
        self.lambda_rest = np.array(lambda_rest)
        #self.Nuissance = s['Nuissance']
        #self.theta = np.concatenate( (self.NGuess, self.bGuess , self.vGuess))
        #self.lb = np.array(Nlow + blow + vlow)
        #self.ub = np.array(NHI + bHI + vHI)
        #self.bounds = [self.lb, self.ub]
  
        self.compile_model()
        self.use_custom_lsf(FWHM=FWHM,grating=grating,life_position=life_position)



    def compile_model(self):
        nclump = int(len(self.zabs))
        line = r.model()

        for i in range(0, nclump):
            line.addline(self.lambda_rest[i], z=self.zabs[i])

        self.line = line

    def model_flux(self, theta, wave):
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

