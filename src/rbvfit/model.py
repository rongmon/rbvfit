from __future__ import print_function
from rbvfit import rb_vfit as r
import numpy as np
from astropy import units as u
from linetools.spectra.lsf import LSF
from astropy.convolution.kernels import CustomKernel
from astropy.convolution import convolve, Gaussian1DKernel

def map_theta2list(theta,ncl=1,nt=1):
    """
    This function maps theta to theta_prime
    Here theta:  list of input [N,b,v] values to create a Voigt profile
         theta_prime :  expanded list of N,b,v values to make sure each transition and nuissance parameter is accounted for
         ncl= number of clumps
         nt= number of transitions                  
    """
    
    # First break theta into individual N,b,v values for all entries
    No_of_clumps=int(len(theta)/3)
    N=theta[0:No_of_clumps]
    b=theta[No_of_clumps:No_of_clumps+No_of_clumps]
    vel=theta[2*No_of_clumps:3*No_of_clumps]
    
    N_prime=[]
    b_prime=[]
    v_prime=[]
    # Now do conversion individually
    
    for i in range(0,nt):  
        N_prime[ncl*i:ncl*i+ncl]=N[0:ncl] 
        b_prime[ncl*i:ncl*i+ncl]=b[0:ncl]
        v_prime[ncl*i:ncl*i+ncl]=vel[0:ncl]

    # If there are Nuissance parameters add them
    if len(theta) > ncl*nt:
        N_prime[ncl*nt:] =N[ncl:] 
        b_prime[ncl*nt:] =b[ncl:] 
        v_prime[ncl*nt:] =vel[ncl:] 
    
    theta_prime=np.concatenate((np.array(N_prime),np.array(b_prime),np.array(v_prime)))
    return theta_prime

        

    


class create_voigt(object):
    """
    This object will create a model Voigt profile spectrum given an input set of paramters.

    Input:    zabs        :  Redshift of absorbers. [should be a list equal to number of independent components]
              lambda_rest :  rest frame wavelength of all independent transitions
              nclump      :  number of clumps [For the main absorber]
              ntransition :  number of transition [e.g. for MgII doublet ntransition =2, for SiII 1190, 1193, 1260, 1526, ntransition=4]

                [Optional]
                FWHM  =    String giving Instrument FWHM in pixels (default = 6.5 pixel)
                           [ If given  FWHM= 'COS'= will take HST COS LSF at given grating and 
                           lifetime position. But will require linetools dependency.]
                grating = HST grating [only required if FWHM= 'COS']
                life_position = HST Lifetime position [only required if FWHM= 'COS']
                
                

    ------------------------------------------------------------------------

    Working example:
       Look up test.ipynb
    
    Written By :  Rongmon Bordoloi  [Fall 2018]
    Edited RB July 12 2019: Made the code flexible to easily add multiple transitions of the same ion and Nuissance parameters to the fit.
    

    """

    def __init__(self, zabs,lambda_rest,nclump,ntransition=1, FWHM = '6.5', grating='G130M',life_position='1'):
        # Setting up model paramters
        self.FWHM = FWHM
        self.grating = grating
        self.life_position=life_position
        self.nclump=nclump
        self.ntransition = ntransition
        #Creating a redshift list that is equal to the length of all transitions and nuissance parameters 
        zlist=np.array(zabs)
        # If there are nuissance paramters
        if len(zlist) > 1:
            zlist=np.append(np.repeat(zlist[0],nclump*ntransition),zlist[1:])
        else:
            zlist=np.repeat(zlist[0],nclump*ntransition)
        self.zabs = zlist

        # Now create a list of rest wavelengths which could be used to create a line object
        lambda_rest=np.array(lambda_rest)
        lam_restlist=[]
        for i in range(0,ntransition):
            lam_restlist=np.append(lam_restlist,np.repeat(lambda_rest[i],nclump))
            # If there are nuissance paramters
            if len(zlist) > 1:
                lam_restlist=np.append(lam_restlist,lambda_rest[ntransition:])
        self.lambda_rest = lam_restlist
  
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
        theta_prime=map_theta2list(theta,self.nclump,self.ntransition)
        ss3, flx = r.create_model_simple(theta_prime, wave, line)

        # Convolve data
        fmodel = convolve(flx, self.kernel ,boundary='extend')  
        return fmodel

    def model_unconvolved(self, theta, wave):       
        line = self.line
        theta_prime=map_theta2list(theta,self.nclump,self.ntransition)
        ss3, flx = r.create_model_simple(theta_prime, wave, line)

        return flx
   

    def model_fit(self, theta, wave):
        line = self.line
        theta_prime=map_theta2list(theta,self.nclump,self.ntransition)
        ss3, flx = r.create_model_simple(theta_prime, wave, line)
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


