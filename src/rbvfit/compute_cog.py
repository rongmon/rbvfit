from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
from rbvfit import rb_vfit as r
from rbvfit import rb_setline as rt




def set_one_absorber(N,b,lam_rest):
    zabs=0.
    line = r.model()
    N=np.array(N)
    b=np.array(b)
    v=np.array([0.])
    
    line.addline(lam_rest, z=zabs)
    wave=np.arange(lam_rest-10.,lam_rest+10.,0.05)
    #ipdb.set_trace()
    theta=np.array([N,b,v])#np.concatenate((N,b,v))
    flx, components = r.model_profile(theta, wave, line)
    W=np.trapz(1.-flx,x=wave)
    return W

def compute_ewlist_from_voigt(Nlist,b,lam_rest):
    Wlist=np.zeros(len(Nlist),)
    for i in range(0,len(Nlist)):
        Wlist[i]=set_one_absorber(Nlist[i],b,lam_rest)
    return Wlist


class compute_cog(object):
    def __init__(self,lam_guess,Nlist,blist):
         """
    This object will create a curve of growth for a given input set of paramters.

    Input:  
              lam_guess :  rest frame wavelength of one  transition
              Nlist      :  array of column densities for which COG is to be computed
              blist :  array of b values for which COG is to be computed

    Output:
              a COG object with all input parameters 
              st: structure containing transition information
              Wlist: matrix containing EW for every logN and b value                
                



    Working example:
       Look up COG Example.ipynb


    """

        self.st=rt.rb_setline(lam_guess,'closest')
        self.Nlist=Nlist
        self.blist=blist

        self.Wlist=np.zeros((len(Nlist),len(blist)))

        for i in range(0, len(blist)):
            print(self.st['wave'])
            self.Wlist[:,i]=compute_ewlist_from_voigt(Nlist,blist[i],self.st['wave'])

    def plot_cog(self):
        #Convert Angstrom to cm
        plt.title(self.st['name'])

        for i in range(0,len(self.blist)):
            plt.plot(np.log10((10**self.Nlist)*self.st['fval']*self.st['wave']*1e-8),np.log10(self.Wlist[:,i]/self.st['wave']),label='b = '+ np.str(self.blist[i]))
        plt.xlabel(r'$log_{10} [N f \lambda]$')
        plt.ylabel(r'$log_{10} [W/ \lambda]$')
        plt.legend()
        plt.show()






