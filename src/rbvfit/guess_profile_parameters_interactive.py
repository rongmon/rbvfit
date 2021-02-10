import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from rbvfit.rb_vfit import rb_veldiff 
# Converting To velocities
from rbvfit import rb_setline as line


def quick_nv_estimate(wave,norm_flx,wrest,f0):
    # All done in rest frame
    spl=2.9979e5  #speed of light
    vel = (wave-wrest*(1.0 + 0.))*spl/(wrest*(1.0 + 0.))
    lambda_r=wave/(1+0.)    
    # check for infinite optical depth
    q=np.where((norm_flx <= 0.))
    norm_flx[q]=0.01

    #compute apparent optical depth
    Tau_a =np.log(1./norm_flx);
    # REMEMBER WE ARE SWITCHING TO VELOCITY HERE
    del_vel_j = np.diff(vel);
    del_vel_j = np.append([del_vel_j[0]], del_vel_j)
    # Column density per pixel as a function of velocity
    nv = Tau_a / ((2.654e-15) * f0 * lambda_r) # in units cm^-2 / (km s^-1), SS91 
    n = nv * del_vel_j  # column density per bin obtained by multiplying differential Nv by bin width
    return vel, n 

class gui_set_clump(object):
    def __init__(self,wave,flux,error,zabs,wrest,xlim=[-600.,600.],**kwargs):
        self.vel=rb_veldiff(wrest,wave/(1.+zabs))
        self.wrest=wrest
        self.zabs=zabs
        self.flux=flux
        self.error=error
        self.wave=wave
    
        self.fig, self.ax = plt.subplots()
        #This is where you feed in your velocity and flux to be fit
        self.ax.step(self.vel,self.flux)
        self.ax.set_xlim(xlim)
        self.w = widgets.HTML()

        self.vel_guess=[]
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid1 = self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        display(self.w)
        
    def input_b_guess(self):
        # Now set up the model fitting paramters.
        # Create the guesses for starting a fit
        n_clouds=int(len(self.vel_guess))
        self.bguess=np.zeros(n_clouds,)
        self.vguess=self.vel_guess
        self.nguess=np.zeros(n_clouds,)
        
        # AOD column guess for the primary line
        str=line.rb_setline(self.wrest,'closest','atom')
        vel,nv=quick_nv_estimate(self.wave/(1.+self.zabs),self.flux,str['wave'],str['fval']);


        for i in range(0,n_clouds):
            qq=np.where( (vel < self.vguess[i]+ 10.) & (vel > self.vguess[i]-10.))
            self.nguess[i]=np.log10(sum(nv[qq]))        

            #Now ask interactively for b values     
            prompt='Guess  b  for line ' +np.str(i+1)+ '/'+np.str(n_clouds) +', vel guess = ' + np.str('%.1f' % self.vguess[i])  +', col guess= '+ np.str('%.1f' % self.nguess[i])+ ': '
            tmp_b =  input(prompt)
            self.bguess[i]= np.double(tmp_b)




    def onclick(self,event):
        self.w.value = 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
        self.vel_guess.append(event.xdata)
        self.ax.plot(event.xdata,event.ydata,'r+')

    def onpress(self,event):
        if event.key=='a':
            self.vel_guess.append(event.xdata)
            self.ax.plot(event.xdata,event.ydata,'r+')    
    
