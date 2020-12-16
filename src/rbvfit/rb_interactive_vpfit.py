from __future__ import print_function
import matplotlib
#matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt 
from rbvfit import model as m
from importlib import reload
reload(m)
from scipy.optimize import curve_fit
from scipy.stats.distributions import  t

import pdb
import emcee
import corner




def compute_error(y,pars,pcov,alpha=0.05):
    # 95% confidence interval = 100*(1-alpha)

    n = len(y)    # number of data points
    p = len(pars) # number of parameters
    dof = max(0, n - p) # number of degrees of freedom
    # student-t value for the dof and confidence level
    tval = t.ppf(1.0-alpha/2., dof) 

    for i, p,var in zip(range(n), pars, np.diag(pcov)):
        sigma = var**0.5
        print(p - sigma*tval,p + sigma*tval)




def set_bounds(nguess,bguess,vguess):

    Nlow=np.zeros((len(nguess,)))
    blow=np.zeros((len(nguess,)))
    vlow=np.zeros((len(nguess,)))


    NHI=np.zeros((len(nguess,)))
    bHI=np.zeros((len(nguess,)))
    vHI=np.zeros((len(nguess,)))

    for i in range(0,len(nguess)):
        Nlow[i]=nguess[i]-2.

        blow[i]=bguess[i]-20.
        if blow[i] < 2.:
            blow[i] = 2.

        vlow[i]=vguess[i]-50.

        NHI[i]=nguess[i]+2.

        bHI[i]=bguess[i]+20.
        if bHI[i] > 200.:
            bHI[i] = 150.

        vHI[i]=vguess[i]+50.
    lb=np.concatenate((Nlow,blow,vlow))
    ub=np.concatenate((NHI,bHI,vHI))
    bounds=[lb,ub]
    return bounds, lb, ub



def vel2shift(vel):
    c = 299792.458;  #% speed of light [km/sec]
    Beta  = Vel/c;
    z = np.sqrt((1.+Beta)/(1.-Beta)) - 1.;
    return z

def map_vel2wave(velgrid,wavegrid,vel):
    return np.interp(vel,velgrid,wavegrid)



def quick_nv_estimate(wave,norm_flx,wrest,f0):
    # All done in rest frame
    spl=2.9979e5  #speed of light
    vel = (wave-wrest*(1.0 + 0.))*spl/(wrest*(1.0 + 0.))
    lambda_r=wave/(1+0.)    
    #compute apparent optical depth
    Tau_a =np.log(1./norm_flx);
    # REMEMBER WE ARE SWITCHING TO VELOCITY HERE
    del_vel_j = np.diff(vel);
    del_vel_j = np.append([del_vel_j[0]], del_vel_j)
    # Column density per pixel as a function of velocity
    nv = Tau_a / ((2.654e-15) * f0 * lambda_r) # in units cm^-2 / (km s^-1), SS91 
    n = nv * del_vel_j  # column density per bin obtained by multiplying differential Nv by bin width
    return vel, n 

def plot_transitions(wave,flux,error,wrest,xrange,ntransition=1,modelflux=[0],individual_components=[0]):
    fig = plt.figure()

    for i in range(0,ntransition):
        ax=fig.add_subplot(ntransition,1,i+1)
        spl=2.9979e5;  #speed of light
        vel = (wave-wrest[i]*(1.0 + 0.))*spl/(wrest[i]*(1.0 + 0))

        ax.step(vel,flux)
        ax.step(vel,error,color='r')
        if len(modelflux) >1:
            ax.plot(vel,modelflux,'-',color='k',lw=2)

        if len(individual_components) > 1:
            temp=individual_components.shape
            nclump= int(temp[1])
            for i in range(0,nclump):
                ax.plot(vel,individual_components[:,i],'g:')
        ax.set_xlim(xrange)
        ax.set_ylim([-0.02,1.8])
        ax.plot([-2500,2500],[0,0],'k:')
        ax.plot([-2500,2500],[1,1],'k:')       
        ax.set_xlabel('vel [km/s]')
        ax.set_ylabel('Normalized Flux')
    plt.show()


        

######## Computing Likelihoods######
def lnprior(theta,lb,ub):
    for index in range(0,len(lb)):
        if (lb[index] > theta[index]) or (ub[index] < theta[index]):
            return -np.inf
            break
    return 0.0


def lnlike(theta, x, y, yerr,model):    
    model = model.model_flux(theta,x)
    inv_sigma2 = 1.0/(yerr**2 )
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))



def lnprob(theta,lb,ub,model, x, y, yerr):
    lp = lnprior(theta,lb,ub)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr,model)


def run_mcmc(wave_final,flux_final,error_final,lb,ub,popt,model):
    ###### Define a lot of walkwers
    length_of_lb=len(lb)
    no_of_Chain=50
    ndim, nwalkers = length_of_lb, no_of_Chain
    guesses = [popt+ 1.e-6*np.random.randn(ndim) for i in range(nwalkers)]


    print("Starting emcee ***********")

    # Starting emcee
    no_of_steps=1000
    burntime=np.round(no_of_steps*.2)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=8,args=(lb,ub,model,wave_final,flux_final,error_final))

    import sys
    width = 30
    #First Burning some steps
    print("Start Burntime Calculations...")
    pos, prob, state = sampler.run_mcmc(guesses, burntime)
    sampler.reset()
    print("Done Burning Steps!")
    print("Now starting the Final Calculations:")
    print("*****************")
    #Now Running mcmc
    for i, result in enumerate(sampler.sample(pos, iterations=no_of_steps)):
        n = int((width+1) * float(i) / no_of_steps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
    sys.stdout.write("\n")

    return sampler,ndim, nwalkers

def plot_corner(sampler,theta,ndim):
    samples=sampler.chain[:, 100:, :].reshape((-1, ndim))#sampler.flatchain
    st=np.percentile(samples,50,axis=0)#=np.median(samples,axis=0)#np.median(sampler.flatchain, axis=0)
    #df = pd.DataFrame(samples)
    #temp=df.mode()
    #st=temp.values[0]

    fig = plt.figure()

    ax=fig.add_subplot(111)
    nfit=int(len(theta)/3)
    N_tile=np.tile("logN",nfit)
    b_tile=np.tile("b",nfit)
    v_tile=np.tile("v",nfit)
    
    tmp=np.append(N_tile,b_tile)
    text_label=np.append(tmp,v_tile)
    
    figure=corner.corner(samples, labels=text_label, truths=st)
    theta_prime=st
    
    
    value1 =np.percentile(samples,32,axis=0)
    
    # This is the empirical mean of the sample:
    value2 = np.percentile(samples,68,axis=0)
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))
    
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(value1[i], color="aqua")
        ax.axvline(value2[i], color="aqua")
    
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value1[xi],color="aqua")
            ax.axvline(value2[xi],color="aqua")
            #ax.axhline(value1[yi], color="g")
            #ax.axhline(value2[yi], color="r")
            #ax.plot(value1[xi], value1[yi], "sg")
            #ax.plot(value2[xi], value2[yi], "sr")

    plt.show()

    return theta_prime, value1, value2, samples



def plot_mcmc_posteriors(samples,model,theta_prime,wave,flux,error
        ,value1,value2,ntransition,wrest,xrange=[-600.,600.]):
    best_fit,f1 = model.model_fit(theta_prime,wave)

    fig = plt.figure(figsize=(16, 9))
    BIGGER_SIZE = 12
    
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    index=np.random.randint(0,high=len(samples), size=100)
    
    
    n_clump=int(len(theta_prime)/3)
    
    best_N=theta_prime[0:n_clump]
    best_b=theta_prime[n_clump:2*n_clump]
    best_v=theta_prime[2*n_clump:3*n_clump]
    
    low_N=value1[0:n_clump]
    low_b=value1[n_clump:2*n_clump]
    low_v=value1[2*n_clump:3*n_clump]
    
    high_N=value2[0:n_clump]
    high_b=value2[n_clump:2*n_clump]
    high_v=value2[2*n_clump:3*n_clump]
    
    
    
    fig = plt.figure()
    for i in range(0,ntransition):
        ax=fig.add_subplot(ntransition,1,i+1)
        spl=2.9979e5;  #speed of light
        vel = (wave-wrest[i]*(1.0 + 0.))*spl/(wrest[i]*(1.0 + 0))

        ax.step(vel,flux,'k-',linewidth=1.)
        ax.step(vel,error,color='r',linewidth=1.)
        
        for i in range(len(index)):
            ax.plot(vel,model.model_flux(samples[index[i],:],wave),color="k", alpha=0.1)


        ax.plot(vel,best_fit,'-',color='k',lw=2)

        for i in range(0,n_clump):
            ax.plot([best_v[i],best_v[i]],[1.05,1.15],'k--',lw=4)
            text1=r'$logN \;= '+ np.str('%.2f' % best_N[i]) +'^{ + ' + np.str('%.2f' % (best_N[i]-low_N[i]))+'}'+ '_{ -' +  np.str('%.2f' % (high_N[i]-best_N[i]))+'}$'
            ax.text(best_v[i],1.2,text1,
                 fontsize=14,rotation=90, rotation_mode='anchor')
            text2=r'$b ='+np.str('%.0f' % best_b[i]) +'^{ + ' + np.str('%.0f' % (best_b[i]-low_b[i]))+'}'+ '_{ -' +  np.str('%.0f' % (high_b[i]-best_b[i]))+'}$'
    
            ax.text(best_v[i]+10,1.2, text2,
                 fontsize=14,rotation=90, rotation_mode='anchor')
    
            ax.plot(vel,f1[:,i],'g:',linewidth=3)
    

        ax.set_xlim(xrange)
        ax.set_ylim([-0.02,1.8])
        ax.plot([-2500,2500],[0,0],'k:')
        ax.plot([-2500,2500],[1,1],'k:')       
        ax.set_xlabel('vel [km/s]')
        ax.set_ylabel('Normalized Flux')
    plt.show()



def rb_interactive_vpfit(wave,flux,error,wrest,zabs,ntransition=1,custom_guess=False,FWHM=6.5,xrange=[-600.,600.],mcmc=False):
    ''' 
    -----------------------------------------------------------------------------------
       This is an interactive routine to fit Voigt profiles to absorption line spectra. 
       It could be used to fit an absorption singlet/multiplet and with or without intervening nuissance parameters. 

       Input:- 
            Must include:-
                 wave    :  wavelength vector [observed frame]
                 flux    :  Normalized flux vector.     
                 error.  :  Normalized error vector
                 wrest.  :  list of rest frame wavelengths for each transition [can add nuissance parameters in the end]
                 zabs.   :  list of redshifts for all absorbers (e.g. first entry for all transitions and additional ones are for nuissance paramters)

            Optional Input:-
                 ntransition   : Number of transitions [default =1]
                 custom_guess  : Custom [N,b,v] guess, if given don't do interactive fit
                 FWHM.         : FWHM in pixels [default 6.5 pixes HST/COS]
                 xrange.       : velocity range to show plots
                 mcmc.         : If set True Performs mcmc fit [default False]



    Written By:   Rongmon Bordoloi       July 22 2019.
    Tested on :   python 3.7
    -----------------------------------------------------------------------------------
    
    Dependencies:
                  numpy, matplotlib, scipy, astropy, rbvfit, emcee, corner.

                  If user wishes to use HST/COS LSF: linteools is needed to be installed.
    -----------------------------------------------------------------------------------
    
    '''


    # Fix infinities

    sq=np.isnan(flux);
    flux[sq]=0;
    sqq=flux<=0;
    flux[sqq]=0;
    q=flux<=0;
    flux[q]=error[q];
    nuissance_flag=False

    # Converting To velocities
    from CGM import rb_setline as line
    # Now create a list of lambda_rest and velocities
    vellist=np.zeros((len(wrest),len(wave)))
    lambda_restlist=np.zeros((len(wrest),))
    
    # If there are nuissance paramters
    zlist=np.array(zabs)
    if len(zlist) > 1:
        zlist=np.append(np.repeat(zlist[0],ntransition),zlist[1:])
        nuissance_flag=True
        n_nuissance= len(zlist)-1
    else:
        zlist=np.repeat(zlist[0],ntransition)

    for i in range(0,len(wrest)):
        str=line.rb_setline(wrest[i],'closest','atom')
        lambda_restlist[i]=str['wave']
        spl=2.9979e5;  #speed of light
        vel = (wave-str['wave']*(1.0 + zlist[0]))*spl/(str['wave']*(1.0 + zlist[0]))
        temp_vel = (wave-str['wave']*(1.0 + zlist[i]))*spl/(str['wave']*(1.0 + zlist[i]))
        vellist[i,:]=temp_vel

    
    # AOD column guess for the primary line
    str=line.rb_setline(wrest[0],'closest','atom')
    vel,nv=quick_nv_estimate(wave/(1.+zlist[0]),flux,str['wave'],str['fval']);

    # Start guessing initial fit interactively
    if custom_guess == False:
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.step(vel,flux)
        ax.step(vel,error,color='r')
        ax.set_xlim(xrange)#[np.min(vel),np.max(vel)])
        ax.set_ylim([-0.02,1.8])
        ax.plot([-2500,2500],[0,0],'k:')
        ax.plot([-2500,2500],[1,1],'k:')       
        ax.set_xlabel('vel [km/s]')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('click to select line centers!')
        xx = plt.ginput(n=-1)
        print('Once you are done, press enter to continue!')
        n_clouds=int(len(xx))


        # Create the guesses for starting a fit
        bguess=np.zeros(n_clouds,)
        vguess=np.zeros(n_clouds,)
        nguess=np.zeros(n_clouds,)
        plt.close()

        for i in range(0,n_clouds):
            vguess[i]=xx[i][0]
            qq=np.where( (vel < vguess[i]+ 10.) & (vel > vguess[i]-10.))
            nguess[i]=np.log10(sum(nv[qq]))        

            #Now ask interactively for b values     
            prompt='Guess  b  for line ' +np.str(i+1)+ '/'+np.str(n_clouds) +', vel guess = ' + np.str('%.1f' % vguess[i])  +', col guess= '+ np.str('%.1f' % nguess[i])+ ': '
            tmp_b =  input(prompt)
            bguess[i]= np.double(tmp_b)

        # If there are nuissance parameters add them now.
        '''
        -----------------------------------------------------------------

        '''

        if nuissance_flag == True:
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.step(vel,flux)
            ax.step(vel,error,color='r')
            ax.set_xlim(xrange)#[np.min(vel),np.max(vel)])
            for i in range(0,n_clouds):
                ax.plot([vguess[i],vguess[i]],[1.1,1.3],'r-',lw=1)
    
            ax.set_ylim([-0.02,1.8])
            ax.plot([-2500,2500],[0,0],'k:')
            ax.plot([-2500,2500],[1,1],'k:')       
            ax.set_xlabel('vel [km/s]')
            ax.set_ylabel('Normalized Flux')
            ax.set_title('Add  # ['+ np.str(n_nuissance)+ '] NUISSANCE parameter Line Center!')
            xx = plt.ginput(n=n_nuissance,show_clicks=True)
            print('Once you are done, press enter to continue!')
            plt.close()
    
            
            # HERE Loop through Each nuissance parameter and compute their new redshifts at the point clicked.
            for i in range(0,n_nuissance):
                str1=line.rb_setline(lambda_restlist[i+ntransition],'closest','atom')
                new_wave=map_vel2wave(vel,wave,xx[i][0])
                # Replace zabs original values with more accurate values
                zabs[i+1]=(new_wave-str1['wave'])/str1['wave']
                
    
                vguess=np.append(vguess,np.array(0.))
    
                # AOD column guess for nuissance lines
                vel1,nv1=quick_nv_estimate(wave/(1.+zabs[i+1]),flux,str1['wave'],str1['fval'])
    
                qq=np.where( (vel1 < vguess[n_clouds+i]+ 10.) & (vel1 > vguess[n_clouds+i]-10.))
                nguess=np.append(nguess,np.log10(sum(nv1[qq])))
    
                #Now ask interactively for b values     
                prompt='Guess  b  for line ' +np.str(wrest[ntransition+i])+ '/'+np.str(n_nuissance) +', zabs = ' + np.str('%.5f' % zabs[1+i])  +', col guess= '+ np.str('%.1f' % nguess[n_clouds+i])+ ': '
                tmp_b =  input(prompt)
                bguess= np.append(bguess,np.double(tmp_b))
    
        



        theta=np.concatenate((nguess,bguess,vguess))
        #pdb.set_trace()
    else:
        theta=custom_guess
        n_clouds=int(len(theta)/3)
        nguess=theta[0:n_clouds]
        bguess=theta[n_clouds:n_clouds+n_clouds]
        vguess=theta[2*n_clouds:3*n_clouds]        


    bounds, lb, ub=set_bounds(nguess,bguess,vguess)
    print(bounds)
    
    #pdb.set_trace()

    # Now starting to set up the Voigt Profile model
    
    s=m.create_voigt(zabs,lambda_restlist,n_clouds,ntransition=ntransition,FWHM=4.5)

    
    outflux= s.model_flux(theta,wave)

    def test_fit(wave,*params):
        return  s.model_flux(params,wave)

 

    #pdb.set_trace()
    popt, pcov = curve_fit(test_fit, wave, flux, p0=(theta),sigma=error,bounds=bounds)
    print(bounds)
    print(popt)
    compute_error(wave,popt,pcov,alpha=0.05)


    model_flux,ind_comp=s.model_fit(popt,wave)

    plot_transitions(wave,flux,error,lambda_restlist,xrange,ntransition=ntransition,modelflux=model_flux,individual_components=ind_comp)

    if mcmc == True:
        plt.close()
        samples,ndim, nwalkers=run_mcmc(wave,flux,error,lb,ub,popt,s)
        theta_prime, values1, values2, samples=plot_corner(samples,popt,ndim)
        plot_mcmc_posteriors(samples,s,theta_prime,wave,flux,error,values1,values2,ntransition,lambda_restlist,xrange=[-600.,600.])




    '''
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.step(vel,flux)
    ax.step(vel,error,color='r')
    ax.plot(vel,outflux,':',color='k',lw=1)
    ax.plot(vel,s.model_flux(popt,wave),color='g',lw=2)
    plt.xlim(xrange)
    plt.ylim([-0.02,1.8])
    ax.plot([-2500,2500],[0,0],'k:')
    ax.plot([-2500,2500],[1,1],'k:')       
    ax.set_xlabel('vel [km/s]')
    ax.set_ylabel('Normalized Flux')
    plt.show()
    '''





    
