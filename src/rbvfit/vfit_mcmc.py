from __future__ import print_function
import emcee
from multiprocessing import Pool
import numpy as np
import corner
import matplotlib.pyplot as plt
import sys
import scipy.optimize as op
from rb_vfit import rb_veldiff as rb_veldiff
from rbvfit import rb_setline as rb
import pdb


def plot_model(wave_obs,fnorm,enorm,fit,model,outfile= False,xlim=[-600.,600.],verbose=False):
        #This model only works if there are no nuissance paramteres
        

        theta_prime=fit.best_theta
        value1=fit.low_theta
        value2=fit.high_theta
        n_clump=model.nclump 
        n_clump_total=np.int(len(theta_prime)/3)

        ntransition=model.ntransition
        zabs=model.zabs

        samples=fit.samples
        model_mcmc=fit.model

        wave_list=np.zeros( len(model.lambda_rest_original),)
        # Use the input lambda rest list to plot correctly
        for i in range(0,len(wave_list)):
            s=rb.rb_setline(model.lambda_rest_original[i],'closest')
            wave_list[i]=s['wave']


        wave_rest=wave_obs/(1+zabs[0])
        
        best_N = theta_prime[0:n_clump_total]
        best_b = theta_prime[n_clump_total:2 * n_clump_total]
        best_v = theta_prime[2 * n_clump_total:3 * n_clump_total]
        
        low_N = value1[0:n_clump_total]
        low_b = value1[n_clump_total:2 * n_clump_total]
        low_v = value1[2 * n_clump_total:3 * n_clump_total]
        
        high_N = value2[0:n_clump_total]
        high_b = value2[n_clump_total:2 * n_clump_total]
        high_v = value2[2 * n_clump_total:3 * n_clump_total]
            


        #Now extracting individual fitted components
        best_fit, f1 = model.model_fit(theta_prime, wave_obs)

        fig, axs = plt.subplots(ntransition, sharex=True, sharey=False,figsize=(12,18 ),gridspec_kw={'hspace': 0})
        
        
        BIGGER_SIZE = 18
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        index = np.random.randint(0, high=len(samples), size=100)
        
        
        if ntransition == 1:
            #When there are no nuissance parameter
            #Now loop through each transition and plot them in velocity space
            vel=rb_veldiff(wave_list[0],wave_rest)
            axs.step(vel, fnorm, 'k-', linewidth=1.)
            axs.step(vel, enorm, color='r', linewidth=1.)
            # Plotting a random sample of outputs extracted from posterior dis
            for ind in range(len(index)):
                axs.plot(vel, model_mcmc(samples[index[ind], :], wave_obs), color="k", alpha=0.1)
            axs.set_ylim([0, 1.6])
            axs.set_xlim(xlim)
            axs.plot(vel, best_fit, color='b', linewidth=3)
            axs.plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
            # plot individual components
            for dex in range(0,np.shape(f1)[1]):
                axs.plot(vel, f1[:, dex], 'g:', linewidth=3)
    
            for iclump in range(0,n_clump):
                axs.plot([best_v[iclump],best_v[iclump]],[1.05,1.15],'k--',lw=4)
                text1=r'$logN \;= '+ np.str('%.2f' % best_N[iclump]) +'^{ + ' + np.str('%.2f' % (best_N[iclump]-low_N[iclump]))+'}'+ '_{ -' +  np.str('%.2f' % (high_N[iclump]-best_N[iclump]))+'}$'
                axs.text(best_v[iclump],1.2,text1,
                     fontsize=14,rotation=90, rotation_mode='anchor')
                text2=r'$b ='+np.str('%.0f' % best_b[iclump]) +'^{ + ' + np.str('%.0f' % (best_b[iclump]-low_b[iclump]))+'}'+ '_{ -' +  np.str('%.0f' % (high_b[iclump]-best_b[iclump]))+'}$'
    
                axs.text(best_v[iclump]+30,1.2, text2,fontsize=14,rotation=90, rotation_mode='anchor')
  
        
        
        
        
        else:
     
            
            #Now loop through each transition and plot them in velocity space
            for i in range(0,ntransition):
                print(wave_list[i])
                vel=rb_veldiff(wave_list[i],wave_rest)
                axs[i].step(vel, fnorm, 'k-', linewidth=1.)
                axs[i].step(vel, enorm, color='r', linewidth=1.)
                #pdb.set_trace()
                # Plotting a random sample of outputs extracted from posterior distribution
                for ind in range(len(index)):
                    axs[i].plot(vel, model_mcmc(samples[index[ind], :], wave_obs), color="k", alpha=0.1)
                axs[i].set_ylim([0, 1.6])
                axs[i].set_xlim(xlim)
                
                
            
                axs[i].plot(vel, best_fit, color='b', linewidth=3)
                axs[i].plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
    
                # plot individual components
                for dex in range(0,np.shape(f1)[1]):
                    axs[i].plot(vel, f1[:, dex], 'g:', linewidth=3)
                
                for iclump in range(0,n_clump):
                    axs[i].plot([best_v[iclump],best_v[iclump]],[1.05,1.15],'k--',lw=4)
                    if i ==0:
                        text1=r'$logN \;= '+ np.str('%.2f' % best_N[iclump]) +'^{ + ' + np.str('%.2f' % (best_N[iclump]-low_N[iclump]))+'}'+ '_{ -' +  np.str('%.2f' % (high_N[iclump]-best_N[iclump]))+'}$'
                        axs[i].text(best_v[iclump],1.2,text1,
                                 fontsize=14,rotation=90, rotation_mode='anchor')
                        text2=r'$b ='+np.str('%.0f' % best_b[iclump]) +'^{ + ' + np.str('%.0f' % (best_b[iclump]-low_b[iclump]))+'}'+ '_{ -' +  np.str('%.0f' % (high_b[iclump]-best_b[iclump]))+'}$'
                
                        axs[i].text(best_v[iclump]+30,1.2, text2,
                                 fontsize=14,rotation=90, rotation_mode='anchor')
        
        if verbose==True:
            from IPython.display import display, Math
    
            samples = fit.sampler.get_chain(discard=100, thin=15, flat=True)
            nfit = int(fit.ndim / 3)
            N_tile = np.tile("logN", nfit)
            b_tile = np.tile("b", nfit)
            v_tile = np.tile("v", nfit)
            tmp = np.append(N_tile, b_tile)
            text_label = np.append(tmp, v_tile)
            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
    
            
                display(Math(txt))

      



        if outfile==False:
            plt.show()
        else:
            outfile_fig =outfile
            fig.savefig(outfile_fig, bbox_inches='tight')




######## Computing Likelihoods######
def lnprior(theta, lb, ub):
    for index in range(0, len(lb)):
        if (lb[index] > theta[index]) or (ub[index] < theta[index]):
            return -np.inf
            break
    return 0.0


def lnlike(theta, model, x, y, yerr):
    model = model(theta, x)
    inv_sigma2 = 1.0 / (yerr ** 2)
    return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, lb, ub, model, x, y, yerr):
    lp = lnprior(theta, lb, ub)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, model, x, y, yerr)


def optimize_guess(model, theta, lb, ub, x, y, yerr):
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, [theta], args=(lb, ub, model, x, y, yerr))
    p = result["x"]
    return p



def set_bounds(nguess,bguess,vguess):

    Nlow=np.zeros((len(nguess,)))
    blow=np.zeros((len(nguess,)))
    vlow=np.zeros((len(nguess,)))


    NHI=np.zeros((len(nguess,)))
    bHI=np.zeros((len(nguess,)))
    vHI=np.zeros((len(nguess,)))

    for i in range(0,len(nguess)):
        Nlow[i]=nguess[i]-2.

        blow[i]=bguess[i]-40.
        if blow[i] < 2.:
            blow[i] = 2.

        vlow[i]=vguess[i]-50.

        NHI[i]=nguess[i]+2.

        bHI[i]=bguess[i]+40.
        if bHI[i] > 200.:
            bHI[i] = 150.

        vHI[i]=vguess[i]+50.
    lb=np.concatenate((Nlow,blow,vlow))
    ub=np.concatenate((NHI,bHI,vHI))
    bounds=[lb,ub]
    return bounds, lb, ub

class vfit(object):
    def __init__(self, model, theta, lb, ub, wave_obs, fnorm, enorm, no_of_Chain=50, no_of_steps=1000,
                 perturbation=1e-6):
        self.wave_obs = wave_obs
        self.fnorm = fnorm
        self.enorm = enorm
        self.model = model
        self.lb = lb
        self.ub = ub
        self.theta = theta
        self.no_of_Chain = no_of_Chain
        self.no_of_steps = no_of_steps
        self.perturbation = perturbation

    def runmcmc(self, optimize=True,verbose=False):
        model = self.model
        theta = self.theta
        lb = self.lb
        ub = self.ub
        wave_obs = self.wave_obs
        fnorm = self.fnorm
        enorm = self.enorm
        no_of_Chain = self.no_of_Chain
        no_of_steps = self.no_of_steps
        perturbation = self.perturbation

        if optimize == True:
            print('Optimizing Guess ***********')
            # Now make a better guess
            popt = optimize_guess(model, theta, lb, ub, wave_obs, fnorm, enorm)
            print('Done ***********')
        else:
            print('Skipping Optimizing Guess ***********')
            print('Using input guess for mcmc ***********')
            popt = theta

        print('Preparing emcee ***********')
        ###### Define a lot of walkers
        length_of_lb = len(lb)
        ndim, nwalkers = length_of_lb, no_of_Chain

        guesses = [popt + perturbation * np.random.randn(ndim) for i in range(nwalkers)]
        print("Starting emcee ***********")
        burntime = np.round(no_of_steps * .2)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,  pool=pool, args=(lb, ub, model, wave_obs, fnorm, enorm))
            pos, prob, state = sampler.run_mcmc(guesses, no_of_steps,progress=True)


        #sampler.reset()
        print("Done!")
        #print("Now starting the Final Calculations:")
        print("*****************")

        #width = 30
        # Now Running mcmc
        #for i, result in enumerate(sampler.sample(pos, iterations=no_of_steps)):
        #    n = int((width + 1) * float(i) / no_of_steps)
        #sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        #sys.stdout.write("\n")
        if verbose==True:
            from IPython.display import display, Math

            samples = sampler.get_chain(discard=100, thin=15, flat=True)
            nfit = int(ndim / 3)
            N_tile = np.tile("logN", nfit)
            b_tile = np.tile("b", nfit)
            v_tile = np.tile("v", nfit)

            tmp = np.append(N_tile, b_tile)
            text_label = np.append(tmp, v_tile)

            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
                display(Math(txt))


        self.sampler = sampler
        self.ndim = ndim
        self.nwalkers = nwalkers

    def plot_corner(self,outfile=False):
        ndim=self.ndim
        #samples = self.sampler.chain[:, 100:, :].reshape((-1, ndim))  # sampler.flatchain
        samples = self.sampler.get_chain(discard=100, thin=15, flat=True)

        st = np.percentile(samples, 50, axis=0)  # =np.median(samples,axis=0)#np.median(sampler.flatchain, axis=0)
        # df = pd.DataFrame(samples)
        # temp=df.mode()
        # st=temp.values[0]

        nfit = int(ndim / 3)
        N_tile = np.tile("logN", nfit)
        b_tile = np.tile("b", nfit)
        v_tile = np.tile("v", nfit)

        tmp = np.append(N_tile, b_tile)
        text_label = np.append(tmp, v_tile)

        figure = corner.corner(samples, labels=text_label, truths=st)
        theta_prime = st

        value1 = np.percentile(samples, 10, axis=0)

        # This is the empirical mean of the sample:
        value2 = np.percentile(samples, 90, axis=0)
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
                ax.axvline(value1[xi], color="aqua")
                ax.axvline(value2[xi], color="aqua")
        # ax.axhline(value1[yi], color="g")
        # ax.axhline(value2[yi], color="r")
        # ax.plot(value1[xi], value1[yi], "sg")
        # ax.plot(value2[xi], value2[yi], "sr")

        self.best_theta=theta_prime
        self.low_theta=value1
        self.high_theta=value2
        self.samples=samples

        if outfile==False:
            plt.show()
        else:
            outfile_fig =outfile
            figure.savefig(outfile_fig, bbox_inches='tight')

            

  