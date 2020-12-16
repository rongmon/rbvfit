from __future__ import print_function
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
import sys
import scipy.optimize as op


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

    def runmcmc(self, optimize=True):
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=8, args=(lb, ub, model, wave_obs, fnorm, enorm))

        print("Start Burntime Calculations ***********")

        pos, prob, state = sampler.run_mcmc(guesses, burntime)
        sampler.reset()
        print("Done Burning Steps!")
        print("Now starting the Final Calculations:")
        print("*****************")
        width = 30
        # Now Running mcmc
        for i, result in enumerate(sampler.sample(pos, iterations=no_of_steps)):
            n = int((width + 1) * float(i) / no_of_steps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        sys.stdout.write("\n")

        self.sampler = sampler
        self.ndim = ndim
        self.nwalkers = nwalkers

    def plot_corner(self,outfile=False):
        ndim=self.ndim
        samples = self.sampler.chain[:, 100:, :].reshape((-1, ndim))  # sampler.flatchain
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


  