from scipy.special import wofz
import numpy as np
import os
from astropy.convolution import convolve, Gaussian1DKernel
from CGM import rb_setline as rb

class line(object):
	name=""
	wave0=0.
	f=0.
	gamma=0.
	def __init__(self,name,lambda0,f,gamma,z):
		self.name=name
		self.wave0=lambda0
		self.f=f
		self.gamma=gamma
		self.z=z

def rb_veldiff(lam_cen,lam_offset):    
	z=(lam_offset/lam_cen) -1.
	C = 299792.458;  #% speed of light [km/sec]
	Beta =((z+1.)**2 - 1.)/(1. + (z+1.)**2.)
	return Beta*C

def chi2(model,spec,err):
	return np.sum((model-spec)**2/err)


def vel2shift(Vel):
#%----------------------------------------------------------------
#% vel2shift function    calculate the red/blue shift (Z)
#%                     from velocity.
#% Input  : - vector of Velocities in km/sec.
#% Output : - red/blue shift (Z).
#% Tested : Matlab 2012
#%     By : Rongmon Bordoloi             Dec 2012
#%----------------------------------------------------------------
	C = 299792.458;  #% speed of light [km/sec]
	Beta  = Vel/C;
	Z = np.sqrt((1.+Beta)/(1.-Beta)) - 1.;
	return Z


def voigt_tau(lambda0,gamma,f,N,b,wv):
	c=29979245800.0 #cm/s
	b_f=b/lambda0*10**13 #Doppler frequency, constant accounts for A,km conversion. Units of Hz
	a=gamma/(4*np.pi*b_f) #Dimensionless damping parameter of intrinsic line shape
	freq0=c/lambda0*10**8
	constant=448898479.507 #sqrt(pi)*e^2/m_e in cm^3/s^2
	constant/=freq0*b*10**5 #10^5 is b from km/s->cm/s
	
	freq=c/wv*10**8 #10^8 is cm/s->A/s conversion
	x=(freq-freq0)/b_f #Dimensionless input to H(a,x)
	H=np.real(wofz(x+1j*a))
	tau=N*f*constant*H #10^5 is b from km/s->cm/s
	#factor=N*f*constant/(freq0*b*10**5)
	#tau=N*f*constant/(freq0*b*10**5)*H #10^5 is b from km/s->cm/s
	return tau
def voigt(lambda0,gamma,f,N,b,vel,Cf,wv):
	z=vel2shift(vel)
	wv_r=wv/(1.+z)
	return (1.-Cf)+Cf*np.exp(-voigt_tau(lambda0,gamma,f,10**N,b,wv_r))




class model(object):
	def __init__(self):
		self.lines=np.array([])



	def addline(self,waverest,z=0.0,method='closest',verbose=False):
		# set a line
		s=rb.rb_setline(waverest,method)
		lambda0=s['wave']
		gamma=s['gamma']
		f=s['fval']
		name=s['name']
		if verbose == True:
			print('Added Line : ' + name +' at z = '+ np.str(z))
		self.lines=np.append(self.lines,line(name,lambda0,f,gamma,z))
	  # return self.lines

#Theta : all N, all b, all vel, all Cf
def create_model_simple(theta,specw,line):
	No_of_clumps=int(len(theta)/3)
	N=theta[0:No_of_clumps]
	b=theta[No_of_clumps:No_of_clumps+No_of_clumps]
	vel=theta[2*No_of_clumps:3*No_of_clumps]		
	voigt_one=np.zeros((len(specw),No_of_clumps),dtype='float')
	voigt_Full=np.zeros((len(specw),),dtype='float')
	for i in range(No_of_clumps):
		Cf=1.
		voigt_one[:,i]=voigt(line.lines[i].wave0,line.lines[i].gamma,line.lines[i].f,N[i],b[i],vel[i],Cf,specw/(1.+line.lines[i].z))
		voigt_Full=np.prod(voigt_one,axis=1)

	return voigt_one, voigt_Full

def model_profile(theta,wave,line):
	voigt_one, voigt_Full=create_model_simple(theta,wave,line)	
	COS_kernel=(6.5/2.355)/3. #6.5 pixels
	g = Gaussian1DKernel(stddev=COS_kernel)
	# Convolve data
	fmodel = convolve(voigt_Full, g,boundary='extend')	
	return fmodel,voigt_one


