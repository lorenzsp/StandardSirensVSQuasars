###################################################
# MCMC of LISA standard sirens with flat LCDM
# execute with: python3 mcmc_flatLCDM.py SSdatasets/median_dataset_15SS.dat
# Note:the running time depends on how many iterations you want, check variable "iterations"
###################################################

import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ellipkinc
import multiprocessing as mp
import corner

import sys

# name of the data set
filename =  str(sys.argv[1])

# remember to seed
np.random.seed(42)

#######################################################
##### Cosmological Model #############################
#######################################################

# Omega matter
Omega_m_ref = 0.3
# Hubble parameter
href = 0.7
# Hubble constant divided by the speed of light
H0 = href/2997.9

################################
# flat LCDM luminosity distance
################################

# numerical integration to get distance for LCDM, which can be estended to wCDM model


# Generate the LCDM model with the given z, Omega_m and h
def integrand(z, params):
    Omega_m, h = params
    h = h/2997.9
    H = h * np.sqrt(Omega_m * (1+z)**3 + (1-Omega_m) )
    return 1/H

# distance luminosity flat LCDM
def getDL_LCDM (z, params):
    I = np.array([ quad (integrand, 0, z[j], args=(params))[0] for j in range (0, len(z)) ])
    return (1.0+z)*I


# flat LCDM luminsity distance with elliptic integral
def T(x):
    ar =  np.arccos((1 + (1-np.sqrt(3))*x )/(1 + (1+np.sqrt(3))*x ) )
    
    return ellipkinc(ar ,(2.+np.sqrt(3))/4. ) /np.power(3,1.0/4.0)

def Ell_DL_LCDM(z, params):
    Omega_m, h = params
    h = h/2997.9
    
    s = np.power( (1.0-Omega_m)/Omega_m , 1.0/3.0 )
    
    return ((1.0+z)/(h * np.sqrt(s * Omega_m)) )*(T(s) - T(s/(1.0+z)) )

# check correctness of luminosity distance integration
zz = np.arange(0.1, 10, 0.1)
MM = np.max( (getDL_LCDM(zz, [Omega_m_ref,href]) - Ell_DL_LCDM(zz, [Omega_m_ref,href]))/Ell_DL_LCDM(zz, [Omega_m_ref,href])  )
print("max relative difference = " ,MM)


###############################################
# Catalogue with data in Gpc
###############################################
data_z_D_dD = pd.read_csv(filename ,sep="\s+", header=None).values # in Gpc
z_SS = data_z_D_dD[:,0]
DL_SS = data_z_D_dD[:,1]*1e3 # transform to Mpc
dDL_SS = data_z_D_dD[:,2]*1e3 # transform to Mpc
print("Imported Data",len(z_SS))
#print(z_SS)


###########################################
# priors, likelihood and posteriors
###########################################
# for a beta prior uncomment the following lines
#MEDIAN = 0.3
#b=100
#a= (1/3 +MEDIAN *(b-2/3))/(1-MEDIAN) # the median value is 0.31

# log_prior of omega matter: flat in omega
def log_prior_Omega(Omega):
    if (Omega<0.0) or (Omega>1.0):
        result = -np.inf
    else:
        result = 0 #stats.beta.logpdf(Omega,a,b) # for beta prior uncomment
    
    return result

# log_prior of h: flat in h
def log_prior_H(h):

    if (h<0.2) or (h>2.2):
        result = -np.inf
    else:
        result = -np.log(2.2-0.2)

    return result

# log likelihood
def log_lik(params, z, D, dD):
    
    return np.sum(-0.5*( (D - Ell_DL_LCDM(z, params))/dD )**2 ) - np.sum(np.log(np.sqrt(2* np.pi)*dD))

# posterior
def log_post(params, z, D, dD):
    Omega, h = params
    if (h<0.2) or (h>2.2) or (Omega<0.0) or (Omega>1.0):
        result = -np.inf
    else:
        result = log_lik(params,z,D, dD)  + log_prior_Omega(Omega) + log_prior_H(h)
    
    return result

###############################
# multiprocessing MCMC 
##############################

# define number of parallel processes
Nprocs = mp.cpu_count()
pool = mp.Pool(Nprocs)

# number of iterations (vary according to how many samples you need for the posterior)
iterations = 1000000

#############################
# MCMC
#############################

# start of the MCMC
pos = [Omega_m_ref, href]

# define sampler
def MCMC_sampler(pos):
    print("Start of the MCMC")
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=(z_SS, DL_SS, dDL_SS), pool=pool) 
    sampler.run_mcmc(pos, iterations, progress=True) # change progress to True if you want to see the progress bar

    # info about the MCMC
    #print("autocorrelation ",sampler.get_autocorr_time())
    #print("acceptance fraction ", sampler.acceptance_fraction() )
    #print("check the convergence by looking at the median of each chain")
    #print(np.median(sampler.get_chain(discard=150, thin=10, flat=False),axis=0))
    flat_samples = sampler.get_chain(discard=150, thin=10, flat=True)

    return flat_samples


# different starts for the MCMC
pos = pos + 1e-2 * np.random.randn(16, 2)

# output
out = MCMC_sampler(pos)

# store samples
flat_samples = out
print("number of samples = ", len(flat_samples))
print("median = ", np.median(flat_samples,axis=0))

# save with name of the dataset + .npy in the same folder of the data set
np.save(filename, flat_samples)

####################################
# preview of the posterior
####################################

labels = [r"Omega_m", r"h"]

fig = corner.corner(flat_samples, labels=labels, truths=[Omega_m_ref, href],DataPoints = False)

plt.savefig("plots/flatLCDM_posterior")
