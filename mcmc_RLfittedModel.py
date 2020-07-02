###################################################
# MCMC of Risaliti Lusso fitted model
# Lorenzo Speri 06/2020
###################################################

import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import multiprocessing as mp
import corner
import h5py

# remember to seed
np.random.seed(42)

#######################################################
##### Cosmological Model #############################
#######################################################

href = 0.7 #  in the article
H0 = href/2997.9

# generate Risaliti Lusso
def DL_RL_article (z,a2,a3):
    """
    input: redshift z, a2, a3 as they are defined in the article
    output: log10( D_L H0 ) where D_L is the luminosity distance
    """
    
    ad = [1, a2, a3];
    F = np.log(10)*(ad[0]* np.log10(1+z) + ad[1]* (np.log10(1+z))**2 + ad[2]* (np.log10(1+z))**3)
    return np.log10(F)


##############################################################
# prior of a2 a3 from the posterior of the image of RL article
# likelihood
# posterior
###############################################################
# obtain the distribution of a2 and a3 from the image of the article
# and use it as a prior for the MCMC
a2_image= [3.233624454148471, 3.68122270742358,3.5240174672489077,3.4082969432314405,]
a3_image = [2.3031624863685938, 0.9683751363140685,1.7928026172300982,1.3936750272628133]

# center of the distribution in a3 a3 from the image
center =  np.array([3.5,1.5])#np.array([3.46872055604, 1.60207998535])

# define the covariance matrix
s = stats.chi2.ppf(0.6826894921370,2)

# factor 2 in front to enlarge the prior
sigma_1 = 4*np.sqrt( (a2_image[0]- a2_image[1])**2 +(a3_image[0]- a3_image[1])**2 ) /(2*np.sqrt(s))
sigma_2 = 4*np.sqrt((a2_image[2]- a2_image[3])**2 +(a3_image[2]- a3_image[3])**2) /(2*np.sqrt(s))

ang = np.arctan((a3_image[0]- a3_image[1])/(a2_image[0]- a2_image[1]) )

M12 = np.array([[1/(sigma_1**2), 0], [0, 1/(sigma_2**2)]])
Rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])

# invers of the covariance matrix ~ Fihser
Mfinal=np.dot(Rot.T, np.dot(M12,Rot) )

Cov = np.array(np.linalg.inv(Mfinal))
print(Cov)
Gamma = np.linalg.det(np.linalg.inv(Mfinal))
factor_prior_a2a3 = (2* np.pi*np.sqrt(Gamma))

# log Prior
def log_prior_a2_a3_normal(a2,a3):
    # input a2 a3 as from the article
    
    Y = np.vstack((a2,a3)).T
    ARG = np.dot((Y-center).dot(Mfinal),(Y-center).T)
    result= -ARG/2 -np.log(factor_prior_a2a3)
    return result

# log likelihood 
def log_likelihood_x_a2_a3(a2, a3,  z, D, dD):
    
    return np.sum(-(D - DL_RL_article(z, a2, a3))**2 /(2*(dD*dD) )) - np.sum(np.log(np.sqrt(2* np.pi)*np.abs(dD))) 

###############################################
# Risaliti Lusso data obtained as described in the article
###############################################
data = np.loadtxt('QSA_JLAdata/QSOdat.dat')
z = data[:,0]
d = data[:,1] #log10(D_L H0)
sigma_d = data[:,2] # sigma[log10(D_L H0)]

##########################################
# Supernovae data
##########################################
# magnitude
red_mag = np.loadtxt('QSA_JLAdata/JLA_MUB.dat')
z_JLA = red_mag[:,0]
# we subtract the calibration value to obtain the log10(DL)
dist_JLA = (red_mag[:,1] -43.1237)/5
# covariance matrix
cov_mat = np.loadtxt('QSA_JLAdata/jla_mub_covmatrix.dat').reshape(31,31) /25
inv_cov = np.linalg.inv(cov_mat)
# uncertainty estimate
sigma_dist_JLA = np.sqrt(np.diag(cov_mat))/5


# log likelihood (combine them with correlations)
def log_lik_JLA(a2, a3):
    ar = dist_JLA - DL_RL_article(z_JLA, a2, a3)
    first = np.dot(ar.T, np.dot(inv_cov,ar) )
    second = np.log( np.sqrt( np.power(2* np.pi,len(cov_mat)) *np.linalg.det(cov_mat) ) )
    return -0.5*first - second

#####################################
# redshift range
####################################
#ind = np.where(data[:,0]<1.4)
#z = data[ind,0]
#d = data[ind,1]
#sigma_d = data[ind,2]

#############################
# MCMC
#############################

# posterior
def log_post(a2_a3, z, d, sigma_d):
    a2, a3 = a2_a3

    log_lik_prop =  log_likelihood_x_a2_a3(a2, a3, z, d, sigma_d) + log_lik_JLA(a2, a3)
    
    log_prior_prop = log_prior_a2_a3_normal(a2, a3)
    return log_lik_prop + log_prior_prop


# different starts of the MCMC
pos =   center+ 1e-4 * np.random.randn(32, 2)


# define sampler
def MCMC_sampler(pos, iterations):
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=(z, d, sigma_d))
    sampler.run_mcmc(pos, iterations, progress=True)
    
    # info about the MCMC
    #print("autocorrelation ",sampler.get_autocorr_time())
    #print("acceptance fraction ", sampler.acceptance_fraction)
    flat_samples = sampler.get_chain(discard=150, thin=10, flat=True)

    return flat_samples


###############################
# parallel computation
##############################

# define number of parallel processes
Nprocs = mp.cpu_count()
pool = mp.Pool(Nprocs)

# number of iterations (vary according to how many samples you need for the posterior)
iterations = 1000000

# different starts for the MCMC
input = [(pos + 1e-2 * np.random.randn(32, 2), iterations) for I in range(0, Nprocs)]

# output
out = pool.starmap(MCMC_sampler, input)
samples = out[0]
for I in range(1,4):
    samples = np.vstack( (samples,out[I]))
print( samples , np.shape(samples))
flat_samples = samples
print("number of samples = ", len(flat_samples))
print("median = ", np.median(flat_samples,axis=0))

# create file
file = h5py.File('RL_Samples.h5','w')

# store samples
file.create_dataset("/samples", data=flat_samples)

file.close()

#############################
# preview posterior
#############################

labels = ["a2", "a3"]

fig = corner.corner(flat_samples, labels=labels, truths=center)

plt.savefig("plots/RL_posterior")