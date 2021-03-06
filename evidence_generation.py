################################################################################
# Calculation of the evidence of the flat LCDM and ALT best-fit model
print("Calculation of the evidence of the flat LCDM and ALT best-fit model \n")
# flat LCDM -> beta prior
# ALT -> Multivariate Gaussian prior
################################################################################

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.special import ellipkinc
import multiprocessing as mp
import h5py

#######################################################
##### Cosmological Models #############################
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

def integrand(z, Omega_m):
    H = H0 * np.sqrt(Omega_m * (1+z)**3 + (1-Omega_m) )
    return 1/H

def getDL_LCDM (z, Omega_m):
    I = np.array([ quad (integrand, 0, z[j], args=(Omega_m))[0] for j in range (0, len(z)) ])
    return (1.0+z)*I

# flat LCDM luminsity distance with elliptic integral

def T(x):
    ar =  np.arccos((1 + (1-np.sqrt(3))*x )/(1 + (1+np.sqrt(3))*x ) )
    
    return ellipkinc(ar ,(2.+np.sqrt(3))/4. ) /np.power(3,1.0/4.0)

def Ell_DL_LCDM(z, Omega_m):
    s = np.power( (1.0-Omega_m)/Omega_m , 1.0/3.0 )
    
    return ((1.0+z)/(H0 * np.sqrt(s * Omega_m)) )*(T(s) - T(s/(1.0+z)) )


# check correctness of luminosity distance integration
zz = np.arange(0.1, 10, 0.1)
MM = np.max( (getDL_LCDM(zz, Omega_m_ref) - Ell_DL_LCDM(zz, Omega_m_ref))/Ell_DL_LCDM(zz, Omega_m_ref)  )
print("max relative difference = " ,MM)

############################
# Risaliti Lusso fitted model
############################

def DL_RL_article (z,a2,a3):
    """
    Gets in the a2 and a3 as they are defined in the article
    """
    
    ad = [1, a2, a3]
    F = np.log(10)*(ad[0]* np.log10(1+z) + ad[1]* np.log10(1+z)**2 + ad[2]* np.log10(1+z)**3)
    return F/H0

###############################################
# Catalogue with data in Gpc
###############################################
data_z_D_dD = pd.read_csv("SS_catalogues.dat" ,sep="\s+").values
z_SS = data_z_D_dD[:,0] # redshift
DL_SS = data_z_D_dD[:,1]*1e3 # luminosity distance in Mpc
dDL_SS = data_z_D_dD[:,2]*1e3 # uncertainty in luminosity distance
print("Imported Data \n")

############################################################
#### Evidences of the different models ####################
# Note on calculating the evidence:
# if you use the code with data which are distributed far from the 
# model prediction, the exponential of the likelihood might be
# round off to approximate zero
############################################################

################
#  Lambda CDM
################

# constants beta prior
MEDIAN = 0.31 
b = 50 # makes it narrower such that the std is 0.05
a = (1/3 +MEDIAN *(b-2/3))/(1-MEDIAN) # the median value is 0.31

print("a = ", a," and b = ", b," of LCDM beta prior")
# prior of the LCDM
def prior_Omega(Omega):
    return stats.beta.pdf(Omega,a,b)

# check normalization of the prior
#print("normalization " ,quad(prior_Omega, 0,1)[0])

# Product of the likelihood and prior
def likelihood_x_Omega(Omega_m, z, D, dD):
    # likelihood with given redshifts and distance luminosities
    # beta prior between 0 and 1
    return (np.product( np.exp(-(D - Ell_DL_LCDM(z, Omega_m))**2 /(2*(dD*dD) ) )/(np.sqrt(2* np.pi)*dD) ))*prior_Omega(Omega_m)

# evidence of the LCDM
def evidence_LCDM(z,D,dD):
    return quad(likelihood_x_Omega, 0., 1., args=(z,D,dD))[0]

Ncheck = 30
#print("check the integration ", evidence_LCDM(z_SS[:Ncheck],DL_SS[:Ncheck],dDL_SS[:Ncheck])/quad(likelihood_x_Omega, 0.0, 1.0, args=(z_SS[:Ncheck],DL_SS[:Ncheck],dDL_SS[:Ncheck]), epsabs=1e-50)[0] )
############################################################
# ALT Model 2 based on Figure 5 of Risaliti Lusso paper
############################################################

#  posterior sample mean and covariance from RLposterior
center = np.array([3.41309063, 1.40248073])
cov = np.array([[ 0.01961402, -0.06005197],
 [-0.06005197, 0.24775043]])

print("\n")
print("Multivariate Gaussian prior for ALT model")
print("mean value",center)
print("covariance matrix", cov)
print("\n")

# invers of the covariance matrix
Mfinal=np.linalg.inv(cov)

# normalization factor of the prior
Gamma = np.linalg.det(cov)
factor_prior_a2a3 = (2* np.pi*np.sqrt(Gamma))

# evidence of the Risaliti Lusso model with analytical calculations
# of the integral of the prior times the likelihood
# referring to the paper: center = mu and Mfinal = Sigma^(-1)
def evidence_a2_a3(z, D, dD):

    x = np.log10(1+z)
    csi = np.log(10)/H0
    psi = D/csi - x
    sigma2 = dD*dD
    c0 = np.sum(psi*psi/sigma2 )
    c11 = np.sum(np.power(x,4)/sigma2)
    c22 = np.sum(np.power(x,6)/sigma2)
    c12 = np.sum(np.power(x,5)/sigma2)
    c01 = -2* np.sum(psi*np.power(x,2)/sigma2)
    c02 = -2* np.sum(psi*np.power(x,3)/sigma2)
    CC = np.array([[c11, c12],[c12, c22]])
    BB = np.array([c01, c02])

    alpha = np.dot(center,np.dot(Mfinal,center.T))
    V = -0.5*csi*csi* BB + np.dot(Mfinal,center.T)
    Denomin = np.product(np.sqrt(2* np.pi)*dD) * factor_prior_a2a3 * np.sqrt(np.linalg.det(csi*csi*CC +Mfinal))
    Argum = -0.5*(csi*csi*c0 + alpha)+ np.dot( V , np.dot(np.linalg.inv(csi*csi*CC+Mfinal), V.T) )/2. 
    return 2*np.pi*np.exp(Argum)/Denomin

# Numerical integration to check the correct implementation of the function evidence
# Normal Prior
def prior_a2_a3_normal(a2,a3):
    # input a2 a3 as from the article
    
    Y = np.vstack((a2,a3)).T
    ARG = np.dot((Y-center).dot(Mfinal),(Y-center).T)
    result= np.exp(-ARG/2)/factor_prior_a2a3
    return result

print("check normalization " ,dblquad(prior_a2_a3_normal, -10.,+10., -10.,+10.)[0])

# Product of the likelihood and prior
def likelihood_x_a2_a3(a2, a3,  z, D, dD):
    
    return np.product( np.exp(-(D - DL_RL_article(z, a2, a3))**2 /(2*(dD*dD) ))/(np.sqrt(2* np.pi)*dD) )*prior_a2_a3_normal(a2,a3)

Ncheck = 10
# the numerical integration with dblquad does not always converge, especially, if the integration region is too wide or the number of Ncheck is too high
print("check the ratio of the numerical and analytical result" ,dblquad(likelihood_x_a2_a3, -5., +5., -5., +5.,  args=(z_SS[:Ncheck],DL_SS[:Ncheck],dDL_SS[:Ncheck]),epsabs=1e-50)[0]/ evidence_a2_a3(z_SS[:Ncheck],DL_SS[:Ncheck],dDL_SS[:Ncheck]) )

#############################################################
# Bayes factor          ratio of the evidences
#############################################################

def Bayes_factor_LCDM_RL(z, D, dD):
    return quad(likelihood_x_Omega, 0., 1., args=(z,D,dD))[0]/evidence_a2_a3(z,D, dD)

# Bayes factor of the fiducial 15 SS 'SSdatasets/median_dataset_15SS.dat'
z_D_dD = pd.read_csv('SSdatasets/median_dataset_15SS.dat' ,sep="\s+", header=None).values # in Gpc

print("bayes factor fiducial dataset ",Bayes_factor_LCDM_RL(z_D_dD[:,0],z_D_dD[:,1]*1e3,z_D_dD[:,2]*1e3 ))
# total number of SS
N_SS = 30

# number of realizations per each N_SS
realizations = 10000

def distribution_bayes_factor(n_z):
    """
    n_z input of number of standard sirens to consider
    output distribution of bayes factor of # realizations
    """

    R = np.arange(0,int((n_z+1)*realizations) ,n_z+1)
    #print("Number of realizations {} ".format(len(R)))
    #print("number of standard sirens per each Bayes {} ".format(n_z))
    #print("check number of standard sirens per each Bayes  ", np.mean([len(z_SS[j:j+n_z]) for j in R]))
    #print("one of them is ", Bayes_factor_LCDM_RL(z_SS[1:1+n_z], DL_SS[1:1+n_z], dDL_SS[1:1+n_z] ))
    #print("catalogue number start ", R)
    #print("\n")
    return np.array([Bayes_factor_LCDM_RL(z_SS[j:j+n_z], DL_SS[j:j+n_z], dDL_SS[j:j+n_z] ) for j in R]) # the plus needed for not re using the data

###############################
# parallel computation
##############################

# set the number of CPUs you want to use
pool = mp.Pool(mp.cpu_count())

# different number of standard sirens
input = [i for i in range(1,N_SS+1)]

# output
out = pool.map(distribution_bayes_factor, input)

BF = np.empty((N_SS,realizations))

for i in range(0,N_SS):
    BF[i,:] = out[i]

######################################
# write to data
#######################################


# figure
plt.figure()
n_ss_vec = np.array(range(0, N_SS)) +1

# median
mu = [np.median( BF[j,:]) for j in range(0, N_SS) ]

# 90 %
low = [stats.scoreatpercentile(BF[j,:],5)  for j in range(0, N_SS) ]
up = [stats.scoreatpercentile(BF[j,:],95)  for j in range(0, N_SS) ]
plt.fill_between(n_ss_vec, low, up,
                 color='green', alpha=0.2, label=r'$90\%$ evidence interval')

# 50%
low = [stats.scoreatpercentile(BF[j,:],25)  for j in range(0, N_SS) ]
up = [stats.scoreatpercentile(BF[j,:],75)  for j in range(0, N_SS) ]
plt.fill_between(n_ss_vec, low, up,
                 color='blue', alpha=0.2, label=r'$50\%$ evidence interval')

# plot
plt.plot(n_ss_vec, mu, '--or', label='Median', ms=5)
plt.yscale('log')

# horizontal lines
plt.hlines(150, -1, N_SS+2,linewidth=2 ,colors='blue',linestyles='dashdot', label='Very strong evidence for $M_1$')
plt.hlines(20, -1, N_SS+2,linewidth=2 , colors='purple' ,linestyles='dashdot' ,label='Strong evidence for $M_1$')

plt.xlabel('$N_{SS}$')
plt.ylabel('$O_{12}$')
plt.grid()
plt.xticks(np.arange(1, 32, step=2))
plt.xlim(0,30)

# ticks adjusted
ax =plt.gca()
import matplotlib as mpl

locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
ax.yaxis.set_major_locator(locmaj)

locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=90) 
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


plt.legend(loc='upper left')#,prop={'size': 6})
plt.tight_layout()
plt.show()

# uncomment the next part to write to file the data 
"""
# create file
file = h5py.File('EvidenceData_BayesPlot.h5','w')

# bayes factor distribution
file.create_dataset("/bayes_factor_dsitribution", data=BF)

# info about priors of the models
file["/bayes_factor_dsitribution"].attrs.create("beta prior (a, b)", np.array([a, b]), np.shape(np.array([a, b]))) 
file["/bayes_factor_dsitribution"].attrs.create("2d Normal prior with mu", center, np.shape(center)) 
file["/bayes_factor_dsitribution"].attrs.create("2d Normal prior with Sigma", np.linalg.inv(Mfinal), np.shape(np.linalg.inv(Mfinal))) 

file.close()
"""