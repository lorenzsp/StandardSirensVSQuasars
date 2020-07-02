#######################################################################
# Posterior distribution of RL model with quasars and SNe observations
#######################################################################


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib as mpl
mpl.use('pgf')
#from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import h5py
import scipy


def corner_contour(X, Y, H, levels):
    ####################################################################
    # Compute the density levels
    # taken from corner plot https://github.com/dfm/corner.py
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0

    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])
    return X2, Y2, H2, V

######################################################################
# nice plot
# settings credits Nils Fisher https://github.com/nilsleiffischer/texfig

from math import sqrt
default_width = 5.78853 # in inches
default_ratio = (sqrt(5.0) - 1.0) / 2.0 # golden mean

mpl.rcParams.update({
    "text.usetex": True,
    'font.size': 14,
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "figure.figsize": [default_width, default_width * default_ratio],
    "pgf.preamble": [
        # put LaTeX preamble declarations here
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        # macros defined here will be available in plots, e.g.:
        r"\newcommand{\vect}[1]{#1}",
        # You can use dummy implementations, since you LaTeX document
        # will render these properly, anyway.
    ],
})
#######################################################################



# https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

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

# factor 4 in front to enlarge the prior
sigma_1 = 4*np.sqrt( (a2_image[0]- a2_image[1])**2 +(a3_image[0]- a3_image[1])**2 ) /(2*np.sqrt(s))
sigma_2 = 4*np.sqrt((a2_image[2]- a2_image[3])**2 +(a3_image[2]- a3_image[3])**2) /(2*np.sqrt(s))

ang = np.arctan((a3_image[0]- a3_image[1])/(a2_image[0]- a2_image[1]) )

M12 = np.array([[1/(sigma_1**2), 0], [0, 1/(sigma_2**2)]])
Rot = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])

# invers of the covariance matrix
Mfinal=np.dot(Rot.T, np.dot(M12,Rot) )

Cov = np.array(np.linalg.inv(Mfinal))

###############################################
# Make a 2d normed histogram

###############################
# main
###############################
# Set contour levels
contour0=0.999936657516334
contour1=0.997300203936740
contour2=0.954499736103642
contour3=0.682689492137086

levels = [contour3, contour2, contour1,contour0]

# figure
fig = plt.figure(figsize=(5,5))

gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)


########################################
# main plot
########################################

# color
cl = cm.get_cmap('copper_r')
newcolors = cl( [0.3, 0.5, 0.7, 1])

# we generated a samples of the prior defined above
# and we import directly the contours
X2, Y2, H2, V = np.load("contour_priorRL.npy",allow_pickle=True)
# contour
ax_main.contour(X2, Y2, H2.T, V, colors=newcolors, linestyles='dotted')

#----------------------
#----------------------
#  file of the posterior sample
file = h5py.File('RL_Samples_a2_a3_withJLA_withCorr_woDelta_wideprior.h5','r')
# bayes factor distribution
x = file["samples"][:,0]
y = file["samples"][:,1]
# info
MEAN = np.mean(file["samples"][:,:], axis=0)
COV = np.cov(file["samples"][:,:], rowvar=0)
print(MEAN)
print(COV)
file.close()

# color
cl = cm.get_cmap('copper_r')
newcolors = cl( [0.3, 0.5, 0.7, 1])
# histogram
H,X,Y=np.histogram2d(x,y,bins=100,normed=True)
from matplotlib import colors
norm = colors.Normalize(vmin=np.min(H), vmax=np.max(H) )
ax_main.hist2d(x, y, bins=100, cmap=cl,norm=colors.Normalize(), cmin=1e-3)
# normalize
norm=H.sum() # Find the norm of the sum
H = H/norm
X2, Y2, H2, V = corner_contour(X,Y, H, levels)
# contour
CS = ax_main.contour(X2, Y2, H2.T, V, colors=newcolors, linestyles='solid')#, colors=newcolors)

# histogram
ax_main.set(xlabel=r"$a_2$", ylabel=r"$a_3$")

# number of ticks
ax_main.yaxis.set_major_locator(plt.MaxNLocator(5))
ax_main.xaxis.set_major_locator(plt.MaxNLocator(5))

# for the a2 a3 posterior
Omm = np.arange(-1, 2, 0.01)
ax_main.plot(np.log(10)*(3/2 - Omm*3/4),np.log(10)*np.log(10)* (7/6 -2* Omm + Omm*Omm*9/8),'-k' )
ax_main.plot(2.9358, 3.54123,'ok', ms=7)

# axes limit
ax_main.set_xlim([1.5, 4.5])
ax_main.set_ylim([-0.5, 7])

# grid 
ax_main.grid()

############################
# secondary plots
############################
n, bins, patches = ax_xDist.hist(x,bins=100,align='mid',density=True)

# color code by height, but you could use any scalar
fracs = n 
# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(vmin=np.min(n),vmax=np.max(n))#

# loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = cl(norm(thisfrac))
    thispatch.set_facecolor(color)

ax_xDist.set(ylabel=r'$p(a_2 |\, \vec{y} \,)$')
ax_xDist.yaxis.set_major_locator(plt.MaxNLocator(5))

ax_xDist.grid()

# CDF
#ax_xCumDist = ax_xDist.twinx()
#ax_xCumDist.hist(x,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid')
#ax_xCumDist.tick_params('y', colors='r')
#ax_xCumDist.set_ylabel('cumulative',color='r')

n, bins, patches = ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid',density=True)


# color code by height, but you could use any scalar
fracs = n
# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(vmin=np.min(n),vmax=np.max(n))#

# loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = cl(norm(thisfrac))
    thispatch.set_facecolor(color)



ax_yDist.set(xlabel=r'$p( a_3 |\, \vec{y} \, )$')
ax_yDist.xaxis.set_major_locator(plt.MaxNLocator(3))
ax_yDist.grid()

# CDF
#ax_yCumDist = ax_yDist.twiny()
#ax_yCumDist.hist(y,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid',orientation='horizontal')
#ax_yCumDist.tick_params('x', colors='r')
#ax_yCumDist.set_xlabel('cumulative',color='r')

plt.tight_layout()
#plt.show()
plt.savefig('plots/RLposterior.pdf')
plt.savefig('plots/RLposterior.pgf')

