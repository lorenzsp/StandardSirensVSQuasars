#######################################################################
# Posterior distribution of LCDM of 15 standard sirens
#######################################################################

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib as mpl
mpl.use('pgf')
from mpl_toolkits.mplot3d import Axes3D
import h5py
import scipy
from matplotlib import colors


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

A = np.load("SSdatasets/median_dataset_15SS.dat.npy")
x = A[:,0]
y = A[:,1]

# test MCMC against chi-squared sigma
# median
print("median omega_m ", np.median(x))
print("median h ", np.median(y))
# one sigma
print("one sigma omega_m = " ,(stats.scoreatpercentile(x,100-15.865525393145703) - stats.scoreatpercentile(x,15.865525393145703))/2, "std =", np.std(x))
print("one sigma h = ", (stats.scoreatpercentile(y,100-15.865525393145703) - stats.scoreatpercentile(y,15.865525393145703))/2, "std =",np.std(y) )
# relative uncertainty
print("delta omega_m /omega_m= " ,(stats.scoreatpercentile(x,100-15.865525393145703) - stats.scoreatpercentile(x,15.865525393145703))/(2*np.median(x)))
print("delta h/h = ", (stats.scoreatpercentile(y,100-15.865525393145703) - stats.scoreatpercentile(y,15.865525393145703))/(2*np.median(y)))

# color
cl = cm.get_cmap('viridis')
newcolors = cl( [0.3, 0.5, 0.7, 1])

# histogram
H,X,Y=np.histogram2d(x,y,bins=100,normed=True)
# normalize
norm=H.sum() # Find the norm of the sum
H = H/norm
X2, Y2, H2, V = corner_contour(X,Y, H, levels)
# contour
ax_main.contour(X2, Y2, H2.T, V, colors=newcolors, linestyles='solid')

# histogram
norm = colors.Normalize(vmin=np.min(H), vmax=np.max(H) )
ax_main.hist2d(x, y, bins=100, cmap=cl,norm=colors.Normalize(), cmin=1e-3)

ax_main.set(xlabel=r"$\Omega _m$", ylabel=r"$h$")

ax_main.hlines(0.7, 0.0, 1,linewidth=1 ,colors='blue',linestyles='solid')
ax_main.vlines(0.3, 0.5, 0.9,linewidth=1 ,colors='blue',linestyles='solid')

# grid 
ax_main.grid()

############################
# secondary plots
############################
n, bins, patches = ax_xDist.hist(x ,bins=100,align='mid',density=True)


# color code by height
fracs = n 
# we need to normalize the data to 0..1 for the full range of the colormap

# color
cl = cm.get_cmap('viridis')
newcolors = cl( [0.3, 0.5, 0.7, 1])
norm = colors.Normalize(vmin=np.min(n),vmax=np.max(n))#

# loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = cl(norm(thisfrac))
    thispatch.set_facecolor(color)

ax_xDist.set(ylabel=r'$p(\Omega _m |\, \vec{y} \,)$')
ax_xDist.yaxis.set_major_locator(plt.MaxNLocator(5))

# line
ax_xDist.vlines(0.3, 0.0, 7.5,linewidth=1 ,colors='blue',linestyles='solid')

# limit
ax_xDist.set_ylim(0.0, 7)

ax_xDist.grid()

# CDF
#ax_xCumDist = ax_xDist.twinx()
#ax_xCumDist.hist(x,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid')
#ax_xCumDist.tick_params('y', colors='r')
#ax_xCumDist.set_ylabel('cumulative',color='r')

n, bins, patches = ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid',density=True)

# color code by height
fracs = n
# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(vmin=np.min(n),vmax=np.max(n))#

# loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = cl(norm(thisfrac))
    thispatch.set_facecolor(color)


ax_yDist.set(xlabel=r'$p( h |\, \vec{y} \, )$')
ax_yDist.xaxis.set_major_locator(plt.MaxNLocator(4))

#ymin, ymax = ax_yDist.get_ylim()
ax_yDist.hlines(0.7, 0.0, 19,linewidth=1 ,colors='blue',linestyles='solid')
ax_yDist.set_xlim(0.0, 19)

ax_yDist.grid()

# CDF
#ax_yCumDist = ax_yDist.twiny()
#ax_yCumDist.hist(y,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid',orientation='horizontal')
#ax_yCumDist.tick_params('x', colors='r')
#ax_yCumDist.set_xlabel('cumulative',color='r')

plt.tight_layout()
plt.savefig('plots/LCDMpost.pdf')
plt.savefig('plots/LCDMpost.pgf')

