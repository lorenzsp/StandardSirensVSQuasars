########################################################################
# histogram of few bayes factor distributions
#######################################################################
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
import h5py
import matplotlib as mpl
mpl.use('pgf')

######################################################################
# nice plot
# settings credits Nils Fisher https://github.com/nilsleiffischer/texfig

from math import sqrt
default_width = 5.78853 # in inches
default_ratio = (sqrt(5.0) - 1.0) / 2.0 # golden mean

mpl.rcParams.update({
    "text.usetex": True,
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

# import data
file = h5py.File('EvidenceData.h5','r')

distribution= file["/bayes_factor_dsitribution"][:,:]

# figure
fig = plt.figure()

# number of colors you want: 6
# this is a vector 
viridis = cm.get_cmap('viridis', 6)
i = 0

for N in [1,3, 10, 15,20, 30]:
    # Bayes factors in logarithmic scale
    O12 = np.log10(distribution[N-1,:])
    # median
    mu=np.median(O12)
    # standard deviation
    sigma=np.std(O12)
    # number of realizations
    nsamp = len(O12)
    # check the number of realizations
    print(len(O12))

    plt.vlines(mu, 0.0, 0.8, color=viridis.colors[i])
    histxs=np.arange(-3,27,step=0.5)
    # here you just plot each time on top
    counts,edges,patches=plt.hist(O12,bins=histxs,density=True, alpha=0.7, color=viridis.colors[i], label=r'$N_{SS} = $ '+ str(N) )
    i += 1

plt.ylim(0,0.5)
plt.xlim(-2,10)
plt.xlabel(r'$\log_{10} (O_{12})$')
plt.ylabel(r'Normalized counts')
plt.xticks(range(-2,11))
plt.legend()
plt.tight_layout()
plt.savefig('plots/histogram.pdf')
plt.savefig('plots/histogram.pgf')