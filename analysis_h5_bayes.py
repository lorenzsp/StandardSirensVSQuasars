#################################################################
# Plot of Bayes factor as a function of number of standard sirens
##################################################################
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
mpl.use('pgf')
from math import sqrt
import h5py
#####################################################################
# nice plot
# settings credits Nils Fisher https://github.com/nilsleiffischer/texfig
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
#####################################################################

# import results
file = h5py.File('EvidenceData.h5','r')

# import samples of bayes factors
distribution= file["/bayes_factor_dsitribution"][:,:]

realizations = np.shape(distribution)[1]
N_SS = np.shape(distribution)[0]
print("N_SS = ", N_SS, " realizations =",realizations)

# median
mu = [np.median( distribution[j,:]) for j in range(0, N_SS) ]

# figure
plt.figure()
n_ss_vec = np.array(range(0, N_SS)) +1

# plot
plt.plot(n_ss_vec, mu, '--or', label='Median', ms=5)

# 90 %
low = [stats.scoreatpercentile(distribution[j,:],5)  for j in range(0, N_SS) ]
up = [stats.scoreatpercentile(distribution[j,:],95)  for j in range(0, N_SS) ]
plt.fill_between(n_ss_vec, low, up,
                 color='green', alpha=0.2, label=r'$90\%$ evidence interval')

# 50%
low = [stats.scoreatpercentile(distribution[j,:],25)  for j in range(0, N_SS) ]
up = [stats.scoreatpercentile(distribution[j,:],75)  for j in range(0, N_SS) ]
plt.fill_between(n_ss_vec, low, up,
                 color='blue', alpha=0.2, label=r'$50\%$ evidence interval')

plt.yscale('log')

# horizontal lines
plt.hlines(150, -1, N_SS+2,linewidth=2 ,colors='blue',linestyles='dashdot', label='Very strong evidence for $M_1$')
plt.hlines(20, -1, N_SS+2,linewidth=2 , colors='purple' ,linestyles='dashdot' ,label='Strong evidence for $M_1$')

plt.xlabel('$N_{SS}$')
plt.ylabel('$O_{12}$')
plt.grid()
plt.xticks(np.arange(1, 32, step=2))
plt.xlim(0,30)

plt.legend(loc='upper left')#,prop={'size': 6})
plt.tight_layout()
plt.savefig("plots/bayes_factor_O12_nSS.pgf")
plt.savefig("plots/bayes_factor_O12_nSS.pdf")
