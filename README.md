# StandardSirensVSQuasars

These python codes allow to reproduce the results shown in the paper ``Testing the Quasar Hubble Diagram with LISA Standard Sirens'' arXiv ---. The goal of this paper is to use Bayesian hypothesis testing to test with LISA standard sirens the deviation from the LCDM found by Risaliti and Lusso with quasars observations.

## Files description

This folder contains:

- mcmc_flatLCDM.py 
    produces the posterior samples matter density  <img src="https://render.githubusercontent.com/render/math?math=\Omega _m"> and Hubble parameter h of a flat <img src="https://render.githubusercontent.com/render/math?math=\Lambda">CDM model given redshift, luminosity distance and luminosity distance uncertainty observations (forour case stored in SSdatasets/median_dataset_15SS.dat)
    
    execute with: python mcmc_flatLCDM.py SSdatasets/median_dataset_15SS.dat
    the posterior samples are stored in the folder SSdatasets with name median_dataset_15SS.dat.h5
    
- evidence_generation.py
    produces the samples of the Bayes factor realizations for each fixed number of standard sirens
    execute: python evidence_generation.py
    output: EvidenceData_BayesPlot.h5
    where each column contains the samples of the Bayes factor realizations, and the column number corresponds to the number of standard sirens

