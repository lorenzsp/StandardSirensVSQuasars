# StandardSirensVSQuasars

The python codes contained in this folder have been used to reproduce the main the results shown in the paper <em>Testing the Quasar Hubble Diagram with LISA Standard Sirens</em> arXiv ---. We implemented a Bayesian hypothesis test to understand how many LISA MBHBs Standard Siren observations are necessary to test the deviation from the <img src="https://render.githubusercontent.com/render/math?math=\Lambda">CDM model found by Risaliti and Lusso with quasars observations [Nat. Astron.3, 272 (2019)]. In addition, we also implemented an MCMC to understand the constraining power of LISA standard sirens on matter density and Hubble parameter for the flat <img src="https://render.githubusercontent.com/render/math?math=\Lambda">CDM model.

## Files description

This folder contains:

- mcmc_flatLCDM.py 
    produces the posterior samples matter density  <img src="https://render.githubusercontent.com/render/math?math=\Omega _m"> and Hubble parameter h of a flat <img src="https://render.githubusercontent.com/render/math?math=\Lambda">CDM model given redshift, luminosity distance and luminosity distance uncertainty observations (for instance the dataset SSdatasets/median_dataset_15SS.dat)
    
    execute with: python mcmc_flatLCDM.py SSdatasets/median_dataset_15SS.dat
    the posterior samples are stored in the folder SSdatasets with name median_dataset_15SS.dat.h5
    
- evidence_generation.py
    produces the samples of the Bayes factor realizations for each fixed number of standard sirens
    
    execute: python evidence_generation.py
    output: EvidenceData_BayesPlot.h5
    where each column contains the samples of the Bayes factor realizations, and the column number corresponds to the number of standard sirens

## Contact

Lorenzo Speri: lorenzo.speri@aei.mpg.de

The data of quasar observations are available from the corresponding author Risaliti and Lusso [Nat. Astron.3, 272 (2019)] upon reasonable request.
The codes used to analyze quasar observations can be provided 

The datasets and the codes used to perform under it is necessary you can contact me and we can to get in contact with the author  for 

