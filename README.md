# StandardSirensVSQuasars

The python codes contained in this folder have been used to reproduce the main the results shown in the paper **Testing the Quasar Hubble Diagram with LISA Standard Sirens** [1]. We implemented a Bayesian hypothesis test to understand how many LISA MBHBs Standard Siren observations are necessary to test the deviation from the <img src="https://render.githubusercontent.com/render/math?math=\Lambda">CDM model found by Risaliti and Lusso with quasars observations [2]. 
In addition, we also implemented an MCMC to understand the constraining power of LISA standard sirens on matter density and Hubble parameter for a flat <img src="https://render.githubusercontent.com/render/math?math=\Lambda">CDM model.

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

    ![](https://github.com/lorenzsp/StandardSirensVSQuasars/blob/master/plots/LCDMpost.pdf)

## Contact

Lorenzo Speri: lorenzo.speri@aei.mpg.de

Our codes used in **Testing the Quasar Hubble Diagram with LISA Standard Sirens** [1] to analyze quasar observations can be also provided, however their usage depends on the quasar data which are available from the corresponding author of [2] upon request.
Please contact the author of [2] for the usage permission of their data and we will provide all the datasets and the codes to reproduce our analysis.


## References

[1]: 

[2]: Risaliti, G., Lusso, E. Cosmological constraints from the Hubble diagram of quasars at high redshifts. Nat Astron 3, 272–277 (2019). https://doi.org/10.1038/s41550-018-0657-z