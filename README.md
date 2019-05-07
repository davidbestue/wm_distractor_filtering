# Working memory & Distractor filtering  

### Psychophysics.  
In this folder I analyze the data coming from psychophysics behaviour.  
Raw data stored in local and Dropbox (paths given).  
The files inside this folder are:
+ ***Create_data_psychophysics.ipynb***    
Creates the DataFrame from raw data  
+ ***Correction_data_psychophysics.ipynb***  
Remove wrong trials and outliers.  
Adds columns of interest.  
Adds the correction from the perceptual biases (not used later on in the analysis).  
+ ***Subject_summary.ipynb***  
Analysis of the interference effects created by the distractors.  
+ ***Serial_effects.ipynb***  
Analysis of the serial effects in the order 1 condition.
+ ***Funciones.ipynb***  
Functions used generally (psychophyis, fMRI and combined).  


### fMRI_beh.  
In this folder I analyze the data coming from fMRI behaviour.  
Raw data stored in local and Dropbox (paths given).  
The files inside this folder are:
+ ***Create_data_fMRI.ipynb***    
Creates the DataFrame from raw data  
+ ***Subject_summary_fMRI.ipynb***  
Analysis of the interference effects created by the distractors.  

As I was not using the corrections in the psychophysics, for the fMRI, the Create_data.ipynb 
also containes the steps to remove wrong trails and outliers.


### Subject_summary_unify.ipynb
Analysis of the combination of psychophysics and fMRI data.
