# INFERRING THE NEURAL BASIS OF BINAURAL DETECTION USING DEEP LEARNING


The article for which this code was developed can be found [here](https://www.biorxiv.org/content/10.1101/2021.01.05.425246v3). The core modules in this project were derived/modified from: [Iten, T. Metger, H.Wilming, L. del Rio, and R. Renner. "Discovering physical concepts with neural networks" (2018)](https://github.com/eth-nn-physics/nn_physical_concepts/). Required software is supplied in an Anaconda environment file ('setupEnvironment.yml'). 


The folders in this repository are:

+ **'example/'** An example project folder. Binaural detection data can be simulated ('makedata.py'). Following this, a deep neural network can be trained ('trainDNN.py'). Results, including statistics and figures, can be produced ('results.py'). An example of this is given for the DNN presented in our article (referred to here as 'demoDNN'). 

+ **'dnnbmld/'** Contents inclue the binaural detection simulation ('ed_BMLD.py') and deep neural network framework ('model.py').

+ **'tf_save/'** The saved location of trained deep neural networks. The 'demoDNN' network can be found here.

+ **'analysis/'** Scripts and files required for quantitative analyses ('analyses.py) and figures ('plots.py').

+ **'data'** and **'tf_log'** These folders are not in the repository. However, they will be created if users opt to store simulation data or record tensorboard logs when training a deep neural network.


*For any queries, please email (samuel.smith@nottingham.ac.uk).*
