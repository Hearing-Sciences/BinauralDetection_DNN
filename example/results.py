#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Modifications copyright (C) 2020 (Samuel Smith [samuel.smith@nottingham.ac.uk] at the University of Nottingham, UK).

#%% SETUP

# Global imports
import sys
from pathlib import Path
import numpy as np
import datetime

# Add parent directory
parentPath = Path().resolve().parent
sys.path.insert(0,str(parentPath))

# Scinet imports
from dnnbmld import *
import dnnbmld.io as io
import dnnbmld.ed_BMLD as edq # ed_BMLD, ed_BMLD_reformattedOriginal , ed_BMLD_original      # # #  need to update in analyses.py too

# Analysis imports
from analysis import *
import analysis.analyses as af
import analysis.plots as pf

# PYTHON imports
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy.stats import norm
from datetime import datetime
import scipy.stats
import PIL # not necessary but mustn't fail


#%% SPECIFIED PARAMETERS

# Network
DNN_Name = "demoDNN"
DNN_KLDiv = [1.32632256e-03, 1.17218494e-03, 6.39855862e-04, 3.44449496e+00,
        3.07843900e+00, 3.90725040e+00, 2.67303491e+00, 3.38256383e+00,
        3.84260225e+00, 1.07347965e-03]
# Default sound parameters
toneAmpPsychometric = np.arange(0,50+1e-10,2.5) # range between 0, 5o, in steps of 2.5
toneFreq_preset = 500
noiseAmp_preset=60
Fs = 20000
# Psychometric analyses
repeatsBeforeRegression = 10
repeatsAfterRegression = 10
# Latent node analyses
repeatsForActivations = 5000
IPDsForCurve = np.linspace(-2*np.pi,2*np.pi+1e-10,41)
noiseAmpForCurve = 60
# Network manipulation
dropoutValues = [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]


#%% SETUP
# Load network
net = nn.Network.from_saved(DNN_Name)       

  
#%% DNN AZIMUTHAL PREDICTIONS
# Setup
stimParams = {"toneFreq": toneFreq_preset,
       "noiseAmp": noiseAmp_preset,
       "transformType": "azimuth"} 
# Analyse
allDTs_DNN,allDTs_ECTheory,allAzis,DTrmse = af.DT_contour(net,stimParams,toneAmpPsychometric,[repeatsBeforeRegression,repeatsAfterRegression],Fs)
# Plot
pf.DT3D(allDTs_DNN,allDTs_ECTheory,allAzis)  


#%% DNN PHASIC PREDICTIONS
# Setup
stimParams = {"toneFreq": toneFreq_preset,
       "noiseAmp": noiseAmp_preset,
       "transformType": "phasic"} 
# Analyse
allDTs_DNN_phasic,allDTs_ECTheory_phasic,allIPDs = af.DT_phasic(net,stimParams,toneAmpPsychometric,[repeatsBeforeRegression,repeatsAfterRegression],Fs)  
# Plot
pf.BMLDPhasic(allDTs_DNN_phasic,allDTs_ECTheory_phasic)


#%% ITD TUNING
# Setup
stimParams = {"toneFreq": toneFreq_preset,
              "toneAmp": -100,
              "noiseAmp": noiseAmpForCurve,
              "transformType": "phasic"}
# Analyse
allActivations_noiseITD, allGaborFitParams_noiseITD, polarityCorrect, allVS_noise, layerActivations_noise = af.IPDTuning(net,stimParams,IPDsForCurve,DNN_KLDiv,repeatsForActivations,Fs,toneFreq_preset)
# Plot
    # Layer 1
allGaborFit_Single1 =pf.plot_earlyITDtuning2(IPDsForCurve,toneFreq_preset,layerActivations_noise[0,:,:],0,[15, 89])
layer1withITDTuning = np.sum(allGaborFit_Single1[2]<0.05)
    # Layer 2 
allGaborFit_Single2 = pf.plot_earlyITDtuning2(IPDsForCurve,toneFreq_preset,layerActivations_noise[1,:,:],1,[45,83],moreSubPlots = 1)
layer2withITDTuning = np.sum(allGaborFit_Single2[2]<0.05)


#%% X-CORR MULTIPANEL PLOTS
# Setup
stimParams = {"toneFreq": toneFreq_preset,
   "toneAmp": 35, 
   "noiseAmp": 60,
   "transformType": "phasic"} 
# Analyse
latentActivations, layerActivations = af.xcorrActivations(net,stimParams,toneAmpPsychometric,repeatsForActivations,DNN_KLDiv,Fs)
[PCC,p] = af.correlatewithmodel(layerActivations,allGaborFit_Single2[0][:,1]) # correlate with model

# Plots
    # layer 2 responses (homophasic/antiphasic stimuli)
pf.xcorr_activtations(layerActivations,allGaborFit_Single2[0]) 
    # x-corr algorithm for comparison
pf.xcorr_model()
pf.xcorr_gilbert()


#%% RATE LEVEL FUNCTIONS
# Setup
stimParams = {"toneFreq": toneFreq_preset,
   "noiseAmp": noiseAmp_preset,
   "transformType": "phasic"} 
# Analyse
allActivations, allRegressionData, bestScale = af.relativeRLFs(net,stimParams,toneAmpPsychometric,repeatsForActivations,DNN_KLDiv,Fs,polarityCorrect)
# Plots
    # comparative responses from Gilbert et al. (2015)
pf.gilbert_RLFs()
    # (central) latent node ITDs and masked RLFs
pf.ITDandRLFs(IPDsForCurve,allActivations_noiseITD,allGaborFitParams_noiseITD, # itd parameters
               toneAmpPsychometric,allActivations, allRegressionData, bestScale, DNN_KLDiv, polarityCorrect) # rlf parameters

# Baseline test
t_node6_NoNpi,p_node6_NoNpi=scipy.stats.ttest_ind(allActivations[0,0,:,5],allActivations[2,0,:,5], equal_var=False)   
    

#%% CENTRAL LATENT NODES - T-Tests
# Setup
stimParams = {"toneFreq": toneFreq_preset,
   "noiseAmp": noiseAmp_preset,
   "transformType": "phasic"}
phasicThresholds = np.mean(allDTs_DNN_phasic,1) # from function: af.DTPhasic
# Analyse
stats_No, stats_Npi, activations =af.PNTypeAnalysis(net,stimParams,phasicThresholds,repeatsForActivations,DNN_KLDiv,Fs,polarityCorrect)        

