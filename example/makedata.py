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
#
# Please set working directory to "/example" folder

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
import dnnbmld.ed_BMLD as edq


#%% PARAMRETERS

stimParams = {"toneFreq": [500,500], # Hz
          "noiseAmp": [60,60], # dB SPL
          "toneAmp": [0,50], # dB SPL
          "transformType": "azimuth", 
          "toneAziPhase": [-90,90], # azimuth angle, degrees
          "noiseAziPhase": [-90,90]} # azimuth angle, degrees
N = 10000 # 1000000  # draw this many random values are drawn
sampleRate = 20000
datasetName = 'dataset1' # Name for dataset


#%% MAKE DATA

edq.BMLD_data( stimParams , drawN=N , Fs=sampleRate , fileName=datasetName )

