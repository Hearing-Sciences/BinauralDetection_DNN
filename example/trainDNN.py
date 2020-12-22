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
import dnnbmld.ed_BMLD as edq


#%% PARAMETERS

#
latentNodes = 10
beta = 0.00001
learningRate = 5e-5
batchSize = 256
noOfLayers = 2
nodesPerLayer = 100
proportionForValidation = 5
epochs = 1000


#%% TRAIN MODEL

## Create unique ID
uniqueCode = ""
for i in range(6):
    uniqueCode += str(np.random.randint(10))
uniqueID = uniqueCode+'_'+str(latentNodes)+'_'+"{:.0e}".format(beta)+'_'+"{:.0e}".format(learningRate)+'_'+str(batchSize)+'_'+str(noOfLayers)+'_'+str(nodesPerLayer)

# Unpack data
trainingData, validationData, _, _, _ = dl.load(5, 'dataset1') 

# Create initialised network object
nnconfig = np.array( np.ones([noOfLayers],dtype=np.int) * nodesPerLayer ).tolist()
net = nn.Network(800, latentNodes, 1, 1, nnconfig, nnconfig, uniqueID)

# Train the network
betaFunction = lambda x: beta #
net.train(epochs, batchSize, learningRate, trainingData, validationData, betaFunction) # 1 write to tensorboard


#%% SAVE MODEL

# Test reconstrutcion loss
rmse = np.sqrt( net.run(validationData, net.recon_loss) )

# Save
net.save(uniqid + "{:.0e}".format(rmse))


#%% TENSORBOARD
# 1) activate anaconda environment
# 2) move to parent directory
# 3) type "tensorboard --logdir tf_log/" 
# 2) localhost:6006 in browser

