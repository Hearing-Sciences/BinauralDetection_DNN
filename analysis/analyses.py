# IMPORTS
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import fsolve
import scipy.stats
import scipy.io as spio
import dnnbmld.ed_BMLD as edq
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
    

#%% PREDICTIVE BEHAVIOUR OF DNNs

###############  MAIN FUNCTIONS ############### 

def DT_contour(DNNnet,stimParams,toneAmpPsychometric,repeats,Fs):  

    # Calculate detection thresholds (3D). 
    # (repeats is a 2 entry list [pre regression repeats, post regression repeats])

    #  Get azimuths to evaluate
    allAzis = sample_based_azimuths(Fs)
    lenAzis = len(allAzis)
    # Initialise arrays
    allDTs_DNN = np.empty((lenAzis,lenAzis,repeats[1]))
    allDTs_ECTheory = np.empty((lenAzis,lenAzis))
    # Set azimuths in dictionary
    for ptAziIndex in range(lenAzis):
        for noAziIndex in range(lenAzis):
            stimParams['toneAziPhase'] = allAzis[ptAziIndex]
            stimParams['noiseAziPhase'] = allAzis[noAziIndex]
            # Repeat calculations of detection threshold
            for repIndex in range(repeats[1]):
                DT_DNN, DT_ECTheory = detection_thresholds(DNNnet,stimParams,toneAmpPsychometric,repeats[0],Fs)
                # Update DNN
                allDTs_DNN[ptAziIndex,noAziIndex,repIndex] = DT_DNN
            # Update EC theory (only need to do once)
            allDTs_ECTheory[ptAziIndex,noAziIndex] = DT_ECTheory   
            
    # RMSE
    meanDTs_DNN = np.mean(allDTs_DNN,2); meanDTs_DNN = np.ravel(meanDTs_DNN)        
    meanDTs_ECTheory = np.ravel(allDTs_ECTheory) 
    DTrmse = np.sqrt(np.mean((meanDTs_DNN-meanDTs_ECTheory)**2))
    
    return allDTs_DNN,allDTs_ECTheory,allAzis, DTrmse

def DT_phasic(DNNnet,stimParams,toneAmpPsychometric,repeats,Fs):     
    
    # Phasic BMLDS (standard NxSy type stimuli)
    
    # IPDs to run
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    # Initialise arrays
    allDTs_DNN = np.empty((lenIPDs,repeats[1]))
    allDTs_ECTheory = np.empty((lenIPDs))
    # Set IPD in dictionary
    for IPDIndex in range(lenIPDs):
        stimParams['toneAziPhase'] = allIPDs[IPDIndex][0]
        stimParams['noiseAziPhase'] = allIPDs[IPDIndex][1]
        # Repeat calculations of detection threshold
        for repIndex in range(repeats[1]):
            DT_DNN, DT_ECTheory = detection_thresholds(DNNnet,stimParams,toneAmpPsychometric,repeats[0],Fs)
            # Update DNN
            allDTs_DNN[IPDIndex,repIndex] = DT_DNN
        # Update EC theory (only need to do once)
        allDTs_ECTheory[IPDIndex] = DT_ECTheory
    return allDTs_DNN,allDTs_ECTheory,allIPDs    
 

############### SUPPLEMENTARY FUNCTIONS ############### 
    
def sample_based_azimuths(Fs):
    # Determines azimuths aligned with the times of samples for a given sample rate (Fs)
    k = 0.0875/343
    bound = ( np.floor(0.00066 * Fs) + 1 ) / Fs
    sampleAlignedAzimuths = []
    for tSampleTime in np.arange(0,bound,1/Fs):
        tFunc = lambda pt_azi : tSampleTime- abs(k*np.sin((np.pi/180)*np.abs(pt_azi))+k*((np.pi/180)*np.abs(pt_azi))) 
        tAzimuth = fsolve(tFunc, 45)
        sampleAlignedAzimuths.append(tAzimuth[0])
    allAzis = [-sampleAlignedAzimuths[i] for i in range(len(sampleAlignedAzimuths)-1,0,-1)]+sampleAlignedAzimuths
    return allAzis

def detection_thresholds(DNNnet,stimParams,toneAmpPsychometric,repeats,sampleRate):                
    # Initialise
    detectionRates_DNN = [];
    for toneAmpIndex in toneAmpPsychometric:
        # Add toneAmp to dictionary
        stimParams['toneAmp'] = toneAmpIndex
        # Generate data
        geninp = edq.BMLD_data(stimParams,drawN=repeats,Fs=sampleRate)
        # Run DNN
        DNNOutput = DNNnet.run(geninp[0],DNNnet.output).ravel() 
        detectionRates_DNN.append(np.mean(DNNOutput))
    # Detection thresholds 
    ppred, pcov = curve_fit(psychometric_curve,toneAmpPsychometric,detectionRates_DNN,p0=0.1)
    DT_DNN =ppred[0]
    DT_ECtheory = 31-geninp[1][0][0];              
    return DT_DNN, DT_ECtheory

def psychometric_curve(pt, DT):
    return norm.cdf(0.5*((0.159*(10**((31-DT)*0.1)))*(10**((pt-23)*0.1))))

def pc_simple(pt, DT):
    return norm.cdf(0.501611*10**(0.1*(pt-DT)))
    
 
#%% ITD
    
def IPDTuning(DNNnet,stimParams,allIPDs,KLDivs,repeats,sampleRate,toneFreq):

    # # LATENT NODE IPDs
    # DNNnet = net
    # allIPDs=IPDsForCurve
    # KLDivs=DNN_KLDiv
    # repeats=repeatsForActivations
    # sampleRate=Fs
    # toneFreq=toneFreq_preset
    
    # SORT KLDivs
    latNodeIndex, latNodeKL = KLDiv_sort(KLDivs)
    
    # LATENT ACTIVATIONS
    allActivations = np.empty((len(allIPDs),repeats,len(latNodeIndex)))
    layerActivations = np.empty((2,len(allIPDs),100))
    for IPD in enumerate(allIPDs):
        if stimParams['toneAmp']<=0:
            stimParams['toneAziPhase'] = 0
            stimParams['noiseAziPhase'] = IPD[1]
        elif stimParams['noiseAmp']<=0:
            stimParams['toneAziPhase'] = IPD[1]
            stimParams['noiseAziPhase'] = 0
        else:
            stimParams['toneAziPhase'] = IPD[1]
            stimParams['noiseAziPhase'] = 0
        geninp = edq.BMLD_data(stimParams,drawN=repeats,Fs=sampleRate)
        activations = DNNnet.run(geninp[0],DNNnet.mu)[:,latNodeIndex]
        allActivations[IPD[0]] = activations     
        # for node in first two layers 
        for layerIndx in range(2):
            layer_activations = DNNnet.run(geninp[0],"encoder/"+str(layerIndx)+"th_enc_layer/act_fun_enc_layer_"+str(layerIndx)+":0")
            layer_activations = np.mean(layer_activations,0)
            layerActivations[layerIndx,IPD[0],:] = layer_activations  
        
    # STANDARDISED VECTOR STRENGTH        
    allVS = np.empty((len(latNodeIndex),3))
    for latNode in range(len(latNodeIndex)):
        # Gen ITD values
        tActivations = np.mean(allActivations[:,:,latNode],1)
        # Random distribution
        permVS = [stanVectorStrength(allIPDs,np.random.permutation(tActivations)) for i in range(10000)]
        sortPermVS = np.sort(permVS)
        sigVSThresh = sortPermVS[int(10000-np.round(500/len(latNodeIndex)))]
        # Calc actual
        tVS = stanVectorStrength(allIPDs,tActivations)
        allVS[latNode,0] = tVS
        allVS[latNode,1] = sigVSThresh
        if tVS > sigVSThresh: # Significant?
            allVS[latNode,2] = 1
        else:
            allVS[latNode,2] = 0
        
    # FIT GABOR
    allITDs = (allIPDs/(2*np.pi)) * (1e6/toneFreq)
    allGaborFit_Double = np.empty((len(latNodeIndex),6))
    allGaborFit_Single = np.empty((len(latNodeIndex),5))
    allGaborFit_p = np.empty((len(latNodeIndex),1))
    for latNode in range(len(latNodeIndex)):
        # Prepare data
        tempXAxis = np.tile(allITDs,(repeats,1)).T
        tempXAxis = tempXAxis.flatten()
        tempYAxis = allActivations[:,:,latNode]
        tempYAxis = tempYAxis.flatten()
        # Fit curve (double peaks)
        ppredDouble, _ = curve_fit(gabor_function_double, tempXAxis.T, tempYAxis.T, p0 = [0.0005,-200,200,1,0,1000], 
                                bounds=((0.0002,-1000,-1000,-np.inf,-np.inf,500), (0.0008,1000,1000,np.inf,np.inf,np.inf))) 
        pYDouble = gabor_function_double(allITDs, *ppredDouble)
        # (single peaks)
        ppredSingle, _ = curve_fit(gabor_function_single, tempXAxis.T, tempYAxis.T, p0 = [0.0005,0,1,0,1000], 
                                bounds=((0.0002,-1000,-np.inf,-np.inf,500), (0.0008,1000,np.inf,np.inf,np.inf)))
        pYSingle = gabor_function_single(allITDs, *ppredSingle)
        # KS Test on L1 norm errors: Hassani, H., & Silva, E. S. (2015). A Kolmogorov-Smirnov based test for comparing the predictive accuracy of two sets of forecasts. Econometrics, 3(3), 590-609.
        # Others: adjusted r squared, F-test?        
        pYDouble = gabor_function_double(allITDs, *ppredDouble)
        pYSingle = gabor_function_single(allITDs, *ppredSingle)
        L1Double = np.abs(pYDouble-np.mean(allActivations[:,:,latNode],1))
        L1Single = np.abs(pYSingle-np.mean(allActivations[:,:,latNode],1))
        if np.mean(L1Double)<np.mean(L1Single):
            KSstat, pVal = scipy.stats.ks_2samp(L1Single,L1Double)
        else:
            pVal = 1
        #
        allGaborFit_Double[latNode,:]=ppredDouble
        allGaborFit_Single[latNode,:]=ppredSingle
        allGaborFit_p[latNode]=pVal
    # Polarity correction
    polarityCorrect = [] 
    for i in range(len(latNodeIndex)):
        if allGaborFit_Single[i,2]>0:
            polarityCorrect.append(1)
        else:
            polarityCorrect.append(-1)
    #
    allGaborFitParams = [allGaborFit_Double,allGaborFit_Single,allGaborFit_p]
    return allActivations, allGaborFitParams, polarityCorrect, allVS , layerActivations

def gabor_function_double(ITD,BF,BITD_left,BITD_right,A,B,s):
    # gabor function
    P=[]
    for itd in ITD:
        leftGabor = A*np.exp((-((itd-BITD_left)**2)/(2*(s**2))))*np.cos(2*np.pi*BF*(itd-BITD_left)) + B
        rightGabor = A*np.exp((-((itd-BITD_right)**2)/(2*(s**2))))*np.cos(2*np.pi*BF*(itd-BITD_right)) + B
        P.append((leftGabor+rightGabor)/2)
    return P

def gabor_function_single(ITD,BF,BITD,A,B,s):
    # gabor function
    P=[]
    for itd in ITD:
        singleGabor = A*np.exp((-((itd-BITD)**2)/(2*(s**2))))*np.cos(2*np.pi*BF*(itd-BITD)) + B
        P.append(singleGabor)
    return P

def stanVectorStrength(allIPDs,tActivations):
    # standardise
    tActivations = tActivations - np.min(tActivations)
    tActivations = tActivations / np.max(tActivations)
    # calc vector strength
    sumSin = 0; sumCos = 0
    for i in range(len(allIPDs)):
        sumSin += (tActivations[i]*np.sin(allIPDs[i]))
        sumCos += (tActivations[i]*np.cos(allIPDs[i]))
    tVS = np.sqrt((sumSin**2)+(sumCos**2))/np.sum(tActivations)
    return tVS

def KLDiv_sort(KLDivs):
    # Indexes
    KLind = np.argsort(KLDivs)
    KLind = np.flip(KLind)
    # Order indexes and values
    orderedKLind = [i for i in KLind if KLDivs[i]>0.1]
    orderedKLDivs = [KLDivs[i] for i in orderedKLind ]
    return orderedKLind, orderedKLDivs


#%% XCORR
def xcorrActivations(DNNnet,stimParams,toneAmpPsychometric,repeats,KLDivs,sampleRate):
   
     # SORT KLDivs
    latNodeIndex, _ = KLDiv_sort(KLDivs)
    
    # LATENT ACTIVATIONS AT DETECTION THRESHOLDS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allActivations = np.empty((len(allIPDs),repeats,len(latNodeIndex)))
    layerActivations = np.empty((len(allIPDs),repeats,100))
    #
    for IPDIndex in range(lenIPDs):
        stimParams['toneAziPhase'] = allIPDs[IPDIndex][0]
        stimParams['noiseAziPhase'] = allIPDs[IPDIndex][1]
        # Tone at threshold
        geninp = edq.BMLD_data(stimParams,drawN=repeats,Fs=sampleRate)
        activations = DNNnet.run(geninp[0],DNNnet.mu)[:,latNodeIndex]
        allActivations[IPDIndex,:,:] = activations 
        # layer 2 vals
        layer_activations = DNNnet.run(geninp[0],"encoder/"+str(1)+"th_enc_layer/act_fun_enc_layer_"+str(1)+":0")
        layerActivations[IPDIndex,:,:] = layer_activations  
    
    return allActivations, layerActivations


def correlatewithmodel(layerActivations,bITDs):
    
    # LOAD .MAT DATA
    mat = spio.loadmat('supplementary/xcorrmodel.mat', squeeze_me=True)
    D = mat['delays'] # array
    A = mat['allSum'] # structure containing an array
    
    # Initialize
    allDNN = []
    allXCORR = []
    
    # Collate and interp
    for i in range(4):
        #
        tDNN = np.mean(layerActivations[i,:,:],0)
        allDNN.extend(tDNN)
        #
        tXCORR = A[i]
        tXCORRinterp = np.interp(bITDs,D,tXCORR)
        allXCORR.extend(tXCORRinterp)
    
    # SET UP FIGURE
    pearson_coef, p_value = scipy.stats.pearsonr(allDNN,allXCORR)

    return pearson_coef, p_value


#%% RLFs
    
def relativeRLFs(DNNnet,stimParams,toneAmpPsychometric,repeats,KLDivs,sampleRate,polarityCorrect):
    
    # Relative RLFs, scale, correlations
    
    # SORT KLDivs
    latNodeIndex, latNodeKL = KLDiv_sort(KLDivs)
    
    # LATENT ACTIVATIONS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allActivations = np.empty((len(allIPDs),len(toneAmpPsychometric),repeats,len(latNodeIndex)))
    for IPDIndex in range(lenIPDs):
        stimParams['toneAziPhase'] = allIPDs[IPDIndex][0]
        stimParams['noiseAziPhase'] = allIPDs[IPDIndex][1]
        for toneAmpIndex in range(len(toneAmpPsychometric)):
            # Add toneAmp to dictionary
            stimParams['toneAmp'] = toneAmpPsychometric[toneAmpIndex]
            # Generate data
            geninp = edq.BMLD_data(stimParams,drawN=repeats,Fs=sampleRate)
            # Run DNN
            activations = DNNnet.run(geninp[0],DNNnet.mu)[:,latNodeIndex]
            for i in range(len(latNodeIndex)):
                activations[:,i] = activations[:,i]*polarityCorrect[i]
            allActivations[IPDIndex,toneAmpIndex] = activations   

    # Data in Gilbert et al. (2015)            
    toneAmpGilbert = np.arange(0,100+1e-10,5)
    sumRLF_NoSo = np.load('supplementary/GilbertData_sumRLF_NoSo.npy')
    sumRLF_NoSpi = np.load('supplementary/GilbertData_sumRLF_NoSpi.npy')
    sumRLF_NpiSpi = np.load('supplementary/GilbertData_sumRLF_NpiSpi.npy')
    sumRLF_NpiSo = np.load('supplementary/GilbertData_sumRLF_NpiSo.npy')
    
    # Average data, calculate difference
    meanActivation = np.mean(allActivations,2)
    meanActivation_relativeNo = meanActivation[1,:,:]-meanActivation[0,:,:]
    meanActivation_relativeNpi = meanActivation[3,:,:]-meanActivation[2,:,:]
    sumRLF_relativeNo = [sumRLF_NoSpi[i]-sumRLF_NoSo[i] for i in range(len(toneAmpGilbert))]
    sumRLF_relativeNpi = [sumRLF_NpiSo[i]-sumRLF_NpiSpi[i] for i in range(len(toneAmpGilbert))]
    
    # Scale/Linear fits
    old_R2 = -2
    for scale in range(50,101):
        #
        tempRegressionData = [[None]*3]*len(latNodeIndex)
        for latNode in range(len(latNodeIndex)):
            # GP data (scale interpolated)
            rr = int(np.floor(scale/5))
            interp_GP = np.arange(0,scale+1e-10,5) # np.linspace(0,rr*5,rr+1) 
            interp_GP_relativeNo = np.interp(interp_GP,toneAmpGilbert,sumRLF_relativeNo)
            interp_GP_relativeNpi = np.interp(interp_GP,toneAmpGilbert,sumRLF_relativeNpi)
            # DNN data (scale interpolated)
            interp_DNN = (interp_GP/scale)*50 # np.linspace(0,50,rr+1)
            interp_DNN_relativeNo = np.interp(interp_DNN,toneAmpPsychometric,meanActivation_relativeNo[:,latNode])
            interp_DNN_relativeNpi = np.interp(interp_DNN,toneAmpPsychometric,meanActivation_relativeNpi[:,latNode])
            # Combine
            allmu_DNN = list(interp_DNN_relativeNo)+list(interp_DNN_relativeNpi)
            allmu_GP = list(interp_GP_relativeNo)+list(interp_GP_relativeNpi)
            # Fit
            coefslope, intercept, r_value, p_value, std_err = scipy.stats.linregress(allmu_DNN,allmu_GP)
            coef = np.polyfit(allmu_DNN,allmu_GP,1)
            poly1d_fn = np.poly1d(coef)  
            # Update data
            tempRegressionData[latNode]=poly1d_fn,r_value,p_value
        # Check average R**2
        collateR2 = [tempRegressionData[i][1] for i in range(len(latNodeIndex))]
        new_R2 = np.mean(collateR2)
        print(new_R2)
        if new_R2 > old_R2:
            # Update
            old_R2 = new_R2
            bestScale = scale
            allRegressionData = tempRegressionData
            
    return allActivations, allRegressionData, bestScale
        
#%%        
def PNTypeAnalysis(DNNnet,stimParams,thresholdLevels,repeats,KLDivs,sampleRate,polarityCorrect):       
    
     # SORT KLDivs
    latNodeIndex, _ = KLDiv_sort(KLDivs)
    
    # LATENT ACTIVATIONS AT DETECTION THRESHOLDS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allActivations = np.empty((len(allIPDs),repeats,len(latNodeIndex)))
    #
    for IPDIndex in range(lenIPDs):
        stimParams['toneAziPhase'] = allIPDs[IPDIndex][0]
        stimParams['noiseAziPhase'] = allIPDs[IPDIndex][1]
        # Tone at threshold
        stimParams['toneAmp'] = thresholdLevels[IPDIndex]
        geninp = edq.BMLD_data(stimParams,drawN=repeats,Fs=sampleRate)
        activations = DNNnet.run(geninp[0],DNNnet.mu)[:,latNodeIndex]
        for i in range(len(latNodeIndex)):
                activations[:,i] = activations[:,i]*polarityCorrect[i]
        allActivations[IPDIndex,:,:] = activations 

    # PERFORM STATISTICS 
    stats_No = np.empty((len(latNodeIndex),2))
    stats_Npi = np.empty((len(latNodeIndex),2))
    for nodeIndex in range(len(latNodeIndex)):
        #
        t_No,p_No=scipy.stats.ttest_ind(allActivations[0,:,nodeIndex],allActivations[1,:,nodeIndex], equal_var=False)
        t_Npi,p_Npi=scipy.stats.ttest_ind(allActivations[2,:,nodeIndex],allActivations[3,:,nodeIndex], equal_var=False)
        #
        stats_No[nodeIndex,0]=t_No
        stats_No[nodeIndex,1]=p_No
        stats_Npi[nodeIndex,0]=t_Npi
        stats_Npi[nodeIndex,1]=p_Npi
    
    return stats_No, stats_Npi, allActivations


def SDTtest(vals1,vals2):
    mu1 = np.mean(vals1)
    mu2 = np.mean(vals2)
    std1 = np.std(vals1)
    std2 = np.std(vals2)
    dpr = (mu2 - mu1) / (std1*std2)**0.5
    return dpr
    
                

#%% HISTOGRAM OF ACTIVATION VALUES
# https://www.quora.com/How-does-one-access-the-intermediate-activation-layers-of-a-deep-net-in-a-TensorFlow

def activationHist(net,data):
    
    # Initialise
    histcounts = [None]*4
    binvals = [None]*4
    # Layers 1/2 (early)
    for layerIndex in range(0,2):
        tActivations=net.run(data,"encoder/"+str(layerIndex)+"th_enc_layer/b_add_enc_layer_"+str(layerIndex)+":0") # or 'act_fun_enc_layer_'
        nOfBins = 200
        tHistcounts, tBinvals = np.histogram(tActivations,nOfBins,density=True)
        histcounts[layerIndex]=tHistcounts; binvals[layerIndex]=tBinvals
    # Layers 3/4 (late)
    for layerIndex in range(0,2):
        tActivations=net.run(data,"decoder/"+str(layerIndex)+"th_dec_layer/b_add_dec_layer_"+str(layerIndex)+":0") # or 'act_fun_dec_layer_'
        nOfBins = 200
        tHistcounts, tBinvals = np.histogram(tActivations,nOfBins,density=True)
        histcounts[layerIndex+2]=tHistcounts; binvals[layerIndex+2]=tBinvals
        
    return histcounts, binvals


#%% ABLATION

def ablate(net_transfer,stimParams,toneAmpPsychometric,allDropValues,repeats,sampleRate):
    
    #
    
    allIPDs = [[0,0],[np.pi,0]]
    layerRefs = [0,1,3,4] # index to layers want to 'ablate'
    
    allDTs_DNN = np.empty((len(layerRefs),len(allDropValues),len(allIPDs),repeats[1]))
    allDTs_EC = np.empty((len(allIPDs)))
    allPVal = np.empty((len(layerRefs),len(allDropValues)))
    for layerIndex in range(len(layerRefs)): 
        for dropIndex in range(len(allDropValues)): 
            
            tDropout = [0,0,0,0,0,0]
            tDropout[layerRefs[layerIndex]] = allDropValues[dropIndex]
            
            for IPDIndex in range (len(allIPDs)):
                
                stimParams['toneAziPhase'] = allIPDs[IPDIndex][0]
                stimParams['noiseAziPhase'] = allIPDs[IPDIndex][1]
                
                for repIndex in range(repeats[1]):
                    
                    detectionRates_DNN = [];
                    for toneAmpIndex in range(len(toneAmpPsychometric)):
                        # Add toneAmp to dictionary
                        stimParams['toneAmp'] = toneAmpPsychometric[toneAmpIndex]
                        # Generate data
                        geninp = edq.BMLD_data(stimParams,drawN=repeats[0],Fs=sampleRate)
                        # Run DNN
                        DNNOutput = net_transfer.run(geninp[0], net_transfer.output,drop=tDropout).ravel()
                        detectionRates_DNN.append(np.mean(DNNOutput))
                    
                    try:
                        ppred, pcov = curve_fit(psychometric_curve,toneAmpPsychometric,detectionRates_DNN,p0=0.1)
                        DT_DNN =ppred[0]
                        DT_ECtheory = 31-geninp[1][0][0]
                        allDTs_DNN[layerIndex,dropIndex,IPDIndex,repIndex]=DT_DNN
                        allDTs_EC[IPDIndex]=DT_ECtheory
                    except RuntimeError:
                        print("Error - curve_fit failed")    
                        
            # Test for sig changes
            _,pVal=scipy.stats.ttest_ind(allDTs_DNN[layerIndex,dropIndex,0,:], allDTs_DNN[layerIndex,dropIndex,1,:], equal_var=False)
            allPVal[layerIndex,dropIndex]=pVal          
            
    return allDTs_DNN, allDTs_EC, allPVal

        
