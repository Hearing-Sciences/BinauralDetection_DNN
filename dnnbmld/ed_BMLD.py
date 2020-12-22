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



#%% TO CONSIDER
# Use of auditory filter (OLD:wfwnB = butter_bandpass_filter(wn, 500+38.35, 500-38.35, 10000, order=6))
# Transforming pure tone based on phase rather than samples?
# More involved/impressive binaural transform. HRTF
# Correct BMLD for sampling limit?
# Is it right to implement an IPD for noise that does not have a frequency ... ?
# Need to sort out noise transform (order of filtering, is odd to say the least)


#%% IMPORT MODULES
import numpy as np
from scipy.signal import butter, lfilter 
from scipy.stats import norm
import _pickle as cPickle
import gzip
import dnnbmld.io as io


#%% MAIN FUNCTIONS
def BMLD_data(stimParams, # using dictionary for stimulus parameters [could replace this approach with **kwargs]
              drawN=None, # draw this many values (only relevant for single number parameters or ranges)
              Fs=20000,
              startingPhase=None, # if None this will be randomised
              fileName=None ): 
    
    # DESCRIPTION. Pre-transform/transform parameters either represent parameters for successive iterations,
    # or ranges within which values should be randomly drawn. In the latter case, randomN specifies how many
    # random iterations will be drawn.
    
    # 1) SETUP PARAMETERS
    # Unpack
    pt_frequency = stimParams["toneFreq"]
    pt_amplitude = stimParams["toneAmp"]
    no_amplitude = stimParams["noiseAmp"]
    transformType = stimParams["transformType"]
    pt_aziORphasic = stimParams["toneAziPhase"]
    no_aziORphasic = stimParams["noiseAziPhase"]

    # Potentially iterate or randomly draw values between range
    if drawN is None:
        N = len(pt_frequency)
    elif not isinstance(pt_frequency, list):
        N = drawN
        pt_frequency = np.ones((N,1))*pt_frequency
        pt_amplitude = np.ones((N,1))*pt_amplitude
        no_amplitude = np.ones((N,1))*no_amplitude
        pt_aziORphasic = np.ones((N,1))*pt_aziORphasic
        no_aziORphasic = np.ones((N,1))*no_aziORphasic
    else:
        N = drawN
        pt_frequency = random_from_range(pt_frequency,N,scale='log')
        pt_amplitude = random_from_range(pt_amplitude,N)
        no_amplitude = random_from_range(no_amplitude,N)
        pt_aziORphasic = random_from_range(pt_aziORphasic,N)
        no_aziORphasic = random_from_range(no_aziORphasic,N)
     
    # Initialise empty lists
    x_in = []; x_out = []; allBMLD = []
    # For each iteration...
    for pt_fre,pt_amp,no_amp,pt_aORp,no_aORp in zip(pt_frequency,pt_amplitude,no_amplitude,pt_aziORphasic,no_aziORphasic):

        # 2) CREATE WAVEFORMS
        # Sampling
        t_vals = np.arange(-0.02,0.04,1/Fs) 
        # Pure tone
        if startingPhase is None: pt_phase = np.random.rand(1)*2*np.pi
        else: pt_phase = startingPhase
        pt = create_pure_tone(t_vals,pt_amp,pt_fre,pt_phase,Fs)
        # White noise
        no = create_white_noise(t_vals)
        
        # 3) TRANSFORM WAVEFORMS
        padding = int(Fs*0.02) # buffer of 20 ms 
        if transformType == 'azimuth':
            pt_left, pt_right, pt_ITD, pt_ILD = transform_azimuth(pt,pt_aORp,Fs,padding)
            no_left, no_right, no_ITD, no_ILD = transform_azimuth(no,no_aORp,Fs,padding)
        elif transformType == 'phasic':
            pt_left, pt_right, pt_ITD = transform_phasic(pt,pt_aORp,pt_fre,Fs,padding)
            pt_ILD = 0
            no_left, no_right, no_ITD = transform_phasic(no,no_aORp,pt_fre,Fs,padding) # IPD of noise that does not have a frequency ... ?
            no_ILD = 0
        # filter_white_noise
        no_left = filter_white_noise(no_left,no_amp+no_ILD,Fs)
        no_right = filter_white_noise(no_right,no_amp-no_ILD,Fs)
        # Combine and concatenate
        waveform_left = pt_left + no_left
        waveform_right = pt_right + no_right
        completeWaveforms = np.concatenate( (waveform_left, waveform_right) )
 
        # 4) CALCULATE "TRUE" BMLD/DETECTION RATE
        tBMLD = BMLD_eqn(pt_ITD, pt_ILD, pt_fre,
                         no_ITD, no_ILD)
        allBMLD.append(tBMLD)
        detectionRate = psychophysical_curve(tBMLD,pt_amp)
                
        # Append data to relevant lists
        x_in.append(completeWaveforms) 
        x_out.append(detectionRate) 
        
    # 5) RESHAPPE AND PACKAGE
    allBMLD = np.reshape(allBMLD,[N,1]) 
    state_list = np.hstack([allBMLD,pt_amplitude,no_amplitude,pt_aziORphasic,no_aziORphasic]).T # compiles parameters (T transposes)
    x_in = np.array(x_in)
    x_out = np.reshape(x_out, [N, 1])
    result = ([x_in, pt_frequency, x_out], state_list, [])
    
    # 6) SAVE
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=4)
        f.close()
    return (result)


#%% SUPPLEMENTARY FUNCTIONS
def random_from_range(givenRange,N,scale='linear'):
    minRange, maxRange = givenRange
    if scale == 'linear':
        randomValues = (maxRange-minRange)*np.random.rand(N)+minRange
    elif scale == 'log':
        randomValues = 10**( (np.random.rand(N)*(np.log10(maxRange)-np.log10(minRange)))+np.log10(minRange) )
    randomValues = np.reshape(randomValues,[N,1]) 
    return randomValues
    
def create_pure_tone(t_vals,pt_amplitude,pt_frequency,pt_phase,Fs):
    desiredRMS = 20e-6*(10**(pt_amplitude/20))
    pt = (np.sin((2*np.pi*pt_frequency*t_vals)+pt_phase)/0.707)*desiredRMS
    return pt

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def create_white_noise(t_vals):#,Fs):
    wn = np.random.normal(0,1,len(t_vals)) 
    return wn

def filter_white_noise(wn,no_amplitude,Fs):
    wn = butter_bandpass_filter(wn, 50, 4999, Fs, order=6)
    wn = (wn/np.sqrt(np.mean(wn**2)))*20e-6*(10**(no_amplitude/20))
    return wn

def transform_azimuth(wf,wf_azimuth,Fs,padding):
    # Transforms a waveform (wf) based on azimuthal angle.
    # Also outputs corresponding ITD/ILD.
    #
    # Temporal offset
    timeOffset_left, timeOffset_right = calculate_phases_at_ears(wf_azimuth) # Temporal
    wf_ITD = timeOffset_left - timeOffset_right # Could round this based on sampling rate
    sampleOffset_left = int( np.round( timeOffset_left*Fs ) )
    sampleOffset_right = int( np.round( timeOffset_right*Fs ) )
    # Level differences
    amplitudeDiff_left = 0 
    amplitudeDiff_right = 0
    wf_ILD = amplitudeDiff_left - amplitudeDiff_right
    # Waveform left/right
    wf_left = wf[((padding)+sampleOffset_left):(sampleOffset_left+(2*padding))] # translate
    wf_left = wf_left*10**(amplitudeDiff_left/20) # amplify
    wf_right = wf[((padding)+sampleOffset_right):(sampleOffset_right+(2*padding))]
    wf_right = wf_right*10**(amplitudeDiff_right/20) # amplify    
    return wf_left, wf_right, wf_ITD, wf_ILD

def calculate_phases_at_ears(azimuth):
    # Based on Woodworth's equations (1978?). Adjusted for sample rate (rounded)
    k = 0.0875/343 # constant (head radius/speed of sound)  
    contralateral = k*np.sin((np.pi/180)*np.abs(azimuth))
    ipsilateral = -k*((np.pi/180)*np.abs(azimuth)) 
    # etsablish left/right ear
    if azimuth >= 0:
        left, right = contralateral, ipsilateral
    else: 
        right, left = contralateral, ipsilateral
    return left, right

def transform_phasic(wf,wf_phasic,wf_fre,Fs,padding):
    # Transforms a waveform (wf) based on phase (unit: RADS).
    # Also outputs corresponding ITD.
    #
    # Anti-phasic
    if wf_phasic == -np.pi:
        wf_left, wf_right = -wf[padding:(2*padding)], wf[padding:(2*padding)]
    elif wf_phasic == np.pi:
        wf_left, wf_right = wf[padding:(2*padding)], -wf[padding:(2*padding)]
    else:
    # Other phases
        wf_left = wf[padding:(2*padding)]
        sampleOffset = int( np.round( (wf_phasic/(2*np.pi))*(1/wf_fre)*Fs ) )
        wf_right = wf[(padding+sampleOffset):((2*padding)+sampleOffset)]
    # ITD 
    wf_ITD = ((wf_phasic)/(2*np.pi))*(1/wf_fre) # Could round this based on sampling rate?  
    return wf_left, wf_right, wf_ITD

def psychophysical_curve(BMLD,pt_amplitude):
    # Reverse engineer psychophysical curve (Egan et al)
    m = 0.159*10**(0.1*BMLD)
    K=1
    dPrime = m*(10**(0.1*(pt_amplitude-23)))**K
    detectionRate = norm.cdf(dPrime/2)
    return detectionRate

def BMLD_eqn(pt_ITD,pt_ILD,pt_fre,
             no_ITD,no_ILD):
    #    Calculated BMLD based on revised BMLD equation 
    #    (Wan, Durlach, Colburn, 2010)
    #
    # Set error parameters
    sig_amperr = 0.25 # amplitude jitter error 
    sig_tderr = 105e-6 # time delay jitter error
    # Set left/right parameters
    tau_s = (pt_ITD)
    tau_n = (no_ITD)
    a_s = 10**((pt_ILD)/20)
    a_n = 10**((no_ILD)/20)
    # Convert to angula freq
    w = pt_fre
    w0 = w * 2 * np.pi       
    # # BMLD equation
    # constant
    k = (1+sig_amperr**2)*np.exp((w0**2)*(sig_tderr**2)); 
    # numerator
    N_a = k*((a_s/a_n)**2+1);
    N_b = 2*(a_s/a_n)*np.cos((w0*tau_s)-(w0*tau_n));
    N=N_a-N_b
    # gamma (envelope autocorr)-formulation 
    if (tau_n > (np.pi/w0)) or (tau_n < -(np.pi/w0)):
        tau0=0
    elif tau_n > 0:
        n = np.floor( (tau_n*w0)/(2*np.pi) );
        tau0 = (2*np.pi*n)/w0
    else:
        n = np.ceil( (tau_n*w0)/(2*np.pi) );
        tau0 = -(2*np.pi*n)/w0
    #
    if (tau_n-tau0) == 0:
        g = 1
    else:
        B = 2*np.pi*(24.7*(((4.37*w)/1000)+1)) # calculate ERB (glasberg)
        g = ((np.sin((B*(tau_n-tau0))/2)**2)/(((B*(tau_n-tau0))/2)**2)) # gamma function (envelope autocorr)
    # denominator
    D = 2*(k-g)*np.max([(a_s/a_n)**2,1])
    # max
    B = np.max( [N/D , 1] ) # monaural processing better than binaural
    B_dB = 10*np.log10(B) # Power quantity
    return np.real(B_dB)
