#    Code to make plots in "results" 
#
#
#
#
#

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy.stats import norm
import analysis.analyses as af
import scipy.stats
import scipy.io as spio
import dnnbmld.ed_BMLD as edq

def DT3D(BMLDsPredictedByDNN,trueBMLDs,allAzis):
    #
    fig = plt.figure(1,figsize=[5.4,1.5])
    fig.clf()
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    tvmin=20;tvmax=30
    lazi = len(allAzis)
        # EC
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('Ground truth',fontsize=8)
    plt.imshow(trueBMLDs,vmin=tvmin,vmax=tvmax,cmap='viridis_r')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    ax.set_xlabel('Tone azimuth (°)')
    ax.set_ylabel('Noise azimuth (°)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0,(lazi-1)/2,lazi-1]);ax.set_xticklabels(['-90',' ','90']);
    ax.set_yticks([0,(lazi-1)/2,lazi-1]);ax.set_yticklabels(['-90',' ','90']);
    ax.set_ylim(-0.5,26.5)
        # DNN
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('DNN',fontsize=8)
    plt.imshow(np.mean(BMLDsPredictedByDNN,2),vmin=tvmin,vmax=tvmax, cmap='viridis_r')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    ax.set_xlabel('Tone azimuth (°)')
    ax.set_ylabel('Noise azimuth (°)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0,(lazi-1)/2,lazi-1]);ax.set_xticklabels(['-90',' ','90']);
    ax.set_yticks([0,(lazi-1)/2,lazi-1]);ax.set_yticklabels(['-90',' ','90']);
    ax.set_ylim(-0.5,26.5)
        # Colorbar
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    cbaxes = fig.add_axes([0.66, 0.5, 0.2, 0.04])  # This is the position for the colorbar
    cb=plt.colorbar(orientation='horizontal',cax = cbaxes)
    # cb.ax.set_xlabel('Detection threshold (dB)', 
    #                   rotation=0, fontsize=7, labelpad=50)
    cb.outline.set_visible(False)
    #
    fig.tight_layout() 


def DT2D(allDTs_DNN,allDTs_ECTheory,allAzis,toneIndex=None,noiseIndex=None):
    #
    if toneIndex == None:
        DNNPlane = allDTs_DNN[:,noiseIndex,:]
        ECTheoryPlane = allDTs_ECTheory[:,noiseIndex]
        DTTitle = "noise azimuth = " + str(int(allAzis[noiseIndex])) + "°"
        XAxLabel = "pure tone azimuth,°"
    elif noiseIndex == None:
        DNNPlane = allDTs_DNN[toneIndex,:,:]
        ECTheoryPlane = allDTs_ECTheory[toneIndex,:]
        DTTitle = "pure tone azimuth = " + str(int(allAzis[toneIndex])) + "°"
        XAxLabel = "noise azimuth,°"
    #
    fig = plt.figure(2,figsize=[7.4,1.8])
    fig.clf()
    plt.rcParams.update({'font.size':8})
    plt.rcParams["font.family"] = "Arial"
    tvmin=19.5;tvmax=33
    # 2D PLOTS
    c2 = [0/256,114/256,189/256]
    c1 = [215/256,83/256,39/256]
    ms=4
    ax = fig.add_subplot(1,3,1)
    ax.plot(allAzis, ECTheoryPlane, color=c1, linestyle='-' , linewidth=3, label='EC theory',markersize=ms)    
    ax.errorbar(allAzis, np.mean(DNNPlane,1), yerr=[mean_confidence_interval(DNNPlane[i,:]) for i in range(len(allAzis))], color=c2, linewidth=0.75,
                markerfacecolor=c2, linestyle='-', marker='o', label='DNN', markersize=ms)     
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_ylim(tvmin,tvmax)
    ax.set_xlabel(XAxLabel)
    ax.set_ylabel('detection threshold, dB')
    ax.set_title(DTTitle)
    handles, labels = plt.gca().get_legend_handles_labels() 
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="upper right",fontsize=8)
    leg.get_frame().set_linewidth(0.0)


def DTPhasic(allDTs_DNN,allDTs_ECTheory):
    # PRE ANALYSIS
    allDTs_DNN_mean = np.mean(allDTs_DNN,1)
    allDTs_DNN_CI = [mean_confidence_interval(allDTs_DNN[i,:]) for i in range(4)]
    # FIGURE
    fig = plt.figure(3,figsize=[7, 2])
    fig.clf()
    plt.rcParams.update({'font.size': 8})
    plt.rcParams["font.family"] = "Arial"
    c1 = [215/256,83/256,39/256]
    c2 = [0/256,114/256,189/256]
    c3 = [.2,.2,.2]
    labels = ['NoSo', 'NoSπ', 'NπSπ', 'NπSo']
    # DETECTION THRESHOLD
    ax = fig.add_subplot(1, 3, 1)
    width = 0.35; barwidth = 0.32
    ax.plot([-0.5,3.5],[0,0],color=(.6,.6,.6),linewidth=.5,zorder=0)
    x = np.arange(len(labels))
    bars1 = ax.bar(x -width/2, allDTs_DNN_mean, barwidth, yerr=allDTs_DNN_CI, label='DNN', color=c2, error_kw=dict(ecolor=[x*0.8 for x in c2]) )
    bars2 = ax.bar(x + width/2, allDTs_ECTheory, barwidth, label='EC theory', color=c1 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=10)
    ax.set_ylim(14,33)
    ax.set_ylabel('detection threshold, dB')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # BMLDs
    ax = fig.add_subplot(1, 3, 2)
    labels_BMLD = [labels[0]+'-'+labels[1],labels[2]+'-'+labels[3]]
    x = np.arange(len(labels_BMLD))  # the label locations
    width = 0.2; barwidth = 0.185
    ax.plot([-0.5,1.5],[0,0],color=(.6,.6,.6),linewidth=.5,zorder=0)
    allDTs_DNN_CI1 = mean_confidence_interval_diff(allDTs_DNN[0,:],allDTs_DNN[1,:])
    allDTs_DNN_CI2 = mean_confidence_interval_diff(allDTs_DNN[2,:],allDTs_DNN[3,:])
    bar1 = ax.bar(x - width, [allDTs_DNN_mean[0]-allDTs_DNN_mean[1],allDTs_DNN_mean[2]-allDTs_DNN_mean[3]], barwidth , 
                    yerr = [allDTs_DNN_CI1,allDTs_DNN_CI2],
                     label='DNN', color=c2, error_kw=dict(ecolor=[x*0.8 for x in c2]) )
    bar2 = ax.bar(x , [allDTs_ECTheory[0]-allDTs_ECTheory[1],allDTs_ECTheory[2]-allDTs_ECTheory[3]], barwidth, label='EC theory', color=c1 )
    bar3 = ax.bar(x+width, [10,22], barwidth, label='GP ACtx', color=c3 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_BMLD,rotation=10)
    ax.set_ylim(0,23)
    ax.set_ylabel('BMLD, dB')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels2 = plt.gca().get_legend_handles_labels() 
    fig.tight_layout() 
    # LEGEND
    ax = fig.add_subplot(1, 3, 3)
    by_label = dict(zip(labels2, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="center left",fontsize=8)
    leg.get_frame().set_linewidth(0.0)
    ax.axis("off")
    
def BMLDPhasic(allDTs_DNN,allDTs_ECTheory):
    # PRE ANALYSIS
    allDTs_DNN_mean = np.mean(allDTs_DNN,1)
    allDTs_DNN_CI = [mean_confidence_interval(allDTs_DNN[i,:]) for i in range(4)]
    # FIGURE
    fig = plt.figure(3,figsize=[1.65, 1.85])
    fig.clf()
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    c1 = [33/256,113/256,181/256]
    c2 = [203/256,24/256,29/256]
    labels = ['NoSo', 'NoSπ', 'NπSπ', 'NπSo']
    # BMLDs
    ax = fig.add_subplot(1, 1, 1)
    labels_BMLD = [labels[0]+'\n- '+labels[1],labels[2]+'\n- '+labels[3]]
    x = np.arange(len(labels_BMLD))  # the label locations
    width = 0.35/2; barwidth = 0.35
    ax.plot([-0.5,1.5],[0,0],color=(.6,.6,.6),linewidth=.5,zorder=0)
    allDTs_DNN_CI1 = mean_confidence_interval_diff(allDTs_DNN[0,:],allDTs_DNN[1,:])
    allDTs_DNN_CI2 = mean_confidence_interval_diff(allDTs_DNN[2,:],allDTs_DNN[3,:])
    bar1 = ax.bar(x - width, [allDTs_DNN_mean[0]-allDTs_DNN_mean[1],allDTs_DNN_mean[2]-allDTs_DNN_mean[3]], barwidth , 
                    yerr = [allDTs_DNN_CI1,allDTs_DNN_CI2],
                     label='DNN', color=c2, error_kw=dict(ecolor=[0 for x in c2]) )
    bar2 = ax.bar(x + width , [allDTs_ECTheory[0]-allDTs_ECTheory[1],allDTs_ECTheory[2]-allDTs_ECTheory[3]], barwidth, label='psychophysics', color=c1 )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_BMLD,rotation=0)
    ax.set_yticks([0,5,10])
    ax.set_ylim(0,15)
    ax.set_ylabel('BMLD, dB')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels2 = plt.gca().get_legend_handles_labels() 
    fig.tight_layout() 
    # LEGEND
    by_label = dict(zip(labels2, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="upper left",fontsize=7)
    leg.get_frame().set_linewidth(0.0)
    # ax.axis("off")


def ITD_tuning(IPDsForRLF,allActivations,allGaborFitParams,KLDivs,stimtype,toneFreq,polarityCorrect):
    #
    nLatNodes = len(allGaborFitParams)
    ITDs = (IPDsForRLF/(2*np.pi)) * (1e6/toneFreq)
    # SORT KLDivs
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    #
    fig = plt.figure(4,figsize=[5.5, .8*nLatNodes])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "Arial"
    #
    for nodeIndex in range(nLatNodes):
        ax = fig.add_subplot(nLatNodes, 3, nodeIndex*3+1)
        ax.text(0.85,0.45,'node '+str(nodeIndex+1)+'\n'+'KL divergence = '+str(np.round(latNodeKL[nodeIndex],2)), 
                fontsize=8, horizontalalignment='center',verticalalignment='center')
        ax.axis("off")
        #
        ax = fig.add_subplot(nLatNodes, 3, nodeIndex*3+2)
        #
        ITDlim = 655.5
        ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        #
        pc = polarityCorrect[nodeIndex]
        #
        tActivationMean = np.mean(allActivations[:,:,nodeIndex],1)*pc
        tCI = [mean_confidence_interval(allActivations[x,:,nodeIndex]) for x in range(len(ITDs))]
        ax.fill_between(ITDs, (tActivationMean-tCI), (tActivationMean+tCI), facecolor=allcols[nodeIndex], edgecolor = 'none', alpha=.4,zorder=1)
        ax.plot(ITDs,tActivationMean,'-',color=allcols[nodeIndex],linewidth=2,zorder=2)
        #
        pVal = allGaborFitParams[2][nodeIndex,:]
        # Gabor calc/plot
        # if pVal>0.05: # BITD_left == BITD_right:
        #     tParams = allGaborFitParams[1][nodeIndex,:]
        #     tGaborFit = af.gabor_function_single(ITDs,*tParams)
        #     tGaborFit = [i*pc for i in tGaborFit]
        #     ax.plot(ITDs,tGaborFit,'--',color=(.0,.0,.0),linewidth=1.5,zorder=3)
        #     ax.plot(tParams[1],af.gabor_function_single([tParams[1]],*tParams)[0]*pc,'^',mew=2, markersize = 6, color='k',mfc='none',mec='k')
        # else:
        tParams = allGaborFitParams[0][nodeIndex,:]
        tGaborFit = af.gabor_function_double(ITDs,*tParams)
        tGaborFit = [i*pc for i in tGaborFit]
        ax.plot(ITDs,tGaborFit,'--',color=(.0,.0,.0),linewidth=1.5,zorder=3)
        ax.plot(tParams[1],af.gabor_function_double([tParams[1]],*tParams)[0]*pc,'o',mew=2, markersize = 6, color='k',mfc='none',mec='k')
        ax.plot(tParams[2],af.gabor_function_double([tParams[2]],*tParams)[0]*pc,'s',mew=2, markersize = 6, color='k',mfc='none',mec='k')
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-0.02,1.02);ax.set_xlim(-2000,2000)
        ymi = np.min(tActivationMean); yma = np.max(tActivationMean); ydi = yma-ymi
        ax.set_ylim(ymi-ydi*0.1,yma+ydi*0.1)
        ax.set_ylabel('activation')
        if nodeIndex == (nLatNodes-1):
            ax.set_xlabel(stimtype+' ITD, µs',fontsize=9)
            ax.set_xticks([-2000,0,2000])
        else:
            ax.set_xticklabels([])       
    fig.tight_layout()

def ITD_tuning2(IPDsForRLF,allActivations,allGaborFitParams,KLDivs,stimtype,toneFreq,polarityCorrect):
    #
    nLatNodes = len(allGaborFitParams)
    ITDs = (IPDsForRLF/(2*np.pi)) * (1e6/toneFreq)
    # SORT KLDivs
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    #
    fig = plt.figure(4,figsize=[2.1,6])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "Arial"
    #
    ax = fig.add_subplot(1,1,1)
    #
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel("latent nodes (ordered by \"informativeness\")")
    ax.set_xticks([])
    ax.set_yticks([])
    #
    for nodeIndex in range(nLatNodes):

        ax = fig.add_subplot(nLatNodes, 1, nodeIndex+1)
        #
        ITDlim = 655.5
        ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        #
        #
        tActivationMean = np.mean(allActivations[:,:,nodeIndex],1)
        tCI = [mean_confidence_interval(allActivations[x,:,nodeIndex]) for x in range(len(ITDs))]
        # normalise
        tmii = np.min(tActivationMean); tmaa = np.max(tActivationMean); tdii = tmaa - tmii
        tActivationMean = (tActivationMean - tmii) / tdii
        tCI = tCI / tdii
        #
        ax.fill_between(ITDs, (tActivationMean-tCI), (tActivationMean+tCI), facecolor=allcols[nodeIndex], edgecolor = 'none', alpha=.4,zorder=1)
        ax.plot(ITDs,tActivationMean,'-',color=allcols[nodeIndex],linewidth=2,zorder=2)
        ax.plot(ITDs,1-tActivationMean,':',dashes=(1, 1),color=allcols[nodeIndex],linewidth=1.25,zorder=2)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-0.02,1.02);ax.set_xlim(-2000,2000)
        ymi = np.min(tActivationMean); yma = np.max(tActivationMean); ydi = yma-ymi
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([])
        ax.set_xticks([-1000,0,1000])
        if nodeIndex == (nLatNodes-1):
            ax.set_xlabel('noise ITD, µs')
            ax.set_xticks([-1000,0,1000])
        else:
            ax.set_xticklabels([])       
    fig.tight_layout()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def ITD_distribution(allGaborFitParams,toneORnoise,KLDivs):
    #
    xvs_tone = [-200,-50,0,50,100,150,200,250,300,350,400]
    yvs_tone = [1,3,2,9,12,5,6,5,21,5,1]
    xvs_noise = [-700,-600,-500,-300,-200,-100,-50,0,50,100,150,200,250,300,350,450,550]
    yvs_noise = [1,1,1,2,3,1,1,5,9,13,14,7,16,13,1,1,1]
    if toneORnoise == "tone":
        xvs = xvs_tone; yvs = yvs_tone
    elif toneORnoise == "noise":
        xvs = xvs_noise; yvs = yvs_noise
    #
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    #
    fig = plt.figure(5,figsize=[3.7, 3.4])
    fig.clf()
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "Arial"
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    # ITDs
    ax = fig.add_subplot(2,2,1)
    width = 40
    # inverted population
    flipxvs = [-x for x in xvs]
    bixvs = np.unique([flipxvs,xvs])
    biyvs = []
    for b in range(0,len(bixvs)):
        tbs = bixvs[b]
        # 
        if tbs in flipxvs:
            tin = flipxvs.index(tbs)
            biyvs.append(yvs[tin])
        else:
            biyvs.append(0)   
    revv = biyvs[::-1]
    #
    ax.bar(bixvs,revv,width,color=(250/256,210/256,208/256),label='GP ACtx (right)')
    ax.bar(bixvs,biyvs,width,bottom=revv,color='none', hatch="///", edgecolor=(174/256,197/256,228/256), ecolor=(174/256,197/256,228/256), label='GP ACtx (inverted)')   #,label='GP ACtx (simulated hemisphere)')
    for latNode in range(nLatNodes):
        jitter = 0.1
        pVal = allGaborFitParams[2][latNode,:]
        # if pVal>0.05:
        #      # single
        #     BITD_single = allGaborFitParams[1][latNode,1]
        #     ax.plot(BITD_single,1+np.random.normal(0, jitter, 1),'^',mew=1,ms=6,color=allcols[latNode],mec='k',label='node '+str(latNode+1))
        # else:
            # left
        BITD_left = allGaborFitParams[0][latNode,1]
        ax.plot(BITD_left,1+np.random.normal(0, jitter, 1),'o',mew=1,ms=6,color=allcols[latNode],mec='k',label='node '+str(latNode+1)+' (left)')
        # right
        BITD_right = allGaborFitParams[0][latNode,2]
        ax.plot(BITD_right,1+np.random.normal(0, jitter, 1),'s',mew=1 ,ms=6,color=allcols[latNode],mec='k',label='node '+str(latNode+1)+' (right)')
    ax.set_xlabel('Best '+toneORnoise+' ITD (μs)')
    ax.set_ylabel('Number of neurons/nodes')
    ax.set_xlim(-1000,1000)
    ax.set_ylim(0,20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # LEGEND
    handles, labels= plt.gca().get_legend_handles_labels()
    # LEGEND
    ax = fig.add_subplot(2, 2, 2)
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="upper left",fontsize=6,ncol=1)
    leg.get_frame().set_linewidth(0.0)
    ax.axis("off")
    #
    fig.tight_layout()
    

def overlaid_NDFs(IPDsForRLF,allGaborFitParams,KLDivs,stimtype,toneFreq,polarityCorrect):
    #
    nLatNodes = len(allGaborFitParams)
    ITDs = (IPDsForRLF/(2*np.pi)) * (1e6/toneFreq)
    # SORT KLDivs
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    #
    fig = plt.figure(11,figsize=[3.7, 3.4])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "Arial"
    ax = fig.add_subplot(2, 2, 1)
    #
    for nodeIndex in range(nLatNodes):
        #
        pc = polarityCorrect[nodeIndex]
        #
        tParams = allGaborFitParams[0][nodeIndex,:]
        # tGaborFit = af.gabor_function_double(ITDs,*tParams)
        tGaborFit = af.gabor_function_single(ITDs,tParams[0],tParams[2],tParams[3],tParams[4],tParams[5])
        tGaborFit = [i*pc for i in tGaborFit]
        # Normalize
        minGab = min(tGaborFit)
        diffGab = max(tGaborFit)-min(tGaborFit)
        tGaborFit = [(i-minGab)/diffGab for i in tGaborFit]
        #
        ax.plot(ITDs,tGaborFit,'-',color=allcols[nodeIndex],linewidth=1,zorder=3)
        #
        # Gab_steepness = [ tGaborFit[i+2]-tGaborFit[i] for i in range(len(tGaborFit)-2) ]
        # minSteep = min(Gab_steepness)
        # diffSteep = max(Gab_steepness)-min(Gab_steepness)
        # Gab_steepness = [(i-minSteep)/diffSteep for i in Gab_steepness]
        # ax.plot( ITDs[1:-1] , Gab_steepness,'-',color='k',linewidth=0.5,zorder=3)
        steepestPoint = tParams[2] - ( 1/(4*tParams[0]) )
        ax.plot( [steepestPoint,steepestPoint], [0,1],':',color=allcols[nodeIndex],linewidth=0.5,zorder=3)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-2000,2000)
        ax.set_ylim(0,1.01)
        ax.set_ylabel('Normalized delay function')
        ax.set_xlabel(stimtype+' ITD, µs',fontsize=9)
        ax.set_xticks([-2000,0,2000]) 
        ax.set_yticks([0, 0.5, 1]) 
    fig.tight_layout()


#%%

def mean_confidence_interval_diff(data1,data2,confidence=0.95):
    a = 1.0 * np.array(data1)
    b = 1.0 * np.array(data2)
    n1 = len(a); n2 = len(b)
    se =  np.sqrt( ((np.std(a)**2)/(n1-1)) + ((np.std(b)**2)/(n2-1)) )
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n1-1)
    return h

# P/N types
# [[11, 13],[6,46],[3,4],[52,6]]
# total = 109



def PNVals(allActivations,KLDivs):
    
    #
    
    # SORT KLDivs
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    # 
    fig = plt.figure(8,figsize=[6.,6.5])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 6})
    plt.rcParams["font.family"] = "Arial"
    #
    nNodes = len(latNodeKL)
    for nodeIndex in range(nNodes):
        #
        for noiseType in range(2):
            #
            suppressed = allActivations[0+noiseType*3,:,nodeIndex]
            homophasic = allActivations[1+noiseType*3,:,nodeIndex]
            antiphasic = allActivations[2+noiseType*3,:,nodeIndex]
            #
            if noiseType == 0: # KLDiv text
                ax = fig.add_subplot(nNodes,4,(nodeIndex*4)+1)
                ax.text(0.7,0.45,'node '+str(nodeIndex+1)+'\n'+'KLdiv='+str(np.round(latNodeKL[nodeIndex],2)),
                        fontsize=10, horizontalalignment='center',verticalalignment='center')
                ax.axis("off")
            #    
            ax = fig.add_subplot(nNodes,4,(nodeIndex*4)+2+noiseType)
            supp_CI = mean_confidence_interval(suppressed)
            homo_CI = mean_confidence_interval(homophasic)          
            anti_CI = mean_confidence_interval(antiphasic)
            tCol = allcols[nodeIndex]
            ax.errorbar([0,1],[np.mean(suppressed),np.mean(homophasic)],yerr=[supp_CI,homo_CI], 
                         label='blank',linestyle='-',color=tCol,mfc=tCol,mec=tCol,marker='o',ms=5,linewidth=1.5)
            ax.errorbar([0,2],[np.mean(suppressed),np.mean(antiphasic)],yerr=[supp_CI,homo_CI], 
                         label='blank',linestyle='--',color=tCol,mfc=tCol,mec=tCol,marker='o',ms=5,linewidth=1.5)
            ax.set_xticks([0,1,2])
            if noiseType == 0:
                labels = ['No','NoSo','NoSπ']
            else:
                labels = ['Nπ','NπSπ','NπSo']
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.4,2.4)
            ax.set_xlabel('            (at threshold)', fontsize=7)
            ax.set_ylabel('activation',fontsize=8)
            ymi = np.min([np.mean(suppressed),np.mean(homophasic),np.mean(antiphasic)])
            yma = np.max([np.mean(suppressed),np.mean(homophasic),np.mean(antiphasic)])
            ydi = yma-ymi
            ax.set_ylim(ymi-ydi,yma+ydi)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.tight_layout()
    
    
#%%
    
def layerHistograms(histcounts,binvals):
    
    fig = plt.figure(10,figsize=[2.2,3])
    fig.clf()
    plt.rcParams.update({'font.size':8})
    plt.rcParams["font.family"] = "Arial"
    allCols = ["#79C000FF","#558D20","#8F7C2F","#4E3D29"]
    
    # Histograms
    ax = fig.add_subplot(2,1,1)
    for layerIndex in [3,2,1,0]:
        #
        xax_v = (binvals[layerIndex][1:]+binvals[layerIndex][0:-1])/2
        yax_v = histcounts[layerIndex]
        #
        # ax.plot(xax_v,yax_v,linestyle='-',linewidth=1.5, color=allCols[layerIndex]) 
        plt.fill_between(xax_v,yax_v,linestyle='-',linewidth=.5, fc=allCols[layerIndex],alpha=.7,ec='k')
        ax.set_yticks([300])
        #        
        # if layerIndex < 2:
        #     ax.text(np.min(xax_v)+0.016*(np.max(xax_v)-np.min(xax_v)),np.max(yax_v),'layer '+str(layerIndex+1),horizontalalignment='left', verticalalignment='top',color=allCols[layerIndex],weight='bold')
        # else:
        #     ax.text(np.min(xax_v)+0.016*(np.max(xax_v)-np.min(xax_v)),np.max(yax_v),'layer '+str(layerIndex+2),horizontalalignment='left', verticalalignment='top',color=allCols[layerIndex],weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(min(binvals[layerIndex]),max(binvals[layerIndex]))
    ax.set_ylim([1e-3,5e0])
    ax.set_xlim([-158,20])
    ax.set_yticks([])
    ax.set_xlabel('pre-activation values')
    ax.set_ylabel('pdf (log scale)')
    # ax.set_xscale('log') 
    ax.set_yscale('log') 
    fig.tight_layout() 

#%%
    
def showAblation(allDTs_DNN, allDTs_EC, allDropVals):
    
    fig = plt.figure(11,figsize=[3,2.5])
    fig.clf()
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    allmarks = ['o','s','^','p']
    allCols = [[121/256,192/256,0],[85/256,141/256,32/256],[143/256,124/256,47/256],[78/256,61/256,41/256]]
    ax = fig.add_subplot(2,1,1)
    
    for layerIndex in range(len(allDTs_DNN)):
        
        if layerIndex < 2:
            tLabel = 'layer '+str(layerIndex+1)
        else: 
            tLabel = 'layer '+str(layerIndex+2)
            
        width = 0.002
        xVals = [100*(x+width*(layerIndex-2)) for x in allDropVals]
        yVals = np.mean(allDTs_DNN[layerIndex,:,0,:],1)-np.mean(allDTs_DNN[layerIndex,:,1,:],1)
        tCI = [mean_confidence_interval_diff(allDTs_DNN[layerIndex,i,0,:],allDTs_DNN[layerIndex,i,1,:]) for i in range(len(allDropVals))]
        ax.hlines(allDTs_EC[0]-allDTs_EC[1],0,100,color =[215/256,83/256,39/256],linewidth = 1.5,linestyles=":",label="human")
        ax.errorbar(xVals,yVals,yerr=tCI,linestyle='-',linewidth=1.5,label=tLabel,
                    marker='o', markersize=3,color = allCols[layerIndex] , ecolor=[1-(1-x)*0.3 for x in  allCols[layerIndex]])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-2,np.max(xVals)+2)
        ax.set_ylim(-7,19)
        ax.set_xlabel('Ablation (%)')
        ax.set_ylabel('NoSo-NoSπ BMLD (dB)')
    # legend 
    handles, labels = plt.gca().get_legend_handles_labels() 
    ax = fig.add_subplot(2,1,2)
    ax.axis("off")
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="lower left",fontsize=7,ncol=1)
    leg.get_frame().set_linewidth(0.0)
    fig.tight_layout() 
    
    
    
#%% CHECK OUT HISTOGRAM
# fig = plt.figure(111)
# fig.clf()
# for i in range(6):
#     #
#     ax = fig.add_subplot(6,2,i*2+1)
#     plt.hist(allActivations[0,:,i], density=True, bins=30, alpha = 0.5)
#     plt.hist(allActivations[1,:,i], density=True, bins=30, alpha = 0.5)
#     plt.hist(allActivations[2,:,i], density=True, bins=30, alpha = 0.5)
#     #
#     ax = fig.add_subplot(6,2,i*2+2)
#     plt.hist(allActivations[3,:,i], density=True, bins=30, alpha = 0.5)
#     plt.hist(allActivations[4,:,i], density=True, bins=30, alpha = 0.5)
#     plt.hist(allActivations[5,:,i], density=True, bins=30, alpha = 0.5)
    
    
#%% NEW CODE #%%
    
    
def plot_overlap(allIPDs,toneFreq,layerActivations):
    
    allITDs = (allIPDs/(2*np.pi)) * (1e6/toneFreq)
    # Get gabor fits (i.e. bITDs)
    allGaborFit_Single = fit_gabor(allITDs,toneFreq,layerActivations)
    bITDorder = np.argsort( allGaborFit_Single[:,1] )
    # Setup fig
    fig = plt.figure(18,figsize=[2.1,5])
    fig.clf()
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "Arial"
    ax = fig.add_subplot(1,1,1)
    # Initialize
    ITDlim = 655.5
    ax.axvspan(-ITDlim, ITDlim, facecolor=(.75,.75,.75), edgecolor='none',zorder=0)
    ax.axvline(x=-1000,linewidth=2, color=(.5,0,0),zorder=1)
    ax.axvline(x=1000,linewidth=2, color=(.5,0,0),zorder=1)
    # MAIN PLOT (after normalization)
    cnt = 0
    for nodeIndex in bITDorder[::-1]:
        tActivationMean = layerActivations[1,:,nodeIndex]
        tActivationMean = ( tActivationMean - min(tActivationMean) ) / ( max(tActivationMean) - min(tActivationMean) )
        ax.plot(allITDs,cnt + tActivationMean,'k-',linewidth=1,zorder=2)
        ax.plot(allGaborFit_Single[nodeIndex,1],cnt + 1,'o',linewidth=2,zorder=3,mew=1, markersize = 2, color=(0,0,.5),mfc=(0,0,.5),mec=(0,0,.5))
        cnt += .075
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0,cnt+1)
    ax.set_xlim(-2000,2000)
    ax.set_xticks([-1000,0,1000])
    ax.set_xlabel('noise ITD, µs')  
    ax.set_ylabel('normalized responses of ' + r"$\bf{layer}$" + ' ' + r"$\bf{2}$" + ' nodes')
    fig.tight_layout()
    # HISTOGRAM
    fig = plt.figure(19,figsize=[2.1,1.6])
    fig.clf()
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "Arial"
    ax = fig.add_subplot(1,1,1)
    ax.axvline(x=-1000,linewidth=2, color=(.5,0,0),zorder=3)
    ax.axvline(x=1000,linewidth=2, color=(.5,0,0),zorder=3)
    plt.hist(allGaborFit_Single[:,1],weights=allGaborFit_Single[:,2],bins=10,range=[-1000, 1000],color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-2000,2000)
    ax.set_xticks([-1000,0,1000])     
    ax.set_xlabel('noise bITD, µs')
    ax.set_ylabel('count')
    #
    fig.tight_layout()
    
    return allGaborFit_Single
    

def plot_grid(allIPDs,toneFreq,layerActivations): 
    allITDs = (allIPDs/(2*np.pi)) * (1e6/toneFreq)
    # OVERLAP
    fig = plt.figure(28,figsize=[2.1,6.5])
    fig.clf()
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "Arial"
    #
    ax = fig.add_subplot(1,1,1)
    #
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-2000,2000)
    ax.set_xticks([-1000,0,1000])
    ax.set_xlabel('noise ITD, µs')
    ax.set_ylabel('normalized responses of ' + r"$\bf{layer}$" + ' ' + r"$\bf{1}$" + ' nodes')      
    # fig.tight_layout()
    #
    for nodeIndex in range(100):
        #
        ax = fig.add_subplot(26,4,nodeIndex+1)
    #     #
        tActivationMean = layerActivations[0,:,nodeIndex]
        tActivationMean = ( tActivationMean - min(tActivationMean) ) / ( max(tActivationMean) - min(tActivationMean) )
        ax.plot(allITDs,tActivationMean,'k-',linewidth=.5,zorder=2)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        #
        # ax.set_xlim(-2000,2000)
        ax.set_xticks([-1000,0,1000],(' ',' ',''))
    #
    fig.tight_layout()
        
    
def fit_gabor(allITDs,toneFreq,layerActivations):
    
    allGaborFit_Single_thresh = np.empty((100,5))
    allGaborFit_Single_R2 = np.empty((100,))
    allGaborFit_Single_p = np.empty((100,))
    for latNode in range(100):
        # Prepare data
        # Fit curve (double peaks)
        try:
            #
            ppredSingle, _ = curve_fit(gabor_function_single, allITDs, layerActivations[:,latNode], p0 = [0.0005,0,1,0,1000], 
                                    bounds=((0.0001,-1000,0,-np.inf,500), (0.0009,1000,np.inf,np.inf,np.inf)), maxfev=5000)
            allGaborFit_Single_thresh[latNode,:]=ppredSingle
            #
            # stats
            yfit0=af.gabor_function_single(allITDs,*ppredSingle) 
            ydata = layerActivations[:,latNode]
            residuals=ydata-yfit0
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata-np.mean(ydata))**2)
            r_squared = 1 - (ss_res / ss_tot)
            allGaborFit_Single_R2[latNode] = r_squared
            # linear test
            linPars, _ = curve_fit(linear_fit, allITDs, layerActivations[:,latNode], p0 = [0, 0])
            lfit0= linear_fit(allITDs,*linPars) 
            ldata = layerActivations[:,latNode]
            residuals_lin=ldata-lfit0
            ss_res_lin = np.sum(residuals_lin**2)
            ss_tot_lin = np.sum((ldata-np.mean(ldata))**2)
            r_squared_lin = 1 - (ss_res_lin / ss_tot_lin)
            # F statistic
            df1 = len( allITDs ) - 2
            df2 = len( allITDs ) - 5
            F = ( (ss_res_lin-ss_res) / (df1 - df2) ) / (ss_res/df2)
            p_value = scipy.stats.f.cdf(F, df1 - df2, df2)
            allGaborFit_Single_p[latNode] = 1 - p_value
        except:
            allGaborFit_Single_thresh[latNode,:]=[0, 0, 0, 0, 0]
            allGaborFit_Single_R2[latNode] = 0
            allGaborFit_Single_p[latNode] = 1
    return allGaborFit_Single_thresh, allGaborFit_Single_R2, allGaborFit_Single_p

def gabor_function_single(ITD,BF,BITD,A,B,s):
    # gabor function
    P=[]
    for itd in ITD:
        singleGabor = A*np.exp((-((itd-BITD)**2)/(2*(s**2))))*np.cos(2*np.pi*BF*(itd-BITD)) + B
        P.append(singleGabor)
    return P

def linear_fit(ITD,m,c):
    # linear fit for statistics
    P=[]
    for itd in ITD:
        singleGabor = m*itd + c
        P.append(singleGabor)
    return P
    
def xcorr_activtations(layerActivations,allGaborFit_Single):
    
    # SET UP FIGURE
    fig = plt.figure(20,figsize=[5,1.75])
    fig.clf()
    allcols = cm.Dark2([1,.66,.33,0])
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    # PHASIC STIM PARAMS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    # FOR EACH PHASIC STIM
    ax = fig.add_subplot(1,3,1)
    for IPDIndex in [0,1,2,3]:
        tActMean = np.mean( layerActivations[IPDIndex,:,:], 0 )
        tCI = [mean_confidence_interval(layerActivations[IPDIndex,:,x]) for x in range(100)]
        tact2x,tact2y = moving_average(allGaborFit_Single[:,1],tActMean,100)
        ax.errorbar(allGaborFit_Single[:,1], 1+tActMean,yerr=tCI,linestyle='none',linewidth=.5,
                    marker='o', markersize=.5,color = (1-(1-allcols[IPDIndex])*0.55))
        # ax.plot(allGaborFit_Single[:,1], 1+tActMean,'.',linewidth=.25,zorder=2,color=allcols[IPDIndex])
        ax.plot(tact2x,[1+tact2y[k] for k in range(len(tact2y))],'-',linewidth=1.5,zorder=4,color=allcols[IPDIndex],label=allLabels[IPDIndex])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # ax.set_xlim([-1100,1100]); ax.set_ylim([0, 1.8]);
    ax.set_xlabel('bITD (µs)')
    ax.set_ylabel('Layer 2 node activations (+1)') 
    ax.set_ylim([0.35, 1.8]);
    # handles, labels = plt.gca().get_legend_handles_labels() 
    # by_label = dict(zip(labels, handles))
    # leg = ax.legend(by_label.values(), by_label.keys(),loc="lower center",fontsize=7,ncol=2)
    # leg.get_frame().set_linewidth(0.0)
    # fig.tight_layout()
    # NoSo - NoSpi
    ax = fig.add_subplot(1,3,2)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    tActMean = np.mean( layerActivations[1,:,:] , 0 ) - np.mean( layerActivations[0,:,:] , 0 )
    tCI = [mean_confidence_interval_diff(layerActivations[1,:,x],layerActivations[0,:,x]) for x in range(100)]
    tact2x,tact2y = moving_average(allGaborFit_Single[:,1],tActMean,100)
    ax.errorbar(allGaborFit_Single[:,1],tActMean,yerr=tCI,linestyle='none',linewidth=.5,
                    marker='o', markersize=.5,color = (1-(1-((allcols[0]+allcols[1])/2))*0.55))
    ax.plot(tact2x,tact2y,'-',linewidth=1.5,zorder=4,color=(allcols[0]+allcols[1])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); ax.set_ylim([-0.35, 0.35]);
    ax.set_xlabel('bITD (µs)')
    ax.set_ylabel('Change in layer 2 activations')
    ax.set_title('NoSπ - NoSo')
    # NpiSpi - NpiSo
    ax = fig.add_subplot(1,3,3)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    tActMean = np.mean( layerActivations[3,:,:] , 0 ) - np.mean( layerActivations[2,:,:] , 0 )
    tCI = [mean_confidence_interval_diff(layerActivations[3,:,x],layerActivations[2,:,x]) for x in range(100)]
    tact2x,tact2y = moving_average(allGaborFit_Single[:,1],tActMean,100)
    ax.errorbar(allGaborFit_Single[:,1],tActMean,yerr=tCI,linestyle='none',linewidth=.5,
                    marker='o', markersize=.5,color = (1-(1-((allcols[3]+allcols[2])/2))*0.55))
    ax.plot(tact2x,tact2y,'-',linewidth=1.5,zorder=4,color=(allcols[3]+allcols[2])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); ax.set_ylim([-0.35, 0.35]);
    ax.set_xlabel('bITD (µs)')
    ax.set_ylabel('Change in layer 2 activations')
    ax.set_title('NπSo - NπSπ')
    ##
    fig.tight_layout()
    
    # SET UP FIGURE
    fig = plt.figure(21,figsize=[5,2.5])
    fig.clf()
    allcols = cm.Dark2([1,.66,.33,0])
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    # # # # XCORR MODEL
     # LOAD .MAT DATA
    mat = spio.loadmat('supplementary/xcorrmodel.mat', squeeze_me=True)
    D = mat['delays'] # array
    A = mat['allSum'] # structure containing an array
    # PHASIC STIM PARAMS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    # FOR EACH PHASIC STIM
    ax = fig.add_subplot(2,3,1)
    for IPDIndex in [0,1,2,3]:
        ax.plot(D,A[IPDIndex,:],'-',linewidth=1.5,zorder=4,color=allcols[IPDIndex],label=allLabels[IPDIndex])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); # 
    mii = np.min(A); maa = np.max(A); di = maa - mii
    ax.set_ylim(mii-di*.05,maa+di*.05)
    ax.set_xlabel('Lag (µs)')
    ax.set_ylabel('X-corr algorithm')
    #
    ax.set_xticks([-1000, 0, 1000])
    ax.set_yticks([])
    #
    # NoSo - NoSpi
    ax = fig.add_subplot(2,3,2)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[1,:]-A[0,:],'-',linewidth=1.5,zorder=4,color=(allcols[0]+allcols[1])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([0])
    maa = np.max(abs(A[1,:]-A[0,:])); 
    ax.set_ylim(-maa*1.75,maa*1.75)
    ax.set_xlabel('Lag (µs)')
    ax.set_ylabel('Change in x-corr')
    ax.set_title('NoSπ - NoSo')
    # NpiSpi - NpiSo
    ax = fig.add_subplot(2,3,3)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[3,:]-A[2,:],'-',linewidth=1.5,zorder=4,color=(allcols[2]+allcols[3])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([0])
    maa = np.max(abs(A[3,:]-A[2,:])); 
    ax.set_ylim(-maa*1.75,maa*1.75)
    ax.set_xlabel('Lag (µs)')
    ax.set_ylabel('Change in x-corr')
    ax.set_title('NπSo - NπSπ')
    #
    
    
    # # # GILBERT
    # LOAD .MAT DATA
    mat = spio.loadmat('supplementary/fig_10_11_gilbert2015', squeeze_me=True)
    D = mat['bITDs'] # array
    A = mat['totalspikecounts'] # structure containing an array
    A[A<1]=np.nan
    # PHASIC STIM PARAMS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    # FOR EACH PHASIC STIM
    ax = fig.add_subplot(2,3,4)
    for IPDIndex in [0,1,2,3]:
        ax.plot(D,A[IPDIndex,:],'.-',linewidth=1.5,ms=4,zorder=4,color=allcols[IPDIndex],label=allLabels[IPDIndex])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); # 
    mii = np.nanmin(A); maa = np.nanmax(A); di = maa - mii
    ax.set_ylim(0,maa+di*.1)
    ax.set_xlabel('bITD (µs)')
    ax.set_ylabel('Spike count')
    #
    ax.set_xticks([-1000, 0, 1000])
    # ax.set_yticks([])
    #
    # NoSo - NoSpi
    ax = fig.add_subplot(2,3,5)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[1,:]-A[0,:],'.-',ms=4,linewidth=1.5,zorder=4,color=(allcols[0]+allcols[1])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([-10,0,10])
    maa = np.nanmax(abs(A[1,:]-A[0,:])); 
    ax.set_ylim(-maa*1.25,maa*1.25)
    ax.set_xlabel('lag (µs)')
    ax.set_ylabel('Change in spiking')
    ax.set_title('NoSπ - NoSo')
    # NpiSpi - NpiSo
    ax = fig.add_subplot(2,3,6)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[3,:]-A[2,:],'.-',ms=4,linewidth=1.5,zorder=4,color=(allcols[2]+allcols[3])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([-5,0,5])
    maa = np.nanmax(abs(A[3,:]-A[2,:])); 
    ax.set_ylim(-maa*1.25,maa*1.25)
    ax.set_xlabel('lag (µs)')
    ax.set_ylabel('Change in spiking')
    ax.set_title('NπSo - NπSπ')

    #
    fig.tight_layout()

def moving_average(x,y,wi=0):
    # gabor function
    mii = -1000
    maa = 1000
    stp =100
    #
    av_x = []
    av_y = []
    for s in range(mii,maa,stp):
        tLogic = (x>=(s-wi))*(x<(s+stp+wi))
        tMean = np.mean(y[tLogic])
        av_x.append(s+stp/2)
        av_y.append(tMean)

    return av_x, av_y

def xcorr_model():
    
    # LOAD .MAT DATA
    mat = spio.loadmat('supplementary/xcorrmodel.mat', squeeze_me=True)
    D = mat['delays'] # array
    A = mat['allSum'] # structure containing an array
    
    # SET UP FIGURE
    fig = plt.figure(80,figsize=[6.6,1.8])
    fig.clf()
    allcols = cm.Dark2([1,.66,.33,0])
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "Arial"
    # PHASIC STIM PARAMS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    # FOR EACH PHASIC STIM
    ax = fig.add_subplot(1,3,1)
    for IPDIndex in [0,1,2,3]:
        ax.plot(D,A[IPDIndex,:],'-',linewidth=1.5,zorder=4,color=allcols[IPDIndex],label=allLabels[IPDIndex])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); # 
    mii = np.min(A); maa = np.max(A); di = maa - mii
    ax.set_ylim(mii-di*.05,maa+di*.05)
    ax.set_xlabel('lag, µs')
    ax.set_ylabel('x-corr algorithm')
    #
    ax.set_xticks([-1000, 0, 1000])
    ax.set_yticks([])
    #
    # NoSo - NoSpi
    ax = fig.add_subplot(1,3,2)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[1,:]-A[0,:],'-',linewidth=1.5,zorder=4,color=(allcols[0]+allcols[1])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([0])
    maa = np.max(abs(A[1,:]-A[0,:])); 
    ax.set_ylim(-maa*1.75,maa*1.75)
    ax.set_xlabel('lag, µs')
    ax.set_ylabel('change in x-corr')
    ax.set_title('NoSπ-NoSo')
    # NpiSpi - NpiSo
    ax = fig.add_subplot(1,3,3)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[3,:]-A[2,:],'-',linewidth=1.5,zorder=4,color=(allcols[2]+allcols[3])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([0])
    maa = np.max(abs(A[3,:]-A[2,:])); 
    ax.set_ylim(-maa*1.75,maa*1.75)
    ax.set_xlabel('lag, µs')
    ax.set_ylabel('change in x-corr')
    ax.set_title('NπSo-NπSπ')
    #
    fig.tight_layout()
    
def xcorr_gilbert():
    
    # LOAD .MAT DATA
    mat = spio.loadmat('supplementary/fig_10_11_gilbert2015', squeeze_me=True)
    D = mat['bITDs'] # array
    A = mat['totalspikecounts'] # structure containing an array
    A[A<1]=np.nan
    
    # SET UP FIGURE
    fig = plt.figure(280,figsize=[6.6,1.9])
    fig.clf()
    allcols = cm.Dark2([1,.66,.33,0])
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "Arial"
    # PHASIC STIM PARAMS
    allIPDs = [[0,0],[np.pi,0],[np.pi,np.pi],[0,np.pi]] # [tone IPD, noise IPD]
    lenIPDs = len(allIPDs)
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    # FOR EACH PHASIC STIM
    ax = fig.add_subplot(1,3,1)
    for IPDIndex in [0,1,2,3]:
        ax.plot(D,A[IPDIndex,:],'.-',linewidth=1.5,ms=6,zorder=4,color=allcols[IPDIndex],label=allLabels[IPDIndex])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); # 
    mii = np.nanmin(A); maa = np.nanmax(A); di = maa - mii
    ax.set_ylim(0,maa+di*.01)
    ax.set_xlabel('bITD, µs')
    ax.set_ylabel('spike counts')
    #
    ax.set_xticks([-1000, 0, 1000])
    # ax.set_yticks([])
    #
    # NoSo - NoSpi
    ax = fig.add_subplot(1,3,2)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[1,:]-A[0,:],'.-',ms=6,linewidth=1.5,zorder=4,color=(allcols[0]+allcols[1])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([-10,0,10])
    maa = np.nanmax(abs(A[1,:]-A[0,:])); 
    ax.set_ylim(-maa*1.25,maa*1.25)
    ax.set_xlabel('lag, µs')
    ax.set_ylabel('change in spiking')
    ax.set_title('NoSπ-NoSo')
    # NpiSpi - NpiSo
    ax = fig.add_subplot(1,3,3)
    ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
    ax.plot(D,A[3,:]-A[2,:],'.-',ms=6,linewidth=1.5,zorder=4,color=(allcols[2]+allcols[3])/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1100,1100]); 
    ax.set_yticks([-5,0,5])
    maa = np.nanmax(abs(A[3,:]-A[2,:])); 
    ax.set_ylim(-maa*1.25,maa*1.25)
    ax.set_xlabel('lag, µs')
    ax.set_ylabel('change in spiking')
    ax.set_title('NπSo-NπSπ')
    #
    fig.tight_layout()


def xcorr_latent_bars(latentActivations,polarityCorrect):
    nLatNodes = 6
    # SET UP FIGURE
    fig = plt.figure(30,figsize=[1.55,6])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, 10))
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "Arial"
    #
    #
    labels_BMLD = ['NoSo', 'NoSπ', 'NπSπ', 'NπSo']
    x = np.arange(len(labels_BMLD))  # the label locations
    barwidth = 0.8
    #
    for nodeIndex in range(nLatNodes):

        ax = fig.add_subplot(nLatNodes, 1, nodeIndex+1)
        #
        ax.axhline(y=0,linewidth=.5, ls='-',color=(.5,.5,.5),zorder=1)
        #
        tbarVals = np.mean(latentActivations[:,:,nodeIndex],1)
        tCI = [mean_confidence_interval(latentActivations[x,:,nodeIndex]) for x in range(4)]
        if polarityCorrect[nodeIndex] == 1:
            bar1 = ax.bar(x, tbarVals, barwidth, yerr = tCI, color=allcols[nodeIndex], error_kw=dict(ecolor=[x*0.8 for x in allcols[nodeIndex]]) )
            tmii = np.min( tbarVals ); tmaa = np.max( tbarVals );
        else:
            bar1 = ax.bar(x, -tbarVals, barwidth, yerr = tCI, color='none', edgecolor=allcols[nodeIndex], error_kw=dict(ecolor=[x*0.8 for x in allcols[nodeIndex]]) , hatch="XXXXX")
            tmii = np.min( -tbarVals ); tmaa = np.max( -tbarVals );
        #
        ax.set_xticks([])
        ax.set_xticks(x,(' ',' ','',''))
        # ax.set_yticks([0])
        if nodeIndex == (nLatNodes-1):
            ax.set_xticks(x)
            ax.set_xticklabels(labels_BMLD,rotation=40)
        tdi = tmaa - tmii;
        ax.set_xlim(-.75,3.75); ax.set_ylim(tmii-tdi*.4,tmaa+tdi*.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    #
    fig.tight_layout()
 
    
def ITD_tuning3(IPDsForRLF,allActivations,allGaborFitParams,KLDivs,stimtype,toneFreq,polarityCorrect):
    #
    nLatNodes = len(allGaborFitParams)
    ITDs = (IPDsForRLF/(2*np.pi)) * (1e6/toneFreq)
    # SORT KLDivs
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    #
    fig = plt.figure(44,figsize=[2,6.6])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "Arial"
    #
    ax = fig.add_subplot(1,1,1)
    #
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('normalized responses of ' + r"$\bf{layer}$" + ' ' + r"$\bf{3}$" + ' nodes')    
    ax.set_xticks([])
    ax.set_yticks([])
    #
    for nodeIndex in range(nLatNodes):

        ax = fig.add_subplot(nLatNodes, 1, nodeIndex+1)
        #
        ITDlim = 655.5
        ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        #
        #
        tActivationMean = np.mean(allActivations[:,:,nodeIndex],1)
        tCI = [mean_confidence_interval(allActivations[x,:,nodeIndex]) for x in range(len(ITDs))]
        # normalise
        tmii = np.min(tActivationMean); tmaa = np.max(tActivationMean); tdii = tmaa - tmii
        tActivationMean = (tActivationMean - tmii) / tdii
        tCI = tCI / tdii
        #
        if polarityCorrect[nodeIndex]==1:
            ax.plot(ITDs,tActivationMean,'-',color=allcols[nodeIndex],linewidth=2,zorder=5)
        else:
            ax.plot(ITDs,1-tActivationMean,':',dashes=(1, 1),color=allcols[nodeIndex],linewidth=2,zorder=5)
        # # Gabor fits
        # tParams = allGaborFitParams[0][nodeIndex,:]
        # tGaborFit = af.gabor_function_double(ITDs,*tParams)
        # if polarityCorrect[nodeIndex]==1:
        #     tGaborFit = [(i - tmii) / tdii for i in tGaborFit]
        #     ax.plot(ITDs,tGaborFit,'-',color=(.0,.0,.0),linewidth=1,zorder=3)
        # else:
        #     tGaborFit = [1 - (i - tmii) / tdii for i in tGaborFit]
        #     ax.plot(ITDs,tGaborFit,'-',color=(.0,.0,.0),linewidth=1,zorder=3)
        # # Single Gabor fits (MAKE THIS A SYMMETRICAL DISTRIBUTION)
        # tParams = allGaborFitParams[0][nodeIndex,:]
        # tGaborFit = af.gabor_function_single(ITDs,tParams[0],tParams[2],tParams[3],tParams[4],tParams[5])
        # if polarityCorrect[nodeIndex]==1:
        #     tGaborFit = [(i - tmii) / tdii for i in tGaborFit]
        #     ax.plot(ITDs,tGaborFit,'-',color=(.0,.0,.0),linewidth=1,zorder=3)
        # else:
        #     tGaborFit = [1 - (i - tmii) / tdii for i in tGaborFit]
        #     ax.plot(ITDs,tGaborFit,'-',color=(.0,.0,.0),linewidth=1,zorder=3)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-0.02,1.02);ax.set_xlim(-1100,1100)
        ymi = np.min(tActivationMean); yma = np.max(tActivationMean); ydi = yma-ymi
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1]); ax.set_yticklabels([])
        ax.set_xticks([-1000,0,1000])
        if nodeIndex == (nLatNodes-1):
            ax.set_xlabel('noise ITD, µs')
            ax.set_xticks([-1000,0,1000])
        else:
            ax.set_xticklabels([])       
    fig.tight_layout()
    
    
def ITD_tuning4(IPDsForRLF,allActivations,allGaborFitParams,KLDivs,stimtype,toneFreq,polarityCorrect):
    #
    nLatNodes = len(allGaborFitParams)
    ITDs = (IPDsForRLF/(2*np.pi)) * (1e6/toneFreq)
    # SORT KLDivs
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    #
    fig = plt.figure(4,figsize=[1.7,6.6])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.family"] = "Arial"
    #
    #
    for nodeIndex in range(nLatNodes):

        ax = fig.add_subplot(nLatNodes, 1, nodeIndex+1)
        #
        ITDlim = 655.5
        ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        #
        if polarityCorrect[nodeIndex] ==1:
            lwww1= 1.5
            lwww2 = 1
        else:
            lwww1 = .5
            lwww2 = 2.5
        #
        tActivationMean = np.mean(allActivations[:,:,nodeIndex],1)
        tCI = [mean_confidence_interval(allActivations[x,:,nodeIndex]) for x in range(len(ITDs))]
        # normalise
        tmii = np.min(tActivationMean); tmaa = np.max(tActivationMean); tdii = tmaa - tmii
        tActivationMean = (tActivationMean - tmii) / tdii
        tCI = tCI / tdii
        #
        ax.fill_between(ITDs, (tActivationMean-tCI), (tActivationMean+tCI), facecolor=(0.5,0.5,0.5), edgecolor = 'none', alpha=.4,zorder=1)
        ax.fill_between(ITDs, (1-tActivationMean-tCI), (1-tActivationMean+tCI), facecolor=(0.5,0.5,0.5), edgecolor = 'none', alpha=.4,zorder=1)
        ax.plot(ITDs,tActivationMean,'-',color='k',linewidth=lwww1,zorder=2)
        ax.plot(ITDs,1-tActivationMean,':',dashes=(1, 1),color='k',linewidth=lwww2,zorder=2)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-0.02,1.02);ax.set_xlim(-1000,1000)
        ymi = np.min(tActivationMean); yma = np.max(tActivationMean); ydi = yma-ymi
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0,1])
        ax.set_xticks([-1000,0,1000])
        if nodeIndex == (nLatNodes-1):
            ax.set_xlabel('noise ITD, µs')
            ax.set_xticks([-1000,0,1000])
        else:
            ax.set_xticklabels([])       
    fig.tight_layout()
    
    
def gilbert_RLFs():
    
    # SET UP FIGURE PARAMS
    fig = plt.figure(77,figsize=[4.15, 1.75])
    fig.clf()
    allcols = cm.Dark2([1,.66,.33,0])
    titles = ['NoSπ-NoSo','NπSo-NπSπ']
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    plt.rcParams.update({'font.size':10})
    plt.rcParams["font.family"] = "Arial"
    
    # DATA in Gilbert et al. (2015)       
    toneAmpGilbert = np.arange(0,100+1e-10,5)
    sumRLF_NoSo = np.load('supplementary/GilbertData_sumRLF_NoSo.npy')
    sumRLF_NoSpi = np.load('supplementary/GilbertData_sumRLF_NoSpi.npy')
    sumRLF_NpiSpi = np.load('supplementary/GilbertData_sumRLF_NpiSpi.npy')
    sumRLF_NpiSo = np.load('supplementary/GilbertData_sumRLF_NpiSo.npy')
    
    #
    ax = fig.add_subplot(1,2,1)
    ax.fill_between(toneAmpGilbert,sumRLF_NoSo,sumRLF_NoSpi,color=(allcols[0]+allcols[1])/2,alpha=.25)
    ax.plot(toneAmpGilbert,sumRLF_NoSo,'-',linewidth=2,zorder=4,color=allcols[0],label=allLabels[0])
    ax.plot(toneAmpGilbert,sumRLF_NoSpi,'-',linewidth=2,zorder=4,color=allcols[1],label=allLabels[1])
    ax.arrow(66,67,0,-16,color=(allcols[0]+allcols[1])/2,length_includes_head=True,
          head_width=3.5, head_length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 100]) 
    ax.set_ylim([0,max(sumRLF_NoSo)*1.15])
    ax.set_xlabel('Tone level (dB SPL)')
    ax.set_ylabel('total spike count')
    ax.set_xticks([0,100])
    # ax.set_yticks([])
    handles, labels = plt.gca().get_legend_handles_labels() 
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="lower left",fontsize=7.5,ncol=1)
    leg.get_frame().set_linewidth(0.0)
    #
    ax = fig.add_subplot(1,2,2)
    ax.fill_between(toneAmpGilbert,sumRLF_NpiSpi,sumRLF_NpiSo,color=(allcols[2]+allcols[3])/2,alpha=.25)
    ax.plot(toneAmpGilbert,sumRLF_NpiSpi,'-',linewidth=2,zorder=4,color=allcols[2],label=allLabels[2])
    ax.plot(toneAmpGilbert,sumRLF_NpiSo,'-',linewidth=2,zorder=4,color=allcols[3],label=allLabels[3])
    ax.arrow(70,43,0,14.7,color=(allcols[2]+allcols[3])/2,length_includes_head=True,
          head_width=3.5, head_length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 100])
    ax.set_ylim([0,max(sumRLF_NpiSo)*1.15])
    ax.set_xlabel('tone level (GP ACtx), dB SPL')
    ax.set_xticks([0,100])
    ax.set_yticklabels([])
    handles, labels = plt.gca().get_legend_handles_labels() 
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="upper left",fontsize=7.5,ncol=1)
    leg.get_frame().set_linewidth(0.0)
    #
    fig.tight_layout()
    
    
def ITDandRLFs(IPDsForRLF,allActivations_noiseITD,allGaborFitParams_noiseITD,toneAmpPsychometric,allActivations, allRegressionData, bestScale, KLDivs, polarityCorrect): 
    
    # KL DIVERGENCE
    _, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    
    # SET UP FIGURE PARAMS
    fig = plt.figure(7,figsize=[3.2, 5.3])
    fig.clf()
    allcols = cm.Dark2([1,.66,.33,0]) # allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    titles = ['NoSπ-NoSo','NπSo-NπSπ']
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    
    # DATA in Gilbert et al. (2015)       
    toneAmpGilbert = np.arange(0,100+1e-10,5)
    sumRLF_NoSo = np.load('supplementary/GilbertData_sumRLF_NoSo.npy')
    sumRLF_NoSpi = np.load('supplementary/GilbertData_sumRLF_NoSpi.npy')
    sumRLF_NpiSpi = np.load('supplementary/GilbertData_sumRLF_NpiSpi.npy')
    sumRLF_NpiSo = np.load('supplementary/GilbertData_sumRLF_NpiSo.npy')
    DTsInGilbertPaper = []
    allLabels = ['NoSo', 'NoSπ' ,'NπSπ' ,'NπSo']
    # PLOT GILBERT RLFS
    ax = fig.add_subplot(len(latNodeKL)+1,3,2)
    ax.fill_between(toneAmpGilbert,sumRLF_NoSo,sumRLF_NoSpi,color=(allcols[0]+allcols[1])/2,alpha=.25)
    ax.plot(toneAmpGilbert,sumRLF_NoSo,'-',linewidth=2,zorder=4,color=allcols[0],label=allLabels[0])
    ax.plot(toneAmpGilbert,sumRLF_NoSpi,'-',linewidth=2,zorder=4,color=allcols[1],label=allLabels[1])
    # ax.arrow(66,67,0,-16,color=(allcols[0]+allcols[1])/2,length_includes_head=True,
          # head_width=3.5, head_length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([15, 100]) 
    ax.set_ylim([20,90])
    # ax.set_xlabel('tone level (GP ACtx), dB SPL')
    # ax.set_ylabel('total spike count')
    ax.set_xticks([15,100])
    # handles, labels = plt.gca().get_legend_handles_labels() 
    # by_label = dict(zip(labels, handles))
    # leg = ax.legend(by_label.values(), by_label.keys(),loc="upper",fontsize=6,ncol=2)
    # leg.get_frame().set_linewidth(0.0)
    #
    ax = fig.add_subplot(len(latNodeKL)+1,3,3)
    ax.fill_between(toneAmpGilbert,sumRLF_NpiSpi,sumRLF_NpiSo,color=(allcols[2]+allcols[3])/2,alpha=.25)
    ax.plot(toneAmpGilbert,sumRLF_NpiSpi,'-',linewidth=2,zorder=4,color=allcols[2],label=allLabels[2])
    ax.plot(toneAmpGilbert,sumRLF_NpiSo,'-',linewidth=2,zorder=4,color=allcols[3],label=allLabels[3])
    # ax.arrow(70,43,0,14.7,color=(allcols[2]+allcols[3])/2,length_includes_head=True,
          # head_width=3.5, head_length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([15, 100])
    ax.set_ylim([20,90])
    # ax.set_xlabel('tone level (GP ACtx), dB SPL')
    ax.set_xticks([15,100])
    ax.set_yticklabels([])
    # handles, labels = plt.gca().get_legend_handles_labels() 
    # by_label = dict(zip(labels, handles))
    # leg = ax.legend(by_label.values(), by_label.keys(),loc="upper",fontsize=6,ncol=2)
    # leg.get_frame().set_linewidth(0.0)
    
    # # ITD PART
    # Plot 1 (comparative GP)
    ax = fig.add_subplot(len(latNodeKL)+1,3,1)
    ax.axvspan(-112.5, 112.5, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
    mat = spio.loadmat('supplementary/GP_ACtx3.mat', squeeze_me=True)
    xq = mat['timedelays_GPACtx3'] # array
    yq = mat['spikes_GPACtx3'] # structure containing an arra
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-0.02,1.98);ax.set_xlim(-1000,1000)
    yq2 = yq[::-1]
    ax.plot(xq,yq,'-',color='r',linewidth=1,zorder=2)
    ax.plot(xq,yq2,'-',color='b',linewidth=1,zorder=2)
    ax.plot(xq,[yq[x]+yq2[x] for x in range(len(xq))],'-',color='k',linewidth=1,zorder=2)
    # actual DNN nodes
    ITDs = (IPDsForRLF/(2*np.pi)) * (1e6/500)
    for nodeIndex in range(nLatNodes):
        ax = fig.add_subplot(len(latNodeKL)+1,3,4+3*nodeIndex)
        ITDlim = 655.5
        ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        if polarityCorrect[nodeIndex] ==1:
            lwww1= 1.5
            lwww2 = 1
        else:
            lwww1 = .5
            lwww2 = 2.5
        tActivationMean = np.mean(allActivations_noiseITD[:,:,nodeIndex],1)
        tCI = [mean_confidence_interval(allActivations_noiseITD[x,:,nodeIndex]) for x in range(len(ITDs))]
        # normalise
        tmii = np.min(tActivationMean[10:31]); tmaa = np.max(tActivationMean[10:31]); tdii = tmaa - tmii
        tActivationMean = (tActivationMean - tmii) / tdii
        tCI = tCI / tdii
        #
        ax.fill_between(ITDs, (tActivationMean-tCI), (tActivationMean+tCI), facecolor=(0.5,0.5,0.5), edgecolor = 'none', alpha=.4,zorder=1)
        ax.fill_between(ITDs, (1-tActivationMean-tCI), (1-tActivationMean+tCI), facecolor=(0.5,0.5,0.5), edgecolor = 'none', alpha=.4,zorder=1)
        ax.plot(ITDs,tActivationMean,'-',color='k',linewidth=lwww1,zorder=2)
        ax.plot(ITDs,1-tActivationMean,':',dashes=(1, 1),color='k',linewidth=lwww2,zorder=2)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-0.02,1.02);ax.set_xlim(-1000,1000)
        ymi = np.min(tActivationMean); yma = np.max(tActivationMean); ydi = yma-ymi
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0,1])
        ax.set_xticks([-1000,0,1000])
        if nodeIndex == (nLatNodes-1):
            ax.set_xlabel('Noise ITD (µs)',fontsize=7)
            ax.set_xticks([-1000,0,1000])
        else:
            ax.set_xticklabels([])     
            
            
    # PLOT LATENT RLFS # # # 
    for No_OR_Npi in range(2):
        for nodeIndex in range(len(latNodeKL)):
            # GRAB FUNCTION
            # Subp[lot]
            ax = fig.add_subplot(len(latNodeKL)+1,3,3+(nodeIndex*3)+2+No_OR_Npi)
            # ACTIVATION
            tx = toneAmpPsychometric
            meanActivation = np.mean(allActivations,2)
            ty1 = meanActivation[(2*No_OR_Npi)+1,:,nodeIndex]
            ty2 = meanActivation[(2*No_OR_Npi),:,nodeIndex]
            tCI1 = [mean_confidence_interval(allActivations[2*No_OR_Npi+1,x,:,nodeIndex]) for x in range(len(toneAmpPsychometric))]
            tCI2 = [mean_confidence_interval(allActivations[2*No_OR_Npi,x,:,nodeIndex]) for x in range(len(toneAmpPsychometric))]
            ax.fill_between(tx, (ty1-tCI1), (ty1+tCI1), facecolor=allcols[2*No_OR_Npi+1], edgecolor = 'none', alpha=.4)
            ax.fill_between(tx, (ty2-tCI2), (ty2+tCI2), facecolor=allcols[2*No_OR_Npi], edgecolor = 'none', alpha=.4)
            if polarityCorrect[nodeIndex]==1:
                ax.plot(tx,ty1,'-',color=allcols[2*No_OR_Npi+1],linewidth=2.5)
                ax.plot(tx,ty2,'-',color=allcols[2*No_OR_Npi],linewidth=2.5)
            else:
                ax.plot(tx,ty1,':',dashes=(1, .5),color=allcols[2*No_OR_Npi+1],linewidth=2.5)
                ax.plot(tx,ty2,':',dashes=(1, .5),color=allcols[2*No_OR_Npi],linewidth=2.5)
            ax.spines['top'].set_visible(False)
            ax.set_xticks([15,50])
            if No_OR_Npi == 1:
                # ax.set_yticks([])
                ax.set_yticklabels([])
            if nodeIndex != 5:
                ax.set_xticklabels([])
            mii = np.min(meanActivation[:,:,nodeIndex])
            maa = np.max(meanActivation[:,:,nodeIndex])
            dii = maa - mii
            ax.set_xlim([15,50])
            ax.set_ylim([mii-dii*0.2, maa+dii*0.2])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    fig.tight_layout()
    
    #% YLABEL
    fig = plt.figure(76,figsize=[3.5, 8])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    ax = fig.add_subplot(4,1,1)
    ax.set_xlabel('Tone level, dB SPL')
    ax.set_ylabel('Layer 3 node activations')  
    ax = fig.add_subplot(4,1,2)
    ax.set_xlabel('Tone level, dB') 
    ax.set_ylabel('Layer 3 normalized responses')
    ax = fig.add_subplot(4,1,3)
    ax.set_ylabel("Normalized \n discharge rate")
    ax = fig.add_subplot(4,1,4)
    ax.set_ylabel("Spike count")
    fig.tight_layout()
    


def layer23weights(net,allIPDs,toneFreq,allGaborFit_Single,polarityCorrect,KLDivs):   
    
    # GRAB WEIGHTS FROM NETWORK
    stimParams = {"toneFreq": 500,
              "toneAmp": -100,
              "noiseAmp": 60,
              "transformType": "phasic",
              'toneAziPhase': 0,
              'noiseAziPhase': 0}
    allITDs = (allIPDs/(2*np.pi)) * (1e6/toneFreq)
    geninp = edq.BMLD_data(stimParams,drawN=10,Fs=20000)
    weights = net.run(geninp[0],"encoder/2th_enc_layer/w:0")
    weights = weights[:,10:]
    bias = net.run(geninp[0],"encoder/2th_enc_layer/b:0")
    bias = bias[10:]
    # LAYER 3 NODE ORDERS
    latNodeIndex, latNodeKL = af.KLDiv_sort(KLDivs)
    nLatNodes = len(latNodeKL)
    # INITIALIZE FIGURES
    fig = plt.figure(94,figsize=[4,2.5])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({'font.size': 7})
    # INITIALIZE SUMMARY ARRAY
    allWeightPop = np.empty((40,nLatNodes))
    # FOR ALL NODES
    for nodeIndex in range(nLatNodes):
        # SUBPLOT
        ax = fig.add_subplot(2, 5, nodeIndex+1)
        # HEAD REGION
        ITDlim = 655.5
        ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        # GRAB WEIGHTS CONNECTING TO A GIVEN LAYER 3 NODE
        bITD = allGaborFit_Single[:,1]
        tnodeweighting = weights[:,latNodeIndex[nodeIndex]]
        # POLARITY CORRECT
        if polarityCorrect[nodeIndex] == 1:
            tWeights = tnodeweighting
            # tWeights = allGaborFit_Single[:,2]*tnodeweighting
        else:
            tWeights = -tnodeweighting
            tWeights = -allGaborFit_Single[:,2]*tnodeweighting
        # +VE CORRECT
        for nn in range(100): # this copmensates for negative weighting (by making positive for an antinphase cosine)
            if tWeights[nn]<0:
                tWeights[nn]  = - tWeights[nn]
                if bITD[nn] < 0:
                    bITD[nn] = allGaborFit_Single[nn,1] + (1/(allGaborFit_Single[nn,0]))*0.5
                else:
                    bITD[nn] = allGaborFit_Single[nn,1] - (1/(allGaborFit_Single[nn,0]))*0.5   
        # PLOT
        ax.plot(bITD,tWeights,'.',color=1-(1-allcols[nodeIndex])*0.6,linewidth=.1,ms=2.5,zorder=2)
        # LINE FIT
        av_x, av_y = moving_average2(bITD,tWeights,200)
        if polarityCorrect[nodeIndex] == 1:
            ax.plot(av_x, av_y,'-',color=allcols[nodeIndex],linewidth=1,zorder=2)
        else:
            ax.plot(av_x, av_y,'--',color=allcols[nodeIndex],linewidth=1,zorder=2)
        #
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([-1000,0,1000])
        # UPDATE
        allWeightPop[:,nodeIndex] = ( av_y / np.sum(av_y) )
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("noise bITD, µs",fontsize=10)
    plt.ylabel(r"$\bf{layer}$" + ' ' + r"$\bf{2/3}$" + ' adjusted weights',fontsize=10)
    fig.tight_layout()
    
    # SUMMARY
    fig = plt.figure(54,figsize=[3,2.45])
    fig.clf()
    allcols = cm.plasma(np.linspace(0, 1, len(KLDivs)))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(1,1,1)
    # # GP DISTRIBUTION
    # xvs = [-700,-600,-500,-300,-200,-100,-50,0,50,100,150,200,250,300,350,450,550]
    # yvs_0 = [1,1,1,2,3,1,1,5,9,13,14,7,16,13,1,1,1]
    # yvs = [yvs_0[i]/1000 for i in range(len(yvs_0))]
    #     # invert population
    # flipxvs = [-x for x in xvs]
    # bixvs = np.unique([flipxvs,xvs])
    # biyvs = []
    # for b in range(0,len(bixvs)):
    #     tbs = bixvs[b]
    #     if tbs in flipxvs:
    #         tin = flipxvs.index(tbs)
    #         biyvs.append(yvs[tin])
    #     else:
    #         biyvs.append(0)   
    # revv = biyvs[::-1]
    #     # plot
    # width = 40
    # ax.bar(bixvs,revv,width,color=(250/256,210/256,208/256),label='GP ACtx (right)')
    # ax.bar(bixvs,biyvs,width,bottom=revv,color='none', hatch="///", edgecolor=(174/256,197/256,228/256), ecolor=(174/256,197/256,228/256), label='GP ACtx (inverted)')
    #     # legend
    # handles, labels= plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # leg = ax.legend(by_label.values(), by_label.keys(),loc="lower left",fontsize=5,ncol=1)
    # leg.get_frame().set_linewidth(0.0)
    # SUMMARY PDF
    ITDlim = 655.5
    ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
    for nodeIndex in range(nLatNodes):
        if polarityCorrect[nodeIndex] == 1:
            ax.plot(av_x, allWeightPop[:,nodeIndex],'-',color=allcols[nodeIndex],linewidth=.75,zorder=2)
        else:
            ax.plot(av_x, allWeightPop[:,nodeIndex],'--',color=allcols[nodeIndex],linewidth=.75,zorder=2)
    ax.plot(av_x, np.mean( allWeightPop , 1 ),'-',color=(0,0,0),linewidth=2,zorder=6)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('noise bITD, µs')
    ax.set_ylabel('probability density')
    ax.set_xticks([-1000,0,1000])
    ax.set_yticks([])
    fig.tight_layout()

def moving_average2(x,y,wi=0):
    # gabor function
    mii = -1000
    maa = 1000
    stp = 50
    #
    av_x = []
    av_y = []
    for s in range(mii,maa,stp):
        tLogic = (x>=(s-wi))*(x<(s+stp+wi))# *(y>np.mean(y))
        tMean = np.sum(y[tLogic]) / (wi/stp)
        av_x.append(s+stp/2)
        av_y.append(tMean)

    return av_x, av_y   

#%% New plots 04/12/20
    
def plot_earlyITDtuning2(allIPDs,toneFreq,layerActivations,layer,unitranks,moreSubPlots = 0):
    
    # Get gabor fits (i.e. bITDs)
    allITDs = (allIPDs/(2*np.pi)) * (1e6/toneFreq)
    allGaborFit_Single, tR2, tP = fit_gabor(allITDs,toneFreq,layerActivations)
    R2order = np.argsort( tR2 )
    # Setup fig
    fig = plt.figure(18+layer,figsize=[6.175, 1.45])
    fig.clf()
    plt.rcParams.update({'font.size': 7})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
        # bestITD colors
    rvb = cm.get_cmap('viridis_r', 100)
    # import matplotlib.colors as mcolors
    # clist = [(0, "blue"), (0.25, "green"), (0.75, "orange"), (1, "red")]
    # rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
   
    # EXAMPLE 1
    unit_indx1 = unitranks[0]
    ax = fig.add_subplot(1,11,(6,8))
        # head space
    ITDlim = 655.5
    ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        # ITD function
    tActivationMean = layerActivations[:,R2order[unit_indx1]]
    ax.plot(allITDs,tActivationMean,'k-',linewidth=1,zorder=2)
        # Gabor fit
    if tP[unit_indx1] < 0.05:
        tParams = allGaborFit_Single[R2order[unit_indx1],:]
        tGaborFit = af.gabor_function_single(allITDs,*tParams)
        gabCol = rvb((allGaborFit_Single[R2order[unit_indx1],1]+1000)/2000)
        ax.plot(allITDs,tGaborFit,'--',color=gabCol,linewidth=1,zorder=1)
        tPeak = af.gabor_function_single([allGaborFit_Single[R2order[unit_indx1],1]],*tParams);
        ax.plot(allGaborFit_Single[R2order[unit_indx1],1],tPeak,'o',linewidth=1.5,zorder=1,mew=1,markersize=6,color=gabCol,mfc=gabCol,mec=gabCol)
        # Format
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    mii = min(tActivationMean[10:31]); maa = max(tActivationMean[10:31]); dii = maa - mii
    ax.set_yticks([mii, maa]); ax.set_yticklabels(['0', '1'])
    ax.set_ylim(mii-dii*.05, maa+dii*.12)
    ax.set_xlim(-1000,1000)
    ax.set_xticks([-1000,0,1000])
    ax.set_xlabel('Noise ITD (µs)')  
    ax.set_ylabel('Normalized response')
    
    # EXAMPLE 2
    unit_indx1 = unitranks[1]
    ax = fig.add_subplot(1,11,(9,11))
        # head space
    ITDlim = 655.5
    ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
        # ITD function
    tActivationMean = layerActivations[:,R2order[unit_indx1]]
    ax.plot(allITDs,tActivationMean,'k-',linewidth=1,zorder=2)
        # Gabor fit
    if tP[unit_indx1] < 0.05:
        tParams = allGaborFit_Single[R2order[unit_indx1],:]
        tGaborFit = af.gabor_function_single(allITDs,*tParams)
        gabCol = rvb((allGaborFit_Single[R2order[unit_indx1],1]+1000)/2000)
        ax.plot(allITDs,tGaborFit,'--',color=gabCol,linewidth=2,zorder=1)
        tPeak = af.gabor_function_single([allGaborFit_Single[R2order[unit_indx1],1]],*tParams);
        ax.plot(allGaborFit_Single[R2order[unit_indx1],1],tPeak,'o',linewidth=1.5,zorder=1,mew=1,markersize=6,color=gabCol,mfc=gabCol,mec=gabCol)
        # Format
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    mii = min(tActivationMean[10:31]); maa = max(tActivationMean[10:31]); dii = maa - mii
    ax.set_yticks([mii, maa]); ax.set_yticklabels(['0', '1'])
    ax.set_ylim(mii-dii*.05, maa+dii*.12)
    ax.set_xlim(-1000,1000)
    ax.set_xticks([-1000,0,1000])
    ax.set_xlabel('Noise ITD (µs)')  
    ax.set_ylabel('Normalized response')
   
    # Bar chart
        # colors for barchart
    ax = fig.add_subplot(1,11,(1,4))
    for i in range(100):
        if tP[R2order[i]] < 0.05:
            ax.bar(i+1,tR2[R2order[i]],1,color = rvb((allGaborFit_Single[R2order[i],1]+1000)/2000) ) # color=[0/256,114/256,189/256])
        else:
            ax.bar(i+1,tR2[R2order[i]],1,color = [0,0,0]) 
            
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.set_yticks([0,1]) 
    ax.set_ylim(0,1)
    ax.set_xlim(0,101)
    ax.set_xticks([1, 50, 100])
    if layer == 0:
        ax.set_xlabel('Layer 1 nodes (ordered)')  
    else:
        ax.set_xlabel('Layer 2 nodes (ordered)')  
    ax.set_ylabel('Variance explained')
        # color bar
    ax = fig.add_subplot(1,11,5)
    ax.axis("off")
    cbaxes = fig.add_axes([0.35, 0.33, 0.015, 0.495]) 
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-1000,vmax=1000)
    sm = plt.cm.ScalarMappable(cmap=rvb, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbaxes, ticks=[-1000,0,1000])
    cb.ax.set_yticklabels(['-1000', '0', '+1000']) 
    cb.ax.tick_params(labelsize=6) 
    cb.ax.set_ylabel('Node bITD (μs)', 
                      rotation=90, fontsize=7, labelpad=0.2)
    ax.yaxis.set_label_position("left")
    cb.outline.set_visible(False)
    #
    fig.tight_layout()
    
    if moreSubPlots == 1:
         # Setup fig
        fig = plt.figure(180+layer,figsize=[2.94, 1.5])
        fig.clf()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = "Arial"
        cnt = 1
        for tUnit in [68,50,90,44,70,97]:
            # EXAMPLE 1
            unit_indx1 = tUnit
            ax = fig.add_subplot(2,3,cnt); 
            cnt = cnt + 1
                # head space
            ITDlim = 655.5
            ax.axvspan(-ITDlim, ITDlim, facecolor=(.9,.9,.9), edgecolor='none',zorder=0)
                # ITD function
            tActivationMean = layerActivations[:,R2order[unit_indx1]]
            ax.plot(allITDs,tActivationMean,'k-',linewidth=1,zorder=2)
                # Gabor fit
            if tP[unit_indx1] < 0.05:
                tParams = allGaborFit_Single[R2order[unit_indx1],:]
                tGaborFit = af.gabor_function_single(allITDs,*tParams)
                gabCol = rvb((allGaborFit_Single[R2order[unit_indx1],1]+1000)/2000)
                ax.plot(allITDs,tGaborFit,'--',color=gabCol,linewidth=1.5,zorder=1)
                tPeak = af.gabor_function_single([allGaborFit_Single[R2order[unit_indx1],1]],*tParams);
                ax.plot(allGaborFit_Single[R2order[unit_indx1],1],tPeak,'o',linewidth=1.5,zorder=1,mew=1,markersize=4,color=gabCol,mfc=gabCol,mec=gabCol)
                # Format
            ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
            mii = min(tActivationMean[10:31]); maa = max(tActivationMean[10:31]); dii = maa - mii
            if cnt > 4:
                ax.set_xticks([-1000,0,1000])
            else:
                ax.set_xticks([-1000,0,1000]); ax.set_xticklabels([' ', ' ',' '])
            if cnt == 2 or cnt == 5:
                ax.set_yticks([mii, maa]); ax.set_yticklabels(['0', '1'])
            else:
                ax.set_yticks([mii, maa]); ax.set_yticklabels([' ', ' '])
            ax.set_ylim(mii-dii*.05, maa+dii*.12)
            ax.set_xlim(-1000,1000)
        fig.tight_layout()
    
    return allGaborFit_Single, tR2, tP
    

def paramdist(allGaborFit_Single1,allGaborFit_Single2):
    
    # HISTOGRAM
    fig = plt.figure(20,figsize=[1.8,1.6])
    fig.clf()
    plt.rcParams.update({'font.size': 9})
    plt.rcParams["font.family"] = "Arial"
    # ax = fig.add_subplot(2,1,1)
    # plt.hist(allGaborFit_Single1[0][allGaborFit_Single1[2]<0.05,1],bins=20,range=[-1000, 1000],color='k')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.set_yticks([])
    # ax.set_yticklabels([])
    # ax.set_xlim(-1050,1050)
    # ax.set_xticks([-1000,0,1000])     
    # ax.set_xlabel('bITD, µs')
    # ax.set_ylabel('count, ' + r"$\bf{layer}$" + ' ' + r"$\bf{1}$")
    ax = fig.add_subplot(1,1,1)
    ITDlim = 655.5
    ax.axvspan(-ITDlim, ITDlim, facecolor=(.85,.85,.85), edgecolor='none',zorder=0)
    plt.hist(allGaborFit_Single2[0][allGaborFit_Single2[2]<0.05,1],bins=20,range=[-1000, 1000],color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-1050,1050)
    ax.set_xticks([-1000,0,1000])     
    ax.set_xlabel('bITD, µs')
    ax.set_ylabel('count (' + r"$\bf{layer}$" + ' ' + r"$\bf{2}$" + ")")
    #
    fig.tight_layout()
    
    # BF v bD
    fig = plt.figure(22,figsize=[2.75,1.6])
    fig.clf()
    plt.rcParams.update({'font.size': 9})
    plt.rcParams["font.family"] = "Arial"
    ax = fig.add_subplot(1,1,1)
    ax.plot(allGaborFit_Single1[0][allGaborFit_Single1[2]<0.05,1],allGaborFit_Single1[0][allGaborFit_Single1[2]<0.05,0]*1e6,
            marker='o',color=[.3,.3, 1],ms=3,linestyle='none',label=r"$\bf{layer}$" + ' ' + r"$\bf{1}$")
    ax.plot(allGaborFit_Single2[0][allGaborFit_Single2[2]<0.05,1],allGaborFit_Single2[0][allGaborFit_Single2[2]<0.05,0]*1e6,
            marker='s',color='k',mfc='none',ms=3,linestyle='none',label=r"$\bf{layer}$" + ' ' + r"$\bf{2}$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(410,620)
    ax.set_yticks([450,500,550])
    # ax.set_yticklabels([])
    ax.set_xlim(-1050,1050)
    ax.set_xticks([-1000,0,1000])     
    ax.set_xlabel('bITD, µs')
    ax.set_ylabel('frequency, Hz')
    handles, labels = plt.gca().get_legend_handles_labels() 
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),loc="upper left",fontsize=6,ncol = 2)
    # leg.get_frame().set_linewidth(0.0)
    fig.tight_layout()
    
    # 2ND LAYER SLOPES
    fig = plt.figure(23,figsize=[1.8,1.6])
    fig.clf()
    plt.rcParams.update({'font.size': 9})
    plt.rcParams["font.family"] = "Arial"
    ax = fig.add_subplot(1,1,1)
    # ALL SLOPES
    allSlope = []
    for i in range(100):
        if allGaborFit_Single2[2][i]<0.05:
            tbITD = allGaborFit_Single2[0][i,1]
            tfreq = allGaborFit_Single2[0][i,0]
            tquartcyc = (1/tfreq)/4
            thalfcyc = (1/tfreq)/2
            #
            tStart = tbITD - tquartcyc
            if tStart >= - 1000:
                allSlope.append(tStart)
            #
            tStart = tbITD + tquartcyc
            if tStart <= 1000:
                allSlope.append(tStart) 
    # plot
    ITDlim = 655.5
    ax.axvspan(-ITDlim, ITDlim, facecolor=(.85,.85,.85), edgecolor='none',zorder=0)
    plt.hist(allSlope,bins=20,range=[-1000, 1000],color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-1050,1050)
    ax.set_xticks([-1000,0,1000])     
    ax.set_xlabel('flanking slopes, µs')
    ax.set_ylabel('count (' + r"$\bf{layer}$" + ' ' + r"$\bf{2}$" + ")")
    #
    fig.tight_layout()
    
    return


