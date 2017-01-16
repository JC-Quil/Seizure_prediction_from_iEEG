### This script convert the sample signals into subsamples and calculate the features ###

# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get specific functions from some other python libraries
from math import floor, log
from scipy.stats import skew, kurtosis
from scipy.io import loadmat   # For loading MATLAB data (.dat) files
from scipy import signal
from pywt import wavedec
from time import time

# This code was adapted from a post from Deep to the Kaggle competition forum here: 
# https://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated/comments

# Convert .mat files into dictionaries
def convertMatToDictionary(path):
    
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return ndata


# Define the frequence bands
def defineEEGFreqs():
    
    '''
    EEG waveforms are divided into frequency groups related to mental activity.
    alpha waves = 8-14 Hz = Awake with eyes closed
    beta waves = 15-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to consciousness and/or perception
    theta waves = 4-7 Hz = Light sleep
    delta waves < 4 Hz = Deep sleep
    '''
    return (np.array([0.2, 4, 8, 15, 30, 45, 70, 180]))  # Frequency levels in Hz


# Calculates the FFT of the epoch signal. Removes the DC component and normalizes the area to 1
def calcNormalizedFFT(epoch, lvl, subsampLen, fs):
    
    lseg = np.round(subsampLen/fs*lvl).astype('int')
    #D = np.absolute(np.fft.fft(epoch, n = (lseg[-1]), axis=0))
    D = np.fft.fft(epoch, n =   (lseg[-1]), axis=0)
    #print "D", D.shape
    D[0,:]=0                          # set the DC component to zero
    D /= D.sum()                      # Normalize each channel               
    #print "fft", D.shape

    return D


# Calculate the power spectral density from the FFT
def calcPSD(fft_epoch):

    psd = np.power(np.abs(fft_epoch), 2)
    #print "PSD", psd.shape
    return psd


# Compute the relative PSD for each frequency band
def calcPSD_band(psd_epoch, lvl, subsampLen, nc, fs):

    e_tot = sum(psd_epoch)
    lseg = np.round(subsampLen/fs*lvl).astype('int')
    psd_band = np.zeros((len(lvl)-1,nc))
    for j in range(0, len(lvl)-1, 1):
        psd_band[j,:] = np.sum(psd_epoch[lseg[j]:lseg[j+1],:], axis=0)/e_tot

    #print psd_band
    return psd_band


# Computes Shannon Entropy of segments corresponding to frequency bands
def calcShannonEntropy(psd_epoch, lvl, subsampLen, nc, fs):
    
    e_tot = sum(psd_epoch)
    rpsd_epoch = psd_epoch/e_tot
    lseg = np.round(subsampLen/fs*lvl).astype('int')
    sentropy = np.zeros((len(lvl)-1,nc))
    for j in range(0, len(lvl)-1, 1):
        sentropy[j,:] = -1*np.sum(np.multiply(rpsd_epoch[lseg[j]:lseg[j+1],:],np.log(rpsd_epoch[lseg[j]:lseg[j+1],:])), axis=0)
    
    return sentropy


# Compute spectral edge frequency
def calcSpectralEdgeFreq(psd_epoch, lvl, subsampLen, nc, fs):
    
    sfreq = fs
    tfreq = 50
    ppow = 0.5
    topfreq = int(round(subsampLen/sfreq*tfreq))+1
    D = psd_epoch
    
    A = np.cumsum(D[:topfreq,:], axis = 0)
    B = A - (A.max(axis = 0)*ppow)
    C = np.argmin(np.abs(B), axis = 0)

    spedge = np.zeros(16)
    for i in range(16):
        spedge[i] = C[i]*sfreq/subsampLen

    return spedge


# Calculate cross-correlation matrix
def corr(data, type_corr):
    
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0

    w,v = np.linalg.eig(C)

    x = np.sort(w)
    x = np.real(x)

    return x


# Compute correlation matrix across channels
def calcCorrelationMatrixChan(epoch):
    
    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    type_corr = 'spearman'
    
    lxchannels = corr(data, type_corr)
    return lxchannels


# Compute correlation matrix across frequencies for each frequency bands
def calcCorrelationMatrixFreq(fft_epoch, lvl, subsampLen, nc, fs):
    
    # Calculate correlation matrix and its eigenvalues (b/w freq)
    dspect = calcPSD(fft_epoch)
    #data = pd.DataFrame(data=dspect[:round(lvl[-2]*subsampLen/fs),:])
    data = dspect[:int(round(lvl[-1]*subsampLen/fs)),:]
 
    lseg = np.round(subsampLen/fs*lvl).astype('int')
    lxfreqbands = np.zeros((len(lvl)-1,nc))
    type_corr = 'spearman'

    for j in range(0, len(lvl)-1, 1):
        
        lxfreqbands[j,:] = corr(pd.DataFrame(data[lseg[j]:lseg[j+1],:]), type_corr)
        
    return lxfreqbands


# Calculate Hjorth activity over epoch
def calcActivity(epoch):

    activity = np.var(epoch, axis=0)
    
    return activity


# Calculate the Hjorth mobility parameter over epoch
def calcMobility(epoch):

    # N.B. the sqrt of the variance is the standard deviation. So we just get std(dy/dt) / std(y)
    mobility = np.divide(
                        np.std(np.diff(epoch, axis=0)), 
                        np.std(epoch, axis=0))
    
    return mobility


# Calculate Hjorth complexity over epoch
def calcComplexity(epoch):

    complexity = np.divide(
        calcMobility(np.diff(epoch, axis=0)), 
        calcMobility(epoch))
        
    return complexity  


# /feature removed from the feature set
def hjorthFD(X, Kmax=3):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter. Kmax is basically the scale size or time offset.
     So you are going to create Kmax versions of your time series.
     The K-th series is every K-th time of the original series.
     This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    L = []
    x = []
    N = len(X)
    for k in range(1,Kmax):
        Lk = []
        
        for m in range(k):
            Lmk = 0
            for i in range(1,int(floor((N-m)/k))):
                Lmk += np.abs(X[m+i*k] - X[m+i*k-k])
                
            Lmk = Lmk*(N - 1)/floor((N - m) / k) / k
            Lk.append(Lmk)
            
        L.append(np.log(np.mean(Lk)))   # Using the mean value in this window to compare similarity to other windows
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s)= np.linalg.lstsq(x, L)  # Numpy least squares solution
    
    return p[0]

# /feature removed from the feature set
def petrosianFD(X, D=None): 
    """Compute Petrosian Fractal Dimension of a time series from either two 
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided, 
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function 
    because D may also be used by other functions whereas computing it here 
    again will slow down.
    
    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    
    if D is None:
        D = np.diff(X)   # Difference between one data point and the next
        
    N_delta= 0; #number of sign changes in derivative of the signal
    for i in range(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1

    n = len(X)
    
    # This code is a little more compact. It gives the same
    # result, but I found that it was actually SLOWER than the for loop
    #N_delta = sum(np.diff(D > 0)) 
    
    return np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))


# Calculate Katz fractal dimension
def katzFD(epoch):
    
    L = np.abs(epoch - epoch[0]).max()
    d = len(epoch)
    
    return (np.log(L)/np.log(d))


# Calculate Higuchi fractal dimension
def higuchiFD(epoch, Kmax = 8):
    '''
    Ported from https://www.mathworks.com/matlabcentral/fileexchange/30119-complete-higuchi-fractal-dimension-algorithm/content/hfd.m
    by Salai Selvam V
    '''
    
    N = len(epoch)
    
    Lmk = np.zeros((Kmax,Kmax))
    
    for k in range(1, Kmax+1):
        
        for m in range(1, k+1):
               
            Lmki = 0
            
            maxI = floor((N-m)/k)
            
            for i in range(1,int(maxI+1)):
                Lmki = Lmki + np.abs(epoch[m+i*k-1]-epoch[m+(i-1)*k-1])
             
            normFactor = (N-1)/(maxI*k)
            Lmk[m-1,k-1] = normFactor * Lmki
    
    Lk = np.zeros((Kmax, 1))
    
    for k in range(1, Kmax+1):
        Lk[k-1,0] = np.sum(Lmk[range(k),k-1])/k/k

    lnLk = np.log(Lk) 
    lnk = np.log(np.divide(1., range(1, Kmax+1)))
    
    fit = np.polyfit(lnk,lnLk,1)  # Fit a line to the curve
     
    return fit[0]   # Grab the slope. It is the Higuchi FD


# Calculate Petrosian fractal dimension /feature removed
def calcPetrosianFD(epoch):
    
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(petrosianFD(epoch[:,j]))    # Petrosan fractal dimension
                   
    return fd


# Calculate Hjorth fractal dimension /feature removed
def calcHjorthFD(epoch):
    
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(hjorthFD(epoch[:,j],3))     # Hjorth exponent
                   
    return fd


# Calculate Higuchi fractal dimension
def calcHiguchiFD(epoch):
    
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(higuchiFD(epoch[:,j]))      # Higuchi fractal dimension               
    
    return fd


# Calculate Katz fractal dimension
def calcKatzFD(epoch):

    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(katzFD(epoch[:,j]))      # Katz fractal dimension
                   
    return fd


# Calculate skewness of the signal
def calcSkewness(epoch):

    sk = skew(epoch)
        
    return sk


# Calculate kurtosis of the signal
def calcKurtosis(epoch):
    
    kurt = kurtosis(epoch)
    
    return kurt


# Compute the relative PSD for each frequency band from welch method
def calcPSDWavelet(epoch, nc):

    psd_wav = np.zeros((7,nc))
    for i in range(16):
        C_7, C_6, C_5, C_4, C_3, C_2, C_1 = wavedec(epoch[:, i], wavelet = 'db6', level = 6)
        C = [C_7, C_6, C_5, C_4, C_3, C_2, C_1]
        for j in range(7):
            D = C[j]
            E = np.power(D, 2)
            psd_wav[j,i] = sum(E)
        psd_wav[:,i] /= psd_wav[:,i].sum()

    return psd_wav


# Calculate mean of the signal
def calcMean(epoch, subsampLen):

    mean_value = np.sum(epoch, axis = 0)/subsampLen

    return mean_value


# Calculate standard deviation of the signal
def calcStdDev(epoch):

    std_dev = np.std(epoch, axis = 0)

    return std_dev


# Control wether an epoch contain more than 80% of non-zero signal
def check_epoch_validity(epoch, nt, nc):

    valid = True
    mat = epoch
    mat = mat[np.all(mat[:,:] != 0, axis = 1),:]
    #print "mat", mat.shape

    if mat.shape[0] < 10000:
        valid = False

    return valid


# Calculate all the features
def calculate_features(file_name):
    
    f = convertMatToDictionary(file_name)
    
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    print('EEG shape = ({} timepoints, {} channels)'.format(nt, nc))
    
    lvl = defineEEGFreqs()
    
    subsampLen = int(floor(fs * 30))  # Grabbing 30-second epochs from within the time series
    numSamps = int(floor(nt / subsampLen));      # Num of 30-sec samples
    sampIdx = range(0,(numSamps+1)*subsampLen,subsampLen)
    
    # Define a feature dictionary with the associated function for calculation 
    functions = { 'shannon entropy': 'calcShannonEntropy(psd_epoch, lvl, subsampLen, nc, fs)'
                , 'power spectral density': 'calcPSD_band(psd_epoch, lvl, subsampLen, nc, fs)'
                , 'spectral edge frequency': 'calcSpectralEdgeFreq(psd_epoch, lvl, subsampLen, nc, fs)'
                , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'
                , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, subsampLen, nc, fs)'
                , 'hjorth activity' : 'calcActivity(epoch)'
                , 'hjorth mobility' : 'calcMobility(epoch)'
                , 'hjorth complexity' : 'calcComplexity(epoch)'
                , 'skewness' : 'calcSkewness(epoch)'
                , 'kurtosis' : 'calcKurtosis(epoch)'
                , 'Katz FD' : 'calcKatzFD(epoch)'
                , 'Higuchi FD' : 'calcHiguchiFD(epoch)'
                , 'PSD wavelet' : 'calcPSDWavelet(epoch, nc)'
                , 'mean' : 'calcMean(epoch, subsampLen)'
                , 'std dev' : 'calcStdDev(epoch)'
                 }
    
    # Initialize a dictionary of pandas dataframes with the features as keys
    feat = {key[0]: pd.DataFrame() for key in functions.items()} 
    for i in range(0,7,1):
        for key in ['shannon entropy', 'power spectral density', 'PSD wavelet', 'correlation matrix (frequency)']:
            sub_key = key + str(i)
            feat[sub_key] = pd.DataFrame()
    #print "feat", feat [debug]

    valid = True # initiate the variable recording the validity of an epoch

    for i in range(1, numSamps+1):
    
        #print('processing file {} epoch {}'.format(file_name,i)) [debug]
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]
        valid = check_epoch_validity(epoch, nt, nc) # Verify the extent of drop-off within the epoch

        if valid:
            fft_epoch = calcNormalizedFFT(epoch, lvl, subsampLen, fs)
            psd_epoch = calcPSD (fft_epoch)
   
            for key in functions.items():
                if key[0] in ['shannon entropy', 'power spectral density', 'PSD wavelet', 'correlation matrix (frequency)']:
                    temp_df = pd.DataFrame(eval(key[1])).T
                    for i in range(0,7,1):
                        sub_key = key[0] + str(i)
                        feat[sub_key] = feat[sub_key].append(temp_df[:][i])
                else:
                    feat[key[0]] = feat[key[0]].append(pd.DataFrame(eval(key[1])).T)
        
        else:
            numSamps = int(numSamps - 1) # Adjust the length of the feature table in case of drop-off

    #print "feat", feat [debug]

    # Attribute the correct epoch number to the rows
    for key in feat.keys():
        feat[key]['Epoch #'] = range(numSamps)
        feat[key] = feat[key].set_index('Epoch #')
    
    return feat
