### Convert the sample signals into features

# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get specific functions from some other python libraries
from math import floor, log
from scipy.stats import skew, kurtosis
from scipy.io import loadmat   # For loading MATLAB data (.dat) files
from scipy import signal

# This code was adapted from a post from Deep to the Kaggle competition forum here: 
# https://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated/comments


def convertMatToDictionary(path):
    
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return ndata


'''
Calculates the FFT of the epoch signal. Removes the DC component and normalizes the area to 1
'''
def calcNormalizedFFT(epoch, lvl, nt, fs):
    
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    #print "D", D.shape
    D[0,:]=0                          # set the DC component to zero
    D /= D.sum()                      # Normalize each channel               

    return D


def defineEEGFreqs():
    
    '''
    EEG waveforms are divided into frequency groups. These groups seem to be related to mental activity.
    alpha waves = 8-13 Hz = Awake with eyes closed
    beta waves = 14-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to conciousness and/or perception (particular 40 Hz)
    theta waves = 4-7 Hz = Light sleep
    delta waves < 3.5 Hz = Deep sleep
    '''
    return (np.array([0.5, 4, 8, 15, 30, 45, 70, 180]))  # Frequency levels in Hz


def calcDSpect(fft_epoch, lvl, nt, nc,  fs):
    
    #D = calcNormalizedFFT(epoch, lvl, nt, fs)
    D = fft_epoch
    lseg = np.round(nt/fs*lvl).astype('int')
    
    dspect = np.zeros((len(lvl)-1,nc))
    for j in range(len(dspect)):
        dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)
        
    return dspect


'''
Computes Shannon Entropy
'''
def calcShannonEntropy(fft_epoch, lvl, nt, nc, fs):
    
    # compute Shannon's entropy
    # segments corresponding to frequency bands
    dspect = calcDSpect(fft_epoch, lvl, nt, nc, fs)
    #print "dspect", dspect
    # Find the shannon's entropy
    spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
    
    return spentropy

'''
Calculate cross-correlation matrix
'''
def corr(data, type_corr):
    
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0
    #print C
    w,v = np.linalg.eig(C)
    #print "w", w
    x = np.sort(w)
    x = np.real(x)
    #print "X", x.dtype
    return x


'''
Compute correlation matrix across channels
'''
def calcCorrelationMatrixChan(epoch):
    
    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    #print "data", data.head()
    type_corr = 'spearman'
    
    lxchannels = corr(data, type_corr)
    
    return lxchannels


'''
Calculate correlation matrix across frequencies
'''
def calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs):
    
    # Calculate correlation matrix and its eigenvalues (b/w freq)
    dspect = calcDSpect(epoch, lvl, nt, nc, fs)
    data = pd.DataFrame(data=dspect)
        
    type_corr = 'spearman'
        
    lxfreqbands = corr(data, type_corr)
        
    return lxfreqbands


'''
Calculate correlation matrix of two successive epoch
'''
def calcCorrelationMatrixEpoch(fft_epoch, fft_pre_epoch, lvl, nt, nc, fs):
    
    # Calculate correlation matrix averaged over the sample
    lseg = np.round(nt/fs*lvl).astype('int')
    #print "lseg", lseg
    epoch = fft_epoch[lseg[4]:lseg[6],:]
    pre_epoch = fft_pre_epoch[lseg[4]:lseg[6],:]
    #print "epoch", epoch.shape


    lxepoch = np.array(np.zeros((2,16)))
    #lxepoch = []

    for i in range(nc):

        data = [pd.DataFrame(epoch[:,i]).T, pd.DataFrame(pre_epoch[:,i]).T]
        data = pd.concat(data, axis = 0).T
        #print data.shape
        type_corr = 'spearman'
        #print "corr", corr(data, type_corr).T.shape
        #lxepoch = np.append(lxepoch, corr(data, type_corr), axis = 1)
        lxepoch[:,i] += corr(data, type_corr).T/9.
        #print "verification", corr(data, type_corr).T/9.
    
    #print "lxepoch inter", lxepoch

    return lxepoch


def calcActivity(epoch):
    '''
    Calculate Hjorth activity over epoch
    '''
    
    # Activity
    activity = np.var(epoch, axis=0)
    
    return activity


def calcMobility(epoch):
    '''
    Calculate the Hjorth mobility parameter over epoch
    '''
      
    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    mobility = np.divide(
                        np.std(np.diff(epoch, axis=0)), 
                        np.std(epoch, axis=0))
    
    return mobility


def calcComplexity(epoch):
    '''
    Calculate Hjorth complexity over epoch
    '''
    
    # Complexity
    complexity = np.divide(
        calcMobility(np.diff(epoch, axis=0)), 
        calcMobility(epoch))
        
    return complexity  


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
    
    # If D has been previously calculated, then it can be passed in here
    #  otherwise, calculate it.
    if D is None:   ## Xin Liu
        D = np.diff(X)   # Difference between one data point and the next
        
    # The old code is a little easier to follow
    N_delta= 0; #number of sign changes in derivative of the signal
    for i in range(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1

    n = len(X)
    
    # This code is a little more compact. It gives the same
    # result, but I found that it was actually SLOWER than the for loop
    #N_delta = sum(np.diff(D > 0)) 
    
    return np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))

def katzFD(epoch):
    ''' 
    Katz fractal dimension 
    '''
    
    L = np.abs(epoch - epoch[0]).max()
    d = len(epoch)
    
    return (np.log(L)/np.log(d))


def logarithmic_n(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.

    Non-integer results are rounded down.

    Args:
    min_n (float): minimum value (must be < max_n)
    max_n (float): maximum value (must be > min_n)
    factor (float): factor used to increase min_n (must be > 1)

    Returns:
    list of integers: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
                      without duplicates
    """
    assert max_n > min_n
    assert factor > 1
    
    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)
    
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    ns = [min_n]
    
    for i in range(max_i+1):
        n = int(np.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
            
    return ns

def dfa(data, nvals= None, overlap=True, order=1, debug_plot=False, plot_file=None):

    total_N = len(data)
    if nvals is None:
        nvals = logarithmic_n(4, 0.1*total_N, 1.2)
        
    # create the signal profile (cumulative sum of deviations from the mean => "walk")
    walk = np.cumsum(data - np.mean(data))
    fluctuations = []
    
    for n in nvals:
        # subdivide data into chunks of size n
        if overlap:
            # step size n/2 instead of n
            d = np.array([walk[i:i+n] for i in range(0,len(walk)-n,n//2)])
        else:
            # non-overlapping windows => we can simply do a reshape
            d = walk[:total_N-(total_N % n)]
            d = d.reshape((total_N//n, n))
            
        # calculate local trends as polynomes
        x = np.arange(n)
        tpoly = np.array([np.polyfit(x, d[i], order) for i in range(len(d))])
        trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
        
        # calculate standard deviation ("fluctuation") of walks in d around trend
        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
        
        # calculate mean fluctuation over all subsequences
        f_n = np.sum(flucs) / len(flucs)
        fluctuations.append(f_n)
        
        
    fluctuations = np.array(fluctuations)
    # filter zeros from fluctuations
    nonzero = np.where(fluctuations != 0)
    nvals = np.array(nvals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(nvals), np.log(fluctuations), 1)
    if debug_plot:
        plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)", fname=plot_file)
        
    return poly[0]

def higuchiFD(epoch, Kmax = 8):
    '''
    Ported from https://www.mathworks.com/matlabcentral/fileexchange/30119-complete-higuchi-fractal-dimension-algorithm/content/hfd.m
    by Salai Selvam V
    '''
    
    N = len(epoch)
    
    Lmk = np.zeros((Kmax,Kmax))
    
    #TODO: I think we can use the Katz code to refactor resampling the series
    for k in range(1, Kmax+1):
        
        for m in range(1, k+1):
               
            Lmki = 0
            
            maxI = floor((N-m)/k)
            
            for i in range(1,int(maxI+1)):
                Lmki = Lmki + np.abs(epoch[m+i*k-1]-epoch[m+(i-1)*k-1])
             
            normFactor = (N-1)/(maxI*k)
            Lmk[m-1,k-1] = normFactor * Lmki
    
    Lk = np.zeros((Kmax, 1))
    
    #TODO: This is just a mean. Let's use np.mean instead?
    for k in range(1, Kmax+1):
        Lk[k-1,0] = np.sum(Lmk[range(k),k-1])/k/k

    lnLk = np.log(Lk) 
    lnk = np.log(np.divide(1., range(1, Kmax+1)))
    
    fit = np.polyfit(lnk,lnLk,1)  # Fit a line to the curve
     
    return fit[0]   # Grab the slope. It is the Higuchi FD


def calcPetrosianFD(epoch):
    
    '''
    Calculate Petrosian fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(petrosianFD(epoch[:,j]))    # Petrosan fractal dimension
                   
    
    return fd

def calcHjorthFD(epoch):
    
    '''
    Calculate Hjorth fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(hjorthFD(epoch[:,j],3))     # Hjorth exponent
                   
    
    return fd


def calcHiguchiFD(epoch):
    
    '''
    Calculate Higuchi fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(higuchiFD(epoch[:,j]))      # Higuchi fractal dimension
                   
    
    return fd

def calcKatzFD(epoch):
    
    '''
    Calculate Katz fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(katzFD(epoch[:,j]))      # Katz fractal dimension
                   
    
    return fd

def calcDFA(epoch):
    
    '''
    Calculate Detrended Fluctuation Analysis
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(dfa(epoch[:,j]))      # DFA
                   
    
    return fd

def calcSkewness(epoch):
    '''
    Calculate skewness
    '''
    # Statistical properties
    # Skewness
    sk = skew(epoch)
        
    return sk


def calcKurtosis(epoch):
    
    '''
    Calculate kurtosis
    '''
    # Kurtosis
    kurt = kurtosis(epoch)
    
    return kurt


# Features added to the initial code are below

# Estimate Spectral Edge Frequency using the Welch method.
def calcSpectralEdgeFreq(Dic_P_spec, lvl, nc, fs):
    
    P_spec = Dic_P_spec['P_spec'].as_matrix()
    f_spec = Dic_P_spec['f_spec'].as_matrix()
    f = np.array(np.zeros(16,))
    #print "P_spec", P_spec.shape
    #print "F", f_spec.shape

    for l in range(16):
        for j in range(10):
            P_temp = P_spec[j,:501]
        #P_temp = P_spec[:501]
            P_pcent = sum(P_temp)
            #print "P_tot", max(P_spec[j,:501])
            P_pcent = 0.8 * P_pcent
            #print "P_pcent", P_pcent
            p = 0
            count_i = 0
            while p < P_pcent:
                count_i +=1 
                p += P_temp[count_i-1]
            #################if count_i
            #print p
            #print "freq", f_spec[0,count_i]
            f[l] += 1./10. * f_spec[0,count_i]
        P_spec = P_spec[:,501:]
        #f[i] += 1./10. * f_spec[i]

    return f


# Calculate the Power Spectral Edge Density over bands of frequency
def calcPSD_Band(Dic_P_spec, lvl, nc,  fs):

    P_spec = Dic_P_spec['P_spec'].as_matrix()
    lseg = np.round(1000/fs*lvl).astype('int')
    #print "lseg", lseg
    
    dspect = np.zeros((nc, len(lvl)-1))
    #print "nc", nc
    for i in range(nc):
        P_temp = P_spec[:,:501]
        #print "bye", P_temp.shape
        for j in range(len(lvl)-1):
            dspect[i,j] = (np.sum(2*np.sum(P_temp[:,lseg[j]:lseg[j+1]], axis=1))/10.)
            #print "hello", dspect[i,j]
        P_spec = P_spec[:,501:]

    return dspect


# Calculate power spectral density using the Welch method.
def calcPSD(epoch, lvl, nt, fs):
    
    P_spec = np.array(np.zeros(1,))
    #print "FS",fs
    #print "P_spec", P_spec.shape
    #print "epoch", epoch[:,1].shape
    for i in range(16):
        f, Pxx_spec = signal.welch(epoch[:, i], fs, nperseg=1000, scaling='spectrum')
        #print "Pxx_spec", Pxx_spec.shape, Pxx_spec[-3:]
        #print "f", f.shape, f[:3]
        #print "Pxx_spec", Pxx_spec[:5]
        P_spec = np.append(P_spec, Pxx_spec, axis = 0)

    P_spec = P_spec[1:]
    #print "P_spec PSD", P_spec[:20]

    return P_spec, f


# Computes Shannon Entropy for the Dyads
def calcShannonEntropyDyad(epoch, lvl, nt, nc, fs):
    
    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)
    #print "dspect", dspect.shape
                           
    # Find the Shannon's entropy
    spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=1)
    #print "spent", spentropyDyd.shape
    return spentropyDyd
    #return dspect

# Computes the entropy change rate
def calcEntChangeRate(value, lvl, nc, fs):

    shan = value.as_matrix()
    
    dspect = np.zeros(nc)
    for i in range(nc):
        for j in range(9):
            dspect[i] = dspect[i]+(abs((shan[j,i]-shan[j+1,i])/shan[j+1,i])/9.)

    return dspect



def calcDSpectDyad(epoch, lvl, nt, nc, fs):
    
    import pywt
    dspect = np.zeros((nc, 4))
    
    for j in range (16):
        cA2, cD3, cD2, cD1 = pywt.wavedec(epoch[:,j], 'db1', level = 3, axis = 0)
        #print "cA2", len(cA2)
        cA2 = np.array(cA2)
        cD3 = np.array(cD3)
        cD2 = np.array(cD2)
        cD1 = np.array(cD1)

        tot_en = sum(np.square(cA2)) + sum(np.square(cD3)) + sum(np.square(cD2)) + sum(np.square(cD1))
        dspect[j, 0] = sum(np.square(cA2))/tot_en
        dspect[j, 1] = sum(np.square(cD3))/tot_en
        dspect[j, 2] = sum(np.square(cD2))/tot_en
        dspect[j, 3] = sum(np.square(cD1))/tot_en

    return dspect


# Record the features for tha samples in a dictionary
def calculate_features(file_name):
    
    f = convertMatToDictionary(file_name)
    
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    print('EEG shape = ({} timepoints, {} channels)'.format(nt, nc))
    
    lvl = defineEEGFreqs()
    
    subsampLen = int(floor(fs * 60))  # Grabbing 60-second epochs from within the time series
    numSamps = int(floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,(numSamps+1)*subsampLen,subsampLen)
     
    functions = {'shannon entropy': 'calcShannonEntropy(fft_epoch, lvl, nt, nc, fs)'
                 ,'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'
                 , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'
                 , 'shannon entropy (dyad)' : 'calcShannonEntropyDyad(epoch, lvl, nt, nc, fs)'
                 , 'hjorth activity' : 'calcActivity(epoch)'
                 , 'hjorth mobility' : 'calcMobility(epoch)'
                 , 'hjorth complexity' : 'calcComplexity(epoch)'
                 , 'skewness' : 'calcSkewness(epoch)'
                 , 'kurtosis' : 'calcKurtosis(epoch)'
                 #, 'Hjorth FD' : 'calcHjorthFD(epoch)'
                 , 'Katz FD' : 'calcKatzFD(epoch)'
                 , 'Higuchi FD' : 'calcHiguchiFD(epoch)'
                # , 'Detrended Fluctuation Analysis' : 'calcDFA(epoch)'  # DFA takes a long time!
                 }
    
    # Initialize a dictionary of pandas dataframes with the features as keys
    feat = {key[0]: pd.DataFrame() for key in functions.items()} 
    Dic_P_spec = {'P_spec': pd.DataFrame()
                , 'f_spec' : pd.DataFrame()
                }

    lxepoch = np.array(np.zeros((2,16)))
    #print "lxepoch initial", lxepoch.shape

    for i in range(1, numSamps+1):
    
        #print('processing file {} epoch {}'.format(file_name,i)) [debug]
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]
        if i > 1:
            fft_pre_epoch = fft_epoch
            fft_epoch = calcNormalizedFFT(epoch, lvl, nt, fs)
        else:
            fft_epoch = calcNormalizedFFT(epoch, lvl, nt, fs)


        epoch_P_spec, epoch_f_spec = eval("calcPSD(epoch, lvl, nt, fs)")
        Dic_P_spec['P_spec']  = Dic_P_spec['P_spec'].append(pd.DataFrame(epoch_P_spec).T)

        pre_epoch = epoch
        if i != 1:
            pre_epoch = eegData[sampIdx[i-2]:sampIdx[i-1], :]
            fft_pre_epoch = calcNormalizedFFT(pre_epoch, lvl, nt, fs)
            lxepoch += calcCorrelationMatrixEpoch(epoch, pre_epoch, lvl, nt, nc, fs)
   
        for key in functions.items():
            feat[key[0]] = feat[key[0]].append(pd.DataFrame(eval(key[1])).T)
    
        

    # This is kludge but it gets the correct epoch number to the rows
    for key in functions.items():
        feat[key[0]]['Epoch #'] = range(numSamps)
        feat[key[0]] = feat[key[0]].set_index('Epoch #')
    
    Dic_P_spec['P_spec']['Epoch #'] = range(numSamps)
    Dic_P_spec['P_spec'] = Dic_P_spec['P_spec'].set_index('Epoch #')
    Dic_P_spec['f_spec']  = pd.DataFrame(epoch_f_spec).T
    feat['SpectralEdgeFreq'] = pd.DataFrame(calcSpectralEdgeFreq(Dic_P_spec, lvl, nc, fs)).T
    feat['PSD_Band'] = pd.DataFrame(calcPSD_Band(Dic_P_spec, lvl, nc, fs)).T
    feat['Autocorr matrix (epoch)'] = pd.DataFrame(lxepoch)
    feat['Ent rate of Change'] = pd.DataFrame(calcEntChangeRate(feat['shannon entropy (dyad)'], lvl, nc, fs)).T


    #return feat, Dic_P_spec
    return feat

'''
Opens a MATLAB file using the Qt file dialog
'''
# We could always just type the filename into this cell, but let's be slick and add a Qt dialog
# to select the file.
#def openfile_dialog():
#     from PyQt4 import QtGui
#     app = QtGui.QApplication([dir])
#     fname = QtGui.QFileDialog.getOpenFileName(None, "Select a MATLAB data file...", '.', filter="MATLAB data file (*.mat);;All files (*)")
#     return str(fname)

#    return 'new_1_1.mat'

#DATA_FILE = openfile_dialog()
#feat, Dic_P_spec = calculate_features(DATA_FILE)
#print "Dic_P_spec", (Dic_P_spec)
#print "feat", (feat)
