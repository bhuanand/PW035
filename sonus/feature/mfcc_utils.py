import numpy as np
import math
import sys
from scipy.fftpack import dct

if sys.version_info >= (3, 0):
    basestring = str

def frameSignal(signal, frameLength, frameStep):
    """
    frame the given signal into overlapping frames.

    frames the signal into 25ms frames which is a standard.
    making sure that signal is devided into even number of frames,
    if not padding with the zeros at the end so it does.
    """
    # length of the entire signal
    signalLength = len(signal)

    # length of frames
    frameLength = round(frameLength)
    
    # normally 10ms will do better. to introduce some overlapping
    frameStep = round(frameStep)

    # calculate the number of frames
    if signalLength <= frameLength:
        # case where frame length > signal length
        numFrames = 1
    else:
        # normal case frame length < signal length
        numFrames = 1 + math.ceil((1.0 * signalLength - frameLength) / frameStep)

    # in case of uneven windows, make even by padding with zero's
    paddingLength = (numFrames - 1) * frameStep + frameLength
    zeroArr = np.zeros((paddingLength - signalLength, ))
    evenSignal = np.concatenate((signal, zeroArr))

    # compute the indices for framing
    indices = np.tile(np.arange(0, frameLength), (numFrames, 1)) + \
       np.transpose(np.tile(np.arange(0, numFrames * frameStep, frameStep), (frameLength, 1)))

    indices = np.array(indices, dtype = np.int32)

    # get the frames
    frames = evenSignal[indices]
    
    # return the frames
    return frames

def windowing(frames, **kwargs):
    """
    apply windowing on input 'signal' giving 
    can give the type of windowing to use in kwargs as:
    typeOfWindow: [hamming,[hanning,[blackman]]]

    return:
    numpy.ndarray of size frames length
    """
    # result array
    func = None

    # by default the hamming windowing is choosen
    window = 'hamming'

    # if typeOfWindow is given in the input choose appropriate windowing
    if kwargs.get('typeOfWindow', False):
        window = kwargs.get('typeOfWindow')
    
    # apply appropriate windowing function and return result array
    if isinstance(window, basestring):
        # check if the window type is string if true then proceed
        if window == 'hanning':
            func = np.hanning
        elif window == 'blackman':
            func = np.blackman
        else:
            func = np.hamming
    else:
        # raise TypeError indicating to provide string as arg for typeOfWindow
        raise TypeError, "typeOfWindow argument must be a string."

    return np.array( map( lambda frame: np.multiply( frame, func( len( frame))), frames))

def preemphasis(signal, emphasiscoeff = 0.97):
    """
    perform preemphasis on the given signal
    y[i] = x[i] - emphasiscoeff * x[i - 1]

    reason:
    often when dealing with the signals of low frequency which are sampled
    at high rate, the adjacent samples tend to yield similar numerical values.
    
    the reason is that the low frequency means they thend to vary very slowly with
    the time, so does the numerical values.

    by subtraction we remove part of signals that did not change in relate to its
    adjacent samples.

    params:
    signal: signal to emphasis
    emphasiscoeff: usually between 0.9 - 1 default: 0.97
    
    return:
    preemphasized signal
    """
    
    return np.append(signal[0], signal[1:] - emphasiscoeff * signal[:-1])

def magnitudeSpectrum(frames, nfft):
    """
    computes the magnitude spectrum of the frames. 
    if frames is of dimension: numFrames * frameLength
    output will be a float array which has dimension:
    numFrames * nfft

    application:
    frames: frames array
    magnetudeSpectrum function will apply the discrete cosine transform 
    on all the frames present in the frames array.

    nfft: 
    number of points in the frame to use for DFT.

    this function internally calls numpy.rfft function
    which computes DFT on purely float values.

    for more details about choosing nfft value see documentation
    for numpy.rfft

    return: 
    float valued array containing DFT of input frames
    dimension: numFrames * nfft
    """

    compMagSpec = np.fft.rfft(frames, nfft)

    return np.absolute(compMagSpec)

def powerSpectrum(magSpecFrames, nfft):
    """
    computes the power spectrum of each frame present in the frames array.

    application:
    frames: frames array
    nfft: number points to be choosen for application of power spectrum

    the power spectrum answers how much of a signal is at the frequency w.

    returns:
    returns a numFrames * nfft array with each row containing the power 
    spectrum of the frame.
    """
    
    return (1.0 / nfft) * (np.square(magSpecFrames))

def logPowerSpectrum(powerSpectrum, norm = 1):
    """
    logPowerSpectrum is normalised aroung norm value.
    so that max value is 1
    """
    powerSpectrum[powerSpectrum <= 1e-30] = 1e-30

    logPowSpectra = 10 * np.log10(powerSpectrum)

    if norm:
        return logPowSpectra - np.max(logPowSpectra)
    else:
        return logPowSpectra


def hertzToMelScale(hertz):
    """
    convert from hertz value to corresponding mel scale value.
    params:
    hertz: numpy array with hz values
    return: numpy array containing corresponding mel values.
    """
    return 2595.0 * np.log10(1 + hertz / 700.00)

def melScaleToHertz(mel):
    """
    convert from mel scale value to hertz.
    params:
    mel: mel value or numpy array of values
    return: numpy array containing frequency values
    """
    return 700.00 * ( 10 ** (mel/ 2595.0) - 1)

def melFilterBanks(numFilters = 26, nfft = 512, samplerate = 16000,\
                       lowFreq = 133.33, highFreq = None):
    """
    returns the mel filterbank.
    numpy array of dimension:
    numFilters * (nfft /2 + 1)

    if DFT is computed for nfft, only (nfft/ 2 + 1) values are choosen.
    """

    highFreq = highFreq or samplerate / 2

    # computing evenly spaced mel values
    low = hertzToMelScale(lowFreq)

    high = hertzToMelScale(highFreq)

    melValues = np.linspace(low, high, numFilters + 2)

    freqValues = melScaleToHertz(melValues)

    # convert the frequencies to fft bins
    fftBin = np.floor( (nfft + 1) * freqValues / samplerate)

    # frequency banks
    filterBank = np.zeros([numFilters, nfft/2 + 1], dtype=np.float32)

    for i in xrange(0, numFilters):
        # case of k < f(m - 1)
        filterBank[i, np.arange(0, int(fftBin[i]))] = 0 

        # case of f(m - 1) <= k <= f(m)
        for j in xrange(int(fftBin[i]), int(fftBin[i + 1])):
            filterBank[i, j] = (j - fftBin[i]) / (fftBin[i + 1] - fftBin[i])

        # case of f(m) <= k <= f(m + 1)
        for j in xrange(int(fftBin[i + 1]), int(fftBin[i + 2])):
            filterBank[i, j] = (fftBin[i + 2] - j) / (fftBin[i + 2] - fftBin[i + 1])
        # case of k > f(m + 2)
        filterBank[i, np.arange(int(fftBin[i + 2]), len(filterBank[i]))] = 0

    return filterBank

def filterBankEnergies(filterBank, powerSpectrum, nfft = 512):
    """
    returns a tuple consisting of filterbank energies and energy present
    in each frame
    """
    energy = np.sum(powerSpectrum, axis = 1)
    
    # keep only first 257 co efficients from power spectrum of
    # each frame
    
    filBankEnergy = np.dot(powerSpectrum, np.transpose(filterBank))

    return filBankEnergy, energy

def logFilterBankEnergies(filterBankEnergy):
    """
    computes the logarithm of each filter vector present
    in the filterBank returns it
    """
    np.seterr(divide='ignore')
    return np.log10(filterBankEnergy)
    
def discreteCosineTransform(logFilterBankEnergy, numCepstrals):
    """
    apply the discrete cosine transform on the 
    given filterbank energies

    used scipy.fftpack.dct function for DCT
    for more information about parameters see scipy.fftpack.dct 
    documentation

    return number of vector * numCepstral array
    """
    
    return dct(logFilterBankEnergy.astype(np.float32), norm = 'ortho')[:, :numCepstrals]

def cepstralLifter(cepstra, lifterCoeff = 22):
    """
    lifts the cepstral values of high frequency DCT coefficients
    """
    
    numFrames, numCepstrals = np.shape(cepstra)
    
    temp = np.arange(numCepstrals)
    
    return (1 + (lifterCoeff / 2) * np.sin(np.pi * temp/ lifterCoeff)) * cepstra

def computeDelta(cepstra, cepstralSpan = 2):
    """
    calculate the differential and accelaration coefficients for the cepstrum.
    
    reason:
    MFCC cepstrum gives the power spectral envelope of single frame.
    but the signal is dynamic in nature.
    we calculate the trajectories of cepstrals of each frame which can
    improve the performance of learning.
    """
    
    # coefficient matrix
    dataVector = np.transpose(np.copy(cepstra))

    # get the number of rows and columns in the matrix
    # rows: indicates the number of co-efficients
    # cols: indicates the number of frames
    rows, cols = dataVector.shape

    # duplicate the first and last rows by 'cepstralSpan' times
    dataVector = np.append( np.tile( dataVector[0], cepstralSpan).reshape(cepstralSpan, cols), dataVector, axis = 0)
    dataVector = np.append( dataVector, np.tile( dataVector[-1], cepstralSpan).reshape(cepstralSpan, cols), axis = 0)

    deltaVector = np.zeros((rows, cols))

    # compute the delta values
    for i in range(cepstralSpan):
        from_one = cepstralSpan + i + 1
        from_two = cepstralSpan - i - 1
        
        deltaVector = np.add( deltaVector, (i + 1) * np.subtract(dataVector[from_one: from_one + rows], dataVector[from_two: from_two + rows]))
    
    deltaVector = deltaVector / (2 * np.sum( np.square( np.arange(1, cepstralSpan + 1))))
    
    return np.transpose(deltaVector)
    
__all__ = [windowing, frameSignal, preemphasis, magnitudeSpectrum, powerSpectrum, hertzToMelScale, melScaleToHertz, melFilterBanks,\
               filterBankEnergies, logFilterBankEnergies, discreteCosineTransform, cepstralLifter, logPowerSpectrum, computeDelta]
