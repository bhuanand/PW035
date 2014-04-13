import mfcc_utils
import numpy as np
import matplotlib.pyplot as mp

def mfcc(signal, samplerate = 16000, windowLength = 0.025, windowStep = 0.01,\
             numCepstrals = 12, numFilters = 26, nfft = 512, lowFreq = 133.33,\
             highFreq = None, emphasisCoeff = 0.97, lifterCoeff = 22):
    """
    computes MFCC - Mel Frequency Cepstral Co efficients of the given
    audio signal.


    params:
    signal: input audio signal to be processed.
    samplerate: sample rate of the signal.
    windowLength: length of window in seconds. default 25ms
    windowStep: step between windows. default 10ms
    numCepstrals: number of MFCC expected. default 13
    numFilters: number of mel spaced filters. default 26
    nfft: number of points to be selected for DFT. default 512
    lowFreq: low band frequency of mel filters.
    highFreq: high band frequency of mel filters.
    emphasisCoeff: preemphasis coefficient. default 0.97 [0.9 - 1.0]
    lifterCoeff: apply lifter to cepstral coefficients. default 22
    
    steps in computing mfcc:
    1. preemphasis
    2. framing of signals
    3. windowing
    4. magnitude spectrum
    5. power spectrum
    6. mel filter banks
    7. discrete cosine transform
    8. lifter
    
    returns:
    numpy array of sice numFrames * numCepstrals
    """
    
    # pre emphasis of signal
    newSignal = mfcc_utils.preemphasis(signal, emphasiscoeff = emphasisCoeff)

    # framing of the signal
    frameLength = windowLength * samplerate
    frameStep = windowStep * samplerate
    frames = mfcc_utils.frameSignal(newSignal, frameLength, frameStep)
    
    # windowing of frames
    windowedFrames = mfcc_utils.windowing(frames, typeOfWindow = 'hamming')

    # magnetude spectrum of frames
    magSpec = mfcc_utils.magnitudeSpectrum(windowedFrames, nfft)

    # power spectrum of frames
    powSpec = mfcc_utils.powerSpectrum(magSpec, nfft)

    # get mel filter banks
    melFilterBank = mfcc_utils.melFilterBanks(numFilters = numFilters, nfft = nfft, samplerate = samplerate, lowFreq = lowFreq, highFreq = highFreq)

    # get filter bank energies
    filterBankEnergy, energy = mfcc_utils.filterBankEnergies(melFilterBank, powSpec)

    # log filter bank energies
    logFilterBankEnergy = mfcc_utils.logFilterBankEnergies(filterBankEnergy)
    
    # discrete cosine transform
    cepstrals = mfcc_utils.discreteCosineTransform(logFilterBankEnergy, numCepstrals)

    # lift the cepstral values
    cepstrals = mfcc_utils.cepstralLifter(cepstrals, lifterCoeff)

    # make the last cepstral as log of energy in that frame
    cepstrals = np.column_stack([cepstrals, np.log10(energy)])

    # get delta coefficients
    delta = mfcc_utils.computeDelta(cepstrals, cepstralSpan = 2)

    # get delta-delta coefficients (accelaration coefficients)
    delta_delta = mfcc_utils.computeDelta(delta, cepstralSpan = 2)

    # now put cepstral - delta - delta delta coeffs together
    res = np.append(cepstrals, delta, axis = 1)
    
    res = np.append(res, delta_delta, axis = 1)

    # remove the nan values in the result
    res = res[ ~np.isnan(res).any(axis = 1)]
    res = res[ ~np.isinf(res).any(axis = 1)]

    return res[ ~np.isneginf(res).any(axis = 1)]







