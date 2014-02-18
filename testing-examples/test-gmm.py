import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import sonus.feature.mfcc as mfcc
import sonus.utils.sonusreader as sonusreader
import sonus.utils.gmm.gmm as gmm

def main():
    audio = sonusreader.SonusReader.from_file('/home/bhuvan/Desktop/.all-folders/sci-python/smashingbaby.wav')

    mfccs = mfcc.mfcc(audio.data, samplerate=audio.samplerate)

    print mfccs[0]

    # crete a GMM object
    # replace option -> method with 'uniform', 'random', 'kmeans'
    GMM = gmm.GaussianMixtureModel(
        data=mfccs, 
        nClusters=4,
        options={
            'method':'kmeans'
            })
    
    print GMM.models[0].mean

if __name__ == '__main__':
    main()










