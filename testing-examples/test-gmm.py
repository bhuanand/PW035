import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import sonus.feature.mfcc as mfcc
import sonus.utils.sonusreader as sonusreader
import sonus.gmm.gmm as gmm

def main():
    audio = sonusreader.SonusReader.from_file(os.path.join(os.path.expanduser('~'),'smashingbaby.wav'))

    mfccs = mfcc.mfcc(audio.data, samplerate=audio.samplerate)

    print len(mfccs)

    # crete a GMM object
    # replace option -> method with 'uniform', 'random', 'kmeans'
    GMM = gmm.GaussianMixtureModel(
        data=mfccs, 
        nClusters=4,
        options={
            'method':'uniform'
            })
    
    GMM.expectationMaximization()
    
    print 'done!'
if __name__ == '__main__':
    main()












