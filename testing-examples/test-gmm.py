import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import sonus.feature.mfcc as mfcc
import sonus.utils.sonusreader as sonusreader
import sonus.gmm.gmm as gmm

def main():
    audio = sonusreader.SonusReader.from_file(os.path.join(os.path.expanduser('~'),'smashingbaby.wav'))

    mfccs = mfcc.mfcc(audio.data, samplerate=audio.samplerate)

    # crete a GMM object
    # replace option -> method with 'uniform', 'random', 'kmeans'
    GMM = gmm.GaussianMixtureModel(
        data=mfccs, 
        nClusters=4,
        options={
            'method':'random'
            })
    
    GMM.expectationMaximization()
    
    # default paths - done
    # specifying file path - done
    # specifying directory as filepath for saving - done
    # specifying a file with wrong content - done
    # specifying a path doesnt exists - done
    # call load before save - done
    # invalid file path + obj not saved - done
    gmm.GaussianMixtureModel.saveobject(GMM)

    newobj = gmm.GaussianMixtureModel.loadobject()

    print GMM.apriori

    print newobj.apriori

    print sum(GMM.apriori)

    print sum(newobj.apriori)
    
if __name__ == '__main__':
    main()
