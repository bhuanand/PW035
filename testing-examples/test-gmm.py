def main():
    audio = sonusreader.SonusReader.from_file('/home/bhuvan/Desktop/.all-folders/sci-python/smashingbaby.wav')

    mfccs = mfcc.mfcc(audio.data, samplerate=audio.samplerate)
    
    print mfccs
    

if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

    import sonus.feature.mfcc as mfcc
    import sonus.utils.sonusreader as sonusreader
    import sonus.utils.gmm.gmm as gmm

    main()
