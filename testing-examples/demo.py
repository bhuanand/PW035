import os
import sys
import test_gmm_win

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import sonus.feature.mfcc as mfcc
import sonus.utils.sonusreader as sonusreader
import sonus.gmm.gmm as gmm

def main():
    kannada = "C:\\Users\\bhuvan\\Desktop\\training set\\Kannada\\wav"
    hindi = "C:\\Users\\bhuvan\\Desktop\\training set\\Hindi\\wav"

    kan_files = test_gmm_win.list_files(['--dirpath=' + kannada])
    hindi_files = test_gmm_win.list_files(['--dirpath=' + hindi])

    nobj = gmm.GaussianMixtureModel.loadobject("C:\\Users\\bhuvan\\sonus\\gmm-uniform\\gmm-object")

    kan_detected = 0
    hin_detected = 0

    for i in range(600, 800):
        a = sonusreader.SonusReader.from_file(os.path.join(kannada, kan_files[i]))

        b = sonusreader.SonusReader.from_file(os.path.join(hindi, hindi_files[i]))

        adata = mfcc.mfcc(a.data, samplerate=a.samplerate)

        bdata = mfcc.mfcc(b.data, samplerate=b.samplerate)

        if nobj.fit(adata) == 0:
            kan_detected = kan_detected + 1

        if nobj.fit(bdata) == 1:
            hin_detected = hin_detected + 1

    print '-' * 30
    print 'Language', '\t\ttested files', '\t\tdetected files'
    print 'Kannada' , '\t\t' + str(200), '\t\t' + str(kan_detected)
    print 'Hindi', '\t\t' + str(200), '\t\t' + str(hin_detected)
    print '-' * 30

if __name__ == '__main__':
    main()
