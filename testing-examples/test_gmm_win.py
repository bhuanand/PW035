import os
import sys
import glob
import getopt
import fnmatch
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import sonus.feature.mfcc as mfcc
import sonus.utils.sonusreader as sonusreader
import sonus.gmm.gmm as gmm

def list_files(arglist):
    try:
        opt_list, args = getopt.getopt(arglist, "f:d:", ["filepath=",
            "dirpath="])
    except getopt.GetoptError, e:
        print str(e)
        sys.exit(1)

    dirpath = None
    filepath = None

    for option, argument in opt_list:
        if option in ("-d", "--dirpath"):
            dirpath = argument
        elif option in ("-f", "--filepath"):
            filepath = argument

    files = list()

    valid_path_formats = ("*.wav",
        "*.mp3",
        "*.ogg")

    if dirpath:
        if os.path.isdir(os.path.expanduser(dirpath)):
            os.chdir(os.path.expanduser(dirpath))
            for format in valid_path_formats:
                files.extend(glob.glob(format))
        else:
            print 'invalid directory path. please specify valid directory path.'
            sys.exit(2)
    elif filepath:
        if os.path.isfile(os.path.expanduser(filepath)):
            for format in valid_path_formats:
                if fnmatch.fnmatch(filepath, format):
                    files.append(filepath)
                    break
        else:
            print 'invalid file path. please specify valid file path.'
            sys.exit(3)
    else:
        print '''please specify either file path or directory path
                "-d", "--dirpath"  for directory path
                "-f", "--filepath" for file path
        '''
        sys.exit(4)

    return files

def main():
    kannada = "C:\\Users\\bhuvan\\Desktop\\training set\\Kannada\\wav"
    hindi = "C:\\Users\\bhuvan\\Desktop\\training set\\Hindi\\wav"

    kan_files = list_files(['--dirpath=' + kannada])
    hindi_files = list_files(['--dirpath=' + hindi])

    numfiles = min(len(kan_files), len(hindi_files))

    len_train_set = int(round(0.6 * numfiles))

    training_set = []
    for i in range(len_train_set):
        training_set.append((os.path.join(kannada, kan_files[i]),
            os.path.join(hindi, hindi_files[i])))

    data = []
    audio1 = sonusreader.SonusReader.from_file(training_set[0][0])
    audio2 = sonusreader.SonusReader.from_file(training_set[0][1])

    mfcc1 = mfcc.mfcc(audio1.data, samplerate=audio1.samplerate)
    mfcc2 = mfcc.mfcc(audio2.data, samplerate=audio2.samplerate)

    data = np.vstack((mfcc1, mfcc2))

    for example in training_set[1:]:
        audio1 = sonusreader.SonusReader.from_file(example[0])
        audio2 = sonusreader.SonusReader.from_file(example[1])

        mfcc1 = mfcc.mfcc(audio1.data, samplerate=audio1.samplerate)
        mfcc2 = mfcc.mfcc(audio2.data, samplerate=audio2.samplerate)

        d = np.vstack((mfcc1, mfcc2))

        data = np.vstack((data, d))

    print 'done'

    GMM = gmm.GaussianMixtureModel(data, 2, options={
        'method':'uniform'
        })

    GMM.expectationMaximization()

    gmm.GaussianMixtureModel.saveobject(GMM, filepath=" C:\\Users\\bhuvan\\sonus\\uniform\\gmm-object")

if __name__ == '__main__':
    main()
