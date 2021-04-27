from pathlib import Path
import joblib
import utils
import os

classnames=[]

def splitAudio(infolder: str, outfolder: str):
    for classname in classnames:
        print("working on class: " + classname)
        frompath = infolder.joinpath(classname)
        topath = outfolder.joinpath(classname)
        #create folder if necessary
        if not (os.path.exists(topath)):
            os.makedirs(topath)

        jobs=[joblib.delayed(utils.computeSplitAudioFile)(wavfile, frompath, topath) for wavfile in utils.getWavListFromFolder(frompath)]
        out=joblib.Parallel(n_jobs=utils.USE_CORES, verbose=1)(jobs)


audioFolder=utils.getAudioFolder()
splitAudioFolder=utils.getSplitAudioFolder()
classnames = utils.getClassList()
splitAudio(audioFolder, splitAudioFolder)
