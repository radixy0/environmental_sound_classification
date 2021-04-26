import glob
import os
from pathlib import Path

import warnings
from sys import platform
import utils
import joblib
import numpy as np
from tqdm import tqdm

import librosa

import model_architecture

warnings.filterwarnings("ignore", category=DeprecationWarning)
utils.silenceTensorflow(3)

def processAudioFile(classname: str, wavfile: str, augment: bool):
    absolutepath = splitAudioFolder.joinpath(classname).joinpath(wavfile+".wav")
    absolutespectrogram=spectrogramFolder.joinpath(classname).joinpath(wavfile+".npy")
    if not os.path.exists(spectrogramFolder.joinpath(classname)):
        os.makedirs(spectrogramFolder.joinpath(classname))

    y, sr = librosa.load(absolutepath)
    audio=[]
    if(augment):
        audio=utils.computeAug(y,sr)
    else:
        audio.append(utils.computeNormalize(y,sr))

    for snd in audio:
        np.save(absolutespectrogram, utils.toSpectrogram(np.nan_to_num(snd), sr))

#pipeline:
#load split audio files from class folders
#augment
#spectrogram

#get classes
class_names=utils.getClassList()

splitAudioFolder=utils.getSplitAudioFolder()
spectrogramFolder=utils.getTrainDataFolder()

answer=utils.ask_user("Augment? [y/n]:")

jobs=[]
for classname in class_names:
    classfolder=splitAudioFolder.joinpath(classname)
    jobs.extend([joblib.delayed(processAudioFile)(classname, wavfile, answer) for wavfile in utils.getWavListFromFolder(classfolder)])
    
print("total tasks: ",len(jobs))
out=joblib.Parallel(n_jobs=utils.USE_CORES, verbose=1)(jobs)