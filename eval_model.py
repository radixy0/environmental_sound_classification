import utils
import nlpaug.augmenter.audio as naa
import librosa
from pydub import AudioSegment
import soundfile as sf
from tqdm import tqdm
import numpy as np
import os
import joblib

def generateSpectrograms():
    folder = utils.getValidationFolder()
    #generate normalized audio into temp1
    print("normalizing dataset..")
    for classindex, classname in enumerate(utils.getClassList()):
        frompath = utils.getValidationFolder().joinpath(classname)
        topath = utils.getTempFolder().joinpath(classname)
        #create path if necessary
        if not (os.path.exists(topath)):
                os.makedirs(topath)

        jobs=[joblib.delayed(utils.computeNormalize)(wavfile, frompath, topath) for wavfile in utils.getWavListFromFolder(frompath)]
        out=joblib.Parallel(n_jobs=utils.USE_CORES, verbose=0)(tqdm(jobs))

    #generate split audio into temp2
    print("splitting into parts..")
    for classindex, classname in enumerate(utils.getClassList()):
        frompath = utils.getTempFolder().joinpath(classname)
        topath = utils.getTempFolder2().joinpath(classname)
        #create folder if necessary
        if not (os.path.exists(topath)):
            os.makedirs(topath)
        jobs=[joblib.delayed(utils.computeSplitAudioFile)(wavfile, frompath, topath) for wavfile in utils.getWavListFromFolder(frompath)]
        out=joblib.Parallel(n_jobs=utils.USE_CORES, verbose=0)(tqdm(jobs))
    #generate spectrograms into eval_specfolder
    print("generating spectrograms..")
    for classindex, classname in enumerate(utils.getClassList()):
        frompath = utils.getTempFolder2().joinpath(classname)
        topath = utils.getValidationSpectrogramFolder().joinpath(classname)
        if not (os.path.exists(topath)):
            os.makedirs(topath)
        jobs = [joblib.delayed(utils.computeSpectrogram)(wavfile, frompath, topath) for wavfile in utils.getWavListFromFolder(frompath)]
        out=joblib.Parallel(n_jobs=utils.USE_CORES, verbose=0)(tqdm(jobs))

    answer = utils.ask_user("Clean up temp files?")
    if(answer): utils.cleanTempFolder()

def getValidationData():
    spectrograms = []
    labels = []

    if not (os.path.isfile("validation_spec/val_specs.npy")):
        for index, classname in enumerate(utils.getClassList()):
            folder = utils.getValidationSpectrogramFolder().joinpath(classname)
            for spectrogram in tqdm(utils.getNpyListFromFolder(folder)):
                abspath = folder.joinpath(spectrogram+".npy")
                spec=np.load(abspath)
                spectrograms.append(spec)
                labels.append(index)
        
        spectrograms=np.asarray(spectrograms)
        labels=np.asarray(labels)

    else:
        spectrograms=np.load("validation_spec/val_specs.npy")
        labels=np.load("validation_spec/val_labels.npy")
        
    sample_shape=spectrograms[0].shape
    input_shape=(len(spectrograms), sample_shape[0], sample_shape[1], 1)
    spectrograms=spectrograms.reshape(input_shape)

    return spectrograms, labels

def main():
    utils.silenceTensorflow(3)
    available_models=utils.getModelList()
    if(available_models==[]): 
        print("No models available! Please use train.py to create one")
        input("Press any key to exit..")
        exit()
    
    print("Available Models: ")
    for index,value in enumerate(available_models):
        print(index,": "+value)
    
    correctAnswer=False
    answer=0
    if(len(available_models)==1):
        correctAnswer=True
        print("\nOnly one model available, using "+available_models[0])

    while not correctAnswer:
        answer=input("\nChoose model to use: ")
        if(answer.isnumeric()):
            answer=int(answer)
            if(answer in range(len(available_models))):
                correctAnswer=True
    
    modelname=available_models[answer]

    print("loading tensorflow..")

    import tensorflow as tf
    from tensorflow.keras import models

    model=models.load_model(utils.getModelFolder().joinpath(modelname))

    print("Using model: ")
    model.summary()

    if not os.path.exists(utils.getValidationSpectrogramFolder()):
        print("no spectrograms for validation! generating..")
        generateSpectrograms()

    spectrograms, labels = getValidationData()
    score = model.evaluate(spectrograms, labels, verbose=1)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

if __name__ == "__main__":
    main()
