import glob
import os
import pathlib

import librosa
import librosa.display
import matplotlib.pyplot as plt
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
import numpy as np
import soundfile as sf
from PIL import Image
from pydub import AudioSegment, effects
from tqdm import tqdm
import shutil

currDirectory = pathlib.Path(__file__).parent.absolute()
audiodatafolder=currDirectory.joinpath("data/audio")
audio_splitfolder=currDirectory.joinpath("data/audio_split")
traindatafolder=currDirectory.joinpath("data/train")
audio_augmentedfolder=currDirectory.joinpath("data/audio_augmented")
model_savefolder=currDirectory.joinpath("model")
validateFolder=currDirectory.joinpath("validation/audio")
validateAugFolder=currDirectory.joinpath("validation/audio_augmented")
validateSplitFolder=currDirectory.joinpath("validation/audio_split")
validateSpectrograms=currDirectory.joinpath("validation_spec")

SPLIT_LENGTH=5 #in s
USE_CORES=-1

def toSpectrogram(data, sr):
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)
    T = np.asarray(S)
    T=T/255
    return T

def toSpectrogram_deprecated(infile: str, outfile:str):

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    plt.ioff()

    y,sr = librosa.load(infile)
    z, _ = librosa.effects.trim(y)
    D = np.abs(librosa.stft(z, n_fft=n_fft,  hop_length=hop_length))
    DB=librosa.amplitude_to_db(D, ref=np.max)
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    ax=plt.axes()
    ax.set_axis_off()
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

    plt.savefig(outfile, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.clf()

def getWavListFromFolder(f: str):
    output=[]
    for filename in glob.glob(os.path.join(f, '*.wav')):
        output.append(pathlib.Path(filename).stem)
    return output

def getJpgListFromFolder(f: str):
    output=[]
    for filename in glob.glob(os.path.join(f, '*.jpg')):
        output.append(pathlib.Path(filename).stem)
    return output

def getNpyListFromFolder(f: str):
    output=[]
    for filename in glob.glob(os.path.join(f, '*.npy')):
        output.append(pathlib.Path(filename).stem)
    return output

def computeSplitAudioFile(wavfile, frompath, topath):
    splitlength=SPLIT_LENGTH*1000 # in ms
    absolutewavfile=frompath.joinpath(wavfile+".wav")
    try:
        newAudio = AudioSegment.from_wav(absolutewavfile)
    except OSError as err:
        None
        print(err, absolutewavfile)

    for i in range(0,len(newAudio),splitlength):
        newAudio2 = newAudio[i:i+splitlength]
        if(len(newAudio2)<splitlength):
            silence=splitlength-len(newAudio2)+1
            newAudio2=newAudio2+AudioSegment.silent(duration=silence)

        filename=wavfile+"_"+str(i)+".wav"
        absfilename=topath.joinpath(filename)
        newAudio2.export(absfilename, format="wav")

def computeNormalize(data, sr):
    norm=naa.NormalizeAug(method='minmax')
    augmented_data=norm.augment(data)
    return augmented_data

def computeAug(data, sr):
    result=[]
    norm=naa.NormalizeAug(method='minmax')   
    aug=naf.Sometimes([
            naa.CropAug(),
            naa.MaskAug(),
            naa.NoiseAug(),
            naa.PitchAug(44100),
            naa.SpeedAug()
        ])
    normalized_data=norm.augment(data)
    result.append(normalized_data)
    augmented_data=aug.augment(data, 2)
    result.extend(augmented_data)
    return result

def computeSpectrogram(wavfile, frompath, topath):
        absolutewavfile=frompath.joinpath(wavfile+".wav")
        specfile=topath.joinpath(wavfile+".npy")
            #print(absolutewavfile)
            #print(specfile)
        y, sr = librosa.load(absolutewavfile)
        np.save(specfile, toSpectrogram(y,sr))

def chunks(lst, n):
    output=[]
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        output.append(lst[i:i + n])
    return output

def ask_user(question: str):
    answered=False
    while not answered:
        answer=input("\n"+question)
        if(answer.lower() == "y" or answer.lower() == "yes"):
            return True
        if(answer.lower() == "n" or answer.lower() == "no"):
            return False

def removeFolderWithProgress(dest: str):
    print("Deleting: ",dest)
    toDelete=[]
    for filename in glob.glob(str(dest)+"/**", recursive=True):
        toDelete.append(pathlib.Path(filename).stem)
    
    toDelete.append(dest)
    for filename in tqdm(toDelete):
        shutil.rmtree(filename, ignore_errors=True)

def removeFolder(dest: str):
    toDelete=[]
    for filename in glob.glob(str(dest)+"/**", recursive=True):
        toDelete.append(pathlib.Path(filename).stem)
    
    toDelete.append(dest)
    for filename in toDelete:
        shutil.rmtree(filename, ignore_errors=True)

def getClassList():
    return [f.name for f in os.scandir(audiodatafolder) if f.is_dir()]

def getAudioFolder():
    return audiodatafolder

def getSplitAudioFolder():
    return audio_splitfolder

def getAugmentedAudioFolder():
    return audio_augmentedfolder

def getTrainDataFolder():
    if not os.path.exists(traindatafolder):
        os.makedirs(traindatafolder)
    return traindatafolder

def getModelFolder():
    if not os.path.exists(model_savefolder):
        os.makedirs(model_savefolder)
    return model_savefolder

def getValidationAugFolder():
    if not os.path.exists(validateAugFolder):
        os.makedirs(validateAugFolder)
    return validateAugFolder

def getValidationSplitFolder():
    if not os.path.exists(validateSplitFolder):
        os.makedirs(validateSplitFolder)
    return validateSplitFolder

def getValidationFolder():
    return validateFolder

def getValidationSpectrogramFolder():
    return validateSpectrograms

def getModelList():
    return os.listdir(getModelFolder())

def silenceTensorflow(level: int):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)  # or any {'0', '1', '2'}