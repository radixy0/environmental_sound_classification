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
audio_dir = currDirectory.joinpath("data/audio")
model_dir = currDirectory.joinpath("model")

SPLIT_LENGTH = 5  # in s
USE_CORES = -1


def chunks(lst, n):
    output = []
    for i in range(0, len(lst), n):
        output.append(lst[i:i + n])
    return output


def ask_user(question: str):
    answered = False
    while not answered:
        answer = input("\n" + question)
        if (answer.lower() == "y" or answer.lower() == "yes"):
            return True
        if (answer.lower() == "n" or answer.lower() == "no"):
            return False


def getAudioFolder():
    return audio_dir


def getModelFolder():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def getModelList():
    return os.listdir(getModelFolder())


def silenceTensorflow(level: int):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)  # or any {'0', '1', '2'}
