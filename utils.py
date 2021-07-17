import librosa
import os
import pathlib
import settings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

currDirectory = pathlib.Path(__file__).parent.absolute()
USE_CORES = -1
preloaded_model=None


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

def getSpectrogram2(file):
    data, rate = librosa.load(file, sr=settings.sr, mono=True)
    melspec = np.mean(librosa.feature.melspectrogram(data, rate).T, axis=0)
    return melspec

def getSpectrogram(file):
    #rate, stereodata = wavfile.read(file)
    data, rate = librosa.load(file, sr=settings.sr, mono=True)

    # convert to mono
    #if stereodata.ndim != 1:
    #    data = stereodata.sum(axis=1) / 2
    #else:
    #    data = stereodata

    plt.ioff()
    mpl.use('Agg')  # to prevent weird memory leak of mpl

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, NFFT=settings.NFFT, noverlap=settings.noverlap)  # , noverlap=NFFT - 1)
    ax.axis('off')
    fig.set_dpi(100)
    fig.set_size_inches(settings.imwidth / 100, settings.imheight / 100)
    fig.canvas.draw()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = np.reshape(mplimage, (int(settings.imwidth), int(settings.imheight), 3))
    # print(width, height)
    # plt.clf()
    plt.close()
    gray = np.dot(imarray[..., :3], [0.299, 0.587, 0.114])
    return gray

def getSpectrogramRaw(data, rate):
    plt.ioff()
    mpl.use('Agg')  # to prevent weird memory leak of mpl

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=settings.sr, NFFT=settings.NFFT, noverlap=settings.noverlap)
    ax.axis('off')
    fig.set_dpi(100)
    fig.set_size_inches(settings.imwidth / 100, settings.imheight / 100)
    fig.canvas.draw()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = np.reshape(mplimage, (int(settings.imwidth), int(settings.imheight), 3))
    # print(width, height)
    # plt.clf()
    plt.close()
    gray = np.dot(imarray[..., :3], [0.299, 0.587, 0.114])

    return gray


def normalizeSpectrogram(array):
    return (array - array.min()) / (array.max() - array.min())


def silenceTensorflow(level: int):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)  # or any {'0', '1', '2'}

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

toHumanLabels = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}
