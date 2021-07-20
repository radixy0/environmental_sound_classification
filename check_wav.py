import librosa
import sys
import os
import settings
import utils
from utils import getSpectrogramRaw
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras


def checkfile(file):
    data, rate = librosa.load(file, sr=settings.sr, mono=True, )
    # length * sr = total length
    window_size_samples = settings.window_size * rate
    slices = []
    for i in range(0, len(data), window_size_samples):
        slices.append(data[i:i + window_size_samples])

    slices = np.asarray(slices)

    # generate spectrograms
    specs = []
    for s in slices:
        graygram = getSpectrogramRaw(s, rate)
        normgram = utils.normalizeSpectrogram(graygram)
        specs.append(normgram)

    model = utils.preloaded_model
    if model is None:
        print("loading model..")
        model = keras.models.load_model("model/model.h5")
        utils.preloaded_model = model

    results = []
    for index, spectrogram in enumerate(specs):
        spectrogram = spectrogram.reshape((1, spectrogram.shape[0], spectrogram.shape[1], 1))
        results.append([model.predict(spectrogram), index * settings.window_size * settings.sr,
                        (index + 1) * settings.window_size * settings.sr])

    return results


def main(file):
    results = checkfile(file)
    # present results


if (__name__ == "__main__"):
    if (len(sys.argv) < 2):
        print("didnt specify file name")
        main("a.wav")
    elif (len(sys.argv) > 2):
        print("too many args")
    else:
        print("reading file: ", sys.argv[1])
        main(sys.argv[1])
