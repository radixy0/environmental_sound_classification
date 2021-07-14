import librosa
import sys
import os
import settings
from utils import getSpectrogram
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras


def checkfile(file):
    data, rate = librosa.load(file, sr=settings.sr, mono=True)
    # length * sr = total length
    window_size_samples = settings.window_size * rate
    slices = []
    for i in range(0, len(data), window_size_samples):
        slices.append(data[i:i + window_size_samples])

    slices = np.asarray(slices)

    # generate spectrograms
    specs = []
    for s in slices:
        normgram = getSpectrogram(s)
        specs.append(normgram)

    model = keras.models.load_model("model/model_resnet18.h5")
    results = []
    for index, spectrogram in enumerate(specs):
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
