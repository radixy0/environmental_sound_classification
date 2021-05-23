import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

imwidth = 450
imheight = 300
NFFT = 512
noverlap = 128

audio_dir = "data/audio/"
val_dir = "data/validation/"
x_path = "data/x_train.npy"
y_path = "data/y_train.npy"

x_val_path = "data/x_val.npy"
y_val_path = "data/y_val.npy"


def getSpectrogram(file):
    #rate, stereodata = wavfile.read(file)
    data, rate = librosa.load(file, sr=22050, mono=True)

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
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, NFFT=NFFT, noverlap=noverlap)  # , noverlap=NFFT - 1)
    ax.axis('off')
    fig.set_dpi(100)
    fig.set_size_inches(imwidth / 100, imheight / 100)
    fig.canvas.draw()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = np.reshape(mplimage, (int(imwidth), int(imheight), 3))
    # print(width, height)
    # plt.clf()
    plt.close()
    gray = np.dot(imarray[..., :3], [0.299, 0.587, 0.114])
    return gray


def normalizeSpectrogram(array):
    return (array - array.min()) / (array.max() - array.min())


def main():
    file_list = [f for f in os.listdir(audio_dir) if '.wav' in f]
    file_list.sort()

    x_train = np.zeros((len(file_list), imwidth, imheight), dtype=np.float64)
    y_train = np.zeros(len(file_list))

    print("preparing files..")
    for i, f in enumerate(tqdm(file_list)):
        # get label
        split = f.split("-")
        y_train[i] = int(split[1])
        # get spectrogram
        try:
            spectrogram = getSpectrogram(audio_dir + f)
        except ValueError as e:
            print("\nvalueerror reading file: ", audio_dir + f)
            print(e)
            continue

        normgram = normalizeSpectrogram(spectrogram)
        x_train[i] = normgram

    np.save(x_path, x_train)
    np.save(y_path, y_train)

    # validation files
    file_list = [f for f in os.listdir(val_dir) if '.wav' in f]
    file_list.sort()

    x_val = np.zeros((len(file_list), imwidth, imheight), dtype=np.float64)
    y_val = np.zeros(len(file_list))

    print("preparing validation files..")
    for i, f in enumerate(tqdm(file_list)):
        # get label
        split = f.split("-")
        y_val[i] = int(split[1])
        # get spectrogram
        try:
            spectrogram = getSpectrogram(val_dir + f)
        except ValueError as e:
            print("\nvalueerror reading file: ", val_dir + f)
            continue

        normgram = normalizeSpectrogram(spectrogram)
        x_val[i] = normgram

    np.save(x_val_path, x_val)
    np.save(y_val_path, y_val)


if __name__ == "__main__":
    main()
