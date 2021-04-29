import os
import numpy as np
import model_architecture
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.io import wavfile

utils.silenceTensorflow(3)
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

audio_dir = "data/audio/"
model_dir = utils.getModelFolder()
imwidth = 375
imheight = 250
num_classes = 10
NFFT = 512


def getSpectrogram(file):
    rate, stereodata = wavfile.read(file)

    # convert to mono
    if stereodata.ndim != 1:
        data = stereodata.sum(axis=1) / 2
    else:
        data = stereodata

    plt.ioff()
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, NFFT=NFFT, noverlap=256)  # , noverlap=NFFT - 1)
    ax.axis('off')
    # plt.rcParams['figure.figsize'] = [0.75, 0.5]
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


def getTrainData():
    x_path = "data/x.npy"
    y_path = "data/y.npy"
    if not (os.path.isfile(x_path)) or not (os.path.isfile(y_path)):
        file_list = [f for f in os.listdir(audio_dir) if '.wav' in f]
        file_list.sort()

        x_train = np.zeros((len(file_list), imwidth, imheight), dtype=np.float64)
        y_train = np.zeros(len(file_list))

        print("preparing files..")

        fig, ax = plt.subplots(1)
        for i, f in enumerate(tqdm(file_list)):
            # get label
            split = f.split("-")
            y_train[i] = int(split[1])
            # get spectrogram
            try:
                spectrogram = getSpectrogram(audio_dir + f)
            except ValueError as e:
                print("\nvalueerror reading file: ", audio_dir + f)
                continue

            normgram = normalizeSpectrogram(spectrogram)
            x_train[i] = normgram

        plt.close()
        np.save(x_path, x_train)
        np.save(y_path, y_train)

    else:
        print("found saved files")
        x_train = np.load(x_path)
        y_train = np.load(y_path)

    return x_train, y_train


def main():
    x_train, y_train = getTrainData()
    x_train = x_train.reshape(x_train.shape[0], imwidth, imheight, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    input_shape = (imwidth, imheight, 1)
    print("x shape", x_train.shape)
    print("y shape: ", y_train.shape)
    print("test: ", x_train[333][20][21][0])

    model = model_architecture.VGG_16(10, input_shape)
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=250, validation_split=0.1, verbose=1, shuffle=True)

    model_filename = "model.h5"
    model.save(model_dir.joinpath(model_filename), include_optimizer=True)

    print("\n\nAll done! Model saved to /model" + model_filename)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model/accuracy.jpg")
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("model/loss.jpg")


if __name__ == "__main__":
    main()
