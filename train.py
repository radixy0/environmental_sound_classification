import os
import numpy as np
import model_architecture
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.io import wavfile

utils.silenceTensorflow(3)
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

audio_dir = "data/audio/"
model_dir = utils.getModelFolder()
val_dir = "data/validation/"
imwidth = 320
imheight = 240
num_classes = 10
NFFT = 2048
noverlap = 512
learning_rate = 0.001
decay = 1e-6
momentum = 0.9
epochs = 250
batch_size = 16


def getSpectrogram(file):
    rate, stereodata = wavfile.read(file)

    # convert to mono
    if stereodata.ndim != 1:
        data = stereodata.sum(axis=1) / 2
    else:
        data = stereodata

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


def getData():
    x_path = "data/x_train.npy"
    y_path = "data/y_train.npy"

    x_val_path = "data/x_val.npy"
    y_val_path = "data/y_val.npy"

    # get training files
    if not (os.path.isfile(x_path)) or not (os.path.isfile(y_path)):
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
                continue

            normgram = normalizeSpectrogram(spectrogram)
            x_train[i] = normgram

        np.save(x_path, x_train)
        np.save(y_path, y_train)

    else:
        print("found saved training files")
        x_train = np.load(x_path)
        y_train = np.load(y_path)

    # get validation files
    if not (os.path.isfile(x_val_path)) or not (os.path.isfile(y_val_path)):
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

    else:
        print("found saved validation files")
        x_val = np.load(x_val_path)
        y_val = np.load(y_val_path)

    return x_train, y_train, x_val, y_val


def main():
    x_train, y_train, x_val, y_val = getData()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    print("x shape", x_train.shape)
    print("y shape: ", y_train.shape)
    print("x val shape: ", x_val.shape)
    print("y val shape: ", y_val.shape)
    print("test: ", x_train[333][20][21][0])  # test random value to see if its in [0,1]

    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(x_val))

    model = model_architecture.VGG16_Untrained(10, input_shape)
    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=1)

    model_filename = "model.h5"
    model.save(model_dir.joinpath(model_filename), include_optimizer=True)

    print("\n\nAll done! Model and Diagrams saved to /model" + model_filename)

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
