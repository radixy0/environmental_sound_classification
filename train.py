import os
import numpy as np
import model_architecture
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.io import wavfile
utils.silenceTensorflow(3)
from tensorflow import keras

audio_dir = "data/audio/"
model_dir = utils.getModelFolder()
imheight = 50
imwidth = 75
num_classes = 10
NFFT = 4096

def getSpectrogram(file, fig, ax):
    rate, stereodata = wavfile.read(file)

    # convert to mono
    if stereodata.ndim != 1:
        data = stereodata.sum(axis=1) / 2
    else:
        data = stereodata

    plt.ioff()
    #fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, NFFT=NFFT, noverlap=256)#, noverlap=NFFT - 1)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = (0.75, 0.5)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = np.reshape(mplimage, (int(width), int(height), 3))
    plt.clf()
    #plt.close()
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

        x_train = np.zeros((len(file_list), imwidth, imheight))
        y_train = np.zeros(len(file_list))

        print("preparing files..")

        fig, ax = plt.subplots(1)
        for i, f in enumerate(tqdm(file_list)):
            # get label
            split = f.split("-")
            y_train[i] = int(split[1])
            # get spectrogram
            try:
                spectrogram = getSpectrogram(audio_dir + f, fig, ax)
            except ValueError as e:
                print("valueerror reading file: ", audio_dir + f)
                continue

            normgram = normalizeSpectrogram(spectrogram) / 255
            if (normgram.shape[0] > 150): continue
            x_train[i] = normgram

        np.save(x_path, x_train)
        np.save(y_path, y_train)

        return x_train, y_train

    else:
        print("found saved files")
        return np.load(x_path), np.load(y_path)


def main():
    x_train, y_train = getTrainData()

    x_train = x_train.reshape(x_train.shape[0], imwidth, imheight, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    input_shape = (imwidth, imheight, 1)
    print("shape", x_train.shape)
    print("label shape: ", y_train.shape)

    model = model_architecture.getModel(2, 10, input_shape)
    history = model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.1, verbose=1)

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
