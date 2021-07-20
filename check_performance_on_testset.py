import os
import settings
import utils
import numpy as np
from tqdm import tqdm
from tensorflow import keras

def main():
    #get files list
    file_list = [f for f in os.listdir(settings.data_folder) if '.wav' in f]
    #create spectrogram and label list
    specs = np.zeros((len(file_list), settings.imwidth, settings.imheight))
    labels = np.zeros(len(file_list))

    print("Creating spectrograms..")

    #fill lists
    for i, f in enumerate(tqdm(file_list)):
        split = f.split("-")
        labels[i] = int(split[1])

        spectrogram=utils.getSpectrogram(settings.data_folder + f)
        normgram = utils.normalizeSpectrogram(spectrogram)
        specs[i] = normgram

    labels = keras.utils.to_categorical(labels, settings.num_classes)

    #load model
    print("loading model..")
    model = keras.models.load_model("model/model.h5")

    #evaluate
    loss, acc = model.evaluate(specs, labels, verbose=0)

    #print
    print("loss: ", loss)
    print("acc: ", acc)

if __name__ == "__main__":
    main()