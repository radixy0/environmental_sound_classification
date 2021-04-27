import glob
import os

import warnings
from sys import platform
import utils
import joblib
import numpy as np
utils.silenceTensorflow(3)
import tensorflow as tf

from tqdm import tqdm
import matplotlib.pyplot as plt

import model_architecture

warnings.filterwarnings("ignore", category=DeprecationWarning)
utils.silenceTensorflow(3)

datafolder=utils.getTrainDataFolder()
model_savefolder=utils.getModelFolder()

class_names=[]

#get classes
class_names = [f.name for f in os.scandir(datafolder) if f.is_dir()]

if not (os.path.isfile("data/specs.npy")):
    print("Loading Data for Training..")
    #load data
    specs = []
    labels = []
    for index, classname in enumerate(class_names):
        foldername = datafolder.joinpath(classname)
        for spectrogram in tqdm(utils.getNpyListFromFolder(foldername)):
            abspath=foldername.joinpath(spectrogram+".npy")
            spec=np.load(abspath)
            if(spec.shape[1]==216):
                specs.append(spec)
                labels.append(index)


    specs=np.asarray(specs, dtype=object).astype('float32')
    labels=np.asarray(labels)

    np.save("data/specs.npy", specs)
    np.save("data/labels.npy", labels)
else:
    specs = np.load("data/specs.npy")
    labels = np.load("data/labels.npy")

#reshape
sample_shape=specs[0].shape
input_shape=(len(specs), sample_shape[0], sample_shape[1], 1)
specs=specs.reshape(input_shape)

input_shape = (len(specs), specs[0].shape[0], specs[0].shape[1], 1)

#get model
model=model_architecture.getModel(2,len(class_names), input_shape)

#start training
print("Training..")
history = model.fit(specs, labels,validation_split=0.1, epochs=25, shuffle=True)

model.save(model_savefolder.joinpath("model.h5"), include_optimizer=True)

print("\n\nAll done! Model saved to /data/model/"+model_filename)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model.savefolder.joinpath("accuracy.jpg"))
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model.savefolder.joinpath("loss.jpg"))