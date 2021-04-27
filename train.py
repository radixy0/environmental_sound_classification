import os

import warnings
import utils
import numpy as np

utils.silenceTensorflow(3)

from tqdm import tqdm
import matplotlib.pyplot as plt

import model_architecture

warnings.filterwarnings("ignore", category=DeprecationWarning)
utils.silenceTensorflow(3)

datafolder = utils.getTrainDataFolder()
model_savefolder = utils.getModelFolder()

class_names = []

# get classes
class_names = [f.name for f in os.scandir(datafolder) if f.is_dir()]

if not (os.path.isfile("data/specs.npy")):
    print("Loading Data for Training..")
    # load data
    specs = []
    labels = []
    for index, classname in enumerate(class_names):
        foldername = datafolder.joinpath(classname)
        for spectrogram in tqdm(utils.getNpyListFromFolder(foldername)):
            abspath = foldername.joinpath(spectrogram + ".npy")
            spec = np.load(abspath)
            # if(spec.shape[1]==216):
            specs.append(spec)
            labels.append(index)

    specs = np.asarray(specs, dtype=object).astype('float32')
    labels = np.asarray(labels)

    np.save("data/specs.npy", specs)
    np.save("data/labels.npy", labels)
else:
    specs = np.load("data/specs.npy")
    labels = np.load("data/labels.npy")

# reshape
sample_shape = specs[0].shape
input_shape = (len(specs), sample_shape[0], sample_shape[1], 1)
specs = specs.reshape(input_shape)

# get model
model = model_architecture.getModel(2, len(class_names), input_shape)

# start training
print("Training..")
history = model.fit(specs, labels, validation_split=0.3, epochs=25, shuffle=True)
model_filename = "model.h5"
model.save(model_savefolder.joinpath(model_filename), include_optimizer=True)

print("\n\nAll done! Model saved to /data/model/" + model_filename)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_savefolder.joinpath("accuracy.jpg"))
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_savefolder.joinpath("loss.jpg"))
