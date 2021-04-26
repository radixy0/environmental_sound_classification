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

import model_architecture

warnings.filterwarnings("ignore", category=DeprecationWarning)
utils.silenceTensorflow(3)

datafolder=utils.getTrainDataFolder()
model_savefolder=utils.getModelFolder()

class_names=[]

#get classes
class_names = [f.name for f in os.scandir(datafolder) if f.is_dir()]

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


specs=np.asarray(specs, dtype=object)
labels=np.asarray(labels)

#reshape
sample_shape=specs[0].shape
input_shape=(len(specs), sample_shape[0], sample_shape[1], 1)
specs=specs.reshape(input_shape)

input_shape = (len(specs), specs[0].shape[0], specs[0].shape[1], 1)

#get model
model=model_architecture.getModel(2,len(class_names), input_shape)

#start training
print("Training..")
model.fit(specs, labels, epochs=250)

model_filename=input("Do you want to give the model a name? End the file with .h5 to use hdf5 format: ")
if(model_filename==""): model_filename="model"

if not (os.path.exists(model_savefolder)):
            os.makedirs(model_savefolder)

model.save(model_savefolder.joinpath(model_filename), include_optimizer=True)

print("\n\nAll done! Model saved to /data/model/"+model_filename)