from train2 import getSpectrogram
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1)
spec=getSpectrogram("data/audio/344-3-0-0.wav", fig,ax)
