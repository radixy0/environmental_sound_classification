import pyaudio
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import settings
from utils import getSpectrogramRaw


def open_mic():
    pa = pyaudio.PyAudio()
    stream = pa.open(channels=1,
                     rate=settings.sr,
                     input=True,
                     frames_per_buffer=settings.chunk_size,
                     format=pyaudio.paInt16)
    return stream, pa


def get_data(stream, pa):
    input_data = stream.read(settings.chunk_size)
    data = np.frombuffer(input_data, np.int16)
    return data


def displayresults(results):
    for index, pred in enumerate(results):
        print(toHumanLabels(index), ": ", pred)


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

if (__name__ == "__main__"):
    stream, pa = open_mic()
    model = keras.models.load_model("model/model.h5")

    for i in range(5):
        data = get_data(stream, pa)
        specgram = getSpectrogramRaw(data)
        results = model.predict(specgram)
        displayresults(results)
