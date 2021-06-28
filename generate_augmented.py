import os
import settings
import librosa
import nlpaug
import random

possible_augs = None # TODO fill this

def main():
    file_list = [f for f in os.listdir(settings.audio_dir) if '.wav' in f]
    file_list.sort()

    for index, file in enumerate(file_list):
        y, sr = librosa.load(settings.audio_dir + file)


if __name__ == "__main__":
    main()
