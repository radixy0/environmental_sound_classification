import os
import numpy as np
import sys
from tqdm import tqdm
import settings
from utils import getSpectrogram, normalizeSpectrogram

aug = False

def main():
    print("include augmented: ", aug)

    file_list = [f for f in os.listdir(settings.audio_dir) if '.wav' in f]

    if(aug):
        file_list += [f for f in os.listdir(settings.aug_dir) if '.wav' in f]

    file_list.sort()

    x_train = np.zeros((len(file_list), settings.imwidth, settings.imheight), dtype=np.float64)
    y_train = np.zeros(len(file_list))

    print("preparing files..")
    for i, f in enumerate(tqdm(file_list)):
        # get label
        split = f.split("-")
        y_train[i] = int(split[1])

        filedir = settings.audio_dir
        if('aug' in f):
            filedir = settings.aug_dir

        # get spectrogram from either audio dir or aug dir
        try:
            spectrogram = getSpectrogram(filedir + f)
        except ValueError as e:
            print("\nerror reading file: ", filedir + f)
            print(e)
            continue

        normgram = normalizeSpectrogram(spectrogram)
        x_train[i] = normgram

    np.save(settings.x_path, x_train)
    np.save(settings.y_path, y_train)


if __name__ == "__main__":
    if(len(sys.argv) > 1):
        if(sys.argv[1] == "-aug"):
            aug=True
    main()
