import os
import sys
import random
import numpy as np
import librosa
import soundfile
import pandas as pd
import settings

def generate(n):
    for i in range(n):
        data=[]
        print("working on {}".format(i))
        files = []
        for j in range(settings.sounds_per_file):
            files.append(random.choice(os.listdir(settings.data_folder)))
        print(files)
        empty = np.zeros((settings.file_len_seconds * settings.sr))
        for file in files:
            y, _ = librosa.load(settings.data_folder + file, sr=settings.sr, mono=True)
            # select random spot
            start = random.randrange(0, len(empty))
            start_copy=start
            # place
            for k in range(len(y)):
                if (start > len(empty) - 1):  # if out of range
                    continue

                if(empty[start] != 0):
                    empty[start] += y[k]
                    empty[start] /= 2
                else:
                    empty[start] += y[k]

                start += 1
            # write data
            split = file.split("-")
            num_class = int(split[1])

            data.append([num_class, start_copy, start])

        # noise
        noise = np.random.normal(0, .01, empty.shape)
        final = empty + noise

        #filename
        filename=settings.out_folder+"gen_"+str(n)
        soundfile.write(filename+".wav", final, settings.sr)

        # generate dataframe and write to csv
        df = pd.DataFrame(data, columns=['Class_ID', 'From', 'To'])
        df.index.name="index"
        df.to_csv(filename+".csv")


if __name__ == '__main__':
    # debug:
    generate(1)
    sys.exit()

    answered = False
    while not answered:
        n = input("how many files? ")
        if n.isnumeric():
            answered = True

    generate(int(n))
