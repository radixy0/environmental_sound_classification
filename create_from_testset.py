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
        print("working on {}".format(i+1))
        files = []
        for j in range(settings.sounds_per_file):
            files.append(random.choice(os.listdir(settings.data_folder)))
        print(files)
        empty = np.zeros((settings.file_len_seconds * settings.sr))

        #put noise
        birds, _ = librosa.load(settings.background_folder + "birds.wav", sr=settings.sr, mono=True)
        cars, _ = librosa.load(settings.background_folder + "cars.wav", sr=settings.sr, mono=True)

        birds=librosa.util.normalize(birds)
        cars=librosa.util.normalize(cars)

        #fill all of empty with birds and cars
        i_birds = random.randrange(0, len(birds))
        i_cars = random.randrange(0, len(cars))

        for k in range(len(empty)):
            empty[k] += birds[i_birds]
            empty[k] += cars[i_cars]
            empty[k] *= settings.background_loudness

            i_birds+=1
            if(i_birds==len(birds)):
                i_birds=0

            i_cars+=1
            if(i_cars == len(cars)):
                i_cars=0


        # put files from testset

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


        #filename
        filename=settings.out_folder+"background_"+str(settings.background_loudness)+"_gen_"+str(i)
        soundfile.write(filename+".wav", empty, settings.sr)

        # generate dataframe and write to csv
        df = pd.DataFrame(data, columns=['Class_ID', 'From', 'To'])
        df.index.name="index"
        df.to_csv(filename+".csv")


if __name__ == '__main__':
    answered = False
    while not answered:
        n = input("how many files? ")
        if n.isnumeric():
            answered = True

    generate(int(n))
