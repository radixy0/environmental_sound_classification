import pandas as pd
import check_wav
import settings
import os
import numpy as np
import utils


def max_index(list):
    return np.argmax(list)


def compare(wavfile, csvfile):
    # load from csv
    df = pd.read_csv(csvfile)

    # get results from file
    results = check_wav.checkfile(wavfile)
    # transform, change first entry of result into predicted class instead of probabilities
    results_without_probs = []

    for [probs, start, end] in results:
        chosen_class = -1
        if(np.max(probs) > 0.9):
            chosen_class = max_index(probs)

        results_without_probs.append([chosen_class, start, end])

    # compare
    hit_count = 0
    miss_count = 0
    for [result, start, end] in results_without_probs:
        #if no hit (class = -1) then continue
        if(result == -1):
            continue

        # check if similar in df
        candidates = df.loc[(df['Class_ID'] == result) & (df['From'] <= start) & (df['To'] >= start)]
        candidates2 = df.loc[(df['Class_ID'] == result) & (df['From'] <= end) & (df['To'] >= end)]
        if not candidates.empty or not candidates2.empty:
            hit_count += 1
            print("correctly found ", utils.toHumanLabels[result], " between ", start / settings.sr, end / settings.sr)
        else:
            miss_count+=1
            print("false positive ", utils.toHumanLabels[result], "between ", start/settings.sr, end/settings.sr)

    return hit_count, miss_count



def main():
    # get everything from generated folder where .wav and .csv exist
    files_and_dirs = os.listdir(settings.out_folder)
    wav_list = [f for f in files_and_dirs if '.wav' in f]
    to_analyze=[]

    for wav in wav_list:
        filename = '.'.join(wav.split('.')[:-1])
        if(filename+".csv" in files_and_dirs):
            to_analyze.append((settings.out_folder + filename+".wav", settings.out_folder + filename+".csv"))

    # load results.csv dataframe
    df = pd.read_csv('results.csv', index_col=0)
    # compare and add results to results.csv
    for wavfile, csvfile in to_analyze:
        hc, mc = compare(wavfile, csvfile)
        bg_noise = wavfile.split('_')[1]
        row_dict = {'bg_noise_rate':bg_noise,'hit_rate':hc,'miss_rate':mc}
        df.loc[len(df.index)] = row_dict


    df.to_csv('results.csv')


if (__name__ == "__main__"):
    main()
