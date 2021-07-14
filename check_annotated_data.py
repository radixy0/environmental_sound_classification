import pandas as pd
import check_wav
import settings
import os



def max_index(list):
    highest_index = 0
    for index, item in enumerate(list):
        if (list[index] > list[highest_index]):
            highest_index = index

    return highest_index


def compare(wavfile, csvfile):
    # load from csv
    df = pd.read_csv(csvfile)

    # get results from file
    results = check_wav.checkfile(wavfile)
    # transform, change first entry of result into predicted class instead of probabilities
    for [probs, start, end] in results:
        probs = max_index(probs)

    # compare
    hit_count = 0
    miss_count = 0
    for [result, start, end] in results:
        # check if similar in df
        candidates = df.loc[(df['Class_ID'] == result) & (df['From'] <= start) & (df['To'] >= start)]
        candidates2 = df.loc[(df['Class_ID'] == result) & (df['From'] <= end) & (df['To'] >= end)]
        if not candidates.empty or not candidates2.empty:
            hit_count += 1
            print("correctly found ", result, " between ", start / check_wav.sr, end / check_wav.sr)
        else:
            miss_count+=1
            print("false positive ", result, "between ", start/check_wav.sr, end/check_wav.sr)

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
    df = pd.read_csv('results.csv')
    # compare and add results to results.csv

    for wavfile, csvfile in to_analyze:
        hc, mc = compare(wavfile, csvfile)
        bg_noise = wavfile.split('_')[1]
        df.append(bg_noise, hc, mc)

    df.to_csv('results.csv', mode='a', header=False)


if (__name__ == "__main__"):
    main()
