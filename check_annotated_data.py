import pandas as pd
import check_wav


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
    for [result, start, end] in results:
        # check if similar in df
        candidates = df.loc[(df['Class_ID'] == result) & (df['From'] <= start) & (df['To'] >= start)]
        candidates2 = df.loc[(df['Class_ID'] == result) & (df['From'] <= end) & (df['To'] >= end)]
        if not candidates.empty or not candidates2.empty:
            hit_count += 1
            print("correctly found ", result, " between ", start / check_wav.sr, end / check_wav.sr)


def main():
    compare("a.wav", "a.csv")


if (__name__ == "__main__"):
    main()
