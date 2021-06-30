import os
import settings
import librosa
import soundfile as sf
import nlpaug.augmenter.audio as naa
from tqdm import tqdm

possible_augs = [naa.NoiseAug(), naa.PitchAug(sampling_rate=settings.sr)]  # TODO fill this

def main():
    file_list = [f for f in os.listdir(settings.audio_dir) if '.wav' in f]
    file_list.sort()

    for index, file in enumerate(tqdm(file_list)):
        y, sr = librosa.load(settings.audio_dir + file)
        filename_parts = file.split("-")
        new_filename_prefix = filename_parts[0] + "-" + filename_parts[1]
        for jndex, aug in enumerate(possible_augs):
            filename = new_filename_prefix + "aug_" + aug.name + ".wav"
            new_y = aug.augment(y)
            sf.write(filename, new_y, settings.sr)


if __name__ == "__main__":
    main()
