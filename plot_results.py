import matplotlib.pyplot as plt
import pandas as pd
import settings

df = pd.read_csv("results.csv")

#make list of available bg noises
bg_column = df['bg_noise_rate']
bg_types = bg_column.drop_duplicates()
available_bg_types = bg_types.tolist()

#make zipped list of bg noise, hitrate avg
bg_hitrate_lst = []
for i in available_bg_types:
    values = df.loc[df['bg_noise_rate'] == i, 'hit_rate']
    avg = values.mean()
    avg = avg/settings.sounds_per_file
    bg_hitrate_lst.append((i, avg))

#make zipped list of bg noise, missrate avg
bg_missrate_lst = []
for i in available_bg_types:
    values = df.loc[df['bg_noise_rate'] == i, 'miss_rate']
    avg = values.mean()
    avg = avg/settings.sounds_per_file
    bg_missrate_lst.append((i, avg))

#sort, just in case
bg_hitrate_lst.sort(key=lambda tup: tup[0])
bg_missrate_lst.sort(key=lambda tup: tup[0])

#plot
plt.plot(*zip(*bg_hitrate_lst), label="Hits")
plt.plot(*zip(*bg_missrate_lst), label="Misses")
plt.xlabel("background noise loudness in %")
plt.legend()
plt.show()
