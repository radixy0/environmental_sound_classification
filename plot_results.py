import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")

#get x,y with bg, hitrate
bg_hitrate = df[['bg_noise_rate', 'hit_rate']]
bg_hitrate_lst = []
#average for each bg range
for i in range(1,11):
    j = i/10
    values = bg_hitrate[bg_hitrate['bg_noise_rate'] == j]
    y_val = values['hit_rate'].mean() / 20
    x_val = j
    bg_hitrate_lst.append((x_val, y_val))

print(bg_hitrate_lst)
#get x,y with bg, missrate
bg_missrate = df[['bg_noise_rate', 'miss_rate']]
bg_missrate_lst = []

#average
for i in range(1,11):
    j = i/10
    values = bg_missrate[bg_missrate['bg_noise_rate'] == j]
    y_val = values['miss_rate'].mean() / 20
    x_val = j
    bg_missrate_lst.append((x_val, y_val))

print(bg_missrate_lst)

#plot
plt.plot(*zip(*bg_hitrate_lst), label="Hits")
plt.plot(*zip(*bg_missrate_lst), label="Misses")
plt.xlabel("background noise loudness in %")
plt.legend()
plt.show()
