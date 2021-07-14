import os
import pandas as pd

results_columns = ['bg_noise_rate', 'hit_rate', 'miss_rate']
df = pd.DataFrame(columns=results_columns)
df.to_csv('results.csv')