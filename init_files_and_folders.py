import os
import pandas as pd

results_columns = ['bg_noise_rate', 'hit_rate', 'miss_rate']
#row_dict = {'bg_noise_rate':[0],'hit_rate':[0],'miss_rate':[0]}
df = pd.DataFrame(columns=results_columns)
df.to_csv('results.csv')
