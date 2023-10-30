import covidcast
from tabulate import tabulate
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from utils import API_KEY, SIGNALS, SHORT_SIGNALS, get_data_path
from plot_utils import plot_bar_nstates, plot_bar_diff_nstates

covidcast.use_api_key(API_KEY)
start_day = date(2021, 2, 27)
end_day = date(2021, 3, 27)
## add one because the enddate is inclusive
interval = (end_day - start_day).days + 1

states = ["wa", "pa", "al"]

avg_ps = []
# sample_sizes = []
for state in states:
    avg_p = []
    for sig in tqdm(SIGNALS):
        file_path = get_data_path(state, sig, start_day, end_day)
        if not os.path.exists(file_path):
            df = covidcast.signal("fb-survey", sig, start_day, end_day, geo_type="state",
                            geo_values = state)
            with open(file_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
        
        avg_p.append(np.mean(df['value'].tolist()))
    avg_ps.append(avg_p)
        # sample_sizes.append(df['sample_size'].tolist())

avg_ps = pd.DataFrame(data = np.array(avg_ps), columns = SHORT_SIGNALS)

k = 5
# for each reason, find the variance among states
vars = np.array([np.var(avg_ps.iloc[:, j]) for j in range(avg_ps.shape[1])])
# display the top k different reasons
idx = vars.argsort()[-k:][::-1]
plot_bar_nstates(avg_ps, states, start_day, end_day, idx)
plot_bar_diff_nstates(avg_ps, states, start_day, end_day, idx, vars)


