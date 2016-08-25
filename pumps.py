import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_numbers = [12, 13, 14, 20, 22, 40]

data = [pd.read_csv('UsageData_{0}.csv'.format(i)) for i in file_numbers]

f, plots = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(30, 20))
for i, d in enumerate(data):
    plots[i%2][int(i/2)].plot(d['timestamp'], d['amount'])

plt.show()
