import torch
import torch.nn as nn
import pandas as pd 
import datetime
from numpy import asarray
import numpy as np

data = pd.read_csv('..\data\CPI_data.csv')
data['Dates'] = pd.to_datetime(data['Dates'])
data = data.sort_values("Dates",inplace=True)

print(type(data))
print(data)
