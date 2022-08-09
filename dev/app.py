#app.py

from dash import dash, dcc, html
import pandas as pd
from datetime import datetime


data = pd.read_csv('..\data\CPI_data.csv')
data['Dates'] = pd.to_datetime(data['Dates'])
daat = data.sort_values("Dates",inplace=True)

app = dash(__name__)

