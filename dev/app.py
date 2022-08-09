#app.py

from dash import dash, dcc, html
import pandas as pd


Data = pd.read_excel("data\CA_Consumer_Price_Index_CPI.xlsx")


if __name__ == '__main__':
    print(Data)