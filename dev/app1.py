#app.py

from distutils.log import debug
from dash import Dash
from dash import html
from dash import  dcc
import pandas as pd
from datetime import datetime


data = pd.read_csv('..\data\CPI_data.csv')
data['Dates'] = pd.to_datetime(data['Dates'])
data = data.sort_values("Dates",inplace=True)

app = Dash(__name__)

app.layout = html.Div(
    children =[
        html.H1(children="CPI-data",
        style = {'textAlign':'center','marginTop':40,'marginBottom':40, 'color':'green'}),
        html.P(
            children="Analyze the CPI rate over months and years in Saudi Arabia",
            style = {'textAlign':'center'}),
            dcc.Graph(
                figure={
                    "data":[
                             {
                            "x":data["Dates"],
                            "y":data["CPIGR"],
                            "type":"lines",
                             },
                            ],
                            "layout":{"title":"average CPIGR"}
                        },      
                      ),
    ]  
)

if __name__=="__main__":
    app.run_server(debug=True)
