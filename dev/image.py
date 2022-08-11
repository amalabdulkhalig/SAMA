import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from keras.models import load_model

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
from tensorflow import keras
model = load_model('..\imagedetection\imagedetection')

app = Dash(__name__)

app.layout = html.Div([
    html.P('Pleae upload an image to detect it',
    style = {'textAlign':'center','marginTop':40,'marginBottom':40, 'color':'green'}
        ),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'alignment':'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload',style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px'}),
])

def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),  
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)
