import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from keras.models import load_model
import numpy as np 
import base64
import cv2
#import image
from PIL import Image
from io import BytesIO
from skimage.transform import resize
import dash_bootstrap_components as dbc


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['dbc.themes.BOOTSTRAP']
model = load_model('model')
classification = ['airplane','autombile','bird','cat','deer','dog','frog','horse','ship','truck']

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = html.Div(
    [
    html.H1('Please upload an image to detect it',
    style = {'textAlign':'center','marginTop':40,'marginBottom':40, 'color':'green'}),
    html.P('I can only see the following classes: airplane, autombile, bird, cat, deer, dog, frog, horse, ship, truck',
    style = {'textAlign':'center','marginTop':40,'marginBottom':40, 'color':'green'}),
    
    #--------------
        dbc.Col(html.Div(dcc.Upload(
        id='upload-image',
        
        children=html.Div([
            html.A('Select Files')
        ],
        style={
            'display': 'inline-block',
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'align':'center'
        },
        
        ),
        # Allow multiple files to be uploaded
        multiple=True
        ),
    ),
               width={"size": 5, "offset": 5},
            
        ),

    #--------------
    
    html.Div(id='output-image-upload',
    style = {'textAlign':'center','marginTop':40,'marginBottom':40, 'color':'green'}),
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    content = base64.b64decode(content_string)
    imgdata = np.array(Image.open(BytesIO(content)))
    #print('imgdata-----------')
    #print(imgdata)
    #print(type(imgdata))

    resized_image = resize(imgdata,(32,32,3))
    #print(resized_image)
    prediction = model.predict(np.array([resized_image]))
    #print(prediction)

    list_index = np.array(range(0,10))
    x = prediction
    for i in range(len(list_index)):
      for j in range(len(list_index)):
        if x[0][list_index[i]]> x[0][list_index[j]]:
          temp = list_index[i]
          list_index[i] = list_index[j]
          list_index[j] = temp
          
    predition = 'This is '+str(round(prediction[0][list_index[0]]*100,2))+'% ' +str(classification[list_index[0]])
    #print(classification,list_index)

    return html.Div([
        html.H5(predition),
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
    app.run_server(debug=True,port=8051)
    