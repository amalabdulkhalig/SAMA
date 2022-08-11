import io
import dash
import time
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import base64
import tensorflow as tf
from matplotlib import image

from glob import glob
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from matplotlib import pyplot
import time


model = keras.models.load_model('model')

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def load_and_preprocess(image):
   image1 = Image.open(image)
   rgb =  Image.new('RGB', image1.size)
   rgb.paste(image1)
   image = rgb
   test_image = image.resize((32,32,3))
   
   return test_image

def np_array_normalise(test_image):
   np_image = np.array(test_image)
   np_image = np_image/255
   final_image = np.expand_dims(np_image, 0)
   return final_image

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'filename'))

def prediction(image):
    final_img = load_and_preprocess(image)
    final_img = np_array_normalise(final_img)
    Y = model.predict(final_img)
    return Y

if __name__ == '__main__':
    app.run_server(debug=True,port=8051)