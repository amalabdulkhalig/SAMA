# Image detection Web app 
# Author: Amal Abdulkhalig 

This web app is built to detect objects in images using the deep learning model Convolutional Neural Network (CNN)
TensorFlow.Keras was utilized to build the model.
the web application was built using the Dash framework 
the classes the model can see are :
classes = ['airplane','autombile','bird','cat','deer','dog','frog','horse','ship','truck']

To run the code: 
- go to the 'dev' directory 
- in the terminal type python app.py 

To retrain the model 
- open the imagemodel.ipynb notebook and run the code cells in order 

To save the trained model run this line 
 model.save("model")

* Note: this model utilizes TensorFlow and the dataset used is big, Therefore it needs a GPU to run