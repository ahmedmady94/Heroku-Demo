import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import joblib

app = Flask(__name__)
trans1=joblib.load('f1')
trans2=joblib.load('f2')
model = tf.keras.models.load_model('HHV_model_2.h5')

def model_predict(x):

    C= x[0]
    H= x[1]
    A= x[2]
    F= int(x[3])
    vector = [C,H,A,0,0,0,0,0,0,0,0,0,0,0,0]
    vector[F]=1 
    x= vector
    
    a=np.array([x,])
    res= trans1.transform(a)
    y=model.predict(res)
    real_y= np.squeeze(trans2.inverse_transform(y))

    return(round(float(real_y),2))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = int_features
    prediction = model_predict(final_features)
    output = prediction

    return render_template('index.html', prediction_text='The Higher Heating value should be {} Mj/kg'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    
    prediction = model.predict([np.array(list(data.values()))])

   
    output = prediction[0]
    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    

