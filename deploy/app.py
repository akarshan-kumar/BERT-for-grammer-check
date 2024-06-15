# import numpy as np
from flask import Flask, request, render_template
# import pickle
import pickle_and_deploy
# import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from random import sample
import tensorflow as tf
from tensorflow.keras.models import load_model
# import transformers
from transformers import pipeline,TFAutoModel, AutoTokenizer
import tensorflow_addons as tfa
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    predictor = [x.strip() for x in request.form.values()]
    # return render_template('index.html', wait_text=f'It takes under 15 seconds to suggest. Please wait!')
    data = pickle_and_deploy.load_data(predictor)
    feature_matrix,test_labels = pickle_and_deploy.to_predictor(data,pipe)
    t,l,preds,logit = pickle_and_deploy.get_preds(feature_matrix,test_labels,model)



    return render_template('index.html', prediction_text=f'Threshold: {t},Label:{predictor[1]},Prediciton: {preds}, Logit: {logit}')


if __name__ == "__main__":
    pipe = pickle_and_deploy.load_pipe()
    model = load_model("pickle//BERT5.hdf5")

    app.run(debug=False)