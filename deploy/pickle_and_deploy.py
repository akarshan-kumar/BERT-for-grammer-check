# -*- coding: utf-8 -*-
"""Pickle and Deploy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zKvP1XrYgyfJDDPjcu3oxVELCkeYF4M9
"""



import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
# from random import sample
import tensorflow as tf
from tensorflow.keras.models import load_model
# import transformers
from transformers import pipeline,TFAutoModel, AutoTokenizer
import tensorflow_addons as tfa
import warnings
warnings.filterwarnings("ignore")

def load_data(ipdata):
    
        
    
    value1 = ipdata[0]
    value2 = ipdata[1]
    data = [[value1, value2]]
    
    df = pd.DataFrame(data, columns = ['SBE', 'Label'])
    
    return df
    
def load_data2():#for testing
  opstr = ''
    
  flag ='2'# input('for using data enter 1 or for providing input enter 2: ')

  if flag=='1':
    size = int(input('sample size to use: '))
  
    if not(isinstance(size,int) and size>0 and size<1189321):#lenght of df
      raise Exception('sample size must belong in the range 1 to 1189321 and of type int')
  
    data=df[['SBE','Label']].sample(size,replace=False)
    # data=df[['SBE','Label']].iloc[:size]
    return data
  
  elif flag=='2':
    
    def take_input():
      value1 = str(input('input sentence below:\n')).strip()
      value2 = int(input('input Label:'))

      try:
        if isinstance(value1,str) and value2 in [1,2]:
          pass
      except:
        raise Exception(f'{value1} has to be string and {value2} has to be 0 or 1')
    
      return value1,value2
    
    sent = []
    label = []
    flag2= True
    while(flag2):
      sentv,labelv = take_input()
      sent.append(sentv)
      label.append(labelv)
      more_input = input('Enter \'y\' for more input: ')
      if not ('y' in more_input.lower()):
        flag2 = False 
        


    data = [[value1, value2] for value1,value2 in zip(sent,label)]

    df = pd.DataFrame(data, columns = ['SBE', 'Label'])

    return df
    
  else:
    print('1')
    raise Exception(f'your input {flag} is neither 1 or 2.')



def load_pipe(model_name = './pickle/distilbert/model',token_name = './pickle/distilbert/tokenizer'):


  BERT = TFAutoModel.from_pretrained(model_name)

  tokenizer = AutoTokenizer.from_pretrained(token_name)

  pipe = pipeline('feature-extraction', model=BERT, 
                  tokenizer=tokenizer,device=1)
  # BERT.save_pretrained('distilbert/model')
  # tokenizer.save_pretrained('distilbert/tokenizer')
  return pipe




def to_predictor(data,pipe):
  
  if isinstance(data,pd.DataFrame):

    features = np.array(pipe(data['SBE'].to_list()),dtype='object')
    lst = []
    for idx in range(np.shape(features)[0]):
      sent_mean = np.mean(features[idx][0],axis =0)
      lst.append(sent_mean)
    feature_matrix= np.array(lst)
    

    test_labels = [ [0,1] if value==1 else [1,0]for value in data['Label'] ]
    test_labels = np.array(test_labels)
    
   
    return feature_matrix.astype('float32'),test_labels

  else:
    raise Exception('Data not of type pandas.DataFrame')




def get_preds(feature_matrix,test_labels,model, printf1 =False,verbose=False):
  opstr = ''
  

  
  y_pr_ts = model.predict(feature_matrix)[:,1]

  y_ts = test_labels[:,1]

  def predict_with_best_t(proba, threshould=0.718):
      predictions = []
      for i in proba:
          if i>=1-threshould:
              predictions.append(1)
          else:
              predictions.append(0)
      return predictions

  predictions = predict_with_best_t(y_pr_ts)

  if verbose:
    opstr += f'Threshold used for prediction is {1-0.718:.3f}\n'

    opstr += f'Label: {y_ts}, Prediction:{predictions}, Logit: {y_pr_ts:.3f}\n'

  if printf1:
    f1=f1_score(y_ts,predictions )*100
    opstr += f'f1 score: {f1:.2f}\n'
    
  
  # print(1-0.718,y_ts,predictions,y_pr_ts)
  return f'{1-0.718:2f}',y_ts,predictions,y_pr_ts

def model_predict(pipe,model):
  data = load_data2()
  feature_matrix,test_labels = to_predictor(data,pipe)
  get_preds(feature_matrix,test_labels,model)






# =============================================================================
# pipe = load_pipe()
# model = load_model("pickle//BERT5.hdf5")
# 
# model_predict(pipe,model)
# 
# =============================================================================
