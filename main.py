
import random
from datetime import datetime
import random, threading, webbrowser
import socket
from flask import Flask, render_template, url_for, request
from flask import Flask, render_template, request, redirect
from flask import Flask, Response, render_template
import numpy as np
import pandas as pd
import time,webbrowser
from fitur import *
import sounddevice as sd
import librosa
from scipy.io.wavfile import write
import wavio as wv

import pandas as pd
import keras
import speech_recognition as sr
import subprocess
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

import multiprocessing
from playsound import playsound
import os
import sounddevice as sd


file = "wav/welcome app, speech emotion, enjoy it.mp3"
os.system("mpg123 " + file)

#model

konf = ({0:'angry', 1:'disgust', 2:'fear',
        3:'happy', 4:'neutral', 5:'sad',
        6:'surprise'})


application = Flask(__name__)
random.seed()  # Initialize the random number generator
msgFromClient       = "200"
bytesToSend         = str.encode(msgFromClient)
serverAddressPort   = ("127.0.0.1", 5000)
bufferSize          = 20000
UDPClientSocket     = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)



'''
from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))'''
model =keras.models.load_model('res2_model (1).h5')

def prediksi(x):
    offset=0.6
    duration=2
    # Sampling frequency
    freq = 39200

    playsound('wav/prediction wait.mp3')
    data, sample_rate = librosa.load(x, sr=freq, duration=duration, offset=offset)
    res1 = extract_features(data, sample_rate)
    res1 = np.array(res1).reshape(1,-1)
    print('res',pd.DataFrame(np.array(res1).reshape(1,-1)))
    res2 = joblib.load('std (1).pkl').transform(res1)
    pred = model.predict( np.expand_dims(res2, axis=2))
    pred1 = np.argmax(pred,axis=1)
    prob= np.max(pred,axis=1)*100
    out = f"{ konf[pred1[0]]} { prob[0] } %"
    print(out)
    joblib.dump(out,'rec.pkl')
    return 

playsound('wav/welcome app, speech emotion, enjoy it.mp3')
app = Flask(__name__ ,
            static_url_path='', 
            static_folder='lib',
            template_folder='template')

@app.route('/lib/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/', methods=["GET","POST"])
def index():
    output = ""
    output2 = joblib.load('rec.pkl')
    if request.method == "POST":
        file = request.files["file"]
        if "file" not in request.files:
            return redirect(request.url)
        if file.filename == "":
            return redirect(request.url)
        if file:
        
            offset=0.6
            duration=3
         
            # Sampling frequency
           
            data, sampling_rate = librosa.load(file,duration=duration, offset=offset)
            
            #librosa.output.write_wav('wav/librosa.wav', data, sampling_rate)
            sd.play(data, sampling_rate)
            print(data.shape)
            d = data.reshape(1,-1)
            m = joblib.load('rf.pkl')
            
            print(m.predict(d))
            '''
            res1 = extract_features(data, sampling_rate )
            res1 = np.array(res1)
        
        
            
            #res1 = extract_features(data, sr=sampling_rate, frame_length=2048, hop_length=512)
            res1 = res1.reshape(1,-1)
            
            #res2 = joblib.load('std.pkl').transform(res1)
            pred = model.predict( np.expand_dims(res1, axis=2))
            print(pred)
            pred1 = np.argmax(pred)
            print(pred1)
            prob= np.max(pred,axis=1)*100
            print(prob)
            out = f"{ konf[pred1]} { prob[0] } %"
    
        output  =out'''
       
        

    return render_template('index.html',output=output,outp=output2)


@app.route('/Record')
def Record():
    print('record')
    file = r"recording.py"
    prog = r"python.exe"
    subprocess.Popen([prog,file])
    time.sleep(5)
    prediksi('wav/recording0.wav')
  
    return "Nothing"






if __name__ == '__main__':
    port = 8001#+ random.randint(0, 999)
    url = "http://127.0.0.1:{0}/".format(port)
    threading.Timer(1.5, lambda: webbrowser.open(url) ).start()
    app.run(host='127.0.0.1',threaded=False,port=8001)