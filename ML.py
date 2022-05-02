
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
import speech_recognition as sr
import subprocess
import joblib
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


def prediksi(x):
    waktu = 3
    freq = 39000
    offset=0.6
    duration = waktu
    playsound('wav/prediction wait.mp3')
    data, sampling_rate = librosa.load(x,duration=duration, offset=offset)
    d = data.reshape(1,-1)
    m = joblib.load('rf.pkl')
    pred = int(m.predict(d))
    out = f"{ konf[pred]} { '100' } %"
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
            sd.play(data, sampling_rate)
            print(data.shape)
            d = data.reshape(1,-1)
            m = joblib.load('rf.pkl')
            pred = int(m.predict(d))
            print(np.array(int(pred)))        
            out = f"{ konf[pred]} { '100' } %"
        output  =out
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