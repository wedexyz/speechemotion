from scipy.io.wavfile import write
import wavio as wv
from playsound import playsound
import sounddevice as sd
import time
playsound('wav/recording progress.mp3')

file = "wav/recording0.wav"
waktu = 3
freq = 39000
duration = waktu
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
sd.wait()
wv.write(file, recording, freq, sampwidth=2)

playsound(file)
