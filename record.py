# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import librosa
import librosa.display
from fitur import *
from playsound import playsound
import pandas as pd
import matplotlib.pyplot as plt

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from keras.models import Sequential,load_model
import keras
model =keras.models.load_model('res_model.h5', custom_objects={"f1_m": f1_m })
print(model.summary())

def create_waveplot(data, sr):
    plt.figure(figsize=(10, 3))
    #plt.title(f'Waveplot for audio with {e} emotion', size=15)
    #librosa.display.waveplot(data, sr=sr)
    plt.plot(data,)
    plt.show()

def create_spectrogram(data, sr):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    #plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


# Sampling frequency
freq = 39200

# Recording duration
duration = 2

# Start recorder with the given values
# of duration and sample frequency
recording = sd.rec(int(duration * freq),
				samplerate=freq, channels=2)

# Record audio for the given number of seconds
sd.wait()

# This will convert the NumPy array to an audio
# file with the given sampling frequency
write("recording0.wav", freq, recording)
# Convert the NumPy array to audio file
wv.write("recording1.wav", recording, freq, sampwidth=2)


offset=0.6
duration=2

data, sample_rate = librosa.load('recording0.wav', sr=freq, duration=duration, offset=offset)
playsound('recording1.wav')
create_waveplot(data, sample_rate)
create_spectrogram(data,sample_rate)

print("ZCR: ", zcr(data).shape)
print("Energy: ", energy(data).shape)
print("Entropy of Energy :", entropy_of_energy(data).shape)
print("RMS :", rmse(data).shape)
print("Spectral Centroid :", spc(data, sample_rate).shape)
# print("Spectral Entropy: ", spc_entropy(data, sampling_rate).shape)
print("Spectral Flux: ", spc_flux(data).shape)
print("Spectral Rollof: ", spc_rollof(data, sample_rate).shape)
print("Chroma STFT: ", chroma_stft(data, sample_rate).shape)
print("MelSpectrogram: ", mel_spc(data, sample_rate).shape)
print("MFCC: ", mfcc(data, sample_rate).shape)

# without augmentation
res1 = extract_features(data, sample_rate)
result = np.array(res1)
print(res1.shape)
# data with noise
noise_data = noise(data, random=True)
res2 = extract_features(noise_data, sample_rate)
result = np.vstack((result, res2)) # stacking vertically

# data with pitching
pitched_data = pitch(data, sample_rate, random=True)
res3 = extract_features(pitched_data, sample_rate)
result = np.vstack((result, res3)) # stacking vertically

# data with pitching and white_noise
new_data = pitch(data, sample_rate, random=True)
data_noise_pitch = noise(new_data, random=True)
res3 = extract_features(data_noise_pitch, sample_rate)
result = np.vstack((result, res3)) # stacking vertically

print(pd.DataFrame(result))
pred =model.predict( np.expand_dims(result, axis=2))
print(pred)
print(np.argmax(pred))