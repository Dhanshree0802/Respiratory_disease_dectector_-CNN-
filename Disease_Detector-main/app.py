from flask import Flask, redirect,render_template,request, url_for,flash
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import librosa
import numpy as np
import random
from keras.models import load_model
from pydub import AudioSegment

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your_secret_key_here'
loaded_model = load_model('my_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contributors')
def contributors():
    return render_template('contributors.html')

@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/predict/<prediction>')
def predict(prediction):
    return render_template('predict.html', prediction=prediction)

@app.route('/record',methods=['POST','GET'])
def record():
    if request.method == 'POST':
        if 'recordedFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['recordedFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            random_filename = str(random.randint(1,10000)) + '.wav'
            file.filename=random_filename
            file_path = os.path.join('uploaded_files', file.filename)
            file.save(file_path)
            mfcc_features = process_single_audio(file_path)
            prediction = predict_label(mfcc_features)
            return redirect(url_for('predict', prediction=prediction))
    return render_template('record.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        if 'coughFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['coughFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploaded_files', file.filename)
            file.save(file_path)
            mfcc_features = process_single_audio(file_path)
            prediction = predict_label(mfcc_features)
            return redirect(url_for('predict', prediction=prediction))
    return render_template('upload.html')

def predict_label(mfcc_features):
    input_data = mfcc_features
    input_data = input_data.reshape(1,*input_data.shape)
    prediction = loaded_model.predict(input_data)
    predicted_label = np.argmax(prediction)
    class_labels = ['Bronchitis', 'Pertussis', 'Asthma', 'Pneumonia']
    return class_labels[predicted_label]


def process_single_audio(audio_path, max_amplitude=0.5, n_mfcc=20, hop_length=512, n_fft=2048):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        audio /= np.max(np.abs(audio))
        audio *= max_amplitude
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        padded_mfcc = pad_sequences([mfccs.T],maxlen=19823, padding='post', dtype='float32')[0]
        return padded_mfcc
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None



if __name__ == '__main__':
    app.run(debug=True)