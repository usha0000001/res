from flask import Flask, render_template, request
import numpy as np
import librosa
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model(r'H:\Dysarthia\Dysarthia.h5')

# Define function for feature extraction
def feature_extraction(audio_path):
    try:
        x, sr = librosa.load(audio_path)
        mean_mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128), axis=1)
        return mean_mfcc
    except EOFError:
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get file from the POST request
        file = request.files['file']
        # Save the file to a temporary location
        file_path = 'temp_audio.wav'
        file.save(file_path)
        # Extract features
        features = feature_extraction(file_path)
        X = features.reshape(-1,16,8,1)
        # Predict
        y_pred = model.predict(X)
        y_pred_class = np.round(y_pred)
        # Interpret the prediction
        if y_pred_class == 1:
            prediction = "The audio is classified as dysarthric."
        else:
            prediction = "The audio is classified as non-dysarthric."
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
