import os
import gdown
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for

# === Download model from Google Drive if not present ===
model_path = "best_sound.keras"
file_id = "1RZhj3etVQeKjzxAEj0FpoVd__K3l3ImA"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# === Flask App Setup ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Load Model ===
model = tf.keras.models.load_model(model_path)

# === Label Mapping ===
labels = {
    0: 'bear', 1: 'cat', 2: 'cow', 3: 'dog', 4: 'donkey',
    5: 'elephant', 6: 'horse', 7: 'lion', 8: 'monkey', 9: 'sheep'
}
safe_animals = {'cat', 'cow', 'dog', 'donkey', 'horse', 'sheep'}
unsafe_animals = {'bear', 'elephant', 'lion', 'monkey'}

# === Audio Preprocessing ===
def extract_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y, top_db=20)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=256, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec + 1e-9, ref=np.max)
        mel_spec_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-9)

        mel_spec_padded = np.zeros((128, 174))
        mel_spec_padded[:, :min(174, mel_spec_norm.shape[1])] = mel_spec_norm[:, :174]
        return np.expand_dims(mel_spec_padded, axis=(0, -1))
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return redirect(request.url)

        file = request.files['audio']
        if file.filename == '':
            return redirect(request.url)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        spectrogram = extract_mel_spectrogram(file_path)
        if spectrogram is not None:
            prediction = model.predict(spectrogram)
            predicted_label = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            animal = labels.get(predicted_label, "Unknown")
            is_safe = animal in safe_animals

            # Load corresponding image
            image_url = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = os.path.join("static", "images", animal + ext)
                if os.path.isfile(test_path):
                    image_url = url_for('static', filename=f'images/{animal}{ext}')
                    break

            return render_template('index.html',
                                   prediction=True,
                                   animal=animal.capitalize(),
                                   confidence=f"{confidence:.2f}",
                                   is_safe=is_safe,
                                   image_url=image_url)

    return render_template('index.html', prediction=False)

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
