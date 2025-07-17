import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input  # Impor preprocess_input

app = Flask(__name__)

# Load the trained model
model = load_model('model_selesai.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define labels for predictions
labels = {
    0: 'Penyakit Layu',
    1: 'Daun Sehat',
    2: 'Penyakit Hama Serangga',
    3: 'Penyakit Bercak Daun',
    4: 'Penyakit Virus Mosaik',
    5: 'Penyakit Daun Kecil',
    6: 'Penyakit Jamur Putih'
}

def getResult(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Sesuaikan ukuran dengan model
    x = img_to_array(img)  # Konversi gambar ke array
    x = np.expand_dims(x, axis=0)  # Tambahkan dimensi batch
    x = preprocess_input(x)  # Gunakan preprocess_input dari EfficientNet
    predictions = model.predict(x)[0]  # Dapatkan prediksi
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render halaman index

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # Ambil file yang diunggah

        # Tentukan path untuk menyimpan file yang diunggah
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(upload_path)  # Simpan file

        # Dapatkan prediksi
        predictions = getResult(upload_path)
        predicted_label = labels[np.argmax(predictions)]  # Dapatkan label yang diprediksi
        return str(predicted_label)  # Kembalikan label yang diprediksi sebagai string
    return None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Production ready