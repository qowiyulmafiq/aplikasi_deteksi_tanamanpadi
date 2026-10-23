import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load model (sesuaikan path ke root folder)
MODEL_PATH = 'best_rice_disease_model.h5'
model = load_model(MODEL_PATH)

CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy',
    'Leaf Blast',
    'Leaf Scald',
    'Narrow Brown Spot'
]

TREATMENTS = {
    'Bacterial Leaf Blight': 'Gunakan varietas tahan hawar, jaga drainase sawah, dan kurangi penggunaan pupuk Nitrogen berlebih.',
    'Brown Spot': 'Berikan pemupukan seimbang (NPK & Kalium), tingkatkan kesuburan tanah, dan gunakan benih bersertifikat.',
    'Healthy': 'Tanaman dalam kondisi sehat. Pertahankan pola pemupukan dan irigasi secara teratur.',
    'Leaf Blast': 'Gunakan fungisida berbahan aktif trisiklazol, hindari penanaman terlalu rapat, dan atur pengairan.',
    'Leaf Scald': 'Gunakan benih bebas penyakit, bersihkan gulma di sekitar pematang, dan hindari genangan air berlebih.',
    'Narrow Brown Spot': 'Berikan pupuk Kalium secukupnya dan semprotkan fungisida jika infeksi meluas di fase vegetatif.'
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada berkas yang diunggah.')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Pilih berkas gambar terlebih dahulu.')

    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_batch)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)

        label = CLASS_NAMES[predicted_class_idx]
        treatment = TREATMENTS.get(label, '-')

        encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{encoded_img}"

        return render_template(
            'index.html',
            image_path=image_data_url,
            label=label,
            confidence=f"{confidence:.2f}%",
            treatment=treatment
        )