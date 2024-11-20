from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/skin_lesion_model.h5'
model = load_model(MODEL_PATH)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            prediction_result = predict(filepath)
            return render_template('index.html', 
                                 prediction=prediction_result['diagnosis'],
                                 confidence=prediction_result['confidence'],
                                 filepath=filepath)
    return render_template('index.html')

SIZE = 64  
# Update class labels to match new model's order
class_labels = [
    'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
]

# Add this mapping dictionary after class_labels
disease_names = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

def predict(image_path):
    img = load_img(image_path, target_size=(SIZE, SIZE)) 
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    full_name = disease_names[predicted_class]
    confidence = float(np.max(predictions))
    return {
        'diagnosis': full_name,
        'confidence': f"{confidence:.2%}"
    }

if __name__ == '__main__':
    app.run(debug=True)