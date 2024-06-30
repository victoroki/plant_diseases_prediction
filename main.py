import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Ensure the UPLOAD_FOLDER directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

def prepare_image(image_path):
    """Prepare image for model prediction"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # Resize image to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded image
        image_array = prepare_image(filepath)
        
        # Predict with the model
        predictions = model.predict(image_array)
        probability = np.max(predictions) * 100  # Get the probability percentage

        # Clean up the saved file
        os.remove(filepath)
        
        return jsonify({'probability': probability})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
