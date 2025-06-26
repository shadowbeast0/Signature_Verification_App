import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('model/signature_model_siamese.keras')
img_height, img_width = 128, 128

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_height, img_width))
    img = cv2.Canny(img, 20, 220)
    img = img.astype('float32') / 255.0
    return img.reshape(img_height, img_width, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['signature']
        if file:
            filename = secure_filename(f"{name}_registered.png")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    result = None
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['signature']
        if file:
            filename = secure_filename(f"{name}_login.png")
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(test_path)
            registered_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_registered.png")
            if os.path.exists(registered_path):
                img1 = preprocess_image(registered_path)
                img2 = preprocess_image(test_path)
                pred = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])[0][0]
                result = "Verified" if pred >= 0.5 else "Forgery Detected"
            else:
                result = "No registered signature found for this name."
    return render_template('login.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
