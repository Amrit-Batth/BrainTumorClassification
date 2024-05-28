import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Define constants
IMAGE_SIZE = (240, 240)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the model
def load_model():
    base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(2500, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(752, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(inputs=base_model.inputs, outputs=output)
    model.load_weights('vgg19.weights.h5')
    return model

model_03 = load_model()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Model loaded. Check http://127.0.0.1:5000/')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_className(classNo):
    return "No Brain Tumor" if classNo == 0 else "Yes Brain Tumor"

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.expand_dims(image, axis=0)
    return image

def getResult(img_path):
    input_img = preprocess_image(img_path)
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return "Invalid file format"

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
