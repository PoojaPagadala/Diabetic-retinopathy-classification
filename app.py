from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

# Custom activation function
@keras.utils.register_keras_serializable()
def swish(x):
    return x * keras.backend.sigmoid(x)

# Custom dropout layer
class FixedDropout(keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if training:
            mask = keras.backend.random_binomial(keras.backend.shape(inputs), 1 - self.rate, seed=self.seed)
            return inputs * mask
        return inputs

# Custom Concatenate Layer
class CustomConcatenate(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomConcatenate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.keras.backend.concatenate(inputs, axis=-1)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], sum(shape[-1] for shape in input_shapes))

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB file size limit
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model with custom objects
model = None  # Initialize the model variable

def load_model_with_custom_objects(model_path):
    global model
    try:
        print("Loading model...")
        keras.config.enable_unsafe_deserialization()  # Enable unsafe deserialization
        with keras.utils.custom_object_scope({
            'FixedDropout': FixedDropout,
            'swish': swish,
            'CustomConcatenate': CustomConcatenate,
            # Add any additional custom layers here
        }):
            model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Call the function to load the model
# load_model_with_custom_objects('C:/Users/pooja/OneDrive/Desktop/DR web files/backend/model/highest93ri.h5')
load_model_with_custom_objects('C:/Users/pooja/OneDrive/Desktop/DR web files/backend/model/H938.h5')

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('dbrt.html')

@app.route('/dbrt')  # New route definition
def dbrt():
    return render_template('dbrt.html')  # Render the same template

@app.route('/predict_dr', methods=['GET', 'POST'])
def predict_dr():
    result = None
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading.', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                print(f"File saved to: {filepath}")

                result = predict_image(filepath)
                print(f"Prediction result: {result}")
                flash('File uploaded and prediction made successfully!', 'success')
            except Exception as e:
                flash(f'An error occurred while processing the image: {str(e)}', 'danger')
                return redirect(request.url)

        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.', 'danger')
            return redirect(request.url)

    return render_template('pd.html', result=result, filename=filename)

def predict_image(filepath):
    if model is None:
        print("Model is not loaded. Please check model loading logic.")
        return "Model not available for predictions."

    try:
        img = Image.open(filepath).resize((299, 299))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        class_labels = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        print(f"Predicted class index: {predicted_class[0]}")
        return class_labels[predicted_class[0]]
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.route('/dr_stages')
def dr_stages():
    return render_template('d.html')

@app.route('/care_tips')
def care_tips():
    return render_template('ct.html')

@app.route('/about_us')
def about_us():
    return render_template('au.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
