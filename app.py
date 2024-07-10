from flask import Flask, request, jsonify, render_template
import os
import tempfile
from PIL import Image
import keras
import numpy as np
import tensorflow as tf

print("Starting Flask app...")

app = Flask(__name__)

print("hello")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file1' not in request.files or 'file2' not in request.files:
        return render_template('index.html', result='Both files need to be uploaded')
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return render_template('index.html', result='Both files need to be selected')
    
    weight = request.form.get('weight')
    height = request.form.get('height')
    gender = request.form.get('gender')
    demographic = request.form.get('demographic')

    if not weight or not height or not gender or not demographic:
        return render_template('index.html', result='All fields (weight, height, gender, demographic) need to be filled')

    if file1 and file2:
        # Define the directory for temporary files
        temp_dir = 'images'
        os.makedirs(temp_dir, exist_ok=True)
        
        file_extension1 = os.path.splitext(file1.filename)[1]
        file_extension2 = os.path.splitext(file2.filename)[1]
        
        temp_file1 = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=file_extension1, delete=False)
        temp_file2 = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=file_extension2, delete=False)
        
        file1.save(temp_file1.name)
        file2.save(temp_file2.name)
        
        print(f"Temporary file created at {temp_file1.name}")
        print(f"Temporary file created at {temp_file2.name}")

        image1 = Image.open(temp_file1.name)
        image2 = Image.open(temp_file2.name)
        results = get_predictions(image1, image2, weight, height, gender, demographic)

        result_text = "Processing complete. Here is your result: " + str(results)
        
        return render_template('index.html', result=result_text)

def preprocess(image1, image2, weight, height, gender, demographic):
    image1 = image1.resize((224, 224)).convert('RGB')
    image2 = image2.resize((224, 224)).convert('RGB')
    
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    tabular_data = np.array([int(weight), int(height), int(gender), int(demographic)])
    
    # Ensure all inputs have a batch dimension
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)
    tabular_data = np.expand_dims(tabular_data, axis=0)

    return image1, image2, tabular_data

def get_predictions(image1, image2, weight, height, gender, demographic):
    model_path = '/code/model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)

    front_image, back_image, tabular_data = preprocess(image1, image2, weight, height, gender, demographic)

    predictions_right_bicep, predictions_left_bicep, predictions_chest, predictions_right_forearm, predictions_left_forearm, predictions_right_quad, predictions_left_quad, predictions_right_calf, predictions_left_calf, predictions_waist, predictions_hips, predictions_body_pose = model.predict([front_image, back_image, tabular_data])
    
    def decode_scalar(vector, output):
        return [np.round(s2g[output].decode(prediction), 1) for prediction in vector]

    predictions = {
        'right_bicep': decode_scalar(predictions_right_bicep, 'right_bicep'),
        'left_bicep': decode_scalar(predictions_left_bicep, 'left_bicep'),
        'chest': decode_scalar(predictions_chest, 'chest'),
        'right_forearm': decode_scalar(predictions_right_forearm, 'right_forearm'),
        'left_forearm': decode_scalar(predictions_left_forearm, 'left_forearm'),
        'right_quad': decode_scalar(predictions_right_quad, 'right_quad'),
        'left_quad': decode_scalar(predictions_left_quad, 'left_quad'),
        'right_calf': decode_scalar(predictions_right_calf, 'right_calf'),
        'left_calf': decode_scalar(predictions_left_calf, 'left_calf'),
        'waist': decode_scalar(predictions_waist, 'waist'),
        'hips': decode_scalar(predictions_hips, 'hips')
    }
    return predictions

class Scalar2Gaussian():
    def __init__(self,min=0.0,max=99.0,sigma=4.0,bins=1000):
        self.min, self.max, self.bins, self.sigma = float(min), float(max), bins, sigma
        self.idxs = np.linspace(self.min,self.max,self.bins)
    def softmax(self, vector):
        e_x = np.exp(vector - np.max(vector))
        return e_x / e_x.sum()

    def code(self,scalar):
        probs = np.exp(-((self.idxs - scalar) / 2*self.sigma)**2)
        probs = probs/probs.sum()
        return probs
  
    def decode(self, vector):
        if np.abs(vector.sum()-1.0) < 1e-3 and np.all(vector>-1e-4):
            # print('Already Probability')
            probs=vector
        else: 
            probs = self.softmax(vector) #make sure vector is not already probabilities
        scalar = np.dot(probs, self.idxs)
        return scalar

    def decode_tensor(self, vector):
        def true_fn():
            return vector

        def false_fn():
            return tf.nn.softmax(vector)

        probs = tf.cond(
        tf.logical_and(
            tf.math.abs(tf.reduce_sum(vector) - 1.0) < 1e-3,
            tf.reduce_all(vector > -1e-4)
        ),
        true_fn,
        false_fn
        )

        scalar = tf.reduce_sum(probs * self.idxs)
        return scalar
  
s2g = {
    'right_bicep': Scalar2Gaussian(min=20.0, max=60.0),
    'left_bicep': Scalar2Gaussian(min=20.0, max=60.0),
    'chest': Scalar2Gaussian(min=60.0, max=170.0),
    'right_forearm': Scalar2Gaussian(min=15.0, max=40.0),
    'left_forearm': Scalar2Gaussian(min=15.0, max=40.0),
    'right_quad': Scalar2Gaussian(min=40.0, max=70.0),
    'left_quad': Scalar2Gaussian(min=40.0, max=70.0),
    'right_calf': Scalar2Gaussian(min=20.0, max=60.0),
    'left_calf': Scalar2Gaussian(min=20.0, max=60.0),
    'waist': Scalar2Gaussian(min=70.0, max=140.0),
    'hips': Scalar2Gaussian(min=80.0, max=110.0)
}

@keras.saving.register_keras_serializable()
def l1_l2_loss(y_true, y_pred, output):
    l1_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)  # Softmax Crossentropy loss
    decoded_y_true = s2g[output].decode_tensor(vector=y_true)
    decoded_y_pred = s2g[output].decode_tensor(vector=y_pred)
    # Ensure decoded outputs have at least one dimension
    decoded_y_true = tf.expand_dims(decoded_y_true, -1)
    decoded_y_pred = tf.expand_dims(decoded_y_pred, -1)
    
    l2_loss = tf.keras.losses.MeanSquaredError()(decoded_y_true, decoded_y_pred)  # MSE loss
    total_loss = l1_loss + l2_loss
    return total_loss

def loss_wrapper(output):
    def loss_fn(y_true, y_pred):
        return l1_l2_loss(y_true, y_pred, output)
    return loss_fn

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
