import os
import tempfile
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import boto3
import keras

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
            probs=vector
        else: 
            probs = self.softmax(vector)
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
    
custom_objects = {
    'l1_l2_loss': l1_l2_loss
}

model_dir = '/code/files/measurements'
model_path = os.path.join(model_dir, 'model.keras')

# Load the model once it's downloaded
model = load_model(model_path, custom_objects=custom_objects)

def preprocess(frontImage, backImage, weight, height, gender, demographic):
    frontImage = frontImage.resize((224, 224)).convert('RGB')
    backImage = backImage.resize((224, 224)).convert('RGB')
    
    frontImage = np.array(frontImage)
    backImage = np.array(backImage)
    
    gender_map = {'male': 0, 'female': 1}
    demographic_map = {"white": 0, "black": 1, "asian": 2, "hispanic": 3}

    tabular_data = np.array([
        int(weight), 
        int(height), 
        gender_map[gender],
        demographic_map[demographic]
    ])
    
    # Ensure all inputs have a batch dimension
    frontImage = np.expand_dims(frontImage, axis=0)
    backImage = np.expand_dims(backImage, axis=0)
    tabular_data = np.expand_dims(tabular_data, axis=0)

    return frontImage, backImage, tabular_data

def get_predictions(frontImage, backImage, weight, height, gender, demographic):
    front_image, back_image, tabular_data = preprocess(frontImage, backImage, weight, height, gender, demographic)

    predictions_right_bicep, predictions_left_bicep, predictions_chest, predictions_right_forearm, predictions_left_forearm, predictions_right_quad, predictions_left_quad, predictions_right_calf, predictions_left_calf, predictions_waist, predictions_hips, predictions_body_pose = model.predict([front_image, back_image, tabular_data])
    
    def decode_scalar(vector, output):
        return np.round(float(s2g[output].decode(vector)), 2) 
    
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
