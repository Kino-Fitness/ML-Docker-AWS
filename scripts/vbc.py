import numpy as np
import os
import tensorflow as tf
import boto3
from tensorflow.keras.models import load_model
# from ultralytics import YOLO

model_dir = '/code/files/vbc'

def get_vbc(frontImage, backImage, weight, height, gender, demographic, waist, hips):

    X_front_images = np.array[(preprocess_image(frontImage).astype(np.float32) / 255) - 0.5]
    X_back_images = np.array[(preprocess_image(backImage).astype(np.float32) / 255) - 0.5]
    X_tabular = np.array[float(height), float(weight), float(waist)/float(hips)]

    return ensemble_predict(X_front_images, X_back_images, X_tabular)

def ensemble_predict(X_front, X_back, X_tabular):
    if not os.path.exists(model_dir):
        load_models()

    predictions = []
    weights = []

    for i, model_path in enumerate(model_dir):
        model = tf.keras.models.load_model(model_path)
        predictions_body_fat = model.predict([X_front, X_back, X_tabular]).flatten()
        weighted_pred = predictions_body_fat * weights[i]
        predictions.append(weighted_pred)
        # del model
        # tf.keras.backend.clear_session()
    
    weighted_avg_prediction = np.sum(predictions, axis=0) / sum(weights)
    return np.round(weighted_avg_prediction, 1)

def load_models():

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for i in range(3):
        model_path = os.path.join(model_dir, f'model_fold_{i}.keras')
        
        if not os.path.exists(model_path):
            bucket_name = 'ml-models-kino'
            file_key = f'vbc/model_fold_{i}.keras'
            local_file_name = model_path

            # Download the .keras file from S3
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            )
            s3.download_file(bucket_name, file_key, local_file_name)

        # Load the model
        load_model(model_path)
    
def preprocess_image(iamge):
    try:
        image = image.convert('RGB')
        # model = YOLO("yolov8n.pt")
        # results = model(image, classes=0)  # or specify custom classes
        # boxes = results[0].boxes
        # coords = boxes.xyxy.tolist()[0]
        # image = image.crop(coords)
        # image = image.resize((224, 224))
        image_array = np.array(image)
        return image_array
    except:
        raise ValueError("image was not able to be preprocessed by YOLO")