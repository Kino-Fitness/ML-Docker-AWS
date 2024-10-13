import os
import boto3
import torch
import shutil

# Model directories and S3 bucket details
model_dir = '/code/files/vbc'
measurements_model_dir = '/code/files/measurements'
bucket_name = 'ml-models-kino'
num_folds = 8

# Safely remove the directories if they exist and recreate them
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)

if os.path.exists(measurements_model_dir):
    shutil.rmtree(measurements_model_dir)
os.makedirs(measurements_model_dir, exist_ok=True)

# Initialize the S3 client without explicit AWS credentials
s3 = boto3.client('s3',
    aws_access_key_id="AKIATIXKZ4OKCOYIXSTC",
    aws_secret_access_key="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"
)

# Download the model files for each fold from S3
for i in range(num_folds):
    model_path = os.path.join(model_dir, f'model_fold_{i}.pt')
    file_key = f'vbc/model_fold_{i}.pt'
    
    try:
        print(f"Downloading {file_key} to {model_path}")
        s3.download_file(bucket_name, file_key, model_path)
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")

# Download the .keras model file from S3
model_path = os.path.join(measurements_model_dir, 'model.keras')
file_key = 'contralateral-measurements/model.keras'

try:
    print(f"Downloading {file_key} to {model_path}")
    s3.download_file(bucket_name, file_key, model_path)
except Exception as e:
    print(f"Error downloading {file_key}: {e}")

# Download the weights file from S3
weights_path = os.path.join(measurements_model_dir, 'model.weights.h5')
file_key = 'contralateral-measurements/model.weights.h5'

try:
    print(f"Downloading {file_key} to {weights_path}")
    s3.download_file(bucket_name, file_key, weights_path)
except Exception as e:
    print(f"Error downloading {file_key}: {e}")
